"""SQLite 数据层 — 研究主题、报告、追踪记录、变更条目的 CRUD"""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from research_buddy.config import DATA_DIR


class Database:
    """SQLite 数据库操作

    所有数据存储在 {DATA_DIR}/research_buddy.db 单文件中。
    首次使用时自动建表。

    使用线程本地连接，确保每个线程有自己的 SQLite 连接，
    避免 "SQLite objects created in a thread can only be used
    in that same thread" 错误。
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(Path(DATA_DIR) / "research_buddy.db")
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    @property
    def conn(self) -> sqlite3.Connection:
        """获取当前线程的 SQLite 连接（线程安全）"""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
            # SSE 图在后台线程收尾，每个线程都重跑 WAL + 整套 DDL
            # 会在 Windows 挂载盘上长时间等锁。每进程只需初始化一次 schema。
            if not self._schema_ready:
                with self._schema_lock:
                    if not self._schema_ready:
                        conn.execute("PRAGMA journal_mode=WAL")
                        self._create_tables(conn)
                        self._schema_ready = True
        return conn

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """建表（IF NOT EXISTS 保证幂等）"""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                tracking_keywords TEXT DEFAULT '[]',
                tracking_cron TEXT DEFAULT '',
                tracking_enabled INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                question TEXT NOT NULL,
                report TEXT NOT NULL DEFAULT '',
                confidence TEXT DEFAULT '',
                sources TEXT DEFAULT '[]',
                research_notes TEXT DEFAULT '[]',
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                search_results_count INTEGER DEFAULT 0,
                reflection_rounds INTEGER DEFAULT 0,
                is_incremental INTEGER DEFAULT 0,
                parent_report_id TEXT DEFAULT '',
                key_facts TEXT DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tracking_logs (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                triggered_at TEXT DEFAULT (datetime('now')),
                status TEXT DEFAULT 'running',
                changes_detected INTEGER DEFAULT 0,
                change_summary TEXT DEFAULT '',
                report_id TEXT DEFAULT '',
                FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS changes (
                id TEXT PRIMARY KEY,
                tracking_log_id TEXT NOT NULL,
                change_type TEXT DEFAULT 'new_info',
                description TEXT DEFAULT '',
                old_content TEXT DEFAULT '',
                new_content TEXT DEFAULT '',
                significance TEXT DEFAULT 'medium',
                FOREIGN KEY (tracking_log_id) REFERENCES tracking_logs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'running',
                question TEXT DEFAULT '',
                style TEXT DEFAULT '',
                result TEXT DEFAULT '{}',
                error TEXT DEFAULT '',
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS article_generations (
                id TEXT PRIMARY KEY,
                external_id TEXT DEFAULT '',
                source_type TEXT DEFAULT 'research',
                status TEXT DEFAULT 'running',
                question TEXT NOT NULL,
                style TEXT DEFAULT '',
                topic_id TEXT DEFAULT '',
                report_id TEXT DEFAULT '',
                report TEXT DEFAULT '',
                confidence TEXT DEFAULT '',
                sources TEXT DEFAULT '[]',
                selected_images TEXT DEFAULT '[]',
                sub_questions TEXT DEFAULT '[]',
                research_notes TEXT DEFAULT '[]',
                writer_model TEXT DEFAULT '',
                judge_model TEXT DEFAULT '',
                config_snapshot TEXT DEFAULT '{}',
                reflection_rounds INTEGER DEFAULT 0,
                reflection_score INTEGER DEFAULT 0,
                reflection_pass INTEGER DEFAULT 0,
                reflection_judge_degraded INTEGER DEFAULT 0,
                stop_reason TEXT DEFAULT '',
                best_report_restored INTEGER DEFAULT 0,
                language_edits TEXT DEFAULT '[]',
                evidence_edits TEXT DEFAULT '[]',
                token_usage TEXT DEFAULT '{}',
                error TEXT DEFAULT '',
                curation_status TEXT DEFAULT 'raw',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS article_versions (
                id TEXT PRIMARY KEY,
                generation_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                stage TEXT DEFAULT '',
                reflection_round INTEGER DEFAULT 0,
                report TEXT DEFAULT '',
                feedback TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (generation_id) REFERENCES article_generations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS article_reviews (
                id TEXT PRIMARY KEY,
                generation_id TEXT NOT NULL,
                reviewer_type TEXT DEFAULT 'human',
                reviewer_model TEXT DEFAULT '',
                overall_score REAL,
                dimension_scores TEXT DEFAULT '{}',
                issue_tags TEXT DEFAULT '[]',
                notes TEXT DEFAULT '',
                include_in_evaluation INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (generation_id) REFERENCES article_generations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_reports_topic ON reports(topic_id);
            CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at);
            CREATE INDEX IF NOT EXISTS idx_tracking_topic ON tracking_logs(topic_id);
            CREATE INDEX IF NOT EXISTS idx_changes_log ON changes(tracking_log_id);
            CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_article_generations_external
                ON article_generations(external_id) WHERE external_id != '';
            CREATE INDEX IF NOT EXISTS idx_article_generations_created
                ON article_generations(created_at);
            CREATE INDEX IF NOT EXISTS idx_article_generations_status
                ON article_generations(status, curation_status);
            CREATE INDEX IF NOT EXISTS idx_article_versions_generation
                ON article_versions(generation_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_article_reviews_generation
                ON article_reviews(generation_id, created_at);
        """)
        self._migrate(conn)
        conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """旧库迁移：为已存在的表补上新列（幂等）。

        CREATE TABLE IF NOT EXISTS 不会给已存在的表加列，
        早期版本建的 reports 表缺 research_notes / token 列。
        """
        cols = {row[1] for row in conn.execute("PRAGMA table_info(reports)").fetchall()}
        for column in ("research_notes", "input_tokens", "output_tokens", "total_tokens"):
            if column not in cols:
                default = "TEXT DEFAULT '[]'" if column == "research_notes" else "INTEGER DEFAULT 0"
                conn.execute(f"ALTER TABLE reports ADD COLUMN {column} {default}")

    # ── Topic CRUD ──────────────────────────────────────

    def create_topic(self, name: str, description: str = "",
                     tracking_keywords: list[str] | None = None,
                     tracking_cron: str = "") -> dict:
        """创建研究主题"""
        topic_id = uuid.uuid4().hex[:12]
        keywords = tracking_keywords or []
        self.conn.execute(
            "INSERT INTO topics (id, name, description, tracking_keywords, tracking_cron) VALUES (?, ?, ?, ?, ?)",
            (topic_id, name, description, json.dumps(keywords, ensure_ascii=False), tracking_cron),
        )
        self.conn.commit()
        return self.get_topic(topic_id)

    def get_topic(self, topic_id: str) -> dict | None:
        """获取单个主题"""
        row = self.conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        if not row:
            return None
        return self._row_to_topic(row)

    def list_topics(self) -> list[dict]:
        """列出所有主题"""
        rows = self.conn.execute("SELECT * FROM topics ORDER BY updated_at DESC").fetchall()
        return [self._row_to_topic(r) for r in rows]

    def update_topic(self, topic_id: str, **kwargs) -> dict | None:
        """更新主题字段"""
        allowed = {"name", "description", "tracking_keywords", "tracking_cron", "tracking_enabled"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_topic(topic_id)

        # JSON 序列化
        if "tracking_keywords" in updates and isinstance(updates["tracking_keywords"], list):
            updates["tracking_keywords"] = json.dumps(updates["tracking_keywords"], ensure_ascii=False)

        # 使用 SQLite datetime('now') 保持与 schema 默认值一致（UTC）
        set_parts = [f"{k} = ?" for k in updates]
        set_parts.append("updated_at = datetime('now')")
        set_clause = ", ".join(set_parts)
        values = list(updates.values()) + [topic_id]
        self.conn.execute(f"UPDATE topics SET {set_clause} WHERE id = ?", values)
        self.conn.commit()
        return self.get_topic(topic_id)

    def delete_topic(self, topic_id: str) -> bool:
        """删除主题（级联删除报告、追踪记录）"""
        cursor = self.conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ── Report CRUD ──────────────────────────────────────

    def create_report(self, topic_id: str, question: str, report: str,
                      confidence: str = "", sources: list | None = None,
                      research_notes: list[str] | None = None,
                      input_tokens: int = 0, output_tokens: int = 0,
                      total_tokens: int = 0,
                      search_results_count: int = 0, reflection_rounds: int = 0,
                      is_incremental: bool = False, parent_report_id: str = "",
                      key_facts: list[str] | None = None) -> dict:
        """创建研究报告"""
        report_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            """INSERT INTO reports
               (id, topic_id, question, report, confidence, sources, research_notes,
                input_tokens, output_tokens, total_tokens,
                search_results_count, reflection_rounds, is_incremental,
                parent_report_id, key_facts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (report_id, topic_id, question, report, confidence,
             json.dumps(sources or [], ensure_ascii=False),
             json.dumps(research_notes or [], ensure_ascii=False),
             int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0),
             search_results_count, reflection_rounds,
             1 if is_incremental else 0, parent_report_id,
             json.dumps(key_facts or [], ensure_ascii=False)),
        )
        # 更新 topic 的 updated_at
        self.conn.execute(
            "UPDATE topics SET updated_at = datetime('now') WHERE id = ?",
            (topic_id,),
        )
        self.conn.commit()
        return self.get_report(report_id)

    def get_report(self, report_id: str) -> dict | None:
        """获取单个报告"""
        row = self.conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if not row:
            return None
        return self._row_to_report(row)

    def list_reports(self, topic_id: str, limit: int = 20) -> list[dict]:
        """列出主题下的报告"""
        rows = self.conn.execute(
            "SELECT * FROM reports WHERE topic_id = ? ORDER BY created_at DESC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        return [self._row_to_report(r) for r in rows]

    def get_latest_report(self, topic_id: str) -> dict | None:
        """获取主题下最新的报告"""
        row = self.conn.execute(
            "SELECT * FROM reports WHERE topic_id = ? ORDER BY created_at DESC LIMIT 1",
            (topic_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_report(row)

    def delete_report(self, report_id: str) -> bool:
        """删除报告"""
        cursor = self.conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ── Tracking Log CRUD ───────────────────────────────

    def create_tracking_log(self, topic_id: str, status: str = "running") -> dict:
        """创建追踪记录"""
        log_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO tracking_logs (id, topic_id, status) VALUES (?, ?, ?)",
            (log_id, topic_id, status),
        )
        self.conn.commit()
        return self.get_tracking_log(log_id)

    def get_tracking_log(self, log_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM tracking_logs WHERE id = ?", (log_id,)).fetchone()
        if not row:
            return None
        return dict(row)

    def update_tracking_log(self, log_id: str, **kwargs) -> dict | None:
        allowed = {"status", "changes_detected", "change_summary", "report_id"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_tracking_log(log_id)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [log_id]
        self.conn.execute(f"UPDATE tracking_logs SET {set_clause} WHERE id = ?", values)
        self.conn.commit()
        return self.get_tracking_log(log_id)

    def list_tracking_logs(self, topic_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tracking_logs WHERE topic_id = ? ORDER BY triggered_at DESC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Change CRUD ─────────────────────────────────────

    def create_change(self, tracking_log_id: str, change_type: str,
                      description: str, old_content: str = "",
                      new_content: str = "", significance: str = "medium") -> dict:
        change_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            """INSERT INTO changes
               (id, tracking_log_id, change_type, description, old_content, new_content, significance)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (change_id, tracking_log_id, change_type, description, old_content, new_content, significance),
        )
        self.conn.commit()
        return dict(self.conn.execute("SELECT * FROM changes WHERE id = ?", (change_id,)).fetchone())

    def list_changes(self, tracking_log_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM changes WHERE tracking_log_id = ? ORDER BY significance",
            (tracking_log_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Helpers ──────────────────────────────────────────

    @staticmethod
    def _row_to_topic(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["tracking_keywords"] = json.loads(d.get("tracking_keywords", "[]"))
        d["tracking_enabled"] = bool(d.get("tracking_enabled", 0))
        return d

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["sources"] = json.loads(d.get("sources", "[]"))
        d["research_notes"] = json.loads(d.get("research_notes", "[]"))
        d["key_facts"] = json.loads(d.get("key_facts", "[]"))
        d["is_incremental"] = bool(d.get("is_incremental", 0))
        return d

    # ── 研究运行记录（断线恢复，重启不丢） ────────────────

    def create_run(self, run_id: str, question: str, style: str = "",
                   created_at: float | None = None) -> None:
        """记录一次研究运行（running 状态）。"""
        self.conn.execute(
            "INSERT INTO runs (run_id, status, question, style, created_at) "
            "VALUES (?, 'running', ?, ?, ?)",
            (run_id, question, style, created_at if created_at is not None else time.time()),
        )
        self.conn.commit()

    def update_run(self, run_id: str, status: str,
                   result: dict | None = None, error: str = "") -> None:
        """更新运行状态（done/error）与结果。"""
        self.conn.execute(
            "UPDATE runs SET status=?, result=?, error=? WHERE run_id=?",
            (status, json.dumps(result or {}, ensure_ascii=False), error, run_id),
        )
        self.conn.commit()

    def update_run_on_connection(self, conn: sqlite3.Connection, run_id: str, status: str,
                                 result: dict | None = None, error: str = "") -> None:
        """复用已初始化的连接快速落盘 SSE 结果。"""
        conn.execute("PRAGMA busy_timeout=200")
        conn.execute(
            "UPDATE runs SET status=?, result=?, error=? WHERE run_id=?",
            (status, json.dumps(result or {}, ensure_ascii=False), error, run_id),
        )
        conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        """取运行记录（含解析后的 result）。"""
        row = self.conn.execute(
            "SELECT run_id, status, question, style, result, error, created_at "
            "FROM runs WHERE run_id=?", (run_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["result"] = json.loads(d.get("result") or "{}")
        except json.JSONDecodeError:
            d["result"] = {}
        return d

    def mark_stale_runs(self, max_age_seconds: float = 2 * 3600) -> int:
        """把服务重启后残留的 running 记录标记为 error（运行已中断）。"""
        cutoff = time.time() - max_age_seconds
        cur = self.conn.execute(
            "UPDATE runs SET status='error', error='服务重启，运行已中断' "
            "WHERE status='running' AND created_at < ?", (cutoff,),
        )
        self.conn.commit()
        return cur.rowcount

    def delete_old_runs(self, max_age_seconds: float = 30 * 60) -> int:
        """删除超龄的已完成/错误记录（保留 running）。"""
        cutoff = time.time() - max_age_seconds
        cur = self.conn.execute(
            "DELETE FROM runs WHERE status != 'running' AND created_at < ?", (cutoff,),
        )
        self.conn.commit()
        return cur.rowcount

    # ── 文章素材档案 ────────────────────────────────────

    def create_article_generation(self, question: str, style: str = "",
                                  source_type: str = "research",
                                  external_id: str = "", topic_id: str = "",
                                  generation_id: str | None = None) -> dict:
        """创建永久文章档案；external_id 用于 SSE/HITL 幂等恢复。"""
        generation_id = generation_id or uuid.uuid4().hex[:12]
        self.conn.execute(
            """INSERT INTO article_generations
               (id, external_id, source_type, status, question, style, topic_id)
               VALUES (?, ?, ?, 'running', ?, ?, ?)""",
            (generation_id, external_id, source_type, question, style, topic_id),
        )
        self.conn.commit()
        return self.get_article_generation(generation_id)

    def backfill_reports_to_article_archive(self) -> int:
        """幂等导入历史知识报告；旧数据没有的模型/配置字段保持为空。"""
        rows = self.conn.execute(
            """SELECT id, topic_id, question, report, confidence, sources,
                      research_notes, input_tokens, output_tokens, total_tokens,
                      reflection_rounds, created_at
               FROM reports ORDER BY created_at"""
        ).fetchall()
        imported = 0
        for row in rows:
            external_id = f"report:{row['id']}"
            exists = self.conn.execute(
                "SELECT 1 FROM article_generations WHERE external_id=?", (external_id,),
            ).fetchone()
            if exists:
                continue
            generation_id = uuid.uuid4().hex[:12]
            token_usage = {
                "input_tokens": int(row["input_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
            }
            self.conn.execute(
                """INSERT INTO article_generations
                   (id, external_id, source_type, status, question, topic_id, report_id,
                    report, confidence, sources, research_notes, reflection_rounds,
                    token_usage, curation_status, created_at, updated_at)
                   VALUES (?, ?, 'knowledge_legacy', 'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           'raw', ?, ?)""",
                (generation_id, external_id, row["question"], row["topic_id"], row["id"],
                 row["report"], row["confidence"], row["sources"], row["research_notes"],
                 int(row["reflection_rounds"] or 0),
                 json.dumps(token_usage, ensure_ascii=False), row["created_at"], row["created_at"]),
            )
            self.conn.execute(
                """INSERT INTO article_versions
                   (id, generation_id, sequence, stage, reflection_round, report, metadata,
                    created_at)
                   VALUES (?, ?, 1, 'legacy_final', ?, ?, ?, ?)""",
                (uuid.uuid4().hex[:12], generation_id, int(row["reflection_rounds"] or 0),
                 row["report"], json.dumps({"backfilled": True}), row["created_at"]),
            )
            imported += 1
        self.conn.commit()
        return imported

    def update_article_generation(self, generation_id: str, **kwargs) -> dict | None:
        """更新档案及其版本快照，不接受 schema 之外的任意字段。"""
        allowed = {
            "status", "report_id", "report", "confidence", "sources",
            "selected_images", "sub_questions", "research_notes", "writer_model",
            "judge_model", "config_snapshot", "reflection_rounds", "reflection_score",
            "reflection_pass", "reflection_judge_degraded", "stop_reason",
            "best_report_restored", "language_edits", "evidence_edits", "token_usage",
            "error", "curation_status", "topic_id", "style",
        }
        versions = kwargs.get("article_versions")
        updates = {key: value for key, value in kwargs.items() if key in allowed}
        json_fields = {
            "sources", "selected_images", "sub_questions", "research_notes",
            "config_snapshot", "language_edits", "evidence_edits", "token_usage",
        }
        bool_fields = {"reflection_pass", "reflection_judge_degraded", "best_report_restored"}
        for key in list(updates):
            if key in json_fields:
                empty_value = {} if key in {"config_snapshot", "token_usage"} else []
                updates[key] = json.dumps(
                    updates[key] or empty_value, ensure_ascii=False,
                )
            elif key in bool_fields:
                updates[key] = 1 if updates[key] else 0
        if updates:
            set_clause = ", ".join(f"{key}=?" for key in updates)
            values = list(updates.values()) + [generation_id]
            self.conn.execute(
                f"UPDATE article_generations SET {set_clause}, updated_at=datetime('now') WHERE id=?",
                values,
            )
        if versions is not None:
            self.conn.execute("DELETE FROM article_versions WHERE generation_id=?", (generation_id,))
            for sequence, version in enumerate(versions, 1):
                self.conn.execute(
                    """INSERT INTO article_versions
                       (id, generation_id, sequence, stage, reflection_round, report, feedback, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (uuid.uuid4().hex[:12], generation_id, sequence,
                     str(version.get("stage", "")), int(version.get("reflection_round", 0) or 0),
                     str(version.get("report", "")), str(version.get("feedback", "")),
                     json.dumps(version.get("metadata", {}), ensure_ascii=False)),
                )
        self.conn.commit()
        return self.get_article_generation(generation_id)

    def get_article_generation(self, generation_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM article_generations WHERE id=?", (generation_id,),
        ).fetchone()
        if not row:
            return None
        record = self._row_to_article_generation(row)
        record["versions"] = self.list_article_versions(generation_id)
        record["reviews"] = self.list_article_reviews(generation_id)
        return record

    def get_article_generation_by_external_id(self, external_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id FROM article_generations WHERE external_id=?", (external_id,),
        ).fetchone()
        return self.get_article_generation(row["id"]) if row else None

    def list_article_generations(self, limit: int = 50, offset: int = 0,
                                 status: str = "", curation_status: str = "") -> list[dict]:
        clauses, values = [], []
        if status:
            clauses.append("status=?")
            values.append(status)
        if curation_status:
            clauses.append("curation_status=?")
            values.append(curation_status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.extend([max(1, min(200, int(limit))), max(0, int(offset))])
        rows = self.conn.execute(
            """SELECT id, external_id, source_type, status, question, style,
                      topic_id, report_id, substr(report, 1, 300) AS report_preview,
                      confidence, writer_model, judge_model, reflection_rounds,
                      reflection_score, reflection_pass, reflection_judge_degraded,
                      stop_reason, best_report_restored, error, curation_status,
                      created_at, updated_at
               FROM article_generations""" + where
            + " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?", values,
        ).fetchall()
        records = [dict(row) for row in rows]
        for record in records:
            for key in ("reflection_pass", "reflection_judge_degraded", "best_report_restored"):
                record[key] = bool(record.get(key))
        return records

    def list_article_versions(self, generation_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM article_versions WHERE generation_id=? ORDER BY sequence",
            (generation_id,),
        ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["metadata"] = self._load_json(record.get("metadata"), {})
            records.append(record)
        return records

    def create_article_review(self, generation_id: str, reviewer_type: str = "human",
                              reviewer_model: str = "", overall_score: float | None = None,
                              dimension_scores: dict | None = None,
                              issue_tags: list[str] | None = None, notes: str = "",
                              include_in_evaluation: bool = False) -> dict:
        if not self.get_article_generation(generation_id):
            raise ValueError("文章档案不存在")
        review_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            """INSERT INTO article_reviews
               (id, generation_id, reviewer_type, reviewer_model, overall_score,
                dimension_scores, issue_tags, notes, include_in_evaluation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (review_id, generation_id, reviewer_type, reviewer_model, overall_score,
             json.dumps(dimension_scores or {}, ensure_ascii=False),
             json.dumps(issue_tags or [], ensure_ascii=False), notes,
             1 if include_in_evaluation else 0),
        )
        self.conn.commit()
        return next(item for item in self.list_article_reviews(generation_id) if item["id"] == review_id)

    def list_article_reviews(self, generation_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM article_reviews WHERE generation_id=? ORDER BY created_at, rowid",
            (generation_id,),
        ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["dimension_scores"] = self._load_json(record.get("dimension_scores"), {})
            record["issue_tags"] = self._load_json(record.get("issue_tags"), [])
            record["include_in_evaluation"] = bool(record.get("include_in_evaluation"))
            records.append(record)
        return records

    @staticmethod
    def _load_json(value, default):
        try:
            return json.loads(value or json.dumps(default))
        except (json.JSONDecodeError, TypeError):
            return default

    @classmethod
    def _row_to_article_generation(cls, row: sqlite3.Row) -> dict:
        record = dict(row)
        for key, default in (
            ("sources", []), ("selected_images", []), ("sub_questions", []),
            ("research_notes", []), ("config_snapshot", {}), ("language_edits", []),
            ("evidence_edits", []), ("token_usage", {}),
        ):
            record[key] = cls._load_json(record.get(key), default)
        for key in ("reflection_pass", "reflection_judge_degraded", "best_report_restored"):
            record[key] = bool(record.get(key))
        return record

    def close(self) -> None:
        conn = getattr(self._local, 'conn', None)
        if conn:
            conn.close()
            self._local.conn = None


# ── 全局单例 ────────────────────────────────────────────

_db: Database | None = None


def get_db() -> Database:
    """获取全局 Database 实例（懒初始化）"""
    global _db
    if _db is None:
        _db = Database()
    return _db
