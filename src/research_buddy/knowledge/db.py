"""SQLite 数据层 — 研究主题、报告、追踪记录、变更条目的 CRUD"""

import json
import sqlite3
import threading
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

    @property
    def conn(self) -> sqlite3.Connection:
        """获取当前线程的 SQLite 连接（线程安全）"""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
            self._create_tables(conn)
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

            CREATE INDEX IF NOT EXISTS idx_reports_topic ON reports(topic_id);
            CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at);
            CREATE INDEX IF NOT EXISTS idx_tracking_topic ON tracking_logs(topic_id);
            CREATE INDEX IF NOT EXISTS idx_changes_log ON changes(tracking_log_id);
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
