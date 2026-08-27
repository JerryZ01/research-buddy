"""reports 表 research_notes 列迁移与持久化测试（真实 SQLite）。"""

import json

import pytest

from research_buddy.knowledge.db import Database


@pytest.fixture
def db(tmp_path):
    """每个测试独立临时库，避免污染 data/ 下的真实库。"""
    return Database(db_path=str(tmp_path / "test.db"))


def test_create_report_persists_research_notes(db):
    db.create_topic("主题")
    topic = db.list_topics()[0]
    record = db.create_report(
        topic_id=topic["id"],
        question="问题",
        report="文章正文",
        confidence="中",
        research_notes=["语义证据评估不可用，仅做了机械校验"],
        input_tokens=1200,
        output_tokens=3400,
        total_tokens=4600,
    )
    assert record["research_notes"] == ["语义证据评估不可用，仅做了机械校验"]
    assert record["confidence"] == "中"
    assert record["input_tokens"] == 1200
    assert record["output_tokens"] == 3400
    assert record["total_tokens"] == 4600

    fetched = db.get_report(record["id"])
    assert fetched["research_notes"] == ["语义证据评估不可用，仅做了机械校验"]
    assert fetched["total_tokens"] == 4600


def test_research_notes_defaults_to_empty_list(db):
    db.create_topic("主题")
    topic = db.list_topics()[0]
    record = db.create_report(topic_id=topic["id"], question="问题", report="正文")
    assert record["research_notes"] == []
    assert record["total_tokens"] == 0


def test_old_schema_without_research_notes_is_migrated(tmp_path):
    """模拟旧库：建表后手工删掉 research_notes 列，再打开应被迁移补回。"""
    import sqlite3

    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE reports (
            id TEXT PRIMARY KEY,
            topic_id TEXT NOT NULL,
            question TEXT NOT NULL,
            report TEXT NOT NULL DEFAULT '',
            confidence TEXT DEFAULT '',
            sources TEXT DEFAULT '[]',
            search_results_count INTEGER DEFAULT 0,
            reflection_rounds INTEGER DEFAULT 0,
            is_incremental INTEGER DEFAULT 0,
            parent_report_id TEXT DEFAULT '',
            key_facts TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE topics (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            tracking_keywords TEXT DEFAULT '[]',
            tracking_cron TEXT DEFAULT '',
            tracking_enabled INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()

    migrated = Database(db_path=path)
    cols = {row[1] for row in migrated.conn.execute("PRAGMA table_info(reports)").fetchall()}
    for column in ("research_notes", "input_tokens", "output_tokens", "total_tokens"):
        assert column in cols, f"旧库迁移后应补上 {column} 列"

    # 迁移后旧数据可正常读取，新列取默认值
    migrated.conn.execute(
        "INSERT INTO reports (id, topic_id, question) VALUES ('r1', 't1', 'q')"
    )
    migrated.conn.commit()
    row = migrated.get_report("r1")
    assert row["research_notes"] == []
    assert row["total_tokens"] == 0
    assert row["report"] == ""


def test_run_persistence_crud_and_stale_marking(db):
    """研究运行记录：CRUD + 重启残留标记 + 超龄清理。"""
    import time as _t

    db.create_run("run1", "问题1", "tech-blog")
    db.create_run("run2", "问题2", created_at=_t.time() - 3 * 3600)  # 旧 running

    # 更新 done
    db.update_run("run1", "done", result={"report": "x", "confidence": "高"})
    r = db.get_run("run1")
    assert r["status"] == "done"
    assert r["result"]["report"] == "x"

    # SSE 后台收尾复用请求开始时已初始化的连接，不在挂载盘上重跑 DDL。
    db.create_run("run3", "问题3")
    db.update_run_on_connection(db.conn, "run3", "done", result={"report": "y"})
    assert db.get_run("run3")["result"]["report"] == "y"

    # 重启残留：旧 running 标记为 error
    marked = db.mark_stale_runs(max_age_seconds=2 * 3600)
    assert marked >= 1
    assert db.get_run("run2")["status"] == "error"

    # 超龄清理：done/error 被删，running 保留
    db.delete_old_runs(max_age_seconds=0)
    assert db.get_run("run1") is None
    assert db.get_run("run2") is None
    assert db.get_run("run3") is None
