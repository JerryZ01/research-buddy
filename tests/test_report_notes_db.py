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
    )
    assert record["research_notes"] == ["语义证据评估不可用，仅做了机械校验"]
    assert record["confidence"] == "中"

    fetched = db.get_report(record["id"])
    assert fetched["research_notes"] == ["语义证据评估不可用，仅做了机械校验"]


def test_research_notes_defaults_to_empty_list(db):
    db.create_topic("主题")
    topic = db.list_topics()[0]
    record = db.create_report(topic_id=topic["id"], question="问题", report="正文")
    assert record["research_notes"] == []


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
    assert "research_notes" in cols

    # 迁移后旧数据可正常读取，新列取默认值
    migrated.conn.execute(
        "INSERT INTO reports (id, topic_id, question) VALUES ('r1', 't1', 'q')"
    )
    migrated.conn.commit()
    row = migrated.get_report("r1")
    assert row["research_notes"] == []
    assert row["report"] == ""
