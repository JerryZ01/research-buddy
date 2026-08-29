"""永久文章素材库的 SQLite 契约测试。"""

from research_buddy.archive import complete_generation, fail_generation
from research_buddy.knowledge.db import Database


def test_article_archive_persists_versions_metadata_and_reviews(tmp_path):
    db = Database(db_path=str(tmp_path / "archive.db"))
    created = db.create_article_generation(
        "解释 Agent 反思机制", "tech-blog", "research_stream", external_id="run-1",
    )

    complete_generation(created["id"], {
        "report": "最终稿",
        "confidence": "高",
        "source_table": [{"title": "官方文档", "url": "https://example.com/doc"}],
        "selected_images": [{"url": "https://example.com/image.png", "alt": "流程图"}],
        "sub_questions": [{"id": "sq_01", "question": "机制是什么"}],
        "research_notes": ["资料截至 2026 年"],
        "reflection_round": 2,
        "reflection_score": 13,
        "reflection_pass": True,
        "reflection_judge_degraded": False,
        "stop_reason": "completed",
        "language_edits": [{"original": "旧", "replacement": "新"}],
        "evidence_edits": [],
        "token_usage": {"total_tokens": 1234},
        "article_versions": [
            {"stage": "synthesizer", "reflection_round": 0, "report": "初稿"},
            {"stage": "reflector", "reflection_round": 1, "report": "最终稿",
             "feedback": "通过", "metadata": {"score": 13}},
        ],
    }, database=db)

    record = db.get_article_generation(created["id"])
    assert record["status"] == "completed"
    assert record["report"] == "最终稿"
    assert record["writer_model"]
    assert record["judge_model"]
    assert record["config_snapshot"]["max_reflection_rounds"] >= 1
    assert record["token_usage"]["total_tokens"] == 1234
    assert [version["stage"] for version in record["versions"]] == ["synthesizer", "reflector"]
    assert record["versions"][1]["metadata"]["score"] == 13

    review = db.create_article_review(
        created["id"], overall_score=8.5,
        dimension_scores={"naturalness": 8}, issue_tags=["标题偏多"],
        notes="可作为候选", include_in_evaluation=True,
    )
    assert review["overall_score"] == 8.5
    assert review["dimension_scores"] == {"naturalness": 8}
    assert review["include_in_evaluation"] is True

    updated = db.update_article_generation(created["id"], curation_status="approved")
    assert updated["curation_status"] == "approved"
    assert db.list_article_generations(curation_status="approved")[0]["id"] == created["id"]
    assert db.get_article_generation_by_external_id("run-1")["id"] == created["id"]


def test_failed_generation_is_retained(tmp_path):
    db = Database(db_path=str(tmp_path / "archive.db"))
    created = db.create_article_generation("失败问题")
    fail_generation(created["id"], "provider unavailable", database=db)

    record = db.get_article_generation(created["id"])
    assert record["status"] == "error"
    assert record["error"] == "provider unavailable"
    assert db.list_article_generations(status="error")[0]["id"] == created["id"]


def test_legacy_knowledge_reports_are_backfilled_once(tmp_path):
    db = Database(db_path=str(tmp_path / "archive.db"))
    topic = db.create_topic("历史主题")
    report = db.create_report(
        topic["id"], "历史问题", "历史最终稿", confidence="中",
        sources=[{"title": "来源", "url": "https://example.com"}],
        research_notes=["旧记录"], input_tokens=10, output_tokens=20, total_tokens=30,
        reflection_rounds=2,
    )

    assert db.backfill_reports_to_article_archive() == 1
    assert db.backfill_reports_to_article_archive() == 0
    record = db.get_article_generation_by_external_id(f"report:{report['id']}")
    assert record["source_type"] == "knowledge_legacy"
    assert record["report"] == "历史最终稿"
    assert record["token_usage"]["total_tokens"] == 30
    assert record["versions"][0]["stage"] == "legacy_final"
