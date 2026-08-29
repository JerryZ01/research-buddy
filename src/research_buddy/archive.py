"""永久文章生成档案；与知识检索库分离，避免未审核稿污染召回。"""

import logging

from research_buddy.config import (
    ARTICLE_EDITOR_ROUNDS,
    ENABLE_ARTICLE_EDITOR,
    ENABLE_LANGUAGE_EDITOR,
    MAX_ARTICLE_TOKENS,
    MAX_IMAGES_IN_ARTICLE,
    MAX_REFLECTION_ROUNDS,
    OPENAI_MODEL,
    REFLECTOR_MODEL,
    WRITER_TEMPERATURE,
)
from research_buddy.knowledge.db import Database, get_db
from research_buddy.utils import normalize_url

logger = logging.getLogger(__name__)


def safe_generation_config() -> dict:
    """返回可持久化配置白名单，绝不记录 API key、base URL 或环境变量全集。"""
    return {
        "max_reflection_rounds": MAX_REFLECTION_ROUNDS,
        "max_article_tokens": MAX_ARTICLE_TOKENS,
        "max_images_in_article": MAX_IMAGES_IN_ARTICLE,
        "writer_temperature": WRITER_TEMPERATURE,
        "language_editor_enabled": ENABLE_LANGUAGE_EDITOR,
        "article_editor_enabled": ENABLE_ARTICLE_EDITOR,
        "article_editor_rounds": ARTICLE_EDITOR_ROUNDS,
    }


def start_generation(question: str, style: str = "", source_type: str = "research",
                     external_id: str = "", topic_id: str = "",
                     database: Database | None = None) -> str:
    """尽力创建 running 档案；档案故障不阻断研究主流程。"""
    try:
        record = (database or get_db()).create_article_generation(
            question=question, style=style, source_type=source_type,
            external_id=external_id, topic_id=topic_id,
        )
        return record["id"]
    except Exception as exc:
        logger.warning("创建文章档案失败，不影响本次研究: %s", exc)
        return ""


def _sources(result: dict) -> list[dict]:
    sources, seen = [], set()
    for item in result.get("source_table", []) or result.get("search_results", []):
        url = str(item.get("url", "")).strip()
        normalized = normalize_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        sources.append({
            "title": str(item.get("title", "")),
            "url": url,
            "source": str(item.get("source", "")),
        })
    return sources


def complete_generation(generation_id: str, result: dict,
                        database: Database | None = None) -> None:
    """用最终 LangGraph state 完成档案，并原样保存各阶段版本。"""
    if not generation_id:
        return
    try:
        (database or get_db()).update_article_generation(
            generation_id,
            status="completed",
            report_id=result.get("saved_report_id", ""),
            report=result.get("report", ""),
            confidence=result.get("confidence", ""),
            sources=_sources(result),
            selected_images=result.get("selected_images", []),
            sub_questions=result.get("sub_questions", []),
            research_notes=result.get("research_notes", []),
            writer_model=OPENAI_MODEL,
            judge_model=REFLECTOR_MODEL,
            config_snapshot=safe_generation_config(),
            reflection_rounds=result.get("reflection_round", 0),
            reflection_score=result.get("reflection_score", 0),
            reflection_pass=result.get("reflection_pass", False),
            reflection_judge_degraded=result.get("reflection_judge_degraded", False),
            stop_reason=result.get("stop_reason", ""),
            best_report_restored=result.get("best_report_restored", False),
            language_edits=result.get("language_edits", []),
            evidence_edits=result.get("evidence_edits", []),
            token_usage=result.get("token_usage", {}),
            article_versions=result.get("article_versions", []),
            error="",
        )
    except Exception as exc:
        logger.warning("完成文章档案失败，不影响报告交付 (id=%s): %s", generation_id, exc)


def fail_generation(generation_id: str, error: str,
                    database: Database | None = None) -> None:
    """保留失败运行，便于分析模型、搜索或流程故障分布。"""
    if not generation_id:
        return
    try:
        (database or get_db()).update_article_generation(
            generation_id, status="error", error=str(error)[:4000],
            writer_model=OPENAI_MODEL, judge_model=REFLECTOR_MODEL,
            config_snapshot=safe_generation_config(),
        )
    except Exception as exc:
        logger.warning("记录文章失败档案失败 (id=%s): %s", generation_id, exc)
