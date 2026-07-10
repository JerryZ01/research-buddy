"""知识存储节点 — 将研究报告保存到知识库"""

import logging

from research_buddy.knowledge.store import get_knowledge_store
from research_buddy.state import ResearchState
from research_buddy.utils import parse_llm_json, create_llm, normalize_url

logger = logging.getLogger(__name__)


def knowledge_store(state: ResearchState) -> dict:
    """知识存储节点：将研究报告保存到知识库

    保存内容：
    - SQLite：报告元数据 + 关键事实
    - ChromaDB：报告文本分块 + 关键事实向量
    """
    question = state.get("question", "")
    report = state.get("report", "")
    topic_id = state.get("topic_id", "")
    is_incremental = state.get("is_incremental", False)

    if not report or not topic_id:
        logger.info("无需存储（缺少报告或 topic_id）")
        return {"messages": ["💾 知识存储跳过：缺少报告或主题"]}

    store = get_knowledge_store()

    # 提取关键事实（从 state 中取）
    key_facts = _extract_key_facts(state)

    # 获取来源信息
    sources = _extract_sources(state)

    # 找到上一份报告 ID（用于增量研究关联）— 通过 KnowledgeStore 门面
    parent_report_id = ""
    if is_incremental:
        latest = store.get_latest_report(topic_id)
        if latest:
            parent_report_id = latest["id"]

    # 保存
    report_record = store.save_report(
        topic_id=topic_id,
        question=question,
        report=report,
        confidence=_extract_confidence(state),
        sources=sources,
        search_results_count=len(state.get("search_results", [])),
        reflection_rounds=state.get("reflection_round", 0),
        is_incremental=is_incremental,
        parent_report_id=parent_report_id,
        key_facts=key_facts,
    )

    report_id = report_record["id"]
    msg = f"💾 研究报告已保存到知识库（ID: {report_id}，{'增量' if is_incremental else '全新'}）"
    logger.info(msg)

    return {
        "saved_report_id": report_id,
        "messages": [msg],
    }


def _extract_key_facts(state: ResearchState) -> list[str]:
    """从研究状态中提取关键事实

    优先级：
    1. state 中已有的 key_facts（reflector 或其他节点设置）
    2. 从报告正文中用 LLM 提取（最精准）
    3. 从搜索结果中截取第一句话（兜底）
    """
    # 1. 如果有显式提取的关键事实
    if state.get("key_facts"):
        return state["key_facts"]

    report = state.get("report", "")

    # 2. 尝试用 LLM 从报告中提取关键事实
    if report and len(report) > 200:
        try:
            llm = create_llm()

            prompt = f"""从以下研究报告中提取 5-8 个最关键的事实要点。
每个要点用一句话概括，只陈述客观事实，不要评价。

研究报告：
{report[:3000]}

请返回 JSON 数组格式（不要包含其他内容）：
```json
["事实1", "事实2", "事实3"]
```"""

            response = llm.invoke(prompt)
            facts = parse_llm_json(response.content)
            if isinstance(facts, list) and facts:
                return [str(f) for f in facts[:8]]
        except Exception:
            logger.debug("LLM 关键事实提取失败，使用兜底策略", exc_info=True)

    # 3. 兜底：从搜索结果中截取第一句话
    facts = []
    for r in state.get("search_results", [])[:5]:
        content = r.get("content", "")
        if content:
            first_sentence = content.split("。")[0].split(". ")[0]
            if first_sentence and len(first_sentence) > 10:
                facts.append(first_sentence.strip())

    return facts[:5]


def _extract_sources(state: ResearchState) -> list[dict]:
    """从搜索结果中提取来源（使用统一的 URL 规范化去重）"""
    sources = []
    seen_urls = set()
    for r in state.get("search_results", []):
        url = r.get("url", "")
        if url:
            normalized = normalize_url(url)
            if normalized not in seen_urls:
                seen_urls.add(normalized)
                sources.append({
                    "title": r.get("title", ""),
                    "url": url,
                })
    return sources


def _extract_confidence(state: ResearchState) -> str:
    """从报告中提取置信度"""
    report = state.get("report", "")
    if "整体置信度：高" in report or "置信度：高" in report:
        return "高"
    elif "整体置信度：中" in report or "置信度：中" in report:
        return "中"
    elif "整体置信度：低" in report or "置信度：低" in report:
        return "低"
    return "中"
