"""知识查询节点 — 从知识库检索历史报告和关键事实"""

from research_buddy.knowledge.store import get_knowledge_store
from research_buddy.state import ResearchState


def knowledge_lookup(state: ResearchState) -> dict:
    """知识查询节点：检索历史知识，为增量研究提供上下文

    如果研究问题关联了某个 topic_id，则查询该主题的历史知识。
    否则，在全局知识库中做语义检索。
    """
    question = state["question"]
    topic_id = state.get("topic_id", "")

    store = get_knowledge_store()

    # 查询知识库
    result = store.lookup(question, topic_id=topic_id or None)

    msgs = []

    if result["has_knowledge"]:
        chunks_count = len(result["chunks"])
        facts_count = len(result["facts"])
        latest = result.get("latest_report")

        msgs.append(f"📚 找到历史知识：{chunks_count} 个相关片段，{facts_count} 条关键事实")

        if latest:
            msgs.append(f"   最新报告：{latest['question']}（{latest['created_at']}）")
            # 追溯增量报告链：如果最新报告是基于另一份报告的增量，
            # 也把 parent 报告摘要加入上下文，让 planner 看到更完整的历史
            parent_summary = _get_parent_chain_summary(latest, store)
            if parent_summary:
                msgs.append(f"   基于历史报告：{parent_summary[:60]}")

        # 构建知识上下文
        knowledge_context = _build_context(result, topic_id, store)
    else:
        msgs.append("📚 未找到历史知识，将进行全新研究")
        knowledge_context = ""

    print(f"📚 知识查询完成：{'有历史知识' if result['has_knowledge'] else '全新研究'}")

    return {
        "knowledge_context": knowledge_context,
        "has_knowledge": result["has_knowledge"],
        "known_source_urls": _extract_source_urls(result),
        "messages": msgs,
    }


def _build_context(lookup_result: dict, topic_id: str, store) -> str:
    """将知识查询结果构建为 LLM 可读的上下文文本

    整合三部分信息：
    1. 主题级摘要（来自 get_knowledge_summary，含完整最新报告前 500 字）
    2. 增量报告链（追溯 parent_report_id，展示历史演变）
    3. 关键事实（精炼，来自向量检索）
    4. 相关报告片段（来自向量检索，截断 500 字保留更多上下文）
    """
    parts = []

    # 1. 主题级摘要（最完整的信息源，取代原来只取 300 字的简陋摘要）
    if topic_id:
        summary = store.get_knowledge_summary(topic_id)
        if summary and summary != "该主题暂无历史研究记录。":
            parts.append(summary)
            parts.append("")

    # 2. 增量报告链（追溯 parent，让 planner 看到研究演变过程）
    latest = lookup_result.get("latest_report")
    if latest and latest.get("parent_report_id"):
        parent_chain = _get_parent_chain(latest, store)
        if parent_chain:
            parts.append("### 历史研究演变")
            for i, report_info in enumerate(parent_chain):
                marker = "（增量）" if report_info.get("is_incremental") else "（首次）"
                parts.append(f"{'  ' * i}→ {report_info['question']} {marker}（{report_info['created_at']}）")
            parts.append("")

    # 3. 关键事实（向量检索，可能和摘要中有重叠但角度不同）
    facts = lookup_result.get("facts", [])
    if facts:
        parts.append("### 语义相关的关键事实")
        for f in facts[:10]:
            parts.append(f"- {f['text']}")
        parts.append("")

    # 4. 相关报告片段（向量检索，500 字保留更多上下文）
    chunks = lookup_result.get("chunks", [])
    if chunks:
        parts.append("### 语义相关的历史报告片段")
        for c in chunks[:5]:
            parts.append(f"> {c['text'][:500]}")
        parts.append("")

    return "\n".join(parts) if parts else ""


def _get_parent_chain(latest_report: dict, store, max_depth: int = 3) -> list[dict]:
    """追溯增量报告链，返回从最早到最新的报告信息列表

    例如：[首次研究, 增量1, 增量2(=latest)]
    """
    chain = []
    current = latest_report
    seen_ids = set()

    for _ in range(max_depth):
        report_id = current.get("id", "")
        if report_id in seen_ids:
            break
        seen_ids.add(report_id)

        chain.append({
            "question": current.get("question", ""),
            "is_incremental": current.get("is_incremental", False),
            "created_at": current.get("created_at", ""),
            "confidence": current.get("confidence", ""),
        })

        parent_id = current.get("parent_report_id", "")
        if not parent_id:
            break

        parent = store.db.get_report(parent_id)
        if not parent:
            break
        current = parent

    # 反转：从最早到最新
    chain.reverse()
    return chain


def _get_parent_chain_summary(latest_report: dict, store) -> str:
    """获取增量报告链的一句话摘要"""
    chain = _get_parent_chain(latest_report, store)
    if len(chain) <= 1:
        return ""
    return f"共 {len(chain)} 份报告，从「{chain[0]['question'][:30]}」演变至今"


def _extract_source_urls(lookup_result: dict) -> list[str]:
    """从查询结果中提取已有知识的来源 URL（用于增量搜索去重）"""
    urls = []
    latest = lookup_result.get("latest_report")
    if latest and isinstance(latest, dict):
        for src in latest.get("sources", []):
            if isinstance(src, dict):
                url = src.get("url", "")
                if url:
                    urls.append(url)
    return urls
