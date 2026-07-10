"""变化分析节点 — 对比新旧搜索结果，用 LLM 识别语义变化"""

import logging

from research_buddy.knowledge.store import get_knowledge_store
from research_buddy.tracking.diff import DiffAnalyzer
from research_buddy.state import ResearchState
from research_buddy.utils import SIGNIFICANCE_EMOJI, summarize_changes

logger = logging.getLogger(__name__)


def diff_analyzer(state: ResearchState) -> dict:
    """变化分析节点：对比新旧报告，识别语义变化

    工作流程：
    1. 获取该主题的最新报告（通过 KnowledgeStore 门面）
    2. 如果有旧报告，用 DiffAnalyzer 分析差异
    3. 输出结构化的变更列表
    """
    topic_id = state.get("topic_id", "")
    new_report = state.get("report", "")

    if not topic_id or not new_report:
        logger.info("变化分析跳过：缺少报告或主题")
        return {"detected_changes": [], "messages": ["📊 变化分析跳过"]}

    store = get_knowledge_store()

    # 获取上一份报告（通过 KnowledgeStore 门面）
    latest = store.get_latest_report(topic_id)
    if not latest:
        logger.info("首次研究，无需对比")
        return {
            "detected_changes": [],
            "messages": ["📊 首次研究，无需对比"],
        }

    old_report = latest.get("report", "")
    if not old_report:
        logger.info("旧报告为空，无需对比")
        return {
            "detected_changes": [],
            "messages": ["📊 旧报告为空，无需对比"],
        }

    # 执行变化分析
    topic = store.get_topic(topic_id)
    topic_name = topic.get("name", "") if topic else ""

    analyzer = DiffAnalyzer()
    result = analyzer.analyze(old_report, new_report, context=topic_name)

    changes = result.get("changes", [])
    similarity = result.get("similarity", 1.0)

    # 保存变更到数据库
    log = store.create_tracking_log(topic_id, status="completed")
    for change in changes:
        store.create_change(
            tracking_log_id=log["id"],
            change_type=change.get("type", "new_info"),
            description=change.get("description", ""),
            old_content=change.get("old_content", ""),
            new_content=change.get("new_content", ""),
            significance=change.get("significance", "medium"),
        )

    # 更新 tracking_log
    store.update_tracking_log(
        log["id"],
        changes_detected=len(changes),
        change_summary=summarize_changes(changes),
    )

    logger.info("变化分析完成：相似度 %.1f%%，检测到 %d 项变化", similarity * 100, len(changes))

    msgs = [
        f"📊 变化分析：相似度 {similarity:.1%}",
        f"   检测到 {len(changes)} 项变化",
    ]
    for c in changes[:5]:
        sig = SIGNIFICANCE_EMOJI.get(c.get("significance", "medium"), "⚪")
        msgs.append(f"   {sig} {c.get('description', '')[:60]}")

    return {
        "detected_changes": changes,
        "similarity": similarity,
        "tracking_log_id": log["id"],
        "messages": msgs,
    }
