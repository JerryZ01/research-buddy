"""变化通知节点 — 检测到重要变化时推送通知"""

from research_buddy.tracking.notifier import get_notifier
from research_buddy.state import ResearchState


def change_notifier(state: ResearchState) -> dict:
    """变化通知节点：检测到重要变化时推送通知

    通知策略：
    - 有 high 级别变化 → 必须通知
    - 有 2+ 条 medium 变化 → 通知
    - 只有 low 变化 → 不通知
    """
    detected_changes = state.get("detected_changes", [])
    topic_id = state.get("topic_id", "")

    if not detected_changes or not topic_id:
        print("📬 通知跳过：无变化或无主题")
        return {"notification_sent": False, "messages": ["📬 无需通知"]}

    # 判断是否需要通知
    should_notify = _should_notify(detected_changes)

    if not should_notify:
        print("📬 通知跳过：变化不显著")
        return {"notification_sent": False, "messages": ["📬 变化不显著，跳过通知"]}

    # 获取主题名称
    from research_buddy.knowledge.store import get_knowledge_store
    store = get_knowledge_store()
    topic = store.get_topic(topic_id)
    topic_name = topic.get("name", "未知主题") if topic else "未知主题"

    # 发送通知
    notifier = get_notifier()
    sent = notifier.send_change_notification(
        topic_name=topic_name,
        topic_id=topic_id,
        changes=detected_changes,
    )

    if sent:
        return {
            "notification_sent": True,
            "messages": [f"📬 通知已发送: {topic_name} ({len(detected_changes)} 项变化)"],
        }
    else:
        return {
            "notification_sent": False,
            "messages": ["📬 通知发送失败或跳过"],
        }


def _should_notify(changes: list[dict]) -> bool:
    """判断是否需要发送通知"""
    if not changes:
        return False

    high_count = sum(1 for c in changes if c.get("significance") == "high")
    medium_count = sum(1 for c in changes if c.get("significance") == "medium")

    # 有高重要性变化 → 通知
    if high_count > 0:
        return True

    # 2+ 条中等重要性变化 → 通知
    if medium_count >= 2:
        return True

    return False
