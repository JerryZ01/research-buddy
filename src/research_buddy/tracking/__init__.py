"""追踪层 — 定时调度、变化检测、智能通知

延迟导入 scheduler，避免导入 tracking 包时触发数据库初始化副作用。
"""

from research_buddy.tracking.diff import DiffAnalyzer
from research_buddy.tracking.notifier import Notifier, get_notifier

__all__ = [
    "DiffAnalyzer",
    "Notifier",
    "get_notifier",
    "TrackingScheduler",
    "get_scheduler",
]


def __getattr__(name):
    """延迟导入 scheduler，避免包级导入触发数据库初始化"""
    if name in ("TrackingScheduler", "get_scheduler"):
        from research_buddy.tracking.scheduler import TrackingScheduler, get_scheduler
        if name == "TrackingScheduler":
            return TrackingScheduler
        return get_scheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
