"""APScheduler 定时调度器 — 按主题配置的 cron 表达式自动触发追踪任务"""

import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore

from research_buddy.knowledge.store import get_knowledge_store
from research_buddy.config import DATA_DIR


class TrackingScheduler:
    """追踪调度器

    管理所有启用追踪的主题，按 cron 表达式定期触发追踪任务。
    与 FastAPI lifespan 集成，应用启动时启动调度器，停止时关闭。
    """

    def __init__(self):
        self._scheduler = AsyncIOScheduler(
            jobstores={"default": MemoryJobStore()},
            timezone="Asia/Shanghai",
        )
        self._started = False

    @property
    def scheduler(self) -> AsyncIOScheduler:
        return self._scheduler

    def start(self) -> None:
        """启动调度器，加载所有启用追踪的主题"""
        if self._started:
            return

        # 加载已启用追踪的主题
        store = get_knowledge_store()
        topics = store.list_topics()
        for topic in topics:
            if topic.get("tracking_enabled") and topic.get("tracking_cron"):
                self.add_tracking_job(topic["id"], topic["tracking_cron"])

        self._scheduler.start()
        self._started = True
        print(f"⏰ 追踪调度器已启动，共 {len(self._scheduler.get_jobs())} 个追踪任务")

    def stop(self) -> None:
        """停止调度器"""
        if self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False
            print("⏰ 追踪调度器已停止")

    def add_tracking_job(self, topic_id: str, cron_expression: str) -> bool:
        """为主题添加追踪任务

        Args:
            topic_id: 主题 ID
            cron_expression: cron 表达式，如 "0 9 * * 1-5"（工作日每天9点）

        Returns: 是否添加成功
        """
        try:
            job_id = f"tracking_{topic_id}"
            # 先移除已有的同 ID 任务
            existing = self._scheduler.get_job(job_id)
            if existing:
                self._scheduler.remove_job(job_id)

            # 解析 cron 表达式
            parts = cron_expression.strip().split()
            if len(parts) != 5:
                print(f"⚠️  无效的 cron 表达式: {cron_expression}")
                return False

            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
                timezone="Asia/Shanghai",
            )

            self._scheduler.add_job(
                _run_tracking,
                trigger=trigger,
                id=job_id,
                args=[topic_id],
                replace_existing=True,
                misfire_grace_time=300,
            )
            print(f"⏰ 添加追踪任务: 主题 {topic_id}, cron: {cron_expression}")
            return True
        except Exception as e:
            print(f"⚠️  添加追踪任务失败: {e}")
            return False

    def remove_tracking_job(self, topic_id: str) -> bool:
        """移除主题的追踪任务"""
        job_id = f"tracking_{topic_id}"
        try:
            self._scheduler.remove_job(job_id)
            print(f"⏰ 移除追踪任务: 主题 {topic_id}")
            return True
        except Exception:
            return False

    def list_jobs(self) -> list[dict]:
        """列出所有追踪任务"""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "job_id": job.id,
                "topic_id": job.id.replace("tracking_", ""),
                "next_run": str(job.next_run_time) if job.next_run_time else None,
                "trigger": str(job.trigger),
            })
        return jobs

    def is_started(self) -> bool:
        return self._started


async def _run_tracking(topic_id: str) -> None:
    """执行一次追踪任务（由调度器调用）

    1. 加载主题和最新报告
    2. 执行追踪工作流（搜索 → diff → 通知）
    3. 保存结果
    """
    from research_buddy.graph import create_knowledge_research_graph
    from research_buddy.knowledge.store import get_knowledge_store
    from research_buddy.tracking.diff import DiffAnalyzer
    from research_buddy.tracking.notifier import get_notifier

    store = get_knowledge_store()
    topic = store.get_topic(topic_id)
    if not topic:
        print(f"⏰ 追踪任务跳过：主题 {topic_id} 不存在")
        return

    print(f"\n⏰ 执行追踪: {topic['name']} ({topic_id})")

    # 创建追踪记录
    log = store.create_tracking_log(topic_id, status="running")

    try:
        # 用追踪关键词搜索最新信息
        keywords = topic.get("tracking_keywords", [])
        if not keywords:
            keywords = [topic["name"]]

        question = f"{topic['name']} 最新动态和变化"

        # 执行追踪搜索
        graph = create_knowledge_research_graph()
        result = {}
        for event in graph.stream({
            "question": question,
            "topic_id": topic_id,
            "is_incremental": True,
        }):
            if isinstance(event, dict):
                for node_name, state_update in event.items():
                    if isinstance(state_update, dict):
                        for key, value in state_update.items():
                            if isinstance(value, list) and key in result and isinstance(result[key], list):
                                result[key].extend(value)
                            else:
                                result[key] = value

        # 分析变化
        new_report = result.get("report", "")
        latest = store.db.get_latest_report(topic_id)

        changes = []
        if latest and new_report:
            analyzer = DiffAnalyzer()
            old_report = latest.get("report", "")
            diff_result = analyzer.analyze(old_report, new_report, topic["name"])
            changes = diff_result.get("changes", [])

            # 保存变更条目
            for change in changes:
                store.create_change(
                    tracking_log_id=log["id"],
                    change_type=change.get("type", "new_info"),
                    description=change.get("description", ""),
                    old_content=change.get("old_content", ""),
                    new_content=change.get("new_content", ""),
                    significance=change.get("significance", "medium"),
                )

        # 更新追踪记录
        store.update_tracking_log(
            log["id"],
            status="completed",
            changes_detected=len(changes),
            change_summary=_summarize_changes(changes),
            report_id=result.get("saved_report_id", ""),
        )

        # 如果有重要变化，发送通知
        if changes:
            high_changes = [c for c in changes if c.get("significance") == "high"]
            if high_changes or len(changes) >= 2:
                notifier = get_notifier()
                notifier.send_change_notification(
                    topic_name=topic["name"],
                    topic_id=topic_id,
                    changes=changes,
                )

        print(f"⏰ 追踪完成: {topic['name']}, 检测到 {len(changes)} 项变化")

    except Exception as e:
        store.update_tracking_log(log["id"], status="failed", change_summary=str(e))
        print(f"⏰ 追踪失败: {topic['name']}, 错误: {e}")


def _summarize_changes(changes: list[dict]) -> str:
    """生成变更摘要"""
    if not changes:
        return "无变化"
    parts = []
    for c in changes:
        sig = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(c.get("significance", "medium"), "⚪")
        parts.append(f"{sig} {c.get('description', '')}")
    return "\n".join(parts)


# ── 全局单例 ────────────────────────────────────────────

_scheduler: TrackingScheduler | None = None


def get_scheduler() -> TrackingScheduler:
    """获取全局 TrackingScheduler 实例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TrackingScheduler()
    return _scheduler
