"""APScheduler 定时调度器 — 按主题配置的 cron 表达式自动触发追踪任务"""

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore

from research_buddy.knowledge.store import get_knowledge_store
from research_buddy.config import DATA_DIR

logger = logging.getLogger(__name__)


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
        logger.info("追踪调度器已启动，共 %d 个追踪任务", len(self._scheduler.get_jobs()))

    def stop(self) -> None:
        """停止调度器"""
        if self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False
            logger.info("追踪调度器已停止")

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
                logger.warning("无效的 cron 表达式: %s", cron_expression)
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
            logger.info("添加追踪任务: 主题 %s, cron: %s", topic_id, cron_expression)
            return True
        except Exception as e:
            logger.warning("添加追踪任务失败: %s", e)
            return False

    def remove_tracking_job(self, topic_id: str) -> bool:
        """移除主题的追踪任务"""
        job_id = f"tracking_{topic_id}"
        try:
            self._scheduler.remove_job(job_id)
            logger.info("移除追踪任务: 主题 %s", topic_id)
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

    使用 create_tracking_graph() 让内置的 diff_analyzer 和 change_notifier
    节点执行变化分析和通知，避免重复实现同一逻辑。

    由于 graph.stream() 是同步阻塞调用，使用 asyncio.to_thread()
    避免阻塞 asyncio 事件循环。
    """
    from research_buddy.graph import create_tracking_graph, get_langfuse_handler
    from research_buddy.knowledge.store import get_knowledge_store

    store = get_knowledge_store()
    topic = store.get_topic(topic_id)
    if not topic:
        logger.warning("追踪任务跳过：主题 %s 不存在", topic_id)
        return

    logger.info("执行追踪: %s (%s)", topic['name'], topic_id)

    # 创建追踪记录
    log = store.create_tracking_log(topic_id, status="running")

    def _do_tracking() -> dict:
        """同步执行追踪（在线程中运行）"""
        # 用追踪关键词搜索最新信息
        keywords = topic.get("tracking_keywords", [])
        if not keywords:
            keywords = [topic["name"]]

        question = f"{topic['name']} 最新动态和变化"

        # 使用 create_tracking_graph() — 内置 diff_analyzer + change_notifier
        graph = create_tracking_graph()
        langfuse_handler = get_langfuse_handler()

        config = {}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        from research_buddy.utils import stream_and_accumulate
        result = stream_and_accumulate(graph, {
            "question": question,
            "topic_id": topic_id,
            "is_incremental": True,
        }, config)

        if langfuse_handler:
            langfuse_handler._langfuse_client.flush()

        return result

    try:
        # 在线程中执行同步的 graph.stream()，避免阻塞事件循环
        result = await asyncio.to_thread(_do_tracking)

        # 更新追踪记录
        changes = result.get("detected_changes", [])
        from research_buddy.utils import summarize_changes
        store.update_tracking_log(
            log["id"],
            status="completed",
            changes_detected=len(changes),
            change_summary=summarize_changes(changes),
            report_id=result.get("saved_report_id", ""),
        )

        logger.info("追踪完成: %s, 检测到 %d 项变化", topic['name'], len(changes))

    except Exception as e:
        store.update_tracking_log(log["id"], status="failed", change_summary=str(e))
        logger.error("追踪失败: %s, 错误: %s", topic['name'], e)


# ── 全局单例 ────────────────────────────────────────────

_scheduler: TrackingScheduler | None = None


def get_scheduler() -> TrackingScheduler:
    """获取全局 TrackingScheduler 实例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TrackingScheduler()
    return _scheduler
