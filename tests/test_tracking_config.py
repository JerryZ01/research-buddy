"""追踪配置与调度器的同步测试。

以前 PUT /topics 只写 SQLite，不碰调度器：UI 提示「已保存」，
GET /tracking/jobs 仍是空，要重启进程才生效；关闭追踪则旧任务继续触发。
"""

import asyncio

from research_buddy import api
from research_buddy.tracking.scheduler import TrackingScheduler

_CRON = "0 9 * * *"


def _topic(enabled: bool = True, cron: str = _CRON) -> dict:
    return {"id": "t1", "name": "主题", "tracking_enabled": enabled, "tracking_cron": cron}


def test_sync_registers_job_when_enabled():
    scheduler = TrackingScheduler()
    assert scheduler.sync_tracking_job(_topic())["scheduled"] is True
    assert [job["topic_id"] for job in scheduler.list_jobs()] == ["t1"]


def test_sync_removes_job_when_disabled():
    scheduler = TrackingScheduler()
    scheduler.sync_tracking_job(_topic())
    result = scheduler.sync_tracking_job(_topic(enabled=False))
    assert result == {"scheduled": False, "reason": "disabled"}
    assert scheduler.list_jobs() == []


def test_sync_removes_job_when_cron_cleared():
    scheduler = TrackingScheduler()
    scheduler.sync_tracking_job(_topic())
    assert scheduler.sync_tracking_job(_topic(cron=""))["scheduled"] is False
    assert scheduler.list_jobs() == []


def test_sync_reports_invalid_cron():
    scheduler = TrackingScheduler()
    result = scheduler.sync_tracking_job(_topic(cron="每天早上九点"))
    assert result == {"scheduled": False, "reason": "invalid_cron"}
    assert scheduler.list_jobs() == []


def test_sync_replaces_existing_job_on_cron_change():
    scheduler = TrackingScheduler()
    scheduler.sync_tracking_job(_topic())
    scheduler.sync_tracking_job(_topic(cron="30 18 * * 1-5"))
    jobs = scheduler.list_jobs()
    assert len(jobs) == 1
    assert "18" in jobs[0]["trigger"]


class _FakeScheduler:
    def __init__(self, scheduled: bool = True, reason: str = ""):
        self.synced: list[dict] = []
        self.removed: list[str] = []
        self._result = {"scheduled": scheduled, "reason": reason}

    def sync_tracking_job(self, topic: dict) -> dict:
        self.synced.append(topic)
        return self._result

    def remove_tracking_job(self, topic_id: str) -> bool:
        self.removed.append(topic_id)
        return True


class _FakeStore:
    def __init__(self, topic: dict | None = None):
        self._topic = topic

    def update_topic(self, topic_id: str, **kwargs) -> dict | None:
        if self._topic is None:
            return None
        return {**self._topic, "id": topic_id, **kwargs}

    def create_topic(self, **kwargs) -> dict:
        return {"id": "new", "tracking_enabled": False, **kwargs}

    def delete_topic(self, _topic_id: str) -> bool:
        return True


def _patch(monkeypatch, store, scheduler):
    monkeypatch.setattr(api, "get_knowledge_store", lambda: store)
    monkeypatch.setattr(api, "get_scheduler", lambda: scheduler)


def test_put_topic_syncs_scheduler(monkeypatch):
    scheduler = _FakeScheduler()
    _patch(monkeypatch, _FakeStore(_topic()), scheduler)

    response = asyncio.run(api.update_topic(
        "t1", api.TopicUpdateRequest(tracking_enabled=True, tracking_cron=_CRON),
    ))

    assert [topic["id"] for topic in scheduler.synced] == ["t1"]
    assert response["tracking_scheduled"] is True
    assert "tracking_warning" not in response


def test_put_topic_surfaces_invalid_cron(monkeypatch):
    scheduler = _FakeScheduler(scheduled=False, reason="invalid_cron")
    _patch(monkeypatch, _FakeStore(_topic(cron="bogus")), scheduler)

    response = asyncio.run(api.update_topic(
        "t1", api.TopicUpdateRequest(tracking_enabled=True, tracking_cron="bogus"),
    ))

    assert response["tracking_scheduled"] is False
    assert "无法解析" in response["tracking_warning"]


def test_put_missing_topic_does_not_touch_scheduler(monkeypatch):
    scheduler = _FakeScheduler()
    _patch(monkeypatch, _FakeStore(None), scheduler)

    response = asyncio.run(api.update_topic("nope", api.TopicUpdateRequest(tracking_enabled=True)))

    assert response.status_code == 404
    assert scheduler.synced == []


def test_delete_topic_removes_job(monkeypatch):
    scheduler = _FakeScheduler()
    _patch(monkeypatch, _FakeStore(_topic()), scheduler)

    asyncio.run(api.delete_topic("t1"))

    assert scheduler.removed == ["t1"]
