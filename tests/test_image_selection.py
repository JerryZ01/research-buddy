"""视觉模型选图测试 — 功能开关、下载、选图校验、降级链。"""

import json
from importlib import import_module

import pytest

images_module = import_module("research_buddy.tools.images")

_FAKE_IMG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


class _FakeResp:
    def __init__(self, content: bytes | None = None, headers: dict | None = None,
                 json_body: dict | None = None):
        self.content = content or b""
        self.headers = headers or {}
        self._json_body = json_body

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_body or {}


class _FakeClient:
    """模拟 httpx.Client：可配置各 URL 的行为。"""

    def __init__(self, failures: set[str] | None = None, oversized: set[str] | None = None):
        self.failures = failures or set()
        self.oversized = oversized or set()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, **_kwargs):
        if url in self.failures:
            raise images_module.httpx.HTTPError("download failed")
        if url in self.oversized:
            return _FakeResp(content=b"x" * (images_module.MAX_DOWNLOAD_BYTES + 1),
                             headers={"content-type": "image/jpeg"})
        return _FakeResp(content=_FAKE_IMG, headers={"content-type": "image/jpeg"})


def _enable_vision(monkeypatch, model: str = "v4-flash-vision-exp"):
    monkeypatch.setattr(images_module, "VISION_MODEL", model)
    monkeypatch.setattr(images_module, "OPENAI_API_KEY", "sk-test")


def _candidate(url: str, sq: str = "sq_01", query: str = "查询") -> dict:
    return {"url": url, "sub_question_id": sq, "query": query}


def _vision_reply(indexes_alts: list[tuple[int, str]]) -> dict:
    return {"choices": [{"message": {"content": json.dumps(
        {"images": [{"index": i, "alt": a} for i, a in indexes_alts]},
        ensure_ascii=False,
    )}}]}


# ── 功能开关 ──────────────────────────────────────────

def test_disabled_without_vision_model(monkeypatch):
    monkeypatch.setattr(images_module, "VISION_MODEL", "")
    monkeypatch.setattr(images_module, "OPENAI_API_KEY", "sk-test")
    assert images_module.select_images([], []) == []


def test_disabled_without_api_key(monkeypatch):
    monkeypatch.setattr(images_module, "VISION_MODEL", "v4-flash-vision-exp")
    monkeypatch.setattr(images_module, "OPENAI_API_KEY", "")
    assert images_module.select_images([], []) == []


# ── 正常选图流程 ─────────────────────────────────────

def test_select_images_returns_picked_images(monkeypatch):
    _enable_vision(monkeypatch)
    client = _FakeClient()
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: client)

    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return _FakeResp(json_body=_vision_reply([(2, "画面内容示意图")]))

    monkeypatch.setattr(images_module.httpx, "post", fake_post)

    sub_questions = [{"id": "sq_01", "question": "两个模型的定位区别？"}]
    candidates = [
        _candidate("https://img.example/a.jpg", "sq_01"),
        _candidate("https://img.example/b.jpg", "sq_01"),
    ]
    picked = images_module.select_images(sub_questions, candidates)

    assert len(picked) == 1
    assert picked[0]["url"] == "https://img.example/b.jpg"  # index=2 → 第二个候选
    assert picked[0]["alt"] == "画面内容示意图"
    assert picked[0]["sub_question_id"] == "sq_01"
    # 请求参数正确：vision 模型 + base64 data URL
    assert captured["json"]["model"] == "v4-flash-vision-exp"
    image_parts = [c for c in captured["json"]["messages"][0]["content"]
                   if c["type"] == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_out_of_range_and_duplicate_indexes_are_dropped(monkeypatch):
    """幻觉输出（编号越界/重复/非对象）必须被丢弃，而不是带进文章。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())
    monkeypatch.setattr(images_module.httpx, "post", lambda *a, **k: _FakeResp(
        json_body=_vision_reply([(9, "越界"), (1, "有效"), (1, "重复"), (2, "")])))
    sub_questions = [{"id": "sq_01", "question": "问题？"}]
    candidates = [_candidate("https://img.example/a.jpg", "sq_01"),
                  _candidate("https://img.example/b.jpg", "sq_01")]
    picked = images_module.select_images(sub_questions, candidates)
    assert len(picked) == 1
    assert picked[0]["url"] == "https://img.example/a.jpg"
    assert picked[0]["alt"] == "有效"


# ── 降级链 ───────────────────────────────────────────

def test_download_failures_are_skipped(monkeypatch):
    _enable_vision(monkeypatch)
    client = _FakeClient(failures={"https://img.example/bad.jpg"})
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(images_module.httpx, "post", lambda *a, **k: _FakeResp(
        json_body=_vision_reply([(1, "好图")])))
    sub_questions = [{"id": "sq_01", "question": "问题？"}]
    candidates = [_candidate("https://img.example/bad.jpg", "sq_01"),
                  _candidate("https://img.example/good.jpg", "sq_01")]
    picked = images_module.select_images(sub_questions, candidates)
    # 坏图跳过，好图照常送选（index 1 = good.jpg）
    assert [p["url"] for p in picked] == ["https://img.example/good.jpg"]


def test_vision_call_failure_degrades_gracefully(monkeypatch):
    """视觉模型调用抛错 → 该子问题无插图，不向调用方抛异常。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())

    def _boom(*a, **k):
        raise images_module.httpx.HTTPError("vision api down")

    monkeypatch.setattr(images_module.httpx, "post", _boom)
    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate("https://img.example/a.jpg", "sq_01")],
    )
    assert picked == []


def test_oversized_image_is_skipped(monkeypatch):
    _enable_vision(monkeypatch)
    client = _FakeClient(oversized={"https://img.example/huge.jpg"})
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(images_module.httpx, "post", lambda *a, **k: _FakeResp(
        json_body=_vision_reply([(1, "x")])))
    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate("https://img.example/huge.jpg", "sq_01")],
    )
    assert picked == []


def test_no_candidates_returns_empty(monkeypatch):
    _enable_vision(monkeypatch)
    assert images_module.select_images([{"id": "sq_01", "question": "问题？"}], []) == []
    assert images_module.select_images([], []) == []
