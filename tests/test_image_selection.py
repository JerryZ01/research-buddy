"""视觉模型选图测试 — 功能开关、下载、选图校验、降级链。"""

import json
from importlib import import_module

import pytest

images_module = import_module("research_buddy.tools.images")

_FAKE_IMG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


class _FakeResp:
    def __init__(self, content: bytes | None = None, headers: dict | None = None,
                 json_body: dict | None = None, status_code: int = 200):
        self.content = content or b""
        self.headers = headers or {}
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise images_module.httpx.HTTPStatusError(
                f"{self.status_code} error", request=None, response=self,
            )

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


# ── 自动降级重试 ─────────────────────────────────────

def _http_status_error(status: int) -> images_module.httpx.HTTPStatusError:
    request = images_module.httpx.Request("POST", "http://vision.example")
    response = images_module.httpx.Response(status, request=request)
    return images_module.httpx.HTTPStatusError(
        f"{status} error", request=request, response=response,
    )


def test_vision_400_halves_batch_and_retries(monkeypatch):
    """请求过大（400/413）→ 减半图片批次重试，而不是直接放弃该子问题。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())

    calls = []

    def flaky_post(url, json=None, **kwargs):
        calls.append(len([c for c in json["messages"][0]["content"]
                          if c["type"] == "image_url"]))
        if len(calls) == 1:
            raise _http_status_error(400)
        return _FakeResp(json_body=_vision_reply([(1, "重试后的图")]))

    monkeypatch.setattr(images_module.httpx, "post", flaky_post)
    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate(f"https://img.example/{i}.jpg", "sq_01") for i in range(4)],
    )
    assert len(calls) == 2
    assert calls[0] == 4 and calls[1] == 2  # 批次从 4 减半到 2
    # 减半后的批次 = [0.jpg, 1.jpg]，index 1 → 0.jpg
    assert [p["url"] for p in picked] == ["https://img.example/0.jpg"]


def test_truncated_json_retries_with_larger_max_tokens(monkeypatch):
    """输出 JSON 被截断 → 加大 max_tokens 重试一次。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())

    calls = []

    def flaky_post(url, json=None, **kwargs):
        calls.append(json.get("max_tokens"))
        if len(calls) == 1:
            # 截断的 JSON：Unterminated string
            return _FakeResp(json_body={"choices": [{"message": {"content": '{"images": [{"index": 1, "alt": "未完成'}}]})
        return _FakeResp(json_body=_vision_reply([(1, "完整输出")]))

    monkeypatch.setattr(images_module.httpx, "post", flaky_post)
    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate("https://img.example/a.jpg", "sq_01")],
    )
    assert calls == [1024, 2048]
    assert [p["url"] for p in picked] == ["https://img.example/a.jpg"]


# ── 插图质量过滤（尺寸/宽高比） ────────────────────────

def _png_bytes(w: int, h: int) -> bytes:
    import struct as _s
    return (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR"
            + _s.pack(">II", w, h) + b"\x00" * 8)


class _SizeClient:
    """按 URL 返回不同尺寸 PNG 的假客户端。"""

    def __init__(self, size_for):
        self.size_for = size_for

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **_kw):
        size = self.size_for(url)
        return _FakeResp(content=_png_bytes(*size), headers={"content-type": "image/png"})


def test_image_size_parses_png():
    assert images_module._image_size(_png_bytes(800, 500)) == (800, 500)
    # 非图片字节（如 HTML）返回 None，不拦截
    assert images_module._image_size(b"<!DOCTYPE html>...") is None


def test_tiny_and_extreme_ratio_images_rejected():
    client = _SizeClient(lambda url: (200, 200) if "tiny" in url else (1200, 100))
    assert images_module._download_image(client, "https://img.example/tiny.png") is None
    assert images_module._download_image(client, "https://img.example/banner.png") is None


def test_ok_size_image_passes():
    client = _SizeClient(lambda url: (800, 500))
    result = images_module._download_image(client, "https://img.example/ok.png")
    assert result is not None
    assert result[1] == "image/png"


def test_global_image_cap_limits_total(monkeypatch):
    """供图总数不得超过 MAX_TOTAL_IMAGES（供图池有上限，文章模型在池内按需配图）。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())
    calls = {"n": 0}

    def fake_post(url, json=None, **kw):
        calls["n"] += 1
        return _FakeResp(json_body=_vision_reply([(1, "图1"), (2, "图2"), (3, "图3")]))

    monkeypatch.setattr(images_module.httpx, "post", fake_post)
    sub_questions = [{"id": f"sq_0{i}", "question": f"问题{i}"} for i in range(1, 6)]
    candidates = [
        _candidate(f"https://img.example/s{i}/a{j}.jpg", f"sq_0{i}")
        for i in range(1, 6) for j in range(1, 4)
    ]
    picked = images_module.select_images(sub_questions, candidates)
    assert calls["n"] == 5                      # 5 个子问题各一次视觉调用
    assert len(picked) == images_module.MAX_TOTAL_IMAGES  # 15 选 → 截断到 12
    assert images_module.MAX_TOTAL_IMAGES >= 8  # 供图池要足够文章模型配图


def test_download_retries_on_403(monkeypatch):
    """403 防盗链后用下一套 header 策略重试，成功则返回图片。"""

    class _Flaky403Client:
        def __init__(self):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **_kw):
            self.calls += 1
            if self.calls == 1:
                return _FakeResp(status_code=403, content=b"", headers={})
            return _FakeResp(content=_png_bytes(800, 500),
                             headers={"content-type": "image/png"})

    client = _Flaky403Client()
    result = images_module._download_image(client, "https://img.example/a.png")
    assert result is not None
    assert client.calls == 2  # 第一次 403 → 换 header 重试成功


def test_wide_chart_image_passes_size_filter():
    """宽型图表（576x152）宽度达标即放行，不被高度阈值误杀。"""
    client = _SizeClient(lambda url: (576, 152))
    result = images_module._download_image(client, "https://img.example/chart.png")
    assert result is not None
