"""视觉模型选图测试 — 功能开关、下载、选图校验、降级链。"""

import json
from importlib import import_module

import pytest

images_module = import_module("research_buddy.tools.images")

_FAKE_IMG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


@pytest.fixture(autouse=True)
def _isolated_image_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(images_module, "IMAGE_CACHE_DIR", tmp_path / "research-images")


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


def _vision_reply(indexes_alts: list[tuple[int, str]],
                  watermark_status: str = "none", overlay: str = "") -> dict:
    return {"choices": [{"message": {"content": json.dumps(
        {"images": [{"index": i, "overlay": overlay, "alt": a,
                     "watermark_status": watermark_status}
                    for i, a in indexes_alts]},
        ensure_ascii=False,
    )}}]}


def _verify_reply(overlay_text: str = "") -> dict:
    """二次复核的响应：抄写内容为空串即视为无水印。"""
    return {"choices": [{"message": {"content": json.dumps(
        {"overlay_text": overlay_text}, ensure_ascii=False)}}]}


def _install_vision(monkeypatch, select_body, verify_body=None):
    """同时接管选图调用和水印复核调用，按 prompt 内容分流。

    select_body / verify_body 可以是响应 dict，也可以是 callable(entry) -> dict，
    便于测试重试逻辑。返回每次调用的记录列表。
    """
    calls: list[dict] = []

    def fake_post(url, json=None, **kwargs):
        parts = json["messages"][0]["content"]
        n_images = sum(1 for c in parts if c["type"] == "image_url")
        prompt_text = next(c["text"] for c in parts if c["type"] == "text")
        is_verify = prompt_text.startswith("你是图片水印审查员")
        entry = {
            "kind": "verify" if is_verify else "select",
            "images": n_images,
            "model": json["model"],
            "max_tokens": json.get("max_tokens"),
            "url": url,
            "auth": (kwargs.get("headers") or {}).get("Authorization"),
        }
        calls.append(entry)
        body = verify_body if is_verify else select_body
        if callable(body):
            return _FakeResp(json_body=body(entry))
        if is_verify:
            return _FakeResp(json_body=body or _verify_reply())
        return _FakeResp(json_body=body)

    monkeypatch.setattr(images_module.httpx, "post", fake_post)
    return calls


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

    calls = _install_vision(monkeypatch, _vision_reply([(2, "画面内容示意图")]))

    sub_questions = [{"id": "sq_01", "question": "两个模型的定位区别？"}]
    candidates = [
        _candidate("https://img.example/a.jpg", "sq_01"),
        _candidate("https://img.example/b.jpg", "sq_01"),
    ]
    picked = images_module.select_images(sub_questions, candidates)

    assert len(picked) == 1
    assert picked[0]["url"] == "https://img.example/b.jpg"  # index=2 → 第二个候选
    assert picked[0]["cached_url"].startswith("/media/research-images/")
    assert picked[0]["alt"] == "画面内容示意图"
    assert picked[0]["watermark_status"] == "none"
    assert picked[0]["sub_question_id"] == "sq_01"
    # 选图请求参数正确：vision 模型 + base64 data URL
    select_calls = [c for c in calls if c["kind"] == "select"]
    assert len(select_calls) == 1
    assert select_calls[0]["model"] == "v4-flash-vision-exp"
    assert select_calls[0]["images"] == 2


def test_every_picked_image_gets_an_independent_watermark_verify(monkeypatch):
    """入选图必须逐张单独复核（一图一请求），复核通过才缓存。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())
    calls = _install_vision(monkeypatch, _vision_reply([(1, "图一"), (2, "图二")]))

    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate("https://img.example/a.jpg", "sq_01"),
         _candidate("https://img.example/b.jpg", "sq_01")],
    )

    verify_calls = [c for c in calls if c["kind"] == "verify"]
    assert len(verify_calls) == 2          # 两张入选图各一次复核
    assert all(c["images"] == 1 for c in verify_calls)   # 每次只发一张图
    assert [p["url"] for p in picked] == ["https://img.example/a.jpg",
                                          "https://img.example/b.jpg"]


def test_verify_overlay_text_rejects_image(monkeypatch):
    """选图放行、但独立复核抄出叠加内容 → 丢弃且不缓存。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())
    _install_vision(monkeypatch, _vision_reply([(1, "看着很干净")]),
                    verify_body=_verify_reply("右下角：知乎 @北方的郎"))

    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate("https://img.example/wm.jpg", "sq_01")],
    )

    assert picked == []
    # 未通过复核的图不能进缓存，否则会被后续文章按哈希复用
    cache_dir = images_module.IMAGE_CACHE_DIR
    assert not cache_dir.exists() or not list(cache_dir.iterdir())


@pytest.mark.parametrize("overlay", ["none", "NULL", "无", "没有水印", "无叠加内容"])
def test_verify_negative_phrasings_count_as_clean(overlay):
    """模型把「无」写成各种否定表述时，不能被误判成有水印。"""
    assert images_module._overlay_detected(overlay) is False


@pytest.mark.parametrize("overlay", [
    "知乎 @北方的郎", "右下角：CSDN", "© 2024 SomeMedia", "二维码",
    "中间有半透明斜纹水印：WIRINGO By Gleam Tech",
])
def test_verify_real_overlay_counts_as_watermark(overlay):
    assert images_module._overlay_detected(overlay) is True


def test_verify_fails_closed_when_call_fails(monkeypatch):
    """复核调用重试后仍失败 → 拒绝该图（fail-closed），不因为出错就放行。"""
    _enable_vision(monkeypatch)
    attempts = []

    def boom(*_a, **_k):
        attempts.append(1)
        raise images_module.httpx.HTTPError("verify api down")

    monkeypatch.setattr(images_module.httpx, "post", boom)
    monkeypatch.setattr(images_module.time, "sleep", lambda _s: None)
    assert images_module.verify_watermark(_FAKE_IMG, "image/jpeg") is False
    # 重试由 _vision_post 承担，次数等于复核的尝试次数
    assert len(attempts) == images_module._VERIFY_ATTEMPTS


def test_verify_fails_closed_on_unparsable_output(monkeypatch):
    """复核输出不是约定 JSON → 按有水印处理。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "post", lambda *a, **k: _FakeResp(
        json_body={"choices": [{"message": {"content": "这张图应该是没问题的"}}]}))
    assert images_module.verify_watermark(_FAKE_IMG, "image/jpeg") is False


def test_verify_accepts_missing_key_as_clean(monkeypatch):
    """overlay_text 为空串 → 确认无水印。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "post", lambda *a, **k: _FakeResp(
        json_body=_verify_reply("")))
    assert images_module.verify_watermark(_FAKE_IMG, "image/jpeg") is True


def test_self_contradictory_select_output_is_rejected(monkeypatch):
    """选图判 none 却自报抄出了叠加文字 → 拒绝（水印漏网的典型形态）。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())
    monkeypatch.setattr(images_module.httpx, "post", lambda *a, **k: _FakeResp(
        json_body=_vision_reply([(1, "矛盾输出")], overlay="右下角：知乎 @北方的郎")))

    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate("https://img.example/a.jpg", "sq_01")],
    )
    assert picked == []


def test_out_of_range_and_duplicate_indexes_are_dropped(monkeypatch):
    """幻觉输出（编号越界/重复/非对象）必须被丢弃，而不是带进文章。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())
    _install_vision(monkeypatch, _vision_reply([(9, "越界"), (1, "有效"),
                                                (1, "重复"), (2, "")]))
    sub_questions = [{"id": "sq_01", "question": "问题？"}]
    candidates = [_candidate("https://img.example/a.jpg", "sq_01"),
                  _candidate("https://img.example/b.jpg", "sq_01")]
    picked = images_module.select_images(sub_questions, candidates)
    assert len(picked) == 1
    assert picked[0]["url"] == "https://img.example/a.jpg"
    assert picked[0]["alt"] == "有效"


@pytest.mark.parametrize("status", ["present", "uncertain", ""])
def test_watermarked_uncertain_or_unclassified_images_are_rejected(monkeypatch, status):
    monkeypatch.setattr(images_module, "VISION_MODEL", "vision-model")
    monkeypatch.setattr(images_module, "VISION_API_KEY", "key")
    monkeypatch.setattr(images_module, "_download_image",
                        lambda _client, _url: (_FAKE_IMG, "image/jpeg"))

    if status:
        response = _vision_reply([(1, "带水印候选")], watermark_status=status)
    else:
        response = {"choices": [{"message": {"content": json.dumps({
            "images": [{"index": 1, "alt": "缺少水印判断"}],
        }, ensure_ascii=False)}}]}
    monkeypatch.setattr(images_module.httpx, "post",
                        lambda *_args, **_kwargs: _FakeResp(json_body=response))

    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate("https://img.example/a.jpg")],
    )
    assert picked == []


# ── 降级链 ───────────────────────────────────────────

def test_download_failures_are_skipped(monkeypatch):
    _enable_vision(monkeypatch)
    client = _FakeClient(failures={"https://img.example/bad.jpg"})
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: client)
    _install_vision(monkeypatch, _vision_reply([(1, "好图")]))
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
    _install_vision(monkeypatch, _vision_reply([(1, "x")]))
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

    def flaky_select(entry):
        calls.append(entry["images"])
        if len(calls) == 1:
            raise _http_status_error(400)
        return _vision_reply([(1, "重试后的图")])

    _install_vision(monkeypatch, flaky_select)
    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate(f"https://img.example/{i}.jpg", "sq_01") for i in range(4)],
    )
    assert calls == [4, 2]  # 批次从 4 减半到 2
    # 减半后的批次 = [0.jpg, 1.jpg]，index 1 → 0.jpg
    assert [p["url"] for p in picked] == ["https://img.example/0.jpg"]


def test_truncated_json_retries_with_larger_max_tokens(monkeypatch):
    """输出 JSON 被截断 → 加大 max_tokens 重试一次。"""
    _enable_vision(monkeypatch)
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())
    calls = []

    def flaky_select(entry):
        calls.append(entry["max_tokens"])
        if len(calls) == 1:
            # 截断的 JSON：Unterminated string
            return {"choices": [{"message": {"content": '{"images": [{"index": 1, "alt": "未完成'}}]}
        return _vision_reply([(1, "完整输出")])

    _install_vision(monkeypatch, flaky_select)
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
    calls = _install_vision(
        monkeypatch,
        lambda entry: _vision_reply([(1, "图1"), (2, "图2"), (3, "图3")]),
    )
    sub_questions = [{"id": f"sq_0{i}", "question": f"问题{i}"} for i in range(1, 6)]
    candidates = [
        _candidate(f"https://img.example/s{i}/a{j}.jpg", f"sq_0{i}")
        for i in range(1, 6) for j in range(1, 4)
    ]
    picked = images_module.select_images(sub_questions, candidates)
    select_calls = [c for c in calls if c["kind"] == "select"]
    assert len(select_calls) == 5    # 5 个子问题各一次视觉选图调用
    assert len(picked) == images_module.MAX_TOTAL_IMAGES
    assert images_module.MAX_TOTAL_IMAGES <= 12


def test_cache_image_is_content_addressed(images_module=images_module):
    first = images_module._cache_image(_FAKE_IMG, "image/jpeg")
    second = images_module._cache_image(_FAKE_IMG, "image/jpeg")
    assert first == second
    assert first.endswith(".jpg")
    assert len(list(images_module.IMAGE_CACHE_DIR.iterdir())) == 1


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


def test_vision_uses_separate_credentials(monkeypatch):
    """VISION_API_KEY/BASE 独立配置时，视觉调用走自己的凭据（文本 key 为空也不影响）。"""
    monkeypatch.setattr(images_module, "VISION_MODEL", "vision-model-x")
    monkeypatch.setattr(images_module, "VISION_API_KEY", "sk-vision")
    monkeypatch.setattr(images_module, "VISION_API_BASE", "https://vision.example/v1")
    monkeypatch.setattr(images_module, "OPENAI_API_KEY", "")   # 文本 key 置空
    monkeypatch.setattr(images_module.httpx, "Client", lambda *a, **k: _FakeClient())
    calls = _install_vision(monkeypatch, _vision_reply([(1, "图")]))

    picked = images_module.select_images(
        [{"id": "sq_01", "question": "问题？"}],
        [_candidate("https://img.example/a.jpg", "sq_01")],
    )
    assert picked  # 即使文本 key 为空也能选图
    assert calls, "应至少发生一次视觉调用"
    for call in calls:                      # 选图和复核都必须走视觉凭据
        assert call["url"] == "https://vision.example/v1/chat/completions"
        assert call["auth"] == "Bearer sk-vision"
        assert call["model"] == "vision-model-x"
