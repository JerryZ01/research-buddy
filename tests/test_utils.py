"""merge_state_update 单元测试 — 锁定全工作流 state merge 的单一语义来源。

覆盖与 stream_and_accumulate、api.py 三个 SSE 生成器共用的合并规则：
- 覆盖语义键（sub_questions / validation_gaps / evidence_assessments）直接替换；
- 其余 list 键在已存在同键 list 时 extend；
- 标量与首现 list 走覆盖。
"""

from research_buddy.utils import merge_state_update


def test_overwrite_keys_are_replaced():
    """sub_questions / validation_gaps / evidence_assessments 用覆盖语义。"""
    result = {
        "sub_questions": [{"id": "sq_01"}],
        "validation_gaps": [{"a": 1}],
        "evidence_assessments": [{"x": 1}],
    }
    merge_state_update(result, {
        "sub_questions": [{"id": "sq_01"}, {"id": "sq_02"}],
        "validation_gaps": [],
        "evidence_assessments": [{"x": 2}],
    })
    assert result["sub_questions"] == [{"id": "sq_01"}, {"id": "sq_02"}]
    assert result["validation_gaps"] == []
    assert result["evidence_assessments"] == [{"x": 2}]


def test_append_lists_are_extended():
    """非覆盖键的 list 在已存在同键 list 时 extend。"""
    result = {"search_results": [{"url": "a"}], "messages": ["start"]}
    merge_state_update(result, {
        "search_results": [{"url": "b"}],
        "messages": ["planner done"],
    })
    assert result["search_results"] == [{"url": "a"}, {"url": "b"}]
    assert result["messages"] == ["start", "planner done"]


def test_scalar_overwrite():
    """标量字段覆盖。"""
    result = {"report": "old", "reflection_round": 0}
    merge_state_update(result, {"report": "new report", "reflection_round": 1})
    assert result["report"] == "new report"
    assert result["reflection_round"] == 1


def test_first_seen_list_is_assigned():
    """首次出现的 list（result 中尚无该键）落入覆盖分支，而非 extend。"""
    result: dict = {}
    merge_state_update(result, {"search_results": [{"url": "first"}]})
    assert result["search_results"] == [{"url": "first"}]
    assert isinstance(result["search_results"], list)


def test_mixed_update_mirrors_sse_behavior():
    """模拟一个节点返回多类字段：覆盖键替换、追加键 extend、标量覆盖、新键赋值。"""
    result = {
        "sub_questions": [{"id": "sq_01"}],
        "search_results": [{"url": "old"}],
        "reflection_round": 1,
    }
    merge_state_update(result, {
        "sub_questions": [{"id": "sq_01"}, {"id": "sq_02"}],  # overwrite
        "search_results": [{"url": "new"}],                   # extend
        "reflection_round": 2,                                # scalar overwrite
        "report": "generated",                                # first-seen -> assign
    })
    assert result["sub_questions"] == [{"id": "sq_01"}, {"id": "sq_02"}]
    assert result["search_results"] == [{"url": "old"}, {"url": "new"}]
    assert result["reflection_round"] == 2
    assert result["report"] == "generated"


# ── Token 用量统计 ─────────────────────────────────────

def test_track_run_tokens_accumulates_and_resets():
    from research_buddy.utils import track_run_tokens, add_tokens, get_current_token_usage

    # 无追踪上下文时 add_tokens 是空操作
    add_tokens({"prompt_tokens": 99, "completion_tokens": 1, "total_tokens": 100})
    assert get_current_token_usage() == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    with track_run_tokens() as usage:
        add_tokens({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        add_tokens({"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})  # 新命名兼容
        add_tokens(None)  # 空 usage 忽略
        add_tokens({"prompt_tokens": "bad", "completion_tokens": 1})  # 脏数据不炸
        assert usage["total_tokens"] == 20
        assert usage["input_tokens"] == 12
        assert usage["output_tokens"] == 8

    # 上下文退出后计数器复位（contextvar reset）
    assert get_current_token_usage() == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_llm_propagates_config_callbacks(monkeypatch):
    """create_llm 的调用必须传播 config 里的 callbacks（Langfuse 依赖此机制）。"""
    import research_buddy.utils as utils
    from langchain_core.callbacks import BaseCallbackHandler

    events = []

    class _Recorder(BaseCallbackHandler):
        def on_llm_start(self, *args, **kwargs):
            events.append("start")

    # 使用本地不可达端口，验证回调传播但不发起真实模型请求。
    monkeypatch.setattr(utils, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(utils, "OPENAI_API_BASE", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(utils, "OPENAI_STRIP_SDK_HEADERS", False)
    llm = utils.create_llm()
    try:
        # 网络会失败，但 on_llm_start 在发起请求前就会触发
        llm.invoke("hi", config={"callbacks": [_Recorder()]})
    except Exception:
        pass
    assert events == ["start"], "config callbacks 必须传播到 LLM 调用（否则 Langfuse 收不到 Generation）"


def test_invoke_llm_retries_on_transient_error():
    """瞬时错误（503/429/5xx）自动重试，成功返回。"""
    from research_buddy.utils import invoke_llm
    calls = {"n": 0}

    class _Flaky:
        def invoke(self, prompt, config=None):
            calls["n"] += 1
            if calls["n"] == 1:
                err = Exception("503 Service Temporarily Unavailable")
                err.status_code = 503
                raise err
            return "ok"

    assert invoke_llm(_Flaky(), "p") == "ok"
    assert calls["n"] == 2


def test_invoke_llm_does_not_retry_non_transient():
    """非瞬时错误（400）不重试，直接抛。"""
    import pytest
    from research_buddy.utils import invoke_llm
    calls = {"n": 0}

    class _Bad:
        def invoke(self, prompt, config=None):
            calls["n"] += 1
            err = Exception("400 Bad Request")
            err.status_code = 400
            raise err

    with pytest.raises(Exception):
        invoke_llm(_Bad(), "p")
    assert calls["n"] == 1


def test_strip_openai_sdk_headers_removes_gateway_blocked_fingerprint():
    import httpx
    from research_buddy.utils import _strip_openai_sdk_headers

    request = httpx.Request("POST", "https://gateway.example/v1/chat/completions", headers={
        "User-Agent": "OpenAI/Python 2.x",
        "X-Stainless-Lang": "python",
        "X-Stainless-Package-Version": "2.x",
        "X-Stainless-Raw-Response": "true",
        "X-Stainless-Helper-Method": "chat.completions.create",
        "X-Request-ID": "keep-me",
    })
    _strip_openai_sdk_headers(request)
    assert request.headers["user-agent"] == "python-httpx"
    assert "x-stainless-lang" not in request.headers
    assert "x-stainless-package-version" not in request.headers
    assert request.headers["x-stainless-raw-response"] == "true"
    assert request.headers["x-stainless-helper-method"] == "chat.completions.create"
    assert request.headers["x-request-id"] == "keep-me"


def test_create_llm_injects_compat_clients_only_when_enabled(monkeypatch):
    import research_buddy.utils as utils

    captured = []

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def with_config(self, _config):
            return self

    sync_client = object()
    async_client = object()
    monkeypatch.setattr(utils, "ChatOpenAI", _FakeLLM)
    monkeypatch.setattr(utils, "_compat_http_clients", lambda: (sync_client, async_client))

    monkeypatch.setattr(utils, "OPENAI_STRIP_SDK_HEADERS", True)
    utils.create_llm()
    assert captured[-1]["http_client"] is sync_client
    assert captured[-1]["http_async_client"] is async_client

    monkeypatch.setattr(utils, "OPENAI_STRIP_SDK_HEADERS", False)
    utils.create_llm()
    assert "http_client" not in captured[-1]
    assert "http_async_client" not in captured[-1]
