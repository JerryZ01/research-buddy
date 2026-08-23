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


def test_llm_propagates_config_callbacks():
    """create_llm 的调用必须传播 config 里的 callbacks（Langfuse 依赖此机制）。"""
    from research_buddy.utils import create_llm
    from langchain_core.callbacks import BaseCallbackHandler

    events = []

    class _Recorder(BaseCallbackHandler):
        def on_llm_start(self, *args, **kwargs):
            events.append("start")

    llm = create_llm()
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
