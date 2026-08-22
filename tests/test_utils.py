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
