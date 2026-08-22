"""报告反思失败策略和代码侧通过条件测试。"""

import json
from importlib import import_module

reflector_module = import_module("research_buddy.nodes.reflector")


class _Response:
    def __init__(self, content: str):
        self.content = content


class _LLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _prompt):
        return _Response(self.content)


def _state(report: str = "报告 https://a.example/source") -> dict:
    return {
        "question": "测试问题",
        "sub_questions": [{"id": "sq_01", "question": "子问题", "search_query": "query"}],
        "search_results": [{
            "sub_question_id": "sq_01", "sub_question": "子问题", "query": "query",
            "title": "source", "url": "https://a.example/source", "content": "evidence", "score": 0.9,
        }],
        "report": report,
        "reflection_round": 0,
        "validation_gaps": [],
    }


def test_parse_failure_does_not_pass(monkeypatch):
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM("not-json"))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")
    result = reflector_module.reflector(_state())
    assert result["reflection_pass"] is False
    assert result["research_complete"] is False
    assert result["validation_gaps"]


def test_code_computes_pass_from_dimensions(monkeypatch):
    evaluation = {
        "completeness": 5, "accuracy": 5, "clarity": 5,
        "total_score": 1, "pass": False, "feedback": "", "supplement_queries": [],
    }
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM(json.dumps(evaluation)))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")
    result = reflector_module.reflector(_state())
    assert result["reflection_pass"] is True
    assert result["stop_reason"] == "completed"


def test_unknown_citation_fails_without_forcing_search(monkeypatch):
    evaluation = {
        "completeness": 5, "accuracy": 5, "clarity": 5,
        "total_score": 15, "pass": True, "feedback": "", "supplement_queries": [],
    }
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM(json.dumps(evaluation)))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")
    result = reflector_module.reflector(_state("报告 https://unknown.example/source"))
    assert result["reflection_pass"] is False
    assert result["validation_gaps"] == []
    assert "不在证据集" in result["reflection_feedback"]


def test_historical_sources_count_as_evidence(monkeypatch):
    """增量/追踪模式引用 knowledge_context 里的历史来源，不能判成「不在证据集」。"""
    evaluation = {
        "completeness": 5, "accuracy": 5, "clarity": 5,
        "total_score": 15, "pass": True, "feedback": "", "supplement_queries": [],
    }
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM(json.dumps(evaluation)))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")

    state = _state("报告引用历史来源 https://history.example/old")
    state["is_incremental"] = True
    state["known_source_urls"] = ["https://history.example/old"]

    result = reflector_module.reflector(state)
    assert result["reflection_pass"] is True
    assert "不在证据集" not in result["reflection_feedback"]


def test_unresolved_validator_gaps_are_preserved(monkeypatch):
    """LLM 没给 supplement_queries 时，validator 的缺口不能被覆盖掉。"""
    evaluation = {
        "completeness": 5, "accuracy": 5, "clarity": 5,
        "total_score": 15, "pass": True, "feedback": "", "supplement_queries": [],
    }
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM(json.dumps(evaluation)))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")

    state = _state()
    state["validation_gaps"] = [{
        "sub_question_id": "sq_01", "question": "子问题",
        "search_query": "官方统计 数据", "reason": "insufficient_domains",
        "priority": "medium", "language": "zh", "region": "CN",
    }]

    result = reflector_module.reflector(state)
    assert result["reflection_pass"] is False
    assert [gap["search_query"] for gap in result["validation_gaps"]] == ["官方统计 数据"]


def test_supplement_gaps_are_attributed_to_a_branch(monkeypatch):
    """报告级补充搜索必须带真实 sub_question_id，否则结果不计入任何分支覆盖率。"""
    evaluation = {
        "completeness": 2, "accuracy": 2, "clarity": 2,
        "total_score": 6, "pass": False, "feedback": "证据不足",
        "supplement_queries": ["official statistics 2026"],
    }
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM(json.dumps(evaluation)))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")

    state = _state()
    state["evidence_assessments"] = [
        {"sub_question_id": "sq_01", "status": "insufficient", "coverage": 0.3},
    ]

    gaps = reflector_module.reflector(state)["validation_gaps"]
    assert gaps[0]["sub_question_id"] == "sq_01"
    assert gaps[0]["search_query"] == "official statistics 2026"


def test_supplements_target_weakest_branch_first(monkeypatch):
    evaluation = {
        "completeness": 2, "accuracy": 2, "clarity": 2,
        "total_score": 6, "pass": False, "feedback": "",
        "supplement_queries": ["q1", "q2"],
    }
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM(json.dumps(evaluation)))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")

    state = _state()
    state["sub_questions"] = [
        {"id": "sq_01", "question": "A", "search_query": "a", "language": "zh", "region": "CN"},
        {"id": "sq_02", "question": "B", "search_query": "b"},
    ]
    state["evidence_assessments"] = [
        {"sub_question_id": "sq_01", "coverage": 0.9},
        {"sub_question_id": "sq_02", "coverage": 0.2},
    ]

    gaps = reflector_module.reflector(state)["validation_gaps"]
    # 覆盖率最低的分支优先拿到补充搜索，并继承该分支的语言/地区
    assert gaps[0]["sub_question_id"] == "sq_02"
    assert gaps[1]["sub_question_id"] == "sq_01"
    assert gaps[1]["language"] == "zh"


def test_parse_failure_gap_is_attributed_and_keeps_inherited(monkeypatch):
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM("not-json"))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")

    state = _state()
    state["validation_gaps"] = [{
        "sub_question_id": "sq_01", "question": "子问题",
        "search_query": "已有缺口查询", "reason": "low_coverage",
        "priority": "medium", "language": "auto", "region": "GLOBAL",
    }]

    gaps = reflector_module.reflector(state)["validation_gaps"]
    queries = [gap["search_query"] for gap in gaps]
    assert "已有缺口查询" in queries
    assert any(gap["reason"] == "reflection_parse_error" for gap in gaps)
    assert all(gap["sub_question_id"] == "sq_01" for gap in gaps)


def test_non_object_evaluation_does_not_pass(monkeypatch):
    """合法 JSON 但不是对象（数组/标量）也必须按未通过处理。"""
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM("[1, 2, 3]"))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")
    result = reflector_module.reflector(_state())
    assert result["reflection_pass"] is False


def test_string_supplement_queries_are_not_split_per_character(monkeypatch):
    evaluation = {
        "completeness": 2, "accuracy": 2, "clarity": 2,
        "total_score": 6, "pass": False, "feedback": "",
        "supplement_queries": "official statistics",
    }
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM(json.dumps(evaluation)))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")
    gaps = reflector_module.reflector(_state())["validation_gaps"]
    assert [gap["search_query"] for gap in gaps] == ["official statistics"]


def test_duplicate_gap_queries_are_merged_once(monkeypatch):
    evaluation = {
        "completeness": 2, "accuracy": 2, "clarity": 2,
        "total_score": 6, "pass": False, "feedback": "",
        "supplement_queries": ["重复查询"],
    }
    monkeypatch.setattr(reflector_module, "create_llm", lambda: _LLM(json.dumps(evaluation)))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")

    state = _state()
    state["validation_gaps"] = [{
        "sub_question_id": "sq_01", "question": "子问题",
        "search_query": "重复查询", "reason": "low_coverage",
        "priority": "medium", "language": "auto", "region": "GLOBAL",
    }]

    gaps = reflector_module.reflector(state)["validation_gaps"]
    assert len(gaps) == 1


# ── 编号引用校验（可发布文章风格） ───────────────────────

_SOURCE_TABLE = [
    {"index": 1, "title": "来源一", "url": "https://a.example/one", "source": "search"},
    {"index": 2, "title": "来源二", "url": "https://b.example/two", "source": "search"},
]


def _passing_eval():
    return {
        "completeness": 5, "accuracy": 5, "clarity": 5,
        "total_score": 15, "pass": True, "feedback": "", "supplement_queries": [],
    }


def _reflect(state, monkeypatch):
    monkeypatch.setattr(reflector_module, "create_llm",
                        lambda: _LLM(json.dumps(_passing_eval())))
    monkeypatch.setattr(reflector_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")
    return reflector_module.reflector(state)


def _state_with_table(report):
    """编号引用测试用 state：search_results 与 source_table 的 URL 对齐。"""
    state = _state(report)
    state["search_results"] = [
        {"sub_question_id": "sq_01", "sub_question": "子问题", "query": "query",
         "title": "来源一", "url": "https://a.example/one", "content": "evidence", "score": 0.9},
        {"sub_question_id": "sq_01", "sub_question": "子问题", "query": "query",
         "title": "来源二", "url": "https://b.example/two", "content": "evidence", "score": 0.8},
    ]
    state["source_table"] = _SOURCE_TABLE
    return state


def test_numbered_citations_pass_with_source_table(monkeypatch):
    state = _state_with_table("结论一[1]，结论二[2]。")
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is True
    assert "不在" not in result["reflection_feedback"]


def test_unknown_citation_number_fails(monkeypatch):
    state = _state_with_table("结论引用了[5]。")
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is False
    assert "不在来源编号表中" in result["reflection_feedback"]


def test_missing_citations_fail_when_table_present(monkeypatch):
    """编号表非空但正文一个 [n] 都没有 → 视为没有引用任何来源。"""
    state = _state_with_table("正文只有论述，没有任何引用编号。")
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is False
    assert "没有引用任何已检索来源 URL" in result["reflection_feedback"]


def test_citation_and_raw_known_url_both_pass(monkeypatch):
    """正文内嵌的裸 URL 只要在证据集内就不判违规（防回归旧风格时误伤）。"""
    state = _state_with_table("结论[1]（来源 https://a.example/one）。")
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is True


def test_historical_sources_citable_via_numbered_table(monkeypatch):
    """增量/追踪模式：编号表里的 knowledge 来源必须可被 [n] 引用。"""
    state = _state("历史结论[1]。")
    state["is_incremental"] = True
    state["known_source_urls"] = ["https://history.example/old"]
    state["source_table"] = [
        {"index": 1, "title": "https://history.example/old",
         "url": "https://history.example/old", "source": "knowledge"},
    ]
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is True
    assert "不在" not in result["reflection_feedback"]
