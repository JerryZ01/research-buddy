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

    def invoke(self, _prompt, **_kwargs):
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


# ── 正文引用风格（无编号引用，仅校验残留 URL） ───────────

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
    """引用风格测试用 state：search_results 与 source_table 的 URL 对齐。"""
    state = _state(report)
    state["search_results"] = [
        {"sub_question_id": "sq_01", "sub_question": "子问题", "query": "query",
         "title": "来源一", "url": "https://a.example/one", "content": "evidence", "score": 0.9},
        {"sub_question_id": "sq_01", "sub_question": "子问题", "query": "query",
         "title": "来源二", "url": "https://b.example/two", "content": "evidence", "score": 0.8},
    ]
    state["source_table"] = _SOURCE_TABLE
    return state


def test_body_without_citations_passes(monkeypatch):
    """正文无 [n] 引用是新的正常形态：不再校验「是否引用了来源」。"""
    state = _state_with_table("正文只有客观论述，没有引用编号。")
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is True


def test_legacy_citation_text_in_body_does_not_fail(monkeypatch):
    """正文残留 [1][2] 文本不判违规（引用校验已移除，风格由 prompt 约束）。"""
    state = _state_with_table("结论一[1]，结论二[2]。")
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is True
    assert "不在" not in result["reflection_feedback"]


def test_raw_known_url_in_body_passes(monkeypatch):
    """正文内嵌的裸 URL 只要在证据集内就不判违规（防回归旧风格时误伤）。"""
    state = _state_with_table("结论（来源 https://a.example/one）。")
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is True


def test_unknown_raw_url_in_body_fails(monkeypatch):
    """正文内嵌不在证据集的 URL → fail-closed 强制修正。"""
    state = _state_with_table("结论（来源 https://unknown.example/outside）。")
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is False
    assert "不在证据集" in result["reflection_feedback"]


def test_historical_sources_stay_in_evidence_set(monkeypatch):
    """增量/追踪模式：历史知识来源必须在证据集里，不算「不在证据集」。"""
    state = _state("历史结论（来源 https://history.example/old）。")
    state["is_incremental"] = True
    state["known_source_urls"] = ["https://history.example/old"]
    state["source_table"] = [
        {"index": 1, "title": "https://history.example/old",
         "url": "https://history.example/old", "source": "knowledge"},
    ]
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is True
    assert "不在" not in result["reflection_feedback"]


# ── 插图 URL 校验（防幻觉图片） ───────────────────────

def test_selected_image_url_is_allowed(monkeypatch):
    """正文嵌入的插图 URL 必须在 selected_images 里才放行。"""
    state = _state_with_table("结论[1]。\n\n![架构示意图](https://img.example/pic.jpg)\n")
    state["selected_images"] = [{
        "url": "https://img.example/pic.jpg", "alt": "架构示意图", "sub_question_id": "sq_01",
    }]
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is True
    assert "不在证据集" not in result["reflection_feedback"]


def test_unselected_image_url_fails(monkeypatch):
    """LLM 嵌入了候选/选中之外的图片 URL → fail-closed 强制修正。"""
    state = _state_with_table("结论[1]。\n\n![未知图](https://img.example/unknown.jpg)\n")
    state["selected_images"] = [{
        "url": "https://img.example/real.jpg", "alt": "真实图", "sub_question_id": "sq_01",
    }]
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is False
    assert "不在证据集" in result["reflection_feedback"]


# ── AI 味硬校验（防模板腔） ─────────────────────────────

def _ai_issues(report: str) -> list[str]:
    return reflector_module._ai_flavor_issues(report)


def test_ai_flavor_meta_comments_detected():
    report = ("本节的关键结论是：X。\n值得注意的是：Y。\n"
              "综上所述，Z。\n这揭示了 W。\n正常段落。")
    issues = _ai_issues(report)
    assert any("元评论" in i for i in issues)


def test_ai_flavor_parallel_overuse_detected():
    report = ("不是A而是B，既C又D，越E越F，不仅G更是H。" * 6)  # 高密度排比
    issues = _ai_issues(report)
    assert any("排比对仗" in i for i in issues)


def test_ai_flavor_formula_ending_detected():
    report = "正文内容。\n\n## 结论\n1. 要点一\n2. 要点二\n3. 要点三\n"
    issues = _ai_issues(report)
    assert any("公式化" in i for i in issues)


def test_ai_flavor_clean_report_passes():
    report = ("这是一段正常的正文，句子长短自然，观点明确。\n"
              "另一个段落继续展开分析，没有模板句式。\n" * 5)
    assert _ai_issues(report) == []


def test_ai_flavor_issues_force_rewrite(monkeypatch):
    """AI 味命中 → 强制不通过，反馈里带具体问题。"""
    state = _state("本节的关键结论是：X。\n值得注意的是：Y。\n综上所述：Z。\n这揭示了 W。\n")
    result = _reflect(state, monkeypatch)
    assert result["reflection_pass"] is False
    assert "元评论" in result["reflection_feedback"]


def test_ai_flavor_self_qa_detected():
    report = ("这场争论为什么如此重要？因为它触及了根本问题。\n"
              "这意味着什么？对于反欺诈系统而言……\n正常段落。")
    issues = _ai_issues(report)
    assert any("自问自答" in i for i in issues)


def test_ai_flavor_guide_sentences_detected():
    report = ("拆解这个定义需要一点耐心。\n让我们先看看背景。\n"
              "这里需要先明确几个概念。\n先说一个结论。\n")
    issues = _ai_issues(report)
    assert any("引导" in i for i in issues)


def test_single_self_qa_does_not_trigger():
    """单处自问自答不误伤（阈值从宽）。"""
    report = "这意味着什么？因为这关系到架构选择。\n正常正文。\n"
    assert _ai_issues(report) == []


def test_ai_flavor_colon_headings_detected():
    report = ("## 自注意力：当每个 token 都成为检索者\n正文。\n"
              "## 多头注意力：从一种相关性到多种相关性\n正文。\n"
              "## 位置编码：给序列注入顺序感\n正文。\n"
              "## 前馈网络：逐位置的变换\n正文。\n")
    issues = _ai_issues(report)
    assert any("冒号标题" in i for i in issues)


def test_mixed_heading_styles_pass():
    report = ("## 为什么 I/O 密集场景不受 GIL 影响\n正文。\n"
              "## 自注意力：当每个 token 都成为检索者\n正文。\n"
              "## 一个被低估的取舍\n正文。\n"
              "## 这值得在生产环境用吗\n正文。\n")
    assert _ai_issues(report) == []


def test_ai_flavor_retrieval_meta_comments_detected():
    """「从检索到的资料看」「有来源提到」等研究过程元评论零容忍。"""
    report = ("从检索到的资料看，LangGraph 支持循环。\n"
              "有来源提到 Checkpointer 的批量写入优化。\n"
              "从检索到的证据来看，EvalScope 支持自定义数据集。\n")
    issues = _ai_issues(report)
    assert any("研究过程元评论" in i for i in issues)


def test_ai_flavor_single_retrieval_meta_comment_triggers():
    """即使只有 1 处「从检索到的资料看」也要重写（零容忍）。"""
    report = "正常正文。\n从检索到的资料看，这一点值得注意。\n"
    issues = _ai_issues(report)
    assert any("研究过程元评论" in i for i in issues)


def test_ai_flavor_not_but_overuse_detected():
    """「不是…而是…」≥3 处触发；2 处不误伤。"""
    report = ("不是 A 而是 B。\n不是 C 而是 D。\n不是 E 而是 F。\n不是 G 而是 H。\n")
    issues = _ai_issues(report)
    assert any("不是…而是…" in i for i in issues)
    # 2 处不触发计数检查；用长正文隔离密度检查（避免短文误伤）
    long_report = "正常论述的正文。" * 200 + "不是 A 而是 B。\n不是 C 而是 D。\n"
    assert _ai_issues(long_report) == []


def test_ai_flavor_triple_parallel_detected():
    """「从 X 到 Y，从 A 到 B，从 C 到 D」三连排比触发。"""
    report = ("这种转变体现在：从顺序执行到有向图执行，从无状态到有状态，"
              "从跑完就结束到随时暂停恢复。\n")
    issues = _ai_issues(report)
    assert any("三连排比" in i for i in issues)
