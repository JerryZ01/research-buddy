"""synthesizer 流式输出与可发布文章风格测试。

核心不变量：LangGraph 必须真的把 writer 注入进 synthesizer。
writer 的注解一旦不是 StreamWriter（比如写成 Callable | None），
LangGraph 会静默跳过注入，writer 恒为 None，report_chunk 事件全部消失，
而所有现有测试仍然会通过 —— 所以这里直接跑图去观察 custom 流。

可发布文章风格不变量：
- 评价性信息（局限/降级/置信度）进入 research_notes / confidence，不进正文
- 文末参考文献由代码生成（编号与正文 [n] 引用一一对应）
"""

from importlib import import_module

from langgraph.graph import StateGraph, START, END

from research_buddy.state import ResearchState

synthesizer_module = import_module("research_buddy.nodes.synthesizer")

_PIECES = ("# 研究报告\n", "## 概述\n正文一段。\n", "结论引用[1]。\n")

_URL_A = "https://a.example/one"
_URL_B = "https://b.example/two"


class _Chunk:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, pieces=None):
        self.pieces = pieces if pieces is not None else _PIECES

    def stream(self, _prompt):
        for piece in self.pieces:
            yield _Chunk(piece)


def _patch_llm(monkeypatch, pieces=None):
    monkeypatch.setattr(synthesizer_module, "create_llm",
                        lambda **_: _FakeLLM(pieces))
    monkeypatch.setattr(synthesizer_module, "get_prompt_from_langfuse",
                        lambda _name, _fallback, **_kwargs: "prompt")


def _graph():
    graph = StateGraph(ResearchState)
    graph.add_node("synthesizer", synthesizer_module.synthesizer)
    graph.add_edge(START, "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


def _run(monkeypatch, state, pieces=None):
    _patch_llm(monkeypatch, pieces)
    return _graph().invoke(state)


def test_report_chunks_reach_custom_stream(monkeypatch):
    _patch_llm(monkeypatch)

    chunks = []
    for mode, payload in _graph().stream(
        {"question": "测试问题", "search_results": []},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom" and isinstance(payload, dict) and payload.get("type") == "report_chunk":
            chunks.append(payload["content"])

    assert chunks, "writer 未被注入：LangGraph 只在注解是 StreamWriter 时才注入该参数"
    assert "".join(chunks).startswith("# 研究报告")


def test_degraded_assessment_moves_to_research_notes(monkeypatch):
    result = _run(monkeypatch, {
        "question": "测试问题",
        "search_results": [],
        "evidence_assessment_degraded": True,
    })
    assert any("语义证据评估不可用" in n for n in result["research_notes"])
    assert "语义证据评估不可用" not in result["report"]
    assert "## 研究局限" not in result["report"]


def test_search_unavailable_moves_to_research_notes(monkeypatch):
    result = _run(monkeypatch, {
        "question": "测试问题",
        "search_results": [],
        "search_unavailable": True,
    })
    assert any("未获得任何新证据" in n for n in result["research_notes"])
    assert "未获得任何新证据" not in result["report"]


def test_budget_exhaustion_lists_gaps_in_notes(monkeypatch):
    result = _run(monkeypatch, {
        "question": "测试问题",
        "search_results": [],
        "stop_reason": "search_budget_exhausted",
        "validation_gaps": [{
            "sub_question_id": "sq_01", "question": "缺口问题",
            "search_query": "q", "reason": "证据不足",
            "priority": "high", "language": "zh", "region": "CN",
        }],
    })
    assert "search_budget_exhausted" in result["research_notes"][0]
    assert any("缺口问题" in n for n in result["research_notes"])
    assert "缺口问题" not in result["report"]


def test_clean_run_has_no_limitations_and_empty_notes(monkeypatch):
    result = _run(monkeypatch, {"question": "测试问题", "search_results": []})
    assert "## 研究局限" not in result["report"]
    assert result["research_notes"] == []


# ── 置信度（代码计算，不进正文） ────────────────────────

def test_confidence_computed_by_code(monkeypatch):
    assert synthesizer_module.compute_confidence({"search_unavailable": True}) == "低"
    assert synthesizer_module.compute_confidence({"evidence_assessment_degraded": True}) == "中"
    assert synthesizer_module.compute_confidence({"stop_reason": "search_budget_exhausted"}) == "中"
    assert synthesizer_module.compute_confidence({"stop_reason": "no_new_queries"}) == "中"
    assert synthesizer_module.compute_confidence({"validation_gaps": [{"search_query": "x"}]}) == "中"
    assert synthesizer_module.compute_confidence({"stop_reason": "evidence_sufficient"}) == "高"
    assert synthesizer_module.compute_confidence({}) == "高"


def test_confidence_never_written_into_report(monkeypatch):
    """mock LLM 输出里即使有置信度文本，也只来自 mock 本身；
    正常 prompt 不再要求模型写置信度，代码只在 state 里产出 confidence。"""
    result = _run(monkeypatch, {"question": "测试问题", "search_results": []},
                  pieces=("# 标题\n", "正文。\n"))
    assert result["confidence"] == "高"
    assert "置信度" not in result["report"]


# ── 编号引用表 + 文末参考文献 ───────────────────────────

def test_source_table_built_from_results_and_known_urls(monkeypatch):
    state = {
        "question": "测试问题",
        "search_results": [
            {"sub_question": "A", "title": "来源一", "url": _URL_A, "content": "x", "score": 0.9},
            {"sub_question": "A", "title": "重复", "url": _URL_A, "content": "y", "score": 0.8},
        ],
        "known_source_urls": [_URL_A, _URL_B],
    }
    result = _run(monkeypatch, state)
    table = result["source_table"]
    # 按 normalize_url 去重；search 结果在前，knowledge 来源在后
    assert [t["url"] for t in table] == [_URL_A, _URL_B]
    assert table[0]["index"] == 1 and table[0]["source"] == "search"
    assert table[1]["index"] == 2 and table[1]["source"] == "knowledge"


def test_references_section_appended_by_code(monkeypatch):
    state = {
        "question": "测试问题",
        "search_results": [
            {"sub_question": "A", "title": "来源一", "url": _URL_A, "content": "x", "score": 0.9},
            {"sub_question": "B", "title": "来源二", "url": _URL_B, "content": "y", "score": 0.8},
        ],
    }
    result = _run(monkeypatch, state)
    report = result["report"]
    assert "## 参考文献" in report
    assert f"1. [来源一]({_URL_A})" in report
    assert f"2. [来源二]({_URL_B})" in report
    # 参考文献排在正文之后
    assert report.index("## 参考文献") > report.index("## 概述")


def test_references_chunk_is_streamed(monkeypatch):
    """参考文献由代码生成后也要走 report_chunk 流，前端才能实时看到。"""
    _patch_llm(monkeypatch)
    chunks = []
    for mode, payload in _graph().stream(
        {"question": "测试问题", "search_results": [
            {"sub_question": "A", "title": "来源一", "url": _URL_A, "content": "x", "score": 0.9},
        ]},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom" and isinstance(payload, dict) and payload.get("type") == "report_chunk":
            chunks.append(payload["content"])
    joined = "".join(chunks)
    assert "## 参考文献" in joined
    assert _URL_A in joined


def test_empty_table_renders_no_references(monkeypatch):
    result = _run(monkeypatch, {"question": "测试问题", "search_results": []})
    assert "## 参考文献" not in result["report"]


# ── 插图（视觉选图） ──────────────────────────────────

def test_selected_images_injected_into_prompt(monkeypatch):
    """选中的插图要注入 prompt 的 image_section，并随 state 返回。"""
    monkeypatch.setattr(synthesizer_module, "select_images", lambda *_, **__: [{
        "url": "https://img.example/pic.jpg",
        "alt": "架构示意图",
        "sub_question_id": "sq_01",
        "query": "查询",
    }])
    captured: dict = {}

    def fake_get_prompt(name, fallback, **kwargs):
        captured["prompt"] = kwargs
        return "prompt"

    monkeypatch.setattr(synthesizer_module, "create_llm", lambda **_: _FakeLLM())
    monkeypatch.setattr(synthesizer_module, "get_prompt_from_langfuse", fake_get_prompt)

    result = _graph().invoke({
        "question": "测试问题",
        "search_results": [],
        "sub_questions": [{"id": "sq_01", "question": "子问题"}],
        "image_candidates": [{"url": "https://img.example/pic.jpg",
                              "sub_question_id": "sq_01", "query": "查询"}],
    })
    assert "https://img.example/pic.jpg" in captured["prompt"]["image_section"]
    assert result["selected_images"][0]["alt"] == "架构示意图"


def test_no_candidates_skips_selection(monkeypatch):
    """没有候选图时不得调用视觉模型（也不得在 prompt 里出现插图指令数据）。"""
    called = {"select": False}

    def fake_select(*args, **kwargs):
        called["select"] = True
        return []

    monkeypatch.setattr(synthesizer_module, "select_images", fake_select)
    result = _run(monkeypatch, {"question": "测试问题", "search_results": []})
    assert called["select"] is False
    assert result["selected_images"] == []
