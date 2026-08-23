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
    # 核心文献筛选默认恒等（原样返回全部来源），各测试按需单独覆盖
    monkeypatch.setattr(synthesizer_module, "curate_core_references",
                        lambda _question, table, **_kwargs: table)


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


# ── 核心文献筛选 ──────────────────────────────────────

def test_core_references_curated_and_renumbered(monkeypatch):
    """筛选后的子集重新编号 1..k，只渲染被选中的来源。"""
    table = [
        {"index": 1, "title": "来源一", "url": _URL_A, "source": "search"},
        {"index": 2, "title": "来源二", "url": _URL_B, "source": "search"},
        {"index": 3, "title": "来源三", "url": "https://c.example/three", "source": "search"},
    ]
    _patch_llm(monkeypatch)
    # 必须先 _patch_llm 再覆盖（_patch_llm 会设置恒等 curation）
    monkeypatch.setattr(synthesizer_module, "curate_core_references",
                        lambda _q, _t, **_k: [table[2], table[0]])  # 只选 3 和 1

    result = _graph().invoke({"question": "测试问题", "search_results": []})

    report = result["report"]
    assert "来源三" in report and "来源一" in report
    assert "来源二" not in report
    # 重新编号：来源三在前（1.），来源一在后（2.）
    assert "1. [来源三](https://c.example/three)" in report
    assert "2. [来源一]" in report
    assert "## 参考文献" in report


def test_curate_core_references_picks_indexes_in_order(monkeypatch):
    """LLM 返回 [3, 1] → 按该顺序返回对应来源，并限制数量。"""
    table = [
        {"index": i, "title": f"来源{i}", "url": f"https://{i}.example/x", "source": "search"}
        for i in range(1, 7)
    ]

    class _Resp:
        content = '{"indexes": [3, 1, 9, 1]}'  # 9 越界、1 重复 → 丢弃

    monkeypatch.setattr(synthesizer_module, "create_llm",
                        lambda **_: type("_LLM", (), {"invoke": lambda self, _p: _Resp()})())
    monkeypatch.setattr(synthesizer_module, "get_prompt_from_langfuse",
                        lambda *_a, **_k: "prompt")

    picked = synthesizer_module.curate_core_references("问题", table, max_refs=4)
    assert [p["index"] for p in picked] == [3, 1]


def test_curate_core_references_fallback_on_failure(monkeypatch):
    """LLM 调用/解析失败 → 降级取来源列表前 max_refs 个。"""
    table = [
        {"index": i, "title": f"来源{i}", "url": f"https://{i}.example/x", "source": "search"}
        for i in range(1, 11)
    ]

    def _boom(*_a, **_k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(synthesizer_module, "create_llm", _boom)
    picked = synthesizer_module.curate_core_references("问题", table, max_refs=3)
    assert [p["index"] for p in picked] == [1, 2, 3]


def test_prompts_require_no_inline_citations(monkeypatch):
    """三个 prompt 都不应再要求 [编号] 引用，也不再注入来源编号表。"""
    for prompt in (synthesizer_module.SYNTHESIZER_PROMPT,
                   synthesizer_module.SYNTHESIZER_INCREMENTAL_PROMPT,
                   synthesizer_module.SYNTHESIZER_REFINE_PROMPT):
        assert "{source_table}" not in prompt
        assert "编号必须来自" not in prompt


def test_prompts_include_mermaid_and_render():
    """技术架构/原理类话题应画 Mermaid 图解；prompt 里的花括号转义不能破坏 format。"""
    for prompt in (synthesizer_module.SYNTHESIZER_PROMPT,
                   synthesizer_module.SYNTHESIZER_INCREMENTAL_PROMPT,
                   synthesizer_module.SYNTHESIZER_REFINE_PROMPT):
        assert "```mermaid" in prompt
        assert "技术图解" in prompt
        # 模拟 fallback 渲染路径：花括号必须正确转义（mermaid 的 { } 节点语法）
        kwargs = {"question": "Q", "search_results": "R", "image_section": "I",
                  "style_section": "测试文风", "image_limit": "8"}
        if "{knowledge_context}" in prompt:
            kwargs["knowledge_context"] = "K"
        if "{report}" in prompt:
            kwargs["report"] = "Rp"
        if "{feedback}" in prompt:
            kwargs["feedback"] = "Fb"
        rendered = prompt.format(**kwargs)
        assert "C{鉴权}" in rendered  # {{鉴权}} 渲染回 {鉴权}
        assert "{{" not in rendered.split("```mermaid")[0]  # 图解之外不应有未转义双花括号


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


# ── 跨轮次复用（性能优化，不改核心逻辑） ───────────────

def test_reuses_selected_images_across_rounds(monkeypatch):
    """重写/回环轮复用上一轮选中的图，不再重新调视觉模型。"""
    def _boom(*a, **k):
        raise AssertionError("已有选中图，不应重新选图")

    monkeypatch.setattr(synthesizer_module, "select_images", _boom)
    _patch_llm(monkeypatch)
    state = {
        "question": "测试问题",
        "search_results": [],
        "image_candidates": [{"url": "https://img.example/pic.jpg",
                              "sub_question_id": "sq_01", "query": "q"}],
        "selected_images": [{"url": "https://img.example/pic.jpg", "alt": "复用图",
                             "sub_question_id": "sq_01", "query": "q"}],
    }
    result = _graph().invoke(state)
    assert result["selected_images"][0]["alt"] == "复用图"


def test_reuses_core_refs_when_sources_unchanged(monkeypatch):
    """来源集（URL 签名）未变时复用上一轮的文献筛选结果，不再调 LLM。"""
    _patch_llm(monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("来源集未变，不应重新筛选文献")

    monkeypatch.setattr(synthesizer_module, "curate_core_references", _boom)
    # 签名 = normalize_url 后的 URL（无 scheme），与 synthesizer 内部计算一致
    expected_sig = synthesizer_module.normalize_url(_URL_A)
    state = {
        "question": "测试问题",
        "search_results": [{"sub_question": "A", "title": "来源一", "url": _URL_A,
                            "content": "x", "score": 0.9}],
        "core_refs": [{"index": 1, "title": "核心", "url": _URL_A, "source": "search"}],
        "core_refs_signature": expected_sig,
    }
    result = _graph().invoke(state)
    assert "1. [核心]" in result["report"]
    assert result["core_refs"][0]["title"] == "核心"


def test_recurates_core_refs_when_sources_changed(monkeypatch):
    """补充搜索带来新来源（签名变化）时重新筛选文献。"""
    _patch_llm(monkeypatch)
    calls = {"n": 0}

    def _fake_curate(q, table, **_k):
        calls["n"] += 1
        return table[:1]

    monkeypatch.setattr(synthesizer_module, "curate_core_references", _fake_curate)
    state = {
        "question": "测试问题",
        "search_results": [{"sub_question": "A", "title": "来源一", "url": _URL_A,
                            "content": "x", "score": 0.9}],
        "core_refs": [{"index": 99, "title": "旧", "url": "https://old.example", "source": "search"}],
        "core_refs_signature": "https://old.example",  # 与当前来源集不同 → 重筛
    }
    _graph().invoke(state)
    assert calls["n"] == 1


# ── 写作风格注入 ──────────────────────────────────────

def test_style_section_injected_into_prompt(monkeypatch):
    """所选风格（如观点锐评）的文风要求要注入 prompt。"""
    captured: dict = {}

    def fake_get_prompt(name, fallback, **kwargs):
        captured["prompt"] = kwargs
        return "prompt"

    monkeypatch.setattr(synthesizer_module, "create_llm", lambda **_: _FakeLLM())
    monkeypatch.setattr(synthesizer_module, "get_prompt_from_langfuse", fake_get_prompt)
    monkeypatch.setattr(synthesizer_module, "curate_core_references",
                        lambda _q, table, **_k: table)

    _graph().invoke({"question": "测试问题", "search_results": [], "style": "essay"})
    assert "犀利" in captured["prompt"]["style_section"]


def test_unknown_style_falls_back_to_default(monkeypatch):
    captured: dict = {}

    def fake_get_prompt(name, fallback, **kwargs):
        captured["prompt"] = kwargs
        return "prompt"

    monkeypatch.setattr(synthesizer_module, "create_llm", lambda **_: _FakeLLM())
    monkeypatch.setattr(synthesizer_module, "get_prompt_from_langfuse", fake_get_prompt)
    monkeypatch.setattr(synthesizer_module, "curate_core_references",
                        lambda _q, table, **_k: table)

    _graph().invoke({"question": "测试问题", "search_results": [], "style": "不存在的风格"})
    assert "专业、克制、自信的技术博客文风" in captured["prompt"]["style_section"]


# ── 标题去模板化（防「名词：副题」式冒号标题） ───────────

def test_collect_headings_finds_md_and_bold_skips_code():
    report = ("## 自注意力：当每个 token 都成为检索者\n正文\n\n"
              "**多头注意力：从一种相关性到多种**\n\n"
              "```\n## 代码里的注释\n```\n\n"
              "### 位置编码\n正文\n")
    texts = [h["text"] for h in synthesizer_module._collect_headings(report)]
    assert texts == ["自注意力：当每个 token 都成为检索者",
                     "多头注意力：从一种相关性到多种", "位置编码"]
    assert "代码里的注释" not in texts


def test_normalize_headings_rewrites_via_llm(monkeypatch):
    report = ("## 自注意力：当每个 token 都成为检索者\n正文\n\n"
              "## 多头注意力：从一种相关性到多种\n正文\n")
    resp = '{"titles": ["当每个 token 都成为检索者", "从一种相关性到多种"]}'

    class _Resp:
        content = resp

    monkeypatch.setattr(synthesizer_module, "create_llm",
                        lambda **_: type("_LLM", (), {"invoke": lambda self, p: _Resp()})())
    monkeypatch.setattr(synthesizer_module, "get_prompt_from_langfuse",
                        lambda *a, **k: "prompt")
    out = synthesizer_module._normalize_headings("问题", report)
    assert "## 当每个 token 都成为检索者" in out
    assert "## 从一种相关性到多种" in out
    assert "：" not in out


def test_normalize_headings_noop_when_already_mixed(monkeypatch):
    """冒号标题占比不足时不触发（不调 LLM）。"""
    report = "## 为什么需要自注意力\n正文\n\n## B：副题\n正文\n\n## 一个被低估的设计\n正文\n"

    def _boom(*a, **k):
        raise AssertionError("冒号占比不足，不应调用 LLM")

    monkeypatch.setattr(synthesizer_module, "create_llm", _boom)
    assert synthesizer_module._normalize_headings("问题", report) == report


def test_normalize_headings_fallback_on_invalid_llm_output(monkeypatch):
    """LLM 输出数量不匹配/仍含冒号时保留原标题，不抛异常。"""
    report = "## A：副题一\n正文\n\n## B：副题二\n正文\n\n## C：副题三\n正文\n"

    class _BadResp:
        content = '{"titles": ["只有一个"]}'  # 数量不匹配

    monkeypatch.setattr(synthesizer_module, "create_llm",
                        lambda **_: type("_LLM", (), {"invoke": lambda self, p: _BadResp()})())
    monkeypatch.setattr(synthesizer_module, "get_prompt_from_langfuse",
                        lambda *a, **k: "prompt")
    assert synthesizer_module._normalize_headings("问题", report) == report
