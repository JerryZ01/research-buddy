import json
from importlib import import_module

from langgraph.graph import END, START, StateGraph

from research_buddy.state import ResearchState

editor = import_module("research_buddy.nodes.article_editor")


def test_validate_and_apply_exact_evidence_edits():
    report = "正文开头。NumPy 的所有运算都会释放 GIL，因此线程可以并行。正文结尾。"
    payload = {"edits": [{
        "quote": "NumPy 的所有运算都会释放 GIL，因此线程可以并行。",
        "replacement": "部分计算密集型扩展在执行原生代码时会释放 GIL。",
        "reason": "证据没有点名具体扩展",
        "evidence_ids": ["E1", "E9"],
        "support_quotes": [{"evidence_id": "E1", "quote": "部分计算密集型扩展"}],
    }]}
    evidence = [{"content": "阻塞 I/O 会释放锁，部分计算密集型扩展也会释放它。"}]
    edits = editor.validate_edits(payload, report, evidence)
    assert edits[0]["evidence_ids"] == ["E1"]
    revised = editor.apply_evidence_edits(report, edits)
    assert "NumPy" not in revised
    assert "部分计算密集型扩展" in revised


def test_invalid_or_hallucinated_edits_are_ignored():
    report = "这是一段足够长的真实文章内容。"
    payload = {"edits": [
        {"quote": "不存在的文章原文片段很长", "replacement": "替换", "reason": "x", "evidence_ids": ["E1"],
         "support_quotes": [{"evidence_id": "E1", "quote": "证据里的原话足够长"}]},
        {"quote": "这是一段足够长的真实文章内容。", "replacement": "https://bad.example", "reason": "x", "evidence_ids": ["E1"],
         "support_quotes": [{"evidence_id": "E1", "quote": "证据里的原话足够长"}]},
        {"quote": "这是一段足够长的真实文章内容。", "replacement": "有证据的替换内容。", "reason": "x", "evidence_ids": ["E1"],
         "support_quotes": [{"evidence_id": "E1", "quote": "模型编造的证据引文"}]},
    ]}
    assert editor.validate_edits(payload, report, [{"content": "证据里的原话足够长。"}]) == []


def test_fact_editor_never_edits_inserted_image_markdown():
    report = "![架构示意图](/media/research-images/abc.png)"
    payload = {"edits": [{
        "quote": report,
        "replacement": "",
        "reason": "模型误判图片为冗余内容",
        "edit_type": "redundant",
        "evidence_ids": [],
        "support_quotes": [],
    }]}
    assert editor.validate_edits(payload, report, [{"content": "有效证据内容"}]) == []


def test_deletion_of_unsupported_claim_does_not_require_support_quote():
    report = "开头。两台服务器不会产生任何新的故障模式。结尾。"
    payload = {"edits": [{
        "quote": "两台服务器不会产生任何新的故障模式。",
        "replacement": "",
        "reason": "证据没有讨论故障模式",
        "evidence_ids": [],
        "support_quotes": [],
    }]}
    edits = editor.validate_edits(payload, report, [{"content": "集群包含控制平面和工作节点。"}])
    assert edits[0]["replacement"] == ""
    assert editor.apply_evidence_edits(report, edits) == "开头。结尾。"


def test_grounding_verdicts_require_complete_unique_indices():
    edits = [{"quote": "a"}, {"quote": "b"}]
    payload = {"verdicts": [
        {"index": 0, "supported": True, "reason": "直接支持"},
        {"index": 1, "supported": False, "reason": "加入了因果"},
    ]}
    assert editor.validate_verdicts(payload, edits) == {0}
    payload["verdicts"].pop()
    try:
        editor.validate_verdicts(payload, edits)
        assert False, "缺少 verdict 应失败"
    except ValueError:
        pass


def test_context_guard_rejects_dangling_colon_and_adjacent_repetition():
    dangling = "它回答的是另一类问题：当容器跨机器运行时需要协调。下一段。"
    edit = {"quote": "当容器跨机器运行时需要协调。", "replacement": ""}
    assert editor.apply_evidence_edits(dangling, [edit]) == dangling

    repeated = "Kubernetes 提供服务发现、自愈和水平扩缩。后续原文需要被替换。"
    edit = {
        "quote": "后续原文需要被替换。",
        "replacement": "Kubernetes 的能力包括服务发现、自愈和水平扩缩。",
    }
    assert editor.apply_evidence_edits(repeated, [edit]) == repeated

    dangling_conjunction = "线程适合等待任务。线程还有共享内存的优势，但前提是任务会释放锁。"
    edit = {"quote": "线程还有共享内存的优势", "replacement": ""}
    assert editor.apply_evidence_edits(dangling_conjunction, [edit]) == dangling_conjunction

    near_duplicate = "绕过的代价是任务和返回值需要序列化，跨进程通信有额外成本。后一句需要替换。"
    edit = {
        "quote": "后一句需要替换。",
        "replacement": "但任务和返回值需要能够序列化，且引入进程通信成本。",
    }
    assert editor.apply_evidence_edits(near_duplicate, [edit]) == near_duplicate


def test_context_guard_rejects_deleting_markdown_structure():
    for report, quote in [
        ("## 一个足够长的章节标题\n正文。", "一个足够长的章节标题"),
        ("- 一个足够长的列表项目内容。\n正文。", "一个足够长的列表项目内容。"),
        ("| 字段 | 一个足够长的表格单元内容 |\n", "一个足够长的表格单元内容"),
    ]:
        edit = {"quote": quote, "replacement": ""}
        assert editor.apply_evidence_edits(report, [edit]) == report


def test_rejected_edits_do_not_trigger_unrelated_duplicate_cleanup():
    duplicate = "部分扩展在执行原生代码时会释放 GIL。"
    report = f"它回答的是另一类问题：当容器跨机器运行时需要协调。{duplicate}{duplicate}"
    rejected = {"quote": "当容器跨机器运行时需要协调。", "replacement": ""}
    revised, applied = editor._apply_evidence_edits_with_audit(report, [rejected])
    assert revised == report
    assert applied == []


def test_exact_duplicate_sentences_removed_but_references_preserved():
    report = ("第一段。部分扩展在执行原生代码时会释放 GIL。\n\n"
              "部分扩展在执行原生代码时会释放 GIL。后续内容。\n"
              "## 参考文献\n1. [来源](https://example.com)\n")
    cleaned = editor.remove_exact_duplicate_sentences(report)
    assert cleaned.count("部分扩展在执行原生代码时会释放 GIL。") == 1
    assert "https://example.com" in cleaned


def test_editor_fails_open_on_bad_model_output(monkeypatch):
    class _Response:
        content = "not-json"

    class _LLM:
        def invoke(self, _prompt, **_kwargs):
            return _Response()

    monkeypatch.setattr(editor, "create_llm", lambda: _LLM())
    report = "这是一段足够长的真实文章内容。"
    revised, edits = editor.edit_article_evidence(
        {"question": "Q", "search_results": [{"title": "S", "content": "E"}],
         "eval_use_local_prompts": True}, report,
    )
    assert revised == report
    assert edits == []


def test_editor_prompt_accepts_empty_edits_json(monkeypatch):
    captured = {}

    class _Response:
        content = json.dumps({"edits": []})

    class _LLM:
        def invoke(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return _Response()

    monkeypatch.setattr(editor, "create_llm", lambda: _LLM())
    report = "这是一段足够长的真实文章内容。"
    revised, edits = editor.edit_article_evidence(
        {"question": "Q", "search_results": [{"title": "S", "content": "E"}],
         "eval_use_local_prompts": True}, report,
    )
    assert revised == report and edits == []
    assert '{"edits": []}' in captured["prompt"]


def test_editor_uses_deduplicated_evidence_ledger(monkeypatch):
    captured = {}

    class _Response:
        content = json.dumps({"edits": []})

    class _LLM:
        def invoke(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return _Response()

    monkeypatch.setattr(editor, "create_llm", lambda: _LLM())
    state = {
        "question": "Q", "eval_use_local_prompts": True,
        "search_results": [
            {"title": "重复来源", "content": "不应出现的原始搜索片段"},
            {"title": "另一条", "content": "也不应出现"},
        ],
        "evidence_ledger": [
            {"id": "E1", "title": "去重来源", "excerpt": "账本中的唯一证据"},
        ],
    }
    editor.edit_article_evidence(state, "这是一段足够长的真实文章内容。", max_rounds=1)
    assert "E1 | 去重来源" in captured["prompt"]
    assert "不应出现的原始搜索片段" not in captured["prompt"]


def test_article_editor_resets_and_streams_only_when_report_changed(monkeypatch):
    report = "原始初稿。"
    revised = "修订后的文章。"
    applied = [{"quote": report, "replacement": revised, "edit_type": "overstated"}]
    monkeypatch.setattr(editor, "ENABLE_ARTICLE_EDITOR", True)
    monkeypatch.setattr(editor, "ARTICLE_EDITOR_ROUNDS", 1)
    monkeypatch.setattr(editor, "edit_article_evidence", lambda *_args, **_kwargs: (revised, applied))

    graph = StateGraph(ResearchState)
    graph.add_node("article_editor", editor.article_editor)
    graph.add_edge(START, "article_editor")
    graph.add_edge("article_editor", END)
    compiled = graph.compile()

    custom = []
    update = None
    for mode, payload in compiled.stream(
        {"report": report}, stream_mode=["custom", "updates"],
    ):
        if mode == "custom":
            custom.append(payload)
        elif mode == "updates":
            update = payload["article_editor"]
    assert custom[0]["type"] == "report_reset"
    assert "".join(item.get("content", "") for item in custom[1:]) == revised
    assert update["report"] == revised
    assert update["evidence_edits"] == applied


def test_article_editor_keeps_stream_untouched_when_no_change(monkeypatch):
    monkeypatch.setattr(editor, "ENABLE_ARTICLE_EDITOR", True)
    monkeypatch.setattr(editor, "ARTICLE_EDITOR_ROUNDS", 1)
    monkeypatch.setattr(
        editor, "edit_article_evidence", lambda _state, report, config=None: (report, []),
    )
    events = []
    result = editor.article_editor(
        {"report": "原稿"}, config=None, writer=lambda event: events.append(event),
    )
    assert events == []
    assert result["report"] == "原稿"
    assert result["evidence_edits"] == []
