import json
from importlib import import_module


editor = import_module("research_buddy.nodes.language_editor")


def test_scan_finds_contrast_template_but_skips_markdown_and_code():
    report = (
        "## 不是标题问题，而是标题内容\n"
        "这不是重新训练模型，而是约束输出后的编辑过程。\n"
        "```python\n不是代码，而是字符串。\n```\n"
        "```text\n首先，代码行。\n首先，代码行。\n首先，代码行。\n```\n"
        "## 参考文献\n不是来源甲，而是来源乙。\n"
    )
    candidates = editor.scan_language_issues(report)
    assert [item["quote"] for item in candidates] == [
        "这不是重新训练模型，而是约束输出后的编辑过程。",
    ]
    assert candidates[0]["issue_type"] == "contrast_template"


def test_validate_requires_exact_candidate_and_preserves_fact_tokens():
    quote = "这意味着 GPT-5 在 2026 年提升了 20%[1]。"
    candidates = [{"quote": quote, "issue_type": "meta_summary", "cue": "这意味着"}]
    valid_payload = {"edits": [{
        "quote": quote,
        "replacement": "GPT-5 在 2026 年提升了 20%[1]。",
        "issue_type": "meta_summary",
        "reason": "删除空泛转折",
    }]}
    valid = editor.validate_language_edits(valid_payload, quote, candidates)
    assert len(valid) == 1

    valid_payload["edits"][0]["replacement"] = "GPT-6 在 2027 年提升了 30%[2]。"
    assert editor.validate_language_edits(valid_payload, quote, candidates) == []


def test_validate_rejects_new_template_and_non_candidate_sentence():
    quote = "值得注意的是，这项机制只处理已经标出的句子。"
    candidates = [{"quote": quote, "issue_type": "empty_transition", "cue": "值得注意的是"}]
    payload = {"edits": [{
        "quote": quote,
        "replacement": "这意味着，这项机制只处理已经标出的句子。",
        "issue_type": "empty_transition",
        "reason": "替换转折",
    }, {
        "quote": "文章中另一句足够长但没有被扫描的内容。",
        "replacement": "另一句内容。",
        "issue_type": "empty_transition",
        "reason": "越界编辑",
    }]}
    report = quote + "文章中另一句足够长但没有被扫描的内容。"
    assert editor.validate_language_edits(payload, report, candidates) == []


def test_no_candidate_skips_llm_call(monkeypatch):
    monkeypatch.setattr(editor, "create_llm", lambda: (_ for _ in ()).throw(AssertionError()))
    report = "机制读取候选句并进行精确替换。"
    revised, edits = editor.edit_article_language({}, report)
    assert revised == report
    assert edits == []


def test_editor_applies_valid_model_edit(monkeypatch):
    report = "这不是重新训练模型，而是约束输出后的编辑过程。"
    replacement = "该过程约束模型输出后再进行编辑。"

    class _Response:
        content = json.dumps({"edits": [{
            "quote": report,
            "replacement": replacement,
            "issue_type": "contrast_template",
            "reason": "删除模板化对比",
        }]}, ensure_ascii=False)

    class _LLM:
        def invoke(self, _prompt, **_kwargs):
            return _Response()

    monkeypatch.setattr(editor, "create_llm", lambda: _LLM())
    revised, edits = editor.edit_article_language(
        {"question": "Q", "eval_use_local_prompts": True}, report,
    )
    assert revised == replacement
    assert edits[0]["quote"] == report


def test_node_streams_reset_only_when_changed(monkeypatch):
    report = "原始文章。"
    revised = "修订文章。"
    applied = [{"quote": report, "replacement": revised, "issue_type": "meta_summary"}]
    monkeypatch.setattr(editor, "ENABLE_LANGUAGE_EDITOR", True)
    monkeypatch.setattr(editor, "edit_article_language", lambda *_args, **_kwargs: (revised, applied))
    events = []
    result = editor.language_editor(
        {"report": report}, config=None, writer=lambda event: events.append(event),
    )
    assert events[0]["type"] == "report_reset"
    assert "".join(item.get("content", "") for item in events[1:]) == revised
    assert result["language_edits"] == applied

    monkeypatch.setattr(editor, "edit_article_language", lambda *_args, **_kwargs: (report, []))
    events.clear()
    result = editor.language_editor(
        {"report": report}, config=None, writer=lambda event: events.append(event),
    )
    assert events == []
    assert result["language_editor_changed"] is False
    assert result["language_candidates_count"] == 0


def test_node_discloses_candidate_when_model_applies_nothing(monkeypatch):
    report = "这不是重新训练模型，而是约束输出后的编辑过程。"
    monkeypatch.setattr(editor, "ENABLE_LANGUAGE_EDITOR", True)
    monkeypatch.setattr(editor, "edit_article_language", lambda *_args, **_kwargs: (report, []))
    result = editor.language_editor(
        {"report": report}, config=None, writer=lambda _event: None,
    )
    assert result["language_candidates_count"] == 1
    assert "没有可安全应用" in result["messages"][0]


def test_api_detail_exposes_edit_diff_and_candidate_count():
    from research_buddy.api import _extract_detail

    detail = _extract_detail("language_editor", {
        "language_editor_changed": True,
        "language_candidates_count": 2,
        "language_edits": [{
            "quote": "不是旧句式，而是新的直接表达。",
            "replacement": "新的直接表达。",
            "issue_type": "contrast_template",
            "reason": "删除模板对比",
        }],
    })
    assert detail["candidates_count"] == 2
    assert detail["edits_count"] == 1
    assert detail["edits_preview"][0]["after"] == "新的直接表达。"
