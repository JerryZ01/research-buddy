import json
from importlib import import_module

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
        {"question": "Q", "search_results": [{"title": "S", "content": "E"}]}, report,
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
        {"question": "Q", "search_results": [{"title": "S", "content": "E"}]}, report,
    )
    assert revised == report and edits == []
    assert '{"edits": []}' in captured["prompt"]
