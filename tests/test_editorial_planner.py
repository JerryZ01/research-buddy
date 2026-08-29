import json
from importlib import import_module

import pytest

planner = import_module("research_buddy.nodes.editorial_planner")


def _payload():
    return {
        "intent": "mechanism",
        "audience": "Python 开发者",
        "thesis": "GIL 限制的是字节码执行权",
        "scope_include": ["CPU 与 I/O 的差异"],
        "scope_exclude": ["无 GIL Python 的未来"],
        "must_cover": [{"point": "阻塞 I/O 释放锁", "evidence_ids": ["E1", "E9"]}],
        "section_plan": [
            {"heading": "锁住的是什么", "purpose": "建立机制", "evidence_ids": ["E1"]},
            {"heading": "等待为何不同", "purpose": "完成对比", "evidence_ids": ["E1", "E2"]},
        ],
        "claims_to_avoid": ["具体切换间隔"],
        "ending": "回到任务类型判断",
    }


def test_normalize_brief_drops_hallucinated_evidence_ids():
    result = planner.normalize_editorial_brief(_payload(), evidence_count=2)
    assert result["must_cover"][0]["evidence_ids"] == ["E1"]
    assert result["section_plan"][1]["evidence_ids"] == ["E1", "E2"]


@pytest.mark.parametrize("payload", [{}, [], {"thesis": "x", "must_cover": [], "section_plan": []}])
def test_normalize_brief_rejects_incomplete_payload(payload):
    with pytest.raises(ValueError):
        planner.normalize_editorial_brief(payload, evidence_count=2)


def test_build_brief_uses_local_prompt_and_validates(monkeypatch):
    class _Response:
        content = json.dumps(_payload(), ensure_ascii=False)

    class _LLM:
        def invoke(self, prompt, **_kwargs):
            assert "E1" in prompt
            return _Response()

    monkeypatch.setattr(planner, "create_llm", lambda: _LLM())
    state = {
        "question": "为什么", "style": "tech-blog",
        "search_results": [
            {"title": "A", "content": "证据一"},
            {"title": "B", "content": "证据二"},
        ],
    }
    result = planner.build_editorial_brief(state, use_local_prompt=True)
    assert result["thesis"] == "GIL 限制的是字节码执行权"


def test_editorial_prompt_limits_sections_and_bans_metaphorical_headings():
    assert "通常规划 2-4 个核心章节" in planner.EDITORIAL_BRIEF_PROMPT
    assert "禁止比喻、拟人、口号" in planner.EDITORIAL_BRIEF_PROMPT


def test_build_brief_fails_open(monkeypatch):
    monkeypatch.setattr(planner, "create_llm", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert planner.build_editorial_brief({"search_results": [{"content": "x"}]}) == {}


def test_evidence_ledger_deduplicates_urls_and_merges_assessment():
    state = {
        "search_results": [
            {
                "sub_question_id": "sq_1", "title": "A", "url": "https://example.com/a?utm_source=x",
                "content": "短证据", "score": 0.5,
            },
            {
                "sub_question_id": "sq_2", "title": "A2", "url": "https://example.com/a",
                "content": "这是同一来源的更长证据内容", "score": 0.9,
            },
        ],
        "evidence_assessments": [
            {"sub_question_id": "sq_1", "status": "sufficient", "contradictions": []},
            {"sub_question_id": "sq_2", "status": "insufficient", "contradictions": ["口径不一致"]},
        ],
    }
    ledger = planner.build_evidence_ledger(state)
    assert len(ledger) == 1
    assert ledger[0]["id"] == "E1"
    assert ledger[0]["excerpt"] == "这是同一来源的更长证据内容"
    assert ledger[0]["sub_question_ids"] == ["sq_1", "sq_2"]
    assert ledger[0]["assessment_status"] == "insufficient"
    assert ledger[0]["contradictions"] == ["口径不一致"]


def test_editorial_node_returns_ledger_and_brief(monkeypatch):
    monkeypatch.setattr(
        planner, "build_editorial_brief",
        lambda _state, config=None, evidence_ledger=None: {"thesis": "核心判断", "section_plan": []},
    )
    result = planner.editorial_planner({
        "search_results": [{"title": "A", "url": "https://a.example", "content": "证据"}],
    })
    assert result["evidence_ledger"][0]["id"] == "E1"
    assert result["editorial_brief"]["thesis"] == "核心判断"
