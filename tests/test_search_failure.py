"""搜索层失败必须可见：不能静默返回空结果让流水线编报告。"""

from importlib import import_module

import pytest

from research_buddy.tools.search import SearchUnavailableError

search_module = import_module("research_buddy.tools.search")
searcher_module = import_module("research_buddy.nodes.searcher")


def _state(**overrides) -> dict:
    state = {
        "question": "测试问题",
        "sub_questions": [{"id": "sq_01", "question": "子问题", "search_query": "query"}],
        "search_results": [],
        "validation_gaps": [],
        "search_history": [],
        "search_round": 0,
        "total_queries": 0,
    }
    state.update(overrides)
    return state


# ── tools/search.py ────────────────────────────────────

def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(search_module, "TAVILY_API_KEY", "")
    with pytest.raises(SearchUnavailableError, match="TAVILY_API_KEY"):
        search_module.search("query")


def test_transient_failure_is_retried_then_raises(monkeypatch):
    attempts = []

    class _Client:
        def search(self, **_kwargs):
            attempts.append(1)
            raise RuntimeError("429 rate limited")

    monkeypatch.setattr(search_module, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(search_module, "_get_tavily_client", lambda: _Client())
    monkeypatch.setattr(search_module.time, "sleep", lambda _s: None)

    with pytest.raises(SearchUnavailableError, match="连续"):
        search_module.search("query")
    assert len(attempts) == search_module._MAX_ATTEMPTS


def test_retry_succeeds_on_second_attempt(monkeypatch):
    attempts = []

    class _Client:
        def search(self, **_kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("connection reset")
            return {"results": [{"title": "t", "url": "https://a.example/x",
                                 "content": "c", "score": 0.7}]}

    monkeypatch.setattr(search_module, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(search_module, "_get_tavily_client", lambda: _Client())
    monkeypatch.setattr(search_module.time, "sleep", lambda _s: None)

    results = search_module.search("query")
    assert len(attempts) == 2
    assert results[0]["url"] == "https://a.example/x"


def test_zero_hits_is_not_a_failure(monkeypatch):
    class _Client:
        def search(self, **_kwargs):
            return {"results": []}

    monkeypatch.setattr(search_module, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(search_module, "_get_tavily_client", lambda: _Client())
    assert search_module.search("query") == []


# ── nodes/searcher.py ──────────────────────────────────

def _boom(_query, **_kwargs):
    raise SearchUnavailableError("TAVILY_API_KEY 未配置，无法执行搜索")


def test_total_failure_without_evidence_aborts(monkeypatch):
    monkeypatch.setattr(searcher_module, "search", _boom)
    with pytest.raises(SearchUnavailableError, match="全部失败"):
        searcher_module.searcher(_state())


def test_total_failure_with_knowledge_degrades(monkeypatch):
    monkeypatch.setattr(searcher_module, "search", _boom)
    output = searcher_module.searcher(_state(
        has_knowledge=True,
        knowledge_context="历史知识：……",
        is_incremental=True,
    ))
    assert output["search_unavailable"] is True
    assert output["stop_reason"] == "search_unavailable"
    assert output["search_results"] == []


def test_total_failure_with_prior_results_does_not_abort(monkeypatch):
    monkeypatch.setattr(searcher_module, "search", _boom)
    prior = [{
        "sub_question_id": "sq_01", "sub_question": "子问题", "query": "old",
        "title": "t", "url": "https://a.example/x", "content": "证据" * 50, "score": 0.8,
    }]
    state = _state(search_results=prior, validation_gaps=[{
        "sub_question_id": "sq_01", "question": "子问题",
        "search_query": "supplement query", "reason": "missing", "priority": "high",
    }])
    output = searcher_module.searcher(state)
    assert output["search_unavailable"] is True
    assert output["stop_reason"] == "search_unavailable"


def test_partial_failure_keeps_going(monkeypatch):
    def _flaky(query, **_kwargs):
        if query == "query-b":
            raise SearchUnavailableError("boom")
        return [{"title": "ok", "url": "https://a.example/ok",
                 "content": "证据" * 50, "score": 0.9}]

    monkeypatch.setattr(searcher_module, "search", _flaky)
    state = _state(sub_questions=[
        {"id": "sq_01", "question": "子问题 A", "search_query": "query-a"},
        {"id": "sq_02", "question": "子问题 B", "search_query": "query-b"},
    ])
    output = searcher_module.searcher(state)
    assert output["search_unavailable"] is False
    assert output["stop_reason"] == ""
    assert len(output["search_results"]) == 1
    assert any("1/2 个查询失败" in msg for msg in output["messages"])


def test_successful_round_clears_unavailable_flag(monkeypatch):
    monkeypatch.setattr(searcher_module, "search", lambda _q, **_k: [
        {"title": "ok", "url": "https://a.example/ok", "content": "证据" * 50, "score": 0.9},
    ])
    output = searcher_module.searcher(_state(search_unavailable=True))
    assert output["search_unavailable"] is False
