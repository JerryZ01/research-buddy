"""证据评估与搜索归属测试。"""

from importlib import import_module

validator_module = import_module("research_buddy.nodes.validator")
searcher_module = import_module("research_buddy.nodes.searcher")
validator_fn = validator_module.validator


def _result(sq_id: str, domain: str, suffix: str, content: str | None = None) -> dict:
    return {
        "sub_question_id": sq_id,
        "sub_question": "测试子问题",
        "query": "test query",
        "title": suffix,
        "url": f"https://{domain}/{suffix}",
        "content": content or ("有效证据内容" * 20),
        "score": 0.8,
    }


def _state(results: list[dict]) -> dict:
    return {
        "question": "测试研究问题",
        "sub_questions": [{"id": "sq_01", "question": "测试子问题", "search_query": "test query"}],
        "search_results": results,
        "search_round": 1,
        "total_queries": 1,
    }


def test_two_independent_sources_are_sufficient(monkeypatch):
    """_llm_assess 返回 None = 语义评估不可用，只按确定性下限判断。"""
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: None)
    result = validator_fn(_state([_result("sq_01", "a.example", "one"), _result("sq_01", "b.example", "two")]))
    assert result["validation_gaps"] == []
    assert result["research_complete"] is True
    assert result["stop_reason"] == "evidence_sufficient"
    # 降级必须被标出来，供 synthesizer 在报告里披露
    assert result["evidence_assessment_degraded"] is True
    assert "语义评估不可用" in result["messages"][0]


def test_branch_skipped_by_evaluator_is_not_sufficient(monkeypatch):
    """评估器答了别的分支却跳过本分支 → fail-closed，不能当作充足。"""
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: {
        "sq_99": {
            "status": "sufficient", "coverage": 0.95,
            "missing_evidence": [], "contradictions": [], "next_queries": [],
        }
    })
    result = validator_fn(_state([_result("sq_01", "a.example", "one"), _result("sq_01", "b.example", "two")]))
    assert result["research_complete"] is False
    assert result["validation_gaps"][0]["sub_question_id"] == "sq_01"
    assert result["validation_gaps"][0]["reason"] == "semantic_assessment_missing"


def test_semantic_gate_uses_soft_threshold(monkeypatch):
    """语义闸用 MIN_SEMANTIC_COVERAGE(0.6) 而非确定性硬闸 0.75：
    复杂话题核心结论可信度 0.6 + sufficient 即通过，不再无限补搜耗尽预算。"""
    assert validator_module.MIN_SEMANTIC_COVERAGE < validator_module.MIN_EVIDENCE_COVERAGE
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: {
        "sq_01": {
            "status": "sufficient",
            "coverage": validator_module.MIN_SEMANTIC_COVERAGE,
            "missing_evidence": [], "contradictions": [], "next_queries": [],
        }
    })
    result = validator_fn(_state([_result("sq_01", "a.example", "one"), _result("sq_01", "b.example", "two")]))
    assert result["validation_gaps"] == []
    assert result["research_complete"] is True
    assert result["stop_reason"] == "evidence_sufficient"


def test_semantic_gate_below_soft_threshold_still_insufficient(monkeypatch):
    """评估器给 partial / coverage 低于软阈值 → 仍判不足并补搜（fail-closed）。"""
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: {
        "sq_01": {
            "status": "partial", "coverage": 0.5,
            "missing_evidence": ["缺少官方统计"],
            "contradictions": [], "next_queries": ["official statistics topic year"],
        }
    })
    result = validator_fn(_state([_result("sq_01", "a.example", "one"), _result("sq_01", "b.example", "two")]))
    assert result["research_complete"] is False
    # reason 是人类可读的缺失证据文本，缺口驱动补搜
    assert result["validation_gaps"][0]["reason"] == "缺少官方统计"
    assert result["evidence_assessment_degraded"] is False


def test_missing_relevance_scores_do_not_inflate_coverage(monkeypatch):
    """所有结果都没有 score 时，相关度是未知而非 0.7，不能白送覆盖度。"""
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: None)
    scored = [_result("sq_01", "a.example", "one"), _result("sq_01", "b.example", "two")]
    unscored = [{**r, "score": 0.0} for r in scored]
    with_scores = validator_fn(_state(scored))["evidence_assessments"][0]["coverage"]
    without_scores = validator_fn(_state(unscored))["evidence_assessments"][0]["coverage"]
    # 有分数：0.5*1 + 0.3*1 + 0.2*0.8 = 0.96；无分数：只算数量与域名 = 0.8
    assert with_scores == 0.96
    assert without_scores == 0.8
    # 未知相关度绝不能比已知的高分更有利
    assert without_scores < with_scores


def test_partial_coverage_without_scores_stays_insufficient(monkeypatch):
    """单一来源 + 无相关度分数：旧实现靠 0.7 默认值凑分，现在必须判不足。"""
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: None)
    single = [{**_result("sq_01", "a.example", "one"), "score": 0.0}]
    result = validator_fn(_state(single))
    assert result["evidence_assessments"][0]["coverage"] == 0.4
    assert result["research_complete"] is False


def test_insufficient_evidence_keeps_original_sub_question_id(monkeypatch):
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: None)
    result = validator_fn(_state([_result("sq_01", "a.example", "one")]))
    assert result["validation_gaps"][0]["sub_question_id"] == "sq_01"
    assert result["research_complete"] is False


def test_semantic_contradiction_forces_more_search(monkeypatch):
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: {
        "sq_01": {
            "status": "sufficient",
            "coverage": 0.9,
            "missing_evidence": [],
            "contradictions": ["两个来源结论相反"],
            "next_queries": ["primary source contradiction evidence"],
        }
    })
    result = validator_fn(_state([_result("sq_01", "a.example", "one"), _result("sq_01", "b.example", "two")]))
    assert result["validation_gaps"][0]["search_query"] == "primary source contradiction evidence"
    assert result["validation_gaps"][0]["priority"] == "high"


def test_budget_exhaustion_is_explicit(monkeypatch):
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: None)
    state = _state([])
    state["search_round"] = validator_module.MAX_SEARCH_ROUNDS
    result = validator_fn(state)
    assert result["stop_reason"] == "search_budget_exhausted"
    assert result["research_complete"] is False


def test_chinese_fallback_query_stays_chinese():
    query = validator_module._fallback_query("中国新能源汽车政策", 0, "insufficient_results", "zh")
    assert "official" not in query
    assert "官方报告" in query


def test_supplemental_search_preserves_branch_and_deduplicates(monkeypatch):
    monkeypatch.setattr(searcher_module, "search", lambda query, **_: [
        {"title": "new", "url": "https://new.example/item", "content": "新证据" * 50, "score": 0.9},
        {"title": "dup", "url": "https://old.example/item?utm_source=test", "content": "旧证据" * 50, "score": 0.8},
    ])
    state = _state([_result("sq_01", "old.example", "item", "旧证据" * 50)])
    state["validation_gaps"] = [{
        "sub_question_id": "sq_01", "question": "测试子问题", "search_query": "new query",
        "reason": "missing", "priority": "high",
    }]
    state["search_history"] = []
    output = searcher_module.searcher(state)
    assert len(output["search_results"]) == 1
    assert output["search_results"][0]["sub_question_id"] == "sq_01"


def test_bilingual_initial_queries_are_both_executed(monkeypatch):
    calls = []

    def fake_search(query, **_):
        calls.append(query)
        return [{
            "title": query,
            "url": f"https://{len(calls)}.example/item",
            "content": (query + " evidence ") * 20,
            "score": 0.8,
        }]

    monkeypatch.setattr(searcher_module, "search", fake_search)
    state = {
        "sub_questions": [{
            "id": "sq_01",
            "question": "比较中国与全球市场",
            "search_query": "中国市场政策",
            "language": "zh",
            "region": "CN",
            "source_preference": "official",
            "search_queries": [
                {"query": "中国市场政策", "language": "zh", "region": "CN"},
                {"query": "global market policy", "language": "en", "region": "GLOBAL"},
            ],
        }],
        "search_results": [],
        "validation_gaps": [],
        "search_history": [],
        "search_round": 0,
        "total_queries": 0,
    }
    output = searcher_module.searcher(state)
    assert set(calls) == {"中国市场政策", "global market policy"}
    assert {result["language"] for result in output["search_results"]} == {"zh", "en"}
    # 初始轮基础搜索不占补搜预算：total_queries 只累计补搜查询
    assert output["total_queries"] == 0


def test_supplement_queries_count_against_budget(monkeypatch):
    """补搜（来自 validation_gaps）的查询才计入 MAX_TOTAL_QUERIES 预算。"""
    calls = []

    def fake_search(query, **_):
        calls.append(query)
        return [{
            "title": query,
            "url": f"https://{len(calls)}.example/item",
            "content": (query + " evidence ") * 20,
            "score": 0.8,
        }]

    monkeypatch.setattr(searcher_module, "search", fake_search)
    # sq_01 已有结果 → 不会重复初始搜索，本轮只有补搜任务
    state = _state([_result("sq_01", "a.example", "existing", "已有内容" * 50)])
    state["total_queries"] = 0
    state["validation_gaps"] = [{
        "sub_question_id": "sq_01", "question": "测试子问题",
        "search_query": "补充查询A", "reason": "low_coverage",
        "priority": "high", "language": "zh", "region": "CN",
    }, {
        "sub_question_id": "sq_01", "question": "测试子问题",
        "search_query": "补充查询B", "reason": "low_coverage",
        "priority": "medium", "language": "zh", "region": "CN",
    }]
    state["search_history"] = []
    output = searcher_module.searcher(state)
    assert len(calls) == 2
    # 两条都是补搜查询，计入预算
    assert output["total_queries"] == 2


def test_searcher_aggregates_image_candidates(monkeypatch):
    """查询级图片 URL 按子问题去重聚合为 image_candidates（视觉选图用）。"""
    calls = []

    def fake_search(query, **_):
        calls.append(query)
        return [{
            "title": query,
            "url": f"https://{len(calls)}.example/item",
            "content": (query + " evidence ") * 20,
            "score": 0.8,
            "images": ["https://img.example/pic.jpg", "https://img.example/pic.jpg",
                       "https://img.example/other.jpg"],
        }]

    monkeypatch.setattr(searcher_module, "search", fake_search)
    state = _state([])
    state["search_history"] = []
    output = searcher_module.searcher(state)
    assert [c["url"] for c in output["image_candidates"]] == [
        "https://img.example/pic.jpg", "https://img.example/other.jpg",
    ]
    assert output["image_candidates"][0]["sub_question_id"] == "sq_01"


def test_image_candidates_dedup_across_rounds(monkeypatch):
    """image_candidates 是追加语义，跨轮次去重由 searcher 基于已有 state 完成。"""
    monkeypatch.setattr(searcher_module, "search", lambda query, **_: [{
        "title": query, "url": "https://x.example/item",
        "content": query * 100, "score": 0.8,
        "images": ["https://img.example/seen.jpg", "https://img.example/new.jpg"],
    }])
    state = _state([])
    state["search_history"] = []
    state["image_candidates"] = [{"url": "https://img.example/seen.jpg",
                                  "sub_question_id": "sq_01", "query": "q"}]
    output = searcher_module.searcher(state)
    assert [c["url"] for c in output["image_candidates"]] == ["https://img.example/new.jpg"]
