import json
from importlib import import_module

import pytest

article_quality = import_module("research_buddy.eval.article_quality")


def _case():
    return {
        "id": "case-1",
        "question": "为什么？",
        "category": "mechanism",
        "style": "tech-blog",
        "expected_points": ["要点"],
        "risk_tags": ["模板腔"],
        "search_results": [{
            "sub_question_id": "sq_01", "sub_question": "机制", "query": "q",
            "language": "zh", "region": "GLOBAL", "title": "来源",
            "url": "https://example.com/a", "content": "冻结证据", "score": 0.9,
        }],
    }


def test_load_cases_normalizes_search_results(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{
        "id": "a", "question": "问题", "evidence": [{"content": "证据"}],
    }]}), encoding="utf-8")
    cases = article_quality.load_cases(path)
    assert cases[0]["search_results"][0]["sub_question_id"] == "sq_01"
    assert cases[0]["style"] == "tech-blog"


@pytest.mark.parametrize("payload", [
    {}, {"cases": []},
    {"cases": [{"id": "a", "question": "", "evidence": []}]},
    {"cases": [
        {"id": "a", "question": "Q", "evidence": [{"content": "x"}]},
        {"id": "a", "question": "Q2", "evidence": [{"content": "y"}]},
    ]},
])
def test_load_cases_rejects_invalid_data(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(article_quality.EvalDataError):
        article_quality.load_cases(path)


def test_capture_and_append_real_research_case(tmp_path):
    result = {
        "question": "真实失败问题",
        "search_results": [
            {"sub_question": "A", "title": "来源", "url": "https://example.com/a",
             "content": "真实搜索摘要", "score": 0.9},
            {"sub_question": "A", "title": "重复", "url": "https://example.com/a",
             "content": "真实搜索摘要", "score": 0.8},
        ],
    }
    case = article_quality.case_from_research_result(
        "online-failure-1", result, risk_tags=["模板结尾"],
    )
    assert len(case["evidence"]) == 1
    path = tmp_path / "regression.json"
    article_quality.append_case(path, case)
    assert article_quality.load_cases(path)[0]["id"] == "online-failure-1"
    with pytest.raises(article_quality.EvalDataError):
        article_quality.append_case(path, case)


def test_deterministic_metrics_exposes_template_problems():
    report = """# 标题

## 背景：先看定义

值得注意的是，这是第一段。不是简单替换，而是系统改变。

## 结论：回到问题

综上所述，这是第二段。不是局部变化，而是整体变化。不是 A，而是 B。
"""
    metrics = article_quality.deterministic_metrics(report)
    assert metrics["template_phrase_count"] == 2
    assert metrics["colon_heading_ratio"] == 1.0
    assert metrics["ai_flavor_issue_count"] >= 1


def test_deterministic_metrics_detects_rhetorical_scaffolding():
    report = ("有人说应该立即迁移，另一边认为完全没必要。两边都忽略了需求。\n\n"
              "为什么迁移？谁来维护？成本是多少？故障怎么办？\n\n"
              "**第一，检查需求。**\n**第二，检查边界。**\n**第三，检查证据。**\n"
              "说白了，先别做。就这么简单。")
    issues = article_quality.deterministic_metrics(report)["ai_flavor_issues"]
    assert any("疑问/反问句过密" in issue for issue in issues)
    assert any("粗体序号段" in issue for issue in issues)
    assert any("假想双方" in issue for issue in issues)
    assert any("刻意口语化" in issue for issue in issues)


def test_deterministic_metrics_detects_internal_evidence_ids():
    metrics = article_quality.deterministic_metrics("E2 说明系统支持自愈，E10 提供另一条事实。")
    assert metrics["internal_evidence_ref_count"] == 2


def test_judge_failure_is_explicit_not_a_default_score(monkeypatch):
    monkeypatch.setattr(article_quality, "create_llm", lambda **_: (_ for _ in ()).throw(RuntimeError("down")))
    result = article_quality.judge_article(_case(), "文章")
    assert result["available"] is False
    assert "scores" not in result
    assert "down" in result["error"]


def test_parse_scores_rejects_missing_dimension():
    with pytest.raises(ValueError):
        article_quality._parse_scores(
            {dimension: 4 for dimension in article_quality.JUDGE_DIMENSIONS[:-1]}, "一段足够长的测试文章内容",
        )


def test_pairwise_result_maps_blind_order_back_to_candidate(monkeypatch):
    class _Response:
        content = json.dumps({
            "winner": "A", "reason": "A 更好", "dimension_winners": {},
            "evidence": [
                {"article": "A", "quote": "候选文章有一段足够长的原文", "reason": "具体"},
                {"article": "B", "quote": "基线文章有一段足够长的原文", "reason": "具体"},
            ],
        }, ensure_ascii=False)

    monkeypatch.setattr(article_quality, "_blind_order", lambda *_: True)
    monkeypatch.setattr(article_quality, "create_llm", lambda **_: object())
    monkeypatch.setattr(article_quality, "invoke_llm", lambda *_: _Response())
    result = article_quality.compare_pair(
        _case(), {"report": "基线文章有一段足够长的原文"},
        {"report": "候选文章有一段足够长的原文"}, 0,
    )
    assert result["winner"] == "candidate"
    assert result["blind_order"] == "candidate-first"


def test_score_judge_rejects_hallucinated_quote():
    payload = {dimension: 4 for dimension in article_quality.JUDGE_DIMENSIONS}
    payload.update({
        "strengths": [{"quote": "这段文字根本不在文章里面", "reason": "虚构引用"}],
        "problems": [],
    })
    with pytest.raises(ValueError, match="不存在的原文"):
        article_quality._parse_scores(payload, "真实文章里只有另外一段足够长的内容")


def test_blind_order_alternates_within_each_case():
    orders = [article_quality._blind_order("stable-case", index) for index in range(4)]
    assert orders[0] != orders[1]
    assert orders[0] == orders[2]


def test_run_evaluation_builds_gate_and_artifacts(monkeypatch, tmp_path):
    def fake_sample(_case_data, rules, run_judge=True, use_editorial_brief=False,
                    use_evidence_editor=False):
        score = 4.0 if rules == "candidate" else 3.0
        return {
            "generation_available": True,
            "report": rules,
            "editorial_brief": {"enabled": use_editorial_brief},
            "evidence_edits": [{"enabled": use_evidence_editor}],
            "elapsed_seconds": 0.1,
            "token_usage": {"total_tokens": 10},
            "metrics": {
                "char_count": len(rules), "ai_flavor_issue_count": 0,
                "template_phrase_count": 0,
            },
            "judge": {
                "available": True,
                "scores": {dimension: score for dimension in article_quality.JUDGE_DIMENSIONS},
                "average": score,
            },
        }

    monkeypatch.setattr(article_quality, "generate_sample", fake_sample)
    monkeypatch.setattr(article_quality, "compare_pair", lambda *_: {
        "available": True, "winner": "candidate", "reason": "更好",
    })
    result = article_quality.run_evaluation(
        [_case()], "baseline", "candidate", samples=2,
    )
    assert result["summary"]["gate"]["passed"] is True
    assert result["summary"]["gate"]["conclusive"] is True
    assert result["summary"]["pairwise"]["wins"]["candidate"] == 2
    json_path, html_path = article_quality.write_artifacts(result, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert "文章质量回归报告" in html_path.read_text(encoding="utf-8")


def test_gate_catches_dimension_regression():
    summary = {
        "baseline": {
            "dimension_averages": {dimension: 4.0 for dimension in article_quality.JUDGE_DIMENSIONS},
            "ai_flavor_issue_rate": 0.0,
        },
        "candidate": {
            "dimension_averages": {dimension: 3.0 for dimension in article_quality.JUDGE_DIMENSIONS},
            "ai_flavor_issue_rate": 0.0,
        },
        "pairwise": {"available_count": 0, "requested_count": 0},
    }
    gate = article_quality._regression_gate(summary)
    assert gate["passed"] is False
    assert gate["conclusive"] is True
    assert len(gate["reasons"]) == len(article_quality.JUDGE_DIMENSIONS)


def test_gate_without_judges_is_deterministic_only():
    summary = {
        "baseline": {
            "dimension_averages": {dimension: None for dimension in article_quality.JUDGE_DIMENSIONS},
            "ai_flavor_issue_rate": 0.0,
            "generation_failure_count": 0,
        },
        "candidate": {
            "dimension_averages": {dimension: None for dimension in article_quality.JUDGE_DIMENSIONS},
            "ai_flavor_issue_rate": 0.0,
            "generation_failure_count": 0,
        },
        "pairwise": {"available_count": 0, "requested_count": 0},
    }
    gate = article_quality._regression_gate(summary, require_judges=False)
    assert gate == {"passed": True, "conclusive": False, "reasons": []}
    rendered = article_quality.render_html({
        "generated_at": "now", "summary": {**summary, "gate": gate}, "cases": [],
    })
    assert "确定性检查通过，质量结论不完整" in rendered


def test_gate_does_not_compare_unmatched_judge_averages():
    summary = {
        "baseline": {
            "sample_count": 3, "judge_available_count": 1,
            "dimension_averages": {dimension: 5.0 for dimension in article_quality.JUDGE_DIMENSIONS},
            "ai_flavor_issue_rate": 0.0,
        },
        "candidate": {
            "sample_count": 3, "judge_available_count": 3,
            "dimension_averages": {dimension: 1.0 for dimension in article_quality.JUDGE_DIMENSIONS},
            "ai_flavor_issue_rate": 0.0,
        },
        "pairwise": {"available_count": 2, "requested_count": 3},
    }
    gate = article_quality._regression_gate(summary)
    assert gate["conclusive"] is False
    assert any("结果不完整" in reason for reason in gate["reasons"])
    assert not any("下降" in reason for reason in gate["reasons"])
