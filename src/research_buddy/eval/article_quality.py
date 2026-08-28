"""可重复的文章质量回归评测。

评测刻意冻结搜索证据，只回放 synthesizer。这样 baseline/candidate 的差异
来自写作规则，而不是 Tavily 结果、validator 路由或搜索预算波动。
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import statistics
import time
from copy import deepcopy
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research_buddy.config import (
    ARTICLE_EVAL_JUDGE_MODEL,
    MAX_ARTICLE_TOKENS,
    OPENAI_MODEL,
    WRITER_TEMPERATURE,
)
from research_buddy.nodes.reflector import _ai_flavor_issues
from research_buddy.nodes.editorial_planner import build_editorial_brief
from research_buddy.nodes.article_editor import edit_article_evidence
from research_buddy.nodes.synthesizer import (
    WRITING_RULES,
    build_source_table,
    synthesizer,
)
from research_buddy.styles import get_style_section
from research_buddy.utils import (
    create_llm,
    invoke_llm,
    normalize_url,
    parse_llm_json,
    track_run_tokens,
)


SCHEMA_VERSION = 1
JUDGE_DIMENSIONS = (
    "content_depth",
    "evidence_fidelity",
    "structure_coherence",
    "naturalness",
    "specificity",
    "reader_value",
)

QUALITY_JUDGE_PROMPT = """你是严格的资深中文编辑。请评估文章，不要因为文字流畅就给高分。

研究问题：
{question}

预期要点：
{expected_points}

冻结证据：
{evidence}

待评文章：
{report}

对以下维度分别打 1-5 分：
- content_depth：是否解释原因、机制和影响，而非资料复述
- evidence_fidelity：关键事实是否能由给定证据支持，是否夸大或偷换结论
- structure_coherence：是否有自然推进的主线，章节是否各有必要且不重复
- naturalness：是否像有经验的人写作，避免模板腔、机械排比和刻意口语化
- specificity：是否使用具体事实、机制、数字或真实例子，避免抽象空话
- reader_value：读者是否真正获得了能理解、判断或行动的信息

评分必须拉开差距，3 分表示合格而非差，5 分只给几乎无需编辑的稿件。
只返回 JSON 对象：
{{
  "content_depth": 1,
  "evidence_fidelity": 1,
  "structure_coherence": 1,
  "naturalness": 1,
  "specificity": 1,
  "reader_value": 1,
  "strengths": [{{"quote": "从文章逐字复制的连续片段", "reason": "具体优点"}}],
  "problems": [{{"quote": "从文章逐字复制的连续片段", "reason": "具体问题"}}],
  "editor_summary": "一句编辑结论"
}}

quote 必须逐字复制文章中真实存在的连续文本（10-120 字），不得改写、拼接或制造乱码。至少提供一条 strength 或 problem。"""

PAIRWISE_JUDGE_PROMPT = """你是资深中文编辑。下面两篇文章回答同一个问题，使用相同证据。
文章顺序已随机化，不要猜测它们来自哪个版本。

研究问题：
{question}

预期要点：
{expected_points}

文章 A：
{report_a}

文章 B：
{report_b}

从内容深度、结构推进、具体性、自然度和读者价值综合比较。事实错误优先于文风。
必须引用具体片段解释选择；差异不明显时应选择 tie，不要强行分胜负。
只返回 JSON 对象：
{{
  "winner": "A|B|tie",
  "reason": "具体理由",
  "dimension_winners": {{
    "content": "A|B|tie",
    "structure": "A|B|tie",
    "style": "A|B|tie"
  }},
  "evidence": [
    {{"article": "A", "quote": "从文章 A 逐字复制的连续片段", "reason": "它如何影响判断"}},
    {{"article": "B", "quote": "从文章 B 逐字复制的连续片段", "reason": "它如何影响判断"}}
  ]
}}

必须至少分别引用 A、B 各一段真实原文。quote 必须是对应文章中逐字存在的 10-120 字连续文本，不得改写、拼接或制造乱码。"""

_TEMPLATE_PHRASES = (
    "值得注意的是", "综上所述", "这揭示了", "本节的关键结论是",
    "随着技术的发展", "在当今数字化时代", "赋能", "底层逻辑", "抓手", "闭环",
)
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;]+")
_HEADING_RE = re.compile(r"^#{2,3}\s+(.+)$", re.M)
_INTERNAL_EVIDENCE_RE = re.compile(r"(?<![A-Za-z0-9])E\d+(?![A-Za-z0-9])")


class EvalDataError(ValueError):
    """评测输入不符合本地 schema。"""


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _coefficient_of_variation(values: list[int]) -> float:
    if len(values) < 2 or not statistics.mean(values):
        return 0.0
    return round(statistics.pstdev(values) / statistics.mean(values), 3)


def load_cases(path: str | Path) -> list[dict]:
    """读取并严格校验冻结证据用例。"""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalDataError(f"无法读取评测用例 {source}: {exc}") from exc
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise EvalDataError("评测文件必须包含非空 cases 列表")

    seen: set[str] = set()
    normalized: list[dict] = []
    for index, raw in enumerate(cases, 1):
        if not isinstance(raw, dict):
            raise EvalDataError(f"第 {index} 条用例不是对象")
        case_id = str(raw.get("id", "")).strip()
        question = str(raw.get("question", "")).strip()
        evidence = raw.get("evidence")
        if not case_id or not question or not isinstance(evidence, list) or not evidence:
            raise EvalDataError(f"第 {index} 条用例缺少 id/question/evidence")
        if case_id in seen:
            raise EvalDataError(f"重复用例 id: {case_id}")
        seen.add(case_id)

        results = []
        for evidence_index, item in enumerate(evidence, 1):
            if not isinstance(item, dict) or not str(item.get("content", "")).strip():
                raise EvalDataError(f"用例 {case_id} 的第 {evidence_index} 条证据缺少 content")
            results.append({
                "sub_question_id": str(item.get("sub_question_id", "sq_01")),
                "sub_question": str(item.get("sub_question", question)),
                "query": str(item.get("query", question)),
                "language": str(item.get("language", "zh")),
                "region": str(item.get("region", "GLOBAL")),
                "title": str(item.get("title", "未命名来源")),
                "url": str(item.get("url", "")),
                "content": str(item["content"]),
                "score": float(item.get("score", 1.0)),
            })
        expected_points = raw.get("expected_points", [])
        risk_tags = raw.get("risk_tags", [])
        if not isinstance(expected_points, list) or not isinstance(risk_tags, list):
            raise EvalDataError(f"用例 {case_id} 的 expected_points/risk_tags 必须是列表")
        normalized.append({
            "id": case_id,
            "question": question,
            "category": str(raw.get("category", "general")),
            "style": str(raw.get("style", "tech-blog")),
            "expected_points": [str(p) for p in expected_points],
            "risk_tags": [str(tag) for tag in risk_tags],
            "search_results": results,
        })
    return normalized


def case_from_research_result(case_id: str, result: dict, category: str = "general",
                              style: str = "tech-blog",
                              expected_points: list[str] | None = None,
                              risk_tags: list[str] | None = None) -> dict:
    """把一次真实研究的完整 state 固化为可提交的冻结证据用例。"""
    question = str(result.get("question", "")).strip()
    search_results = result.get("search_results", [])
    if not case_id.strip() or not question or not isinstance(search_results, list) or not search_results:
        raise EvalDataError("研究结果必须包含 question 和非空 search_results")
    evidence = []
    seen: set[tuple[str, str]] = set()
    for item in search_results:
        if not isinstance(item, dict) or not str(item.get("content", "")).strip():
            continue
        key = (normalize_url(str(item.get("url", ""))), str(item.get("content", ""))[:160])
        if key in seen:
            continue
        seen.add(key)
        evidence.append({key: item.get(key) for key in (
            "sub_question_id", "sub_question", "query", "language", "region",
            "title", "url", "content", "score",
        )})
    if not evidence:
        raise EvalDataError("研究结果中没有可用的搜索证据")
    return {
        "id": case_id.strip(),
        "question": question,
        "category": category,
        "style": style,
        "risk_tags": risk_tags or [],
        "expected_points": expected_points or [],
        "evidence": evidence,
    }


def append_case(path: str | Path, case: dict) -> None:
    """追加用例并拒绝重复 ID；写入后重新加载，保证文件仍符合 schema。"""
    target = Path(path)
    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvalDataError(f"目标用例文件不是合法 JSON: {exc}") from exc
    else:
        payload = {"schema_version": SCHEMA_VERSION, "cases": []}
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise EvalDataError("目标用例文件缺少 cases 列表")
    if any(isinstance(item, dict) and item.get("id") == case.get("id") for item in cases):
        raise EvalDataError(f"用例 id 已存在: {case.get('id')}")
    cases.append(case)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        load_cases(temp)
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()


def deterministic_metrics(report: str) -> dict:
    """无需模型、完全可重复的文风和结构指标。"""
    body = report.split("\n## 参考文献", 1)[0]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    prose_paragraphs = [p for p in paragraphs if not p.startswith(("#", "```", "|"))]
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(body) if len(s.strip()) >= 4]
    headings = _HEADING_RE.findall(body)
    colon_headings = [heading for heading in headings if "：" in heading]
    openers: dict[str, int] = {}
    for sentence in sentences:
        opener = re.sub(r"\s+", "", sentence)[:6]
        if opener:
            openers[opener] = openers.get(opener, 0) + 1
    repeated_openers = {key: count for key, count in openers.items() if count >= 3}
    template_hits = {
        phrase: body.count(phrase) for phrase in _TEMPLATE_PHRASES if phrase in body
    }
    issues = _ai_flavor_issues(body) + _evaluation_style_issues(body)
    return {
        "char_count": len(body),
        "paragraph_count": len(prose_paragraphs),
        "heading_count": len(headings),
        "colon_heading_ratio": round(len(colon_headings) / len(headings), 3) if headings else 0.0,
        "paragraph_length_cv": _coefficient_of_variation([len(p) for p in prose_paragraphs]),
        "sentence_length_cv": _coefficient_of_variation([len(s) for s in sentences]),
        "template_phrase_count": sum(template_hits.values()),
        "template_phrase_hits": template_hits,
        "repeated_sentence_openers": repeated_openers,
        "internal_evidence_ref_count": len(_INTERNAL_EVIDENCE_RE.findall(body)),
        "ai_flavor_issue_count": len(issues),
        "ai_flavor_issues": issues,
    }


def _evaluation_style_issues(body: str) -> list[str]:
    """评测专用的高精度模式；未验证前不改变生产 reflector 行为。"""
    issues = []
    question_count = len(re.findall(r"[？?]", body))
    question_density = question_count * 1000 / max(len(body), 1)
    if question_count >= 4 and question_density >= 2.0:
        issues.append(f"疑问/反问句过密（{question_count} 处，约 {question_density:.1f} 处/千字）")
    ordinal_blocks = re.findall(
        r"^\*\*(?:第[一二三四五六七八九十]+|首先|其次|最后)[^\n]{0,40}\*\*",
        body, re.M,
    )
    if len(ordinal_blocks) >= 3:
        issues.append(f"存在 {len(ordinal_blocks)} 个粗体序号段，结构像模板化清单")
    staged_opposition = len(re.findall(r"有人说|另一边|两边都|一派认为|另一派认为", body))
    if staged_opposition >= 2:
        issues.append("用假想双方观点搭建开场，信息密度偏低")
    colloquial_fillers = sum(body.count(phrase) for phrase in ("说白了", "就这么简单"))
    if colloquial_fillers >= 2:
        issues.append("反复使用刻意口语化的过渡或收束")
    return issues


def _format_evidence(case: dict, max_chars: int = 14000) -> str:
    parts = []
    for index, item in enumerate(case["search_results"], 1):
        parts.append(
            f"[{index}] {item['title']}\nURL: {item['url']}\n{item['content']}"
        )
    return "\n\n".join(parts)[:max_chars]


def _validated_annotations(value: Any, report: str, label: str) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError(f"Judge {label} 不是列表")
    annotations = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Judge {label} 条目不是对象")
        quote = str(item.get("quote", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not 10 <= len(quote) <= 120 or quote not in report:
            raise ValueError(f"Judge {label} 引用了不存在的原文: {quote[:60]!r}")
        if not reason:
            raise ValueError(f"Judge {label} 缺少 reason")
        annotations.append({"quote": quote, "reason": reason})
    return annotations


def _parse_scores(payload: Any, report: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Judge 返回值不是 JSON 对象")
    scores = {}
    for dimension in JUDGE_DIMENSIONS:
        value = payload.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
            raise ValueError(f"Judge 维度 {dimension} 无效: {value!r}")
        scores[dimension] = float(value)
    strengths = _validated_annotations(payload.get("strengths"), report, "strengths")
    problems = _validated_annotations(payload.get("problems"), report, "problems")
    if not strengths and not problems:
        raise ValueError("Judge 没有提供任何可验证的原文批注")
    return {
        "available": True,
        "scores": scores,
        "average": _mean(list(scores.values())),
        "strengths": strengths,
        "problems": problems,
        "editor_summary": str(payload.get("editor_summary", "")),
    }


def judge_article(case: dict, report: str) -> dict:
    """分层 Judge；失败显式返回 available=False，不制造占位分。"""
    base_prompt = QUALITY_JUDGE_PROMPT.format(
        question=case["question"],
        expected_points="\n".join(f"- {p}" for p in case["expected_points"]) or "（未设置）",
        evidence=_format_evidence(case),
        report=report[:24000],
    )
    last_error: Exception | None = None
    for attempt in range(2):
        prompt = base_prompt
        if attempt:
            prompt += ("\n\n上一次输出因原文引文不精确而无效。请重新独立评分；quote 只复制文章中一段"
                       "不含代码的短句，逐字复制，不要修正、概括或拼接。")
        try:
            response = invoke_llm(create_llm(model=ARTICLE_EVAL_JUDGE_MODEL), prompt)
            return _parse_scores(parse_llm_json(response.content), report)
        except Exception as exc:
            last_error = exc
    return {"available": False, "attempts": 2,
            "error": f"{type(last_error).__name__}: {last_error}"}


def _blind_order(case_id: str, sample_index: int) -> bool:
    # 每题首个方向由稳定哈希决定，后续采样严格交替，避免小样本恰好全是同一顺序。
    digest = hashlib.sha256(case_id.encode()).digest()
    return bool((digest[0] + sample_index) % 2)


def compare_pair(case: dict, baseline: dict, candidate: dict, sample_index: int) -> dict:
    """随机化文章顺序后做 A/B 盲评，并把结果映射回版本名。"""
    candidate_first = _blind_order(case["id"], sample_index)
    report_a = candidate["report"] if candidate_first else baseline["report"]
    report_b = baseline["report"] if candidate_first else candidate["report"]
    base_prompt = PAIRWISE_JUDGE_PROMPT.format(
        question=case["question"],
        expected_points="\n".join(f"- {p}" for p in case["expected_points"]) or "（未设置）",
        report_a=report_a[:24000],
        report_b=report_b[:24000],
    )
    last_error: Exception | None = None
    for attempt in range(2):
        prompt = base_prompt
        if attempt:
            prompt += ("\n\n上一次输出因原文引文不精确而无效。重新比较；A、B 各只引用一段"
                       "不含代码的短句，逐字复制，不要修正、概括或拼接。")
        try:
            response = invoke_llm(create_llm(model=ARTICLE_EVAL_JUDGE_MODEL), prompt)
            payload = parse_llm_json(response.content)
            if not isinstance(payload, dict) or payload.get("winner") not in {"A", "B", "tie"}:
                raise ValueError("A/B Judge winner 无效")
            evidence = payload.get("evidence")
            if not isinstance(evidence, list):
                raise ValueError("A/B Judge 缺少 evidence")
            validated_evidence = []
            seen_articles = set()
            for item in evidence:
                if not isinstance(item, dict) or item.get("article") not in {"A", "B"}:
                    raise ValueError("A/B Judge evidence 条目无效")
                article = item["article"]
                quote = str(item.get("quote", "")).strip()
                source_report = report_a if article == "A" else report_b
                if not 10 <= len(quote) <= 120 or quote not in source_report:
                    raise ValueError(f"A/B Judge 引用了 {article} 中不存在的原文: {quote[:60]!r}")
                seen_articles.add(article)
                validated_evidence.append({
                    "article": article, "quote": quote,
                    "reason": str(item.get("reason", "")).strip(),
                })
            if seen_articles != {"A", "B"}:
                raise ValueError("A/B Judge 必须分别引用 A 和 B")
            winner = payload["winner"]
            if winner != "tie":
                a_version = "candidate" if candidate_first else "baseline"
                winner = a_version if winner == "A" else ("baseline" if a_version == "candidate" else "candidate")
            return {
                "available": True,
                "winner": winner,
                "reason": str(payload.get("reason", "")),
                "dimension_winners": payload.get("dimension_winners", {}),
                "evidence": validated_evidence,
                "blind_order": "candidate-first" if candidate_first else "baseline-first",
            }
        except Exception as exc:
            last_error = exc
    return {"available": False, "attempts": 2,
            "error": f"{type(last_error).__name__}: {last_error}"}


def _source_signature(source_table: list[dict]) -> str:
    return "|".join(sorted(
        normalize_url(item.get("url", "")) for item in source_table if item.get("url")
    ))


def generate_sample(case: dict, rules: str, run_judge: bool = True,
                    use_editorial_brief: bool = False,
                    use_evidence_editor: bool = False) -> dict:
    """用真实 synthesizer 回放一份冻结证据。"""
    state = {
        "question": case["question"],
        "style": case["style"],
        "search_results": case["search_results"],
        "writing_rules_override": rules,
        "style_section_override": get_style_section(case["style"]),
        "eval_use_local_prompts": True,
        "evidence_assessments": [],
        "validation_gaps": [],
        "selected_images": [],
        "image_candidates": [],
    }
    table = build_source_table(state)
    state["core_refs"] = table
    state["core_refs_signature"] = _source_signature(table)
    started = time.perf_counter()
    usage: dict = {}
    try:
        with track_run_tokens() as tracked_usage:
            usage = tracked_usage
            if use_editorial_brief:
                state["editorial_brief"] = build_editorial_brief(
                    state, config={}, use_local_prompt=True,
                )
            # synthesizer 的流式正文会 print；评测结果统一进 artifact，不污染 CLI。
            with redirect_stdout(io.StringIO()):
                result = synthesizer(state, {}, writer=lambda _event: None)
            if use_evidence_editor:
                edited_report, evidence_edits = edit_article_evidence(
                    # 历史回归样本使用两轮定向编辑；生产默认一轮的
                    # 延迟/费用限制不应改变已有评测协议。
                    state, result.get("report", ""), config={}, max_rounds=2,
                )
                result["report"] = edited_report
                state["evidence_edits"] = evidence_edits
            usage = dict(tracked_usage)
    except Exception as exc:
        return {
            "generation_available": False,
            "generation_error": f"{type(exc).__name__}: {exc}",
            "report": "",
            "editorial_brief": state.get("editorial_brief", {}),
            "evidence_edits": state.get("evidence_edits", []),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "token_usage": dict(usage),
            "metrics": deterministic_metrics(""),
            "judge": {"available": False, "skipped": True,
                      "error": "文章生成失败，未运行 Judge"},
        }
    report = result.get("report", "")
    sample = {
        "generation_available": True,
        "report": report,
        "editorial_brief": state.get("editorial_brief", {}),
        "evidence_edits": state.get("evidence_edits", []),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "token_usage": dict(usage),
        "metrics": deterministic_metrics(report),
    }
    sample["judge"] = judge_article(case, report) if run_judge else {"available": False, "skipped": True}
    return sample


def _variant_summary(cases: list[dict], variant: str) -> dict:
    samples = [sample for case in cases for sample in case[variant]["samples"]]
    generated_samples = [s for s in samples if s.get("generation_available", True)]
    judge_samples = [s for s in samples if s["judge"].get("available")]
    dimensions = {
        dimension: _mean([s["judge"]["scores"][dimension] for s in judge_samples])
        for dimension in JUDGE_DIMENSIONS
    }
    return {
        "sample_count": len(samples),
        "generation_failure_count": len(samples) - len(generated_samples),
        "judge_available_count": len(judge_samples),
        "dimension_averages": dimensions,
        "judge_average": _mean([s["judge"]["average"] for s in judge_samples]),
        "ai_flavor_issue_rate": _mean([
            1.0 if s["metrics"]["ai_flavor_issue_count"] else 0.0 for s in generated_samples
        ]),
        "template_phrase_average": _mean([
            float(s["metrics"]["template_phrase_count"]) for s in generated_samples
        ]),
        "internal_evidence_ref_average": _mean([
            float(s["metrics"].get("internal_evidence_ref_count", 0)) for s in generated_samples
        ]),
        "average_chars": _mean([float(s["metrics"]["char_count"]) for s in generated_samples]),
        "average_seconds": _mean([float(s["elapsed_seconds"]) for s in samples]),
        "total_tokens": sum(int(s["token_usage"].get("total_tokens", 0)) for s in samples),
    }


def _regression_gate(summary: dict, max_dimension_drop: float = 0.25,
                     require_judges: bool = True) -> dict:
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    reasons = []
    judge_results_complete = True
    if candidate.get("generation_failure_count", 0):
        reasons.append(f"Candidate 有 {candidate['generation_failure_count']} 次文章生成失败")
    if require_judges:
        for name, variant in (("Baseline", baseline), ("Candidate", candidate)):
            if variant.get("judge_available_count", 0) < variant.get("sample_count", 0):
                judge_results_complete = False
                reasons.append(
                    f"{name} Judge 结果不完整"
                    f"（{variant.get('judge_available_count', 0)}/{variant.get('sample_count', 0)}）"
                )
        pairwise = summary.get("pairwise", {})
        if pairwise.get("available_count", 0) < pairwise.get("requested_count", 0):
            judge_results_complete = False
            reasons.append(
                "A/B Judge 结果不完整"
                f"（{pairwise.get('available_count', 0)}/{pairwise.get('requested_count', 0)}）"
            )
    if judge_results_complete:
        for dimension in JUDGE_DIMENSIONS:
            before = baseline["dimension_averages"].get(dimension)
            after = candidate["dimension_averages"].get(dimension)
            if before is not None and after is not None and after < before - max_dimension_drop:
                reasons.append(f"{dimension} 下降 {before - after:.2f}（{before:.2f} → {after:.2f}）")
    before_rate = baseline.get("ai_flavor_issue_rate")
    after_rate = candidate.get("ai_flavor_issue_rate")
    if before_rate is not None and after_rate is not None and after_rate > before_rate:
        reasons.append(f"AI 模板硬规则命中率上升（{before_rate:.1%} → {after_rate:.1%}）")
    before_refs = baseline.get("internal_evidence_ref_average", 0) or 0
    after_refs = candidate.get("internal_evidence_ref_average", 0) or 0
    if after_refs > before_refs:
        reasons.append(f"正文内部证据编号增加（平均 {before_refs:.2f} → {after_refs:.2f}）")
    comparisons = summary.get("pairwise", {})
    if comparisons.get("available_count", 0) >= 3 and comparisons.get("candidate_loss_rate", 0) > 0.4:
        reasons.append(f"A/B 盲评候选败率过高（{comparisons['candidate_loss_rate']:.1%}）")
    return {
        "passed": not reasons,
        "conclusive": require_judges and not any(
            "Judge 结果不完整" in reason for reason in reasons
        ),
        "reasons": reasons,
    }


def run_evaluation(cases: list[dict], baseline_rules: str, candidate_rules: str,
                   samples: int = 3, run_judges: bool = True,
                   baseline_label: str = "baseline",
                   candidate_label: str = "candidate",
                   baseline_editorial_brief: bool = False,
                   candidate_editorial_brief: bool = False,
                   baseline_evidence_editor: bool = False,
                   candidate_evidence_editor: bool = False,
                   progress: Callable[[str], None] | None = None) -> dict:
    """运行完整 baseline/candidate 回归。"""
    if samples < 1:
        raise ValueError("samples 必须至少为 1")
    case_results = []
    for case in cases:
        result = {
            "id": case["id"], "question": case["question"],
            "category": case["category"], "style": case["style"],
            "risk_tags": case["risk_tags"], "expected_points": case["expected_points"],
            "search_results": case["search_results"],
            "baseline": {"label": baseline_label, "samples": []},
            "candidate": {"label": candidate_label, "samples": []},
            "comparisons": [],
        }
        for sample_index in range(samples):
            if progress:
                progress(f"[{case['id']}] 采样 {sample_index + 1}/{samples}: baseline")
            baseline = generate_sample(
                case, baseline_rules, run_judge=run_judges,
                use_editorial_brief=baseline_editorial_brief,
                use_evidence_editor=baseline_evidence_editor,
            )
            if progress:
                progress(f"[{case['id']}] 采样 {sample_index + 1}/{samples}: candidate")
            candidate = generate_sample(
                case, candidate_rules, run_judge=run_judges,
                use_editorial_brief=candidate_editorial_brief,
                use_evidence_editor=candidate_evidence_editor,
            )
            result["baseline"]["samples"].append(baseline)
            result["candidate"]["samples"].append(candidate)
            if run_judges:
                if (baseline.get("generation_available", True)
                        and candidate.get("generation_available", True)):
                    if progress:
                        progress(f"[{case['id']}] 采样 {sample_index + 1}/{samples}: A/B Judge")
                    result["comparisons"].append(
                        compare_pair(case, baseline, candidate, sample_index)
                    )
                else:
                    result["comparisons"].append({
                        "available": False,
                        "error": "baseline 或 candidate 文章生成失败，未运行 A/B Judge",
                    })
        case_results.append(result)

    pairwise = [comparison for case in case_results for comparison in case["comparisons"]
                if comparison.get("available")]
    wins = {winner: sum(c["winner"] == winner for c in pairwise)
            for winner in ("baseline", "candidate", "tie")}
    summary = {
        "baseline": _variant_summary(case_results, "baseline"),
        "candidate": _variant_summary(case_results, "candidate"),
        "pairwise": {
            "requested_count": len(case_results) * samples if run_judges else 0,
            "available_count": len(pairwise),
            "wins": wins,
            "candidate_win_rate": round(wins["candidate"] / len(pairwise), 3) if pairwise else None,
            "candidate_loss_rate": round(wins["baseline"] / len(pairwise), 3) if pairwise else None,
        },
    }
    summary["gate"] = _regression_gate(summary, require_judges=run_judges)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": OPENAI_MODEL,
            "judge_model": ARTICLE_EVAL_JUDGE_MODEL,
            "writer_temperature": WRITER_TEMPERATURE,
            "max_article_tokens": MAX_ARTICLE_TOKENS,
            "samples_per_variant": samples,
            "judges_enabled": run_judges,
            "baseline_label": baseline_label,
            "candidate_label": candidate_label,
            "baseline_editorial_brief": baseline_editorial_brief,
            "candidate_editorial_brief": candidate_editorial_brief,
            "baseline_evidence_editor": baseline_evidence_editor,
            "candidate_evidence_editor": candidate_evidence_editor,
            "baseline_rules_sha256": hashlib.sha256(baseline_rules.encode()).hexdigest(),
            "candidate_rules_sha256": hashlib.sha256(candidate_rules.encode()).hexdigest(),
            "case_ids": [case["id"] for case in cases],
        },
        "prompt_snapshots": {
            "baseline_writing_rules": baseline_rules,
            "candidate_writing_rules": candidate_rules,
        },
        "summary": summary,
        "cases": case_results,
    }


def rejudge_evaluation(result: dict, cases: list[dict],
                       progress: Callable[[str], None] | None = None) -> dict:
    """复用已有文章重新运行 Judge，不再次调用写作模型。"""
    revised = deepcopy(result)
    case_lookup = {case["id"]: case for case in cases}
    for case_result in revised.get("cases", []):
        case_id = case_result.get("id")
        if case_id not in case_lookup:
            raise EvalDataError(f"重评找不到冻结用例: {case_id}")
        case = case_lookup[case_id]
        case_result["search_results"] = case["search_results"]
        case_result["comparisons"] = []
        for sample_index, (baseline, candidate) in enumerate(zip(
                case_result["baseline"]["samples"],
                case_result["candidate"]["samples"]), 1):
            if progress:
                progress(f"[{case_id}] 重评 {sample_index}: baseline / candidate")
            baseline["judge"] = judge_article(case, baseline["report"])
            candidate["judge"] = judge_article(case, candidate["report"])
            if baseline.get("generation_available", True) and candidate.get("generation_available", True):
                case_result["comparisons"].append(
                    compare_pair(case, baseline, candidate, sample_index - 1)
                )

    pairwise = [comparison for case in revised["cases"] for comparison in case["comparisons"]
                if comparison.get("available")]
    wins = {winner: sum(c["winner"] == winner for c in pairwise)
            for winner in ("baseline", "candidate", "tie")}
    requested = sum(len(case["comparisons"]) for case in revised["cases"])
    summary = {
        "baseline": _variant_summary(revised["cases"], "baseline"),
        "candidate": _variant_summary(revised["cases"], "candidate"),
        "pairwise": {
            "requested_count": requested,
            "available_count": len(pairwise),
            "wins": wins,
            "candidate_win_rate": round(wins["candidate"] / len(pairwise), 3) if pairwise else None,
            "candidate_loss_rate": round(wins["baseline"] / len(pairwise), 3) if pairwise else None,
        },
    }
    summary["gate"] = _regression_gate(summary, require_judges=True)
    revised["summary"] = summary
    revised["config"]["judges_enabled"] = True
    revised["config"]["judge_model"] = ARTICLE_EVAL_JUDGE_MODEL
    revised["rejudged_at"] = datetime.now(timezone.utc).isoformat()
    return revised


def _score_text(judge: dict) -> str:
    if not judge.get("available"):
        return "Judge 跳过或失败"
    return " · ".join(f"{key} {value:g}" for key, value in judge["scores"].items())


def render_html(result: dict) -> str:
    """生成完全离线的单文件 HTML，可直接双击查看。"""
    summary = result["summary"]
    gate = summary["gate"]
    case_sections = []
    for case in result["cases"]:
        samples = []
        for index, (baseline, candidate) in enumerate(zip(
                case["baseline"]["samples"], case["candidate"]["samples"]), 1):
            comparison = case["comparisons"][index - 1] if case["comparisons"] else {}
            samples.append(f"""
            <details {'open' if index == 1 else ''}>
              <summary>采样 {index} · A/B: {html.escape(str(comparison.get('winner', '未评')))}</summary>
              <p class="reason">{html.escape(str(comparison.get('reason', comparison.get('error', ''))))}</p>
              <div class="columns">
                <section><h3>Baseline</h3><p>{html.escape(_score_text(baseline['judge']))}</p>
                  <p>硬规则问题 {baseline['metrics']['ai_flavor_issue_count']} · {baseline['elapsed_seconds']}s</p>
                  <details><summary>编辑简报</summary><pre>{html.escape(json.dumps(baseline.get('editorial_brief', {}), ensure_ascii=False, indent=2))}</pre></details>
                  <details><summary>证据编辑</summary><pre>{html.escape(json.dumps(baseline.get('evidence_edits', []), ensure_ascii=False, indent=2))}</pre></details>
                  <pre>{html.escape(baseline['report'])}</pre></section>
                <section><h3>Candidate</h3><p>{html.escape(_score_text(candidate['judge']))}</p>
                  <p>硬规则问题 {candidate['metrics']['ai_flavor_issue_count']} · {candidate['elapsed_seconds']}s</p>
                  <details><summary>编辑简报</summary><pre>{html.escape(json.dumps(candidate.get('editorial_brief', {}), ensure_ascii=False, indent=2))}</pre></details>
                  <details><summary>证据编辑</summary><pre>{html.escape(json.dumps(candidate.get('evidence_edits', []), ensure_ascii=False, indent=2))}</pre></details>
                  <pre>{html.escape(candidate['report'])}</pre></section>
              </div>
            </details>""")
        case_sections.append(f"""
        <article>
          <h2>{html.escape(case['question'])}</h2>
          <p class="meta">{html.escape(case['id'])} · {html.escape(case['category'])} · {html.escape(case['style'])} · {' / '.join(map(html.escape, case['risk_tags']))}</p>
          {''.join(samples)}
        </article>""")

    def value(path: str, default: Any = "-") -> Any:
        current: Any = summary
        for key in path.split("."):
            current = current.get(key) if isinstance(current, dict) else None
        return default if current is None else current

    gate_class = "pass" if gate["passed"] else "fail"
    if not gate.get("conclusive", True):
        gate_text = "确定性检查通过，质量结论不完整" if gate["passed"] else "确定性检查未通过，质量结论不完整"
    else:
        gate_text = "通过" if gate["passed"] else "未通过"
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in gate["reasons"])
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Buddy 文章质量回归</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f5f6f8;color:#17191d;line-height:1.55}}main{{max-width:1500px;margin:auto;padding:28px}}
h1,h2,h3{{letter-spacing:0}}.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:20px 0}}
.metric,article{{background:white;border:1px solid #dfe2e7;border-radius:6px;padding:16px}}.metric strong{{display:block;font-size:1.4rem}}
.gate.pass{{color:#176b3a}}.gate.fail{{color:#a52626}}article{{margin:18px 0}}.meta,.reason,section>p{{color:#626874;font-size:.88rem}}
details{{border-top:1px solid #e4e6ea;padding:12px 0}}summary{{cursor:pointer;font-weight:650}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
section{{min-width:0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#fafafa;border:1px solid #e1e3e7;padding:16px;max-height:720px;overflow:auto;font:14px/1.7 ui-monospace,monospace}}
@media(max-width:900px){{main{{padding:14px}}.columns{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>文章质量回归报告</h1><p>生成时间：{html.escape(result['generated_at'])}</p>
<div class="summary">
<div class="metric"><span>回归门禁</span><strong class="gate {gate_class}">{gate_text}</strong></div>
<div class="metric"><span>Baseline Judge</span><strong>{value('baseline.judge_average')}</strong></div>
<div class="metric"><span>Candidate Judge</span><strong>{value('candidate.judge_average')}</strong></div>
<div class="metric"><span>Candidate A/B 胜率</span><strong>{value('pairwise.candidate_win_rate')}</strong></div>
<div class="metric"><span>有效盲评</span><strong>{value('pairwise.available_count')}</strong></div>
</div><ul>{reasons}</ul>{''.join(case_sections)}
</main></body></html>"""


def write_artifacts(result: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "results.json"
    html_path = output / "report.html"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(result), encoding="utf-8")
    return json_path, html_path


def default_baseline_rules() -> str:
    return WRITING_RULES
