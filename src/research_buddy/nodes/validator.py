"""证据评估节点 - 结合确定性指标和 LLM 语义判断决定是否继续搜索。"""

import json
import logging

from langchain_core.runnables import RunnableConfig
from urllib.parse import urlsplit

from research_buddy.config import (
    MAX_SEARCH_ROUNDS,
    MAX_TOTAL_QUERIES,
    MIN_DISTINCT_DOMAINS,
    MIN_EVIDENCE_COVERAGE,
    MIN_RESULTS_PER_SUB_QUESTION,
    MIN_SEARCH_CONTENT_LENGTH,
    MIN_SEMANTIC_COVERAGE,
    OPENAI_API_KEY,
)
from research_buddy.state import ResearchState, ValidationGap
from research_buddy.utils import create_llm, get_prompt_from_langfuse, normalize_url, parse_llm_json

logger = logging.getLogger(__name__)


EVIDENCE_EVALUATOR_PROMPT = """你是研究证据评估器。根据每个子问题的搜索证据，判断核心问题是否已被现有证据回答。

研究问题：{question}

证据包：
{evidence_payload}

判断标准：
- coverage 表示「核心结论的可信度」，不是「所有细节的完备度」。
  只要现有证据足以支撑一个可信的核心结论（通常 0.6 以上），就应给 sufficient。
  研究预算有限，为追求细节完备而反复要求补搜会耗尽预算、让整篇报告无法出稿。
- 只有在核心结论确实缺乏支撑时才标 partial / insufficient，并说明缺什么。
- missing_evidence 只列会影响结论的关键缺失（最多 3 条），无关紧要的细节不要写。
- 标记来源矛盾（contradictions）：确实存在矛盾才写，不要为了凑数。
- next_queries 必须简洁、具体、与历史查询不同，并匹配目标信息源的语言和地区
- 中国本地证据优先中文查询，全球/学术证据可使用英文；不要生成中英混杂的机械后缀
- status 只能是 sufficient、partial、insufficient
- coverage 是 0-1 之间的小数

只返回 JSON 数组：
```json
[
  {{
    "sub_question_id": "sq_01",
    "status": "sufficient",
    "coverage": 0.7,
    "missing_evidence": ["缺少官方统计"],
    "contradictions": [],
    "next_queries": ["official statistics topic year"]
  }}
]
```"""


def _domain(url: str) -> str:
    try:
        host = (urlsplit(url if "://" in url else f"https://{url}").hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


def _fallback_query(original_query: str, round_num: int, reason: str,
                    language: str = "auto") -> str:
    is_chinese = language == "zh" or any("\u4e00" <= char <= "\u9fff" for char in original_query)
    if is_chinese:
        suffixes = {0: "官方报告 数据", 1: "独立来源 分析", 2: "一手资料 统计"}
        suffix = suffixes.get(min(round_num, 2), "最新证据")
        if reason == "insufficient_domains":
            suffix = "独立来源"
    else:
        suffixes = {0: "official report data", 1: "independent sources analysis", 2: "primary source statistics"}
        suffix = suffixes.get(min(round_num, 2), "evidence update")
        if reason == "insufficient_domains":
            suffix = "independent sources"
    return f"{original_query} {suffix}".strip()


def _llm_assess(question: str, evidence_payload: list[dict],
                     config: RunnableConfig | None = None) -> dict[str, dict] | None:
    """批量语义评估。

    返回 None 表示评估器整体不可用（没配 key、请求失败、输出完全无法使用），
    调用方据此降级到纯确定性判断并在报告里披露；返回 dict 表示评估器给出了
    至少一个分支的结论，此时 dict 里缺失的分支按「未通过语义评估」处理，
    而不是当作充足 —— 否则 LLM 故障会让证据门槛变宽而不是变严。
    """
    if not OPENAI_API_KEY or not evidence_payload:
        return None
    try:
        prompt = get_prompt_from_langfuse(
            "research-buddy-evidence-evaluator",
            EVIDENCE_EVALUATOR_PROMPT,
            question=question,
            evidence_payload=json.dumps(evidence_payload, ensure_ascii=False),
        )
        response = create_llm().invoke(prompt, config=config)
        parsed = parse_llm_json(response.content)
        if not isinstance(parsed, list):
            raise ValueError("evidence evaluator must return a list")
        validated = {}
        for item in parsed:
            if not isinstance(item, dict) or not item.get("sub_question_id"):
                continue
            status = item.get("status")
            if status not in {"sufficient", "partial", "insufficient"}:
                continue
            try:
                coverage = max(0.0, min(1.0, float(item.get("coverage", 0))))
            except (TypeError, ValueError):
                coverage = 0.0
            validated[item["sub_question_id"]] = {
                "status": status,
                "coverage": coverage,
                "missing_evidence": [str(v) for v in item.get("missing_evidence", [])[:5]],
                "contradictions": [str(v) for v in item.get("contradictions", [])[:5]],
                "next_queries": [str(v).strip() for v in item.get("next_queries", [])[:3] if str(v).strip()],
            }
        if not validated:
            # 结构对但一条都没通过校验，说明 schema 不匹配，没有任何分支级信号可用
            logger.warning("证据语义评估返回 %d 条但全部无效，降级为确定性指标", len(parsed))
            return None
        return validated
    except Exception as exc:
        logger.warning("证据语义评估失败，使用确定性指标降级: %s", exc)
        return None


def validator(state: ResearchState, config: RunnableConfig | None = None) -> dict:
    """评估每个研究分支的证据覆盖，并生成可追溯的补充搜索任务。"""
    sub_questions = state.get("sub_questions", [])
    search_results = state.get("search_results", [])
    search_round = state.get("search_round", 0)
    total_queries = state.get("total_queries", 0)

    results_by_id: dict[str, list] = {}
    question_to_id = {
        sq.get("question", ""): sq.get("id", f"sq_{index:02d}")
        for index, sq in enumerate(sub_questions, 1)
    }
    for result in search_results:
        sq_id = result.get("sub_question_id") or question_to_id.get(result.get("sub_question", ""), "")
        if sq_id:
            results_by_id.setdefault(sq_id, []).append(result)

    deterministic = []
    evidence_payload = []
    for index, sq in enumerate(sub_questions, 1):
        sq_id = sq.get("id", f"sq_{index:02d}")
        seen_urls = set()
        valid_results = []
        for result in results_by_id.get(sq_id, []):
            url_key = normalize_url(result.get("url", ""))
            if len(result.get("content", "")) < MIN_SEARCH_CONTENT_LENGTH:
                continue
            if url_key and url_key in seen_urls:
                continue
            if url_key:
                seen_urls.add(url_key)
            valid_results.append(result)

        domains = {_domain(result.get("url", "")) for result in valid_results}
        domains.discard("")
        positive_scores = [float(r.get("score", 0)) for r in valid_results if float(r.get("score", 0)) > 0]
        count_score = min(1.0, len(valid_results) / max(MIN_RESULTS_PER_SUB_QUESTION, 1))
        domain_score = min(1.0, len(domains) / max(MIN_DISTINCT_DOMAINS, 1))
        if positive_scores:
            relevance = sum(positive_scores) / len(positive_scores)
            coverage = round(0.5 * count_score + 0.3 * domain_score + 0.2 * min(1.0, relevance), 3)
        else:
            # 所有结果都没有相关度分数时，相关度是「未知」而不是「还行」。
            # 之前默认 0.7 会白送 0.14 覆盖度，让无 score 的分支凑到 0.94。
            # 这里既不给默认值也不把 0.2 的权重重新分摊（那等于按 relevance=1 算），
            # 直接丢掉这一项：覆盖度上限变成 0.8，仍高于默认阈值 0.75，
            # 所以数量和域名都达标的分支照样能过，但拿不到任何相关度加成。
            relevance = None
            coverage = round(0.5 * count_score + 0.3 * domain_score, 3)
        hard_floor = (
            len(valid_results) >= MIN_RESULTS_PER_SUB_QUESTION
            and len(domains) >= MIN_DISTINCT_DOMAINS
            and coverage >= MIN_EVIDENCE_COVERAGE
        )
        reason = ""
        if len(valid_results) < MIN_RESULTS_PER_SUB_QUESTION:
            reason = "insufficient_results"
        elif len(domains) < MIN_DISTINCT_DOMAINS:
            reason = "insufficient_domains"
        elif coverage < MIN_EVIDENCE_COVERAGE:
            reason = "low_coverage"

        deterministic.append({
            "sub_question_id": sq_id,
            "question": sq.get("question", ""),
            "search_query": sq.get("search_query", ""),
            "valid_results": len(valid_results),
            "distinct_domains": len(domains),
            "coverage": coverage,
            "hard_floor": hard_floor,
            "reason": reason,
            "language": sq.get("language", "auto"),
            "region": sq.get("region", "GLOBAL"),
        })
        evidence_payload.append({
            "sub_question_id": sq_id,
            "question": sq.get("question", ""),
            "original_query": sq.get("search_query", ""),
            "language": sq.get("language", "auto"),
            "region": sq.get("region", "GLOBAL"),
            "source_preference": sq.get("source_preference", "general"),
            "metrics": {"valid_results": len(valid_results), "distinct_domains": len(domains), "coverage": coverage},
            "evidence": [
                {
                    "title": result.get("title", "")[:160],
                    "url": result.get("url", ""),
                    "content": result.get("content", "")[:700],
                    "score": result.get("score", 0),
                }
                for result in valid_results[:6]
            ],
        })

    semantic = _llm_assess(state.get("question", ""), evidence_payload, config=config)
    assessment_available = semantic is not None
    if not assessment_available and evidence_payload:
        logger.warning("语义证据评估不可用，本轮只用确定性下限判断，报告需披露该降级")

    assessments = []
    gaps: list[ValidationGap] = []
    for item in deterministic:
        llm_item = (semantic or {}).get(item["sub_question_id"], {})
        if not assessment_available:
            # 评估器整体不可用：只保留确定性下限，不假装语义已验证
            semantic_sufficient = True
        elif not llm_item:
            # 评估器答了别的分支却跳过了这个分支 —— fail-closed，不当作充足
            semantic_sufficient = False
        else:
            semantic_sufficient = (
                llm_item.get("status") == "sufficient"
                # 语义闸用独立软阈值：LLM 的 coverage 是核心结论可信度，
                # 与确定性覆盖度（MIN_EVIDENCE_COVERAGE）是两回事，不能用同一个硬标准。
                and llm_item.get("coverage", 0) >= MIN_SEMANTIC_COVERAGE
            )
        contradictions = llm_item.get("contradictions", [])
        sufficient = item["hard_floor"] and semantic_sufficient and not contradictions
        final_coverage = min(item["coverage"], llm_item.get("coverage", 1.0)) if llm_item else item["coverage"]
        missing = llm_item.get("missing_evidence", [])
        assessments.append({
            "sub_question_id": item["sub_question_id"],
            "status": "sufficient" if sufficient else "insufficient",
            "coverage": final_coverage,
            "valid_results": item["valid_results"],
            "distinct_domains": item["distinct_domains"],
            "missing_evidence": missing,
            "contradictions": contradictions,
        })
        if not sufficient:
            next_queries = llm_item.get("next_queries", [])
            query = next_queries[0] if next_queries else _fallback_query(
                item["search_query"], search_round, item["reason"], item["language"]
            )
            fallback_reason = item["reason"] or (
                "semantic_assessment_missing" if assessment_available and not llm_item
                else "semantic_coverage_gap"
            )
            reason = "; ".join(missing or contradictions) or fallback_reason
            gaps.append({
                "sub_question_id": item["sub_question_id"],
                "question": item["question"],
                "search_query": query,
                "reason": reason,
                "priority": "high" if contradictions or final_coverage < 0.5 else "medium",
                "language": item["language"],
                "region": item["region"],
            })

    budget_exhausted = search_round >= MAX_SEARCH_ROUNDS or total_queries >= MAX_TOTAL_QUERIES
    previous_reason = state.get("stop_reason", "")
    stop_reason = ""
    if gaps and previous_reason in {"no_new_queries", "search_unavailable"}:
        # 补搜已经无路可走（没有新查询 / 搜索层挂了），保留原因让路由直接去综合
        stop_reason = previous_reason
    elif gaps and budget_exhausted:
        stop_reason = "search_budget_exhausted"
    elif not gaps:
        stop_reason = "evidence_sufficient"

    logger.info(
        "证据评估完成：%d/%d 个分支充足，搜索轮次 %d，查询数 %d%s",
        len(assessments) - len(gaps), len(assessments), search_round, total_queries,
        "（语义评估不可用，仅确定性判断）" if not assessment_available else "",
    )
    msg = f"证据评估：{len(assessments) - len(gaps)}/{len(assessments)} 个分支充足"
    if not assessment_available and evidence_payload:
        msg += "（⚠️ 语义评估不可用，仅按来源数/域名/覆盖度判断）"
    return {
        "validation_gaps": gaps,
        "evidence_assessments": assessments,
        "evidence_assessment_degraded": bool(evidence_payload) and not assessment_available,
        "stop_reason": stop_reason,
        "research_complete": not gaps,
        "messages": [msg],
    }
