"""反思节点 - LLM 自评报告质量，决定是否需要修正"""

import logging
import re

from research_buddy.config import MAX_REFLECTION_ROUNDS
from research_buddy.state import ResearchState
from research_buddy.utils import parse_llm_json, create_llm, get_prompt_from_langfuse, normalize_url

logger = logging.getLogger(__name__)


REFLECTOR_PROMPT = """你是一个研究质量评审专家。请评估以下研究报告的质量，从三个维度打分（1-5 分）：

## 研究问题
{question}

## 子问题
{sub_questions}

## 搜索结果数量
{result_count} 条

## 可用来源索引
{source_index}

## 证据覆盖评估
{evidence_status}

## 研究报告
{report}

{user_feedback_section}

## 评分维度
1. **完整性**（1-5）：是否回答了所有子问题，有无遗漏
2. **准确性**（1-5）：论点是否有充分来源支撑，有无凭空推测。报告应以 [编号] 引用来源（编号来自可用来源索引），正文不应内嵌 URL
3. **清晰度**（1-5）：结构是否清晰，逻辑是否连贯

## 输出格式
请返回如下 JSON（不要包含其他内容）：
```json
{{
  "completeness": 4,
  "accuracy": 3,
  "clarity": 4,
  "total_score": 11,
  "pass": false,
  "feedback": "报告缺少对XX子问题的深入分析，建议补充搜索...",
  "supplement_queries": ["English supplement search query 1", "English supplement search query 2"]
}}
```

- 总分 >= 12 时 pass 设为 true
- pass 为 false 时必须提供 feedback 和 supplement_queries
- supplement_queries 应匹配目标信息源的语言和地区，保持简短、具体
- 如果有用户反馈，优先针对用户反馈的不足生成补充搜索词"""


def _supplement_targets(sub_questions: list[dict],
                        evidence_assessments: list[dict]) -> list[dict]:
    """给报告级补充搜索挑归属分支，覆盖率最低的排在前面。

    补充缺口必须带真实的 sub_question_id：validator 只统计 sub_question_id 非空的
    结果（results_by_id 会丢掉空 id），所以 sub_question_id="" 的补搜结果不计入
    任何分支的覆盖率，validator 下一轮又产出同样的缺口，白烧搜索预算。
    """
    branches = {
        sq["id"]: {
            "sub_question_id": sq["id"],
            "question": sq.get("question", ""),
            "language": sq.get("language", "auto"),
            "region": sq.get("region", "GLOBAL"),
        }
        for sq in sub_questions if sq.get("id")
    }
    if not branches:
        return []

    ranked = sorted(
        (a for a in evidence_assessments if a.get("sub_question_id") in branches),
        key=lambda a: a.get("coverage", 0),
    )
    targets = [branches[a["sub_question_id"]] for a in ranked]
    # 没有评估结果（例如首轮解析失败）时按规划顺序兜底
    targets.extend(b for b in branches.values() if b not in targets)
    return targets


def _merge_gaps(primary: list[dict], inherited: list[dict]) -> list[dict]:
    """合并新缺口与上游未解决的缺口，按搜索词去重。

    validation_gaps 是覆盖语义，reflector 直接返回自己的列表会把 validator 标出的
    缺口整段擦掉：一旦 LLM 没给 supplement_queries，缺口就消失，路由改走
    revise_report，用完全相同的证据再写一遍报告。
    """
    merged = []
    seen = set()
    for gap in [*primary, *inherited]:
        key = " ".join(str(gap.get("search_query", "")).lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(gap)
    return merged


def reflector(state: ResearchState) -> dict:
    """反思节点：LLM 评估报告质量

    返回：
    - reflection_pass: 是否通过
    - reflection_feedback: 反馈/改进建议
    - reflection_round: 当前轮次 +1
    - validation_gaps: 如果未通过，生成补充搜索任务
    """
    question = state["question"]
    sub_questions = state.get("sub_questions", [])
    search_results = state.get("search_results", [])
    report = state.get("report", "")
    current_round = state.get("reflection_round", 0)
    user_feedback = state.get("user_feedback", "")
    evidence_assessments = state.get("evidence_assessments", [])
    # validator 标出但还没解决的缺口，必须带到本节点的输出里，不能被覆盖掉
    inherited_gaps = list(state.get("validation_gaps", []))
    supplement_targets = _supplement_targets(sub_questions, evidence_assessments)

    # 格式化子问题
    sq_text = "\n".join(
        f"- {sq.get('question', '')}（搜索词：{sq.get('search_query', '')}）"
        for sq in sub_questions
    )

    # 用户反馈部分
    if user_feedback:
        user_feedback_section = f"## 用户反馈（必须优先处理）\n{user_feedback}"
    else:
        user_feedback_section = ""

    source_table = state.get("source_table", []) or []
    if source_table:
        source_index = "\n".join(
            f"- [{item.get('index', '')}] {item.get('title', '')}: {item.get('url', '')}"
            for item in source_table
        )
    else:
        source_index = "\n".join(
            f"- {result.get('title', '')}: {result.get('url', '')}"
            for result in search_results[:30]
        )
    evidence_status = "\n".join(
        f"- {item.get('sub_question_id', '')}: {item.get('status', '')}, "
        f"coverage={item.get('coverage', 0)}, missing={item.get('missing_evidence', [])}"
        for item in evidence_assessments
    )

    logger.info("正在反思评估报告质量...")

    llm = create_llm()
    prompt = get_prompt_from_langfuse(
        "research-buddy-reflector", REFLECTOR_PROMPT,
        question=question,
        sub_questions=sq_text,
        result_count=len(search_results),
        source_index=source_index,
        evidence_status=evidence_status,
        report=report,
        user_feedback_section=user_feedback_section,
    )

    response = llm.invoke(prompt)

    # 解析 LLM 返回的 JSON
    try:
        evaluation = parse_llm_json(response.content)
    except Exception as exc:
        logger.warning("反思节点解析失败，按未通过处理: %s", exc)
        next_round = current_round + 1
        fallback_target = supplement_targets[0] if supplement_targets else {
            "sub_question_id": "", "question": question,
            "language": "auto", "region": "GLOBAL",
        }
        return {
            "reflection_pass": False,
            "reflection_feedback": "反思结果无法解析，未将报告标记为通过",
            "reflection_round": next_round,
            "reflection_score": 0,
            "validation_gaps": _merge_gaps([{
                "sub_question_id": fallback_target["sub_question_id"],
                "question": fallback_target["question"],
                "search_query": f"{question} reliable evidence",
                "reason": "reflection_parse_error",
                "priority": "high",
                "language": fallback_target["language"],
                "region": fallback_target["region"],
            }], inherited_gaps),
            "stop_reason": "reflection_budget_exhausted" if next_round >= MAX_REFLECTION_ROUNDS else "",
            "research_complete": False,
        }

    if not isinstance(evaluation, dict):
        # parse_llm_json 成功不等于拿到了对象：模型可能返回数组或标量
        logger.warning("反思结果不是 JSON 对象（%s），按未通过处理", type(evaluation).__name__)
        evaluation = {}

    feedback = str(evaluation.get("feedback", "") or "")
    raw_supplements = evaluation.get("supplement_queries", [])
    # 模型有时把 supplement_queries 写成一个字符串，直接 enumerate 会逐字符展开
    if isinstance(raw_supplements, str):
        raw_supplements = [raw_supplements]
    elif not isinstance(raw_supplements, list):
        raw_supplements = []
    supplement_queries = [str(query).strip() for query in raw_supplements if str(query).strip()]
    dimensions = {}
    for name in ("completeness", "accuracy", "clarity"):
        try:
            dimensions[name] = max(1, min(5, int(evaluation.get(name, 1))))
        except (TypeError, ValueError):
            dimensions[name] = 1
    total_score = sum(dimensions.values())
    passed = total_score >= 12 and min(dimensions.values()) >= 3

    # 证据集 = 本次检索结果 + 历史知识的来源 + 视觉模型选中的插图 URL。
    # 增量/追踪模式下 synthesizer 被要求引用 knowledge_context 里的历史来源，
    # 这些 URL 不在 search_results 里；只用 search_results 当证据集会把每一条
    # 历史引用都判成「不在证据集」，导致增量研究每轮必然不通过直到耗尽预算。
    # 插图 URL 出现在正文的 ![alt](url) 中，也要放行（但只放行被选中的图，
    # LLM 若嵌入候选之外的图片 URL 仍会被判违规）。
    known_urls = {normalize_url(result.get("url", "")) for result in search_results}
    known_urls.update(normalize_url(url) for url in state.get("known_source_urls", []))
    known_urls.update(
        normalize_url(img.get("url", "")) for img in state.get("selected_images", [])
    )
    known_urls.discard("")

    # 编号引用表：synthesizer 构建，正文 [n] 与文末参考文献的单一事实来源。
    source_table = state.get("source_table", []) or []
    table_by_index = {
        int(item["index"]): item for item in source_table
        if item.get("index") is not None
    }

    citation_issues = []

    # 1) 编号引用检查：正文必须用 [n] 引用，编号必须在编号表内。
    #    用负向后行断言排除图片语法 ![alt](url) 里的方括号，避免 alt 文本
    #    中的数字被误判为引用编号。
    report_cites = [int(n) for n in re.findall(r"(?<!\!)\[(\d+)\]", report)]
    unknown_cites = sorted({n for n in report_cites if n not in table_by_index})
    cited_urls = {
        normalize_url(table_by_index[n]["url"])
        for n in set(report_cites) if n in table_by_index
    }

    # 2) 正文裸 URL 检查：LLM 不应把 URL 内嵌进正文（可发布文章风格）。
    raw_urls = {
        normalize_url(url.rstrip(".,);]，。；）】"))
        for url in re.findall(r"https?://[^\s<>]+", report)
    }
    raw_urls.discard("")

    if table_by_index and not report_cites:
        citation_issues.append("报告没有引用任何已检索来源 URL")
    if unknown_cites:
        citation_issues.append(f"报告包含 {len(unknown_cites)} 个不在来源编号表中的引用编号")
    if cited_urls - known_urls:
        citation_issues.append(
            f"报告引用了 {len(cited_urls - known_urls)} 个不在证据集中的来源"
        )
    if raw_urls - known_urls:
        citation_issues.append(f"报告包含 {len(raw_urls - known_urls)} 个不在证据集中的 URL")
    if citation_issues:
        passed = False
        feedback = "\n".join(citation_issues + ([feedback] if feedback else []))

    if inherited_gaps:
        passed = False
        feedback = "报告生成时仍存在未解决证据缺口。\n" + feedback

    # 有用户反馈时一律不通过（用户要求优先），无论评分多高
    if user_feedback and passed:
        logger.info("存在用户反馈，强制不通过以处理用户要求")
        passed = False

    logger.info("评分: %d/15 → %s", total_score, "✅ 通过" if passed else "⚠️  需要修正")

    # 如果未通过，生成补充搜索任务；同时保留 validator 尚未解决的缺口
    report_gaps = []
    if not passed:
        for index, query in enumerate(supplement_queries):
            target = (supplement_targets[index % len(supplement_targets)]
                      if supplement_targets else
                      {"sub_question_id": "", "question": f"报告级补充搜索 {index + 1}",
                       "language": "auto", "region": "GLOBAL"})
            report_gaps.append({
                "sub_question_id": target["sub_question_id"],
                "question": target["question"],
                "search_query": query,
                "reason": "report_quality_gap",
                "priority": "high",
                "language": target["language"],
                "region": target["region"],
            })
    gaps = _merge_gaps(report_gaps, inherited_gaps) if not passed else []

    # 合并用户反馈到 reflection_feedback
    if user_feedback:
        feedback = f"[用户要求] {user_feedback}\n[评估反馈] {feedback}"

    result_msg = f"🔄 反思: 评分 {total_score}/15 → {'✅ 通过' if passed else '⚠️ 需要修正'}"
    if gaps:
        queries = [gap.get("search_query", "") for gap in gaps[:3]]
        result_msg += f"，待补充证据 {len(gaps)} 项: {', '.join(q for q in queries if q)}"

    next_round = current_round + 1
    stop_reason = state.get("stop_reason", "")
    if not passed and next_round >= MAX_REFLECTION_ROUNDS:
        stop_reason = "reflection_budget_exhausted"

    return {
        "reflection_pass": passed,
        "reflection_feedback": feedback,
        "reflection_round": next_round,
        "reflection_score": total_score,
        "validation_gaps": gaps,
        "stop_reason": "completed" if passed else stop_reason,
        "research_complete": passed,
        "messages": [result_msg],
    }
