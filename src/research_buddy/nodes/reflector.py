"""反思节点 - LLM 自评报告质量，决定是否需要修正"""

import logging
import re

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from research_buddy.config import MAX_REFLECTION_ROUNDS, OPENAI_MODEL, REFLECTOR_MODEL
from research_buddy.state import ResearchState
from research_buddy.utils import (create_llm, get_prompt_from_langfuse,
                                  invoke_llm, is_metaphorical_heading,
                                  normalize_url, parse_llm_json)

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
2. **准确性**（1-5）：论点是否有充分来源支撑，有无凭空推测。报告应基于检索到的来源撰写；正文不应内嵌 URL，也不应出现 `[编号]` 引用标注（来源统一放在文末参考文献）
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


# ── AI 味硬校验（防模板腔） ─────────────────────────────

# 元评论/模板句：命中数量 ≥ _META_COMMENT_LIMIT 视为模板腔
_META_COMMENT_PATTERNS = [
    r"本节.{0,14}(?:关键|核心|重要)?(?:结论|发现|判断)",
    r"值得(?:注意|指出|一提|强调)的是",
    r"综上所述|总而言之|总的来说|一言以蔽之",
    r"这(?:揭示了|恰恰说明|直接说明)",
]
_META_COMMENT_LIMIT = 3

# 研究过程元评论（可发布文章绝不该出现）：检索/来源/资料类自指。
# 与 _META_COMMENT_PATTERNS 分开统计，命中 1 处即重写（这类话零容忍）。
_RETRIEVAL_META_PATTERNS = [
    r"从检索到的(?:资料|结果|证据)(?:看|中|里)",
    r"有(?:来源|资料)(?:提到|指出|显示|称)",
    r"检索(?:结果|到的|回来的)",
    r"多个(?:来源|资料)(?:指出|显示|提到|认为)",
    r"据(?:资料|检索|来源)(?:显示|称|来看)",
    r"公开(?:资料|信息)(?:称|显示)",
]

# 排比对仗句式：密度（每千字命中数）≥ _PARALLEL_DENSITY_LIMIT 视为滥用
_PARALLEL_PATTERNS = [
    r"不是.{0,20}而是",
    r"既.{0,16}又.{0,16}",
    r"越.{0,8}越.{0,8}",
    r"不(?:仅|只).{0,12}更(?:是|要|重要)",
]
_PARALLEL_DENSITY_LIMIT = 4.0  # 每千字

# 「不是…而是…」单独计数（检测器最敏感的句式之一，整篇 ≥3 处即模板腔）
_NOT_BUT_RE = re.compile(r"不是[^，。；\n]{1,32}而是")
_NOT_BUT_LIMIT = 2

# 「从 X 到 Y，从 A 到 B，从 C 到 D」式三连排比（AI 招牌句式，出现即重写）
_TRIPLE_PARALLEL_PATTERNS = [
    r"(?:从[^，。；\n]{1,24}到[^，。；\n]{1,24}，){2,}(?:从[^，。；\n]{1,24}到[^，。；\n]{1,24})",
]

# 公式化结尾：「…结论」标题 + 数字列表出现在文末 600 字内
_FORMULA_TAIL_HEADING = re.compile(r"#{1,3}\s*[^#\n]{0,18}(?:总结|结论)")
_FORMULA_TAIL_LIST = re.compile(r"(?:^|\n)\s*(?:1[.、．]|①)")

# 自问自答句式：「……为什么如此重要？因为它……」「这意味着什么？对于……」
_SELFQA_PATTERNS = [
    r"[？?](?:因为|答案(?:是|在于)|关键在于|本质上是因为)",
    r"[？?](?:对于|对一个)",
]
# 引导句/教学腔：「拆解这个定义需要一点耐心」「让我们先看看」
_GUIDE_PATTERNS = [
    r"拆解[^。]{0,20}需要一点耐心",
    r"让我们(?:先|来看|回到)",
    r"这里需要(?:先|明确|区分)",
    r"先说(?:一个|个结论)",
]
# 自问自答 ≥2 处，或引导句 ≥3 处，或合计 ≥3 处 → 模板腔
_SELFQA_LIMIT = 2
_GUIDE_LIMIT = 3
_GUIDE_OR_SELFQA_COMBINED_LIMIT = 3

# 冒号式小标题（「名词：副题」）占比 ≥ 50% 且 ≥2 个 → 标题模板腔。
# 同时覆盖 markdown（##/###）与独立一行的粗体标题。
_COLON_HEADING_RATIO_LIMIT = 0.5
_COLON_HEADING_MIN = 2
_HEADING_MD_RE = re.compile(r"^(#{2,3})\s*(.+?)\s*$", re.M)
_HEADING_BOLD_RE = re.compile(r"^\*\*(.+?)\*\*\s*$", re.M)


def _ai_flavor_issues(report: str) -> list[str]:
    """检测文章的 AI 模板痕迹，返回问题描述列表（空 = 通过）。

    阈值从宽：只拦截明显的模板腔，不误伤正常的个别句式。
    """
    if not report:
        return []
    issues: list[str] = []

    meta_count = sum(len(re.findall(p, report)) for p in _META_COMMENT_PATTERNS)
    if meta_count >= _META_COMMENT_LIMIT:
        issues.append(
            f"存在 {meta_count} 处元评论/模板句（「本节的关键结论是」「值得注意的是」「综上所述」等），"
            "请重写时删除这些句式，直接用内容说话"
        )

    # 研究过程元评论：零容忍（出现 1 处即重写）
    retrieval_meta_count = sum(len(re.findall(p, report)) for p in _RETRIEVAL_META_PATTERNS)
    if retrieval_meta_count:
        issues.append(
            f"存在 {retrieval_meta_count} 处研究过程元评论（「从检索到的资料看」「有来源提到」"
            "「检索到的证据」等），可发布文章不应出现这类自指检索过程的表述，请全部删除"
        )

    parallel_count = sum(len(re.findall(p, report)) for p in _PARALLEL_PATTERNS)
    density = parallel_count * 1000 / max(len(report), 1)
    if density >= _PARALLEL_DENSITY_LIMIT:
        issues.append(
            f"排比对仗句式过密（约 {density:.1f} 处/千字，如「不是…而是…」「既…又…」），"
            "请重写时让句子长短自然交错"
        )

    # 「不是…而是…」单独计数（检测器最敏感，≥3 处即模板腔）
    not_but_count = len(_NOT_BUT_RE.findall(report))
    if not_but_count > _NOT_BUT_LIMIT:
        issues.append(
            f"「不是…而是…」句式出现 {not_but_count} 处（整篇应 ≤2 处），"
            "请重写时改用其他表达方式，如直接陈述或让步句式"
        )

    # 「从 X 到 Y，从 A 到 B，从 C 到 D」三连排比
    triple_count = sum(len(re.findall(p, report)) for p in _TRIPLE_PARALLEL_PATTERNS)
    if triple_count:
        issues.append(
            f"存在 {triple_count} 处「从 X 到 Y，从 A 到 B，从 C 到 D」式三连排比，"
            "请重写时打破这种整齐的递进结构"
        )

    tail = report[-600:]
    if _FORMULA_TAIL_HEADING.search(tail) and _FORMULA_TAIL_LIST.search(tail):
        issues.append(
            "结尾是公式化的「…结论：1. 2. 3.」结构，请改用散文收束或按问题语义给出行之有效的建议"
        )

    selfqa_count = sum(len(re.findall(p, report)) for p in _SELFQA_PATTERNS)
    guide_count = sum(len(re.findall(p, report)) for p in _GUIDE_PATTERNS)
    if (selfqa_count >= _SELFQA_LIMIT or guide_count >= _GUIDE_LIMIT
            or selfqa_count + guide_count >= _GUIDE_OR_SELFQA_COMBINED_LIMIT):
        issues.append(
            f"存在 {selfqa_count} 处自问自答句式（「这意味着什么？因为…」）和 "
            f"{guide_count} 处引导/教学腔（「拆解这个定义需要一点耐心」「让我们先看看」），"
            "请重写时直接陈述，不要自问自答、不要铺设引导句"
        )

    # 冒号式小标题过密（「名词：副题」整篇重复，markdown 与粗体都算）
    md_headings = _HEADING_MD_RE.findall(report)
    headings = ([text for _level, text in md_headings]
                + _HEADING_BOLD_RE.findall(report))
    if len(headings) >= _COLON_HEADING_MIN:
        colon_headings = [h for h in headings if "：" in h]
        if len(colon_headings) >= _COLON_HEADING_MIN \
                and len(colon_headings) / len(headings) >= _COLON_HEADING_RATIO_LIMIT:
            issues.append(
                f"{len(colon_headings)}/{len(headings)} 个小标题都是「名词：副题」式冒号标题"
                "（如「自注意力：当每个 token 都成为检索者」），请改成直接、克制的"
                "陈述或名词短语，不要为了形式变化改成反问或比喻"
            )

    metaphor_headings = [heading for heading in headings if is_metaphorical_heading(heading)]
    if metaphor_headings:
        issues.append(
            f"存在 {len(metaphor_headings)} 个比喻、拟人或口号式标题"
            f"（如「{metaphor_headings[0]}」），请改成直接说明本节对象或判断的克制标题"
        )

    h2_count = sum(level == "##" for level, _text in md_headings)
    h3_count = sum(level == "###" for level, _text in md_headings)
    if h3_count > 2 and h3_count >= h2_count:
        issues.append(
            f"文章使用了 {h3_count} 个三级小标题，结构切分过细；请合并相邻内容，"
            "只在长章节内部确有多个独立层次时保留三级标题"
        )

    return issues


def reflector(state: ResearchState, config: RunnableConfig | None = None,
              *, writer: StreamWriter = None) -> dict:
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
    unprocessed_user_feedback = bool(
        user_feedback and state.get("report_feedback_signature", "") != user_feedback
    )
    evidence_assessments = state.get("evidence_assessments", [])
    # validator 标出但还没解决的缺口，必须带到本节点的输出里，不能被覆盖掉
    inherited_gaps = list(state.get("validation_gaps", []))
    supplement_targets = _supplement_targets(sub_questions, evidence_assessments)
    evidence_signature = "|".join(sorted({
        normalize_url(result.get("url", ""))
        for result in search_results if result.get("url")
    } | {
        normalize_url(url) for url in state.get("known_source_urls", []) if url
    }))
    best_is_eligible = (
        state.get("best_evidence_signature", "") == evidence_signature
        and state.get("best_feedback_signature", "") == user_feedback
    )
    eligible_best_report = state.get("best_report", "") if best_is_eligible else ""
    eligible_best_rank = state.get("best_quality_rank", -1) if best_is_eligible else -1
    eligible_best_score = state.get("best_reflection_score", 0) if best_is_eligible else 0
    eligible_best_round = state.get("best_reflection_round", 0) if best_is_eligible else 0

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

    # 独立 Judge 是软依赖：服务异常或返回格式损坏时，仍继续执行下方的
    # URL、证据缺口和 AI 味确定性硬校验，不能因评审服务故障丢掉文章。
    judge_degraded = False
    parse_failed = False
    try:
        llm = (create_llm(model=REFLECTOR_MODEL)
               if REFLECTOR_MODEL != OPENAI_MODEL else create_llm())
        response = invoke_llm(llm, prompt, config=config)
    except Exception as exc:
        logger.warning("反思 Judge 不可用，降级为确定性规则检查: %s", exc)
        judge_degraded = True
        evaluation = {
            "completeness": 4,
            "accuracy": 4,
            "clarity": 4,
            "feedback": "独立 Judge 不可用，本轮仅执行确定性规则检查",
            "supplement_queries": [],
        }
    else:
        try:
            evaluation = parse_llm_json(response.content)
        except Exception as exc:
            logger.warning("反思结果无法解析，按未通过处理: %s", exc)
            parse_failed = True
            evaluation = {
                "feedback": "反思结果无法解析，未将报告标记为通过",
                "supplement_queries": [],
            }

    if not isinstance(evaluation, dict):
        logger.warning("反思结果不是 JSON 对象（%s），按未通过处理",
                       type(evaluation).__name__)
        parse_failed = True
        evaluation = {
            "feedback": "反思结果格式异常，未将报告标记为通过",
            "supplement_queries": [],
        }

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
    # 正文不内嵌来源 URL（来源统一放在文末参考文献，由代码生成），
    # 只检查正文里残留的裸 URL 是否都在证据集内——防 LLM 回归旧式内嵌链接
    # 或幻觉 URL。插图 URL 出现在正文的 ![alt](url) 中，也要放行（但只放行
    # 被选中的图，LLM 若嵌入候选之外的图片 URL 仍会被判违规）。
    known_urls = {normalize_url(result.get("url", "")) for result in search_results}
    known_urls.update(normalize_url(url) for url in state.get("known_source_urls", []))
    known_urls.update(
        normalize_url(img.get("url", "")) for img in state.get("selected_images", [])
    )
    known_urls.update(
        normalize_url(img.get("cached_url", "")) for img in state.get("selected_images", [])
    )
    known_urls.discard("")

    citation_issues = []

    # 正文裸 URL 检查：可发布文章风格下正文不应出现任何 URL（参考文献
    # 由代码追加，其中的 URL 都在证据集内，天然通过）。
    raw_urls = {
        normalize_url(url.rstrip(".,);]，。；）】"))
        for url in re.findall(r"https?://[^\s<>]+", report)
    }
    raw_urls.discard("")

    if raw_urls - known_urls:
        citation_issues.append(f"报告包含 {len(raw_urls - known_urls)} 个不在证据集中的 URL")

    # AI 味硬校验（防模板腔）：元评论句、排比对仗密度、公式化结尾。
    # 阈值从宽设置，避免误伤正常行文；命中才强制重写。
    citation_issues.extend(_ai_flavor_issues(report))

    if citation_issues:
        passed = False
        feedback = "\n".join(citation_issues + ([feedback] if feedback else []))

    if inherited_gaps:
        passed = False
        feedback = "报告生成时仍存在未解决证据缺口。\n" + feedback

    # 只在当前报告尚未纳入最新反馈时强制重写。反馈进入新稿后不能继续
    # 永久阻止通过，否则 HITL 会重复改同一条要求直到耗尽反思预算。
    if unprocessed_user_feedback and passed:
        logger.info("当前报告尚未纳入最新用户反馈，强制不通过")
        passed = False

    logger.info("评分: %d/15 → %s", total_score, "✅ 通过" if passed else "⚠️  需要修正")

    # 如果未通过，生成补充搜索任务；同时保留 validator 尚未解决的缺口
    report_gaps = []
    if not passed:
        if parse_failed:
            target = supplement_targets[0] if supplement_targets else {
                "sub_question_id": "", "question": question,
                "language": "auto", "region": "GLOBAL",
            }
            report_gaps.append({
                "sub_question_id": target["sub_question_id"],
                "question": target["question"],
                "search_query": f"{question} reliable evidence",
                "reason": "reflection_parse_error",
                "priority": "high",
                "language": target["language"],
                "region": target["region"],
            })
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

    # 硬校验通过的稿件永远优先于高分但含未知 URL/模板问题/未解决缺口的稿件；
    # 同一层级再比较三维评分。这样重写轮变差时不会覆盖已经更好的历史版本。
    hard_failure = bool(
        parse_failed or citation_issues or inherited_gaps or unprocessed_user_feedback
    )
    quality_rank = total_score + (0 if hard_failure else 100)
    best_report = eligible_best_report
    best_rank = eligible_best_rank
    best_score = eligible_best_score
    best_round = eligible_best_round
    if report and quality_rank > best_rank:
        best_report = report
        best_rank = quality_rank
        best_score = total_score
        best_round = next_round

    terminal = passed or next_round >= MAX_REFLECTION_ROUNDS
    restored = bool(terminal and best_report and best_report != report)
    delivered_report = best_report if restored else report
    delivered_score = best_score if restored else total_score
    if restored and writer:
        writer({"type": "report_reset"})
        for offset in range(0, len(delivered_report), 2000):
            writer({"type": "report_chunk", "content": delivered_report[offset:offset + 2000]})
    if restored:
        result_msg += f"；最终恢复第 {best_round} 轮的历史最佳稿"

    return {
        "report": delivered_report,
        "reflection_pass": passed,
        "reflection_feedback": feedback,
        "reflection_round": next_round,
        "reflection_score": delivered_score,
        "reflection_judge_degraded": judge_degraded,
        "article_versions": [{
            "stage": "reflector",
            "reflection_round": next_round,
            "report": delivered_report,
            "feedback": feedback,
            "metadata": {
                "score": delivered_score,
                "passed": passed,
                "judge_degraded": judge_degraded,
                "best_report_restored": restored,
            },
        }],
        "best_report": best_report,
        "best_quality_rank": best_rank,
        "best_reflection_score": best_score,
        "best_reflection_round": best_round,
        "best_evidence_signature": evidence_signature,
        "best_feedback_signature": user_feedback,
        "best_report_restored": restored,
        "validation_gaps": gaps,
        "stop_reason": "completed" if passed else stop_reason,
        "research_complete": passed,
        "messages": [result_msg],
    }
