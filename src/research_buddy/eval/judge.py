"""LLM-as-Judge 自动评分 - 对研究报告进行多维度评估"""

import json as _json
import logging

from langfuse import Langfuse
from langchain_core.runnables import RunnableConfig

from research_buddy.config import ARTICLE_EVAL_JUDGE_MODEL
from research_buddy.utils import create_llm, parse_llm_json, get_prompt_from_langfuse, invoke_llm

logger = logging.getLogger(__name__)

# 评分维度和范围
_SCORE_DIMENSIONS = ["relevance", "completeness", "accuracy"]
_MIN_SCORE = 1
_MAX_SCORE = 5

JUDGE_PROMPT = """你是一个研究质量评估专家。请评估以下研究报告的质量。

## 研究问题
{question}

## 预期要点
{expected_points}

## 实际报告
{report}

## 评分维度

请从以下三个维度打分（1-5 分）：

1. **相关性（relevance）**：报告是否紧扣研究问题，有无偏题
   - 1分：完全偏题
   - 3分：部分相关但有偏移
   - 5分：完全紧扣问题

2. **完整性（completeness）**：预期要点是否被报告覆盖
   - 1分：几乎没有覆盖
   - 3分：覆盖了一半要点
   - 5分：所有要点都被充分覆盖

3. **准确性（accuracy）**：论点是否有来源支撑，有无凭空推测
   - 1分：大量无根据推测
   - 3分：部分有来源但不够充分
   - 5分：所有论点都有明确来源支撑

## 输出格式
请返回如下 JSON（不要包含其他内容）：
```json
{{
  "relevance": 4,
  "completeness": 3,
  "accuracy": 4,
  "reasoning": "简要说明各维度评分理由"
}}
```"""


def _default_scores(reason: str) -> dict:
    """评分不可用时的默认分数，parse_failed 让调用方把它排除在汇总之外。"""
    return {
        "relevance": 3,
        "completeness": 3,
        "accuracy": 3,
        "reasoning": reason,
        "parse_failed": True,
    }


def judge_report(question: str, expected_points: list[str], report: str,
                 config: RunnableConfig | None = None) -> dict:
    """LLM-as-Judge 评估报告质量

    Args:
        question: 研究问题
        expected_points: 预期要点列表
        report: 实际生成的研究报告
        config: LangGraph RunnableConfig，透传给 LLM 调用（Langfuse 观测）

    Returns:
        评分字典 {"relevance": int, "completeness": int, "accuracy": int, "reasoning": str, "parse_failed": bool}
        parse_failed=True 表示解析失败，分数为默认值
    """
    points_text = "\n".join(f"- {p}" for p in expected_points)

    llm = create_llm(model=ARTICLE_EVAL_JUDGE_MODEL)

    prompt = get_prompt_from_langfuse(
        "research-buddy-judge", JUDGE_PROMPT,
        question=question,
        expected_points=points_text,
        report=report,
    )
    # 与其他节点一致：走 invoke_llm 的瞬时错误（429/5xx）自动重试，而不是一次 503 就挂掉
    response = invoke_llm(llm, prompt, config=config)

    # 使用统一的 parse_llm_json
    try:
        scores = parse_llm_json(response.content)
    except Exception:
        # 解析失败，给默认中等分，但标记 parse_failed 让调用方知道
        logger.warning("Judge JSON 解析失败，使用默认分数")
        return _default_scores("评分解析失败，使用默认分数")

    # parse_llm_json 成功不等于拿到了字典：模型可能返回 JSON 数组或标量，
    # 那样后面的 scores.get() 会 AttributeError 把整个评估跑挂。
    if not isinstance(scores, dict):
        logger.warning("Judge 返回的不是 JSON 对象（%s），使用默认分数", type(scores).__name__)
        return _default_scores("评分格式不是 JSON 对象，使用默认分数")

    # 验证分数范围和维度完整性
    for dim in _SCORE_DIMENSIONS:
        val = scores.get(dim, 3)
        if isinstance(val, bool) or not isinstance(val, (int, float)) \
                or val < _MIN_SCORE or val > _MAX_SCORE:
            logger.warning("Judge 维度 %s 分数 %s 超出范围，修正为 3", dim, val)
            scores[dim] = 3
        else:
            scores[dim] = int(val)

    scores.setdefault("parse_failed", False)
    return scores


def score_trace(trace_id: str, scores: dict) -> None:
    """将评分写入 Langfuse trace

    Args:
        trace_id: Langfuse trace ID
        scores: judge_report 返回的评分字典
    """
    langfuse = Langfuse()

    for dimension in _SCORE_DIMENSIONS:
        if dimension in scores:
            langfuse.create_score(
                trace_id=trace_id,
                name=dimension,
                value=scores[dimension],
                comment=scores.get("reasoning", ""),
            )

    # 总分
    total = sum(scores.get(d, 0) for d in _SCORE_DIMENSIONS)
    langfuse.create_score(
        trace_id=trace_id,
        name="total",
        value=total,
        comment=f"总分 {total}/15",
    )


# ── 流程节点 Judge（对整个流程中间产物打分，而非只看最终报告） ──

# 每个流程 Judge 的评分维度（1-5 分），与 judge_report 的容错模式保持一致。
_PLAN_DIMENSIONS = ["necessity", "coverage", "granularity"]
_EVIDENCE_DIMENSIONS = ["sufficiency", "relevance", "diversity"]
_BRIEF_DIMENSIONS = ["clarity", "actionability", "evidence_alignment"]

PLAN_JUDGE_PROMPT = """你是研究规划评审专家。评估研究问题被拆解成子问题的质量。

## 研究问题
{question}

## 子问题列表
{sub_questions}

## 评分维度（1-5 分）
1. **必要性（necessity）**：每个子问题是否都是回答主问题所必需，有无冗余或偏题
2. **覆盖完整性（coverage）**：子问题集合是否完整覆盖主问题的核心方面
3. **粒度合理性（granularity）**：子问题粒度是否合适，能否独立检索、互不重叠

## 输出格式
只返回 JSON 对象（不要其他内容）：
```json
{{
  "necessity": 4,
  "coverage": 3,
  "granularity": 4,
  "reasoning": "简要说明各维度评分理由"
}}
```"""

EVIDENCE_JUDGE_PROMPT = """你是研究证据评审专家。评估搜索结果对研究问题的证据支持是否充分。

## 研究问题
{question}

## 子问题
{sub_questions}

## 搜索结果
{search_results}

## 评分维度（1-5 分）
1. **充分性（sufficiency）**：每个子问题是否有足够数量的有效证据
2. **相关性（relevance）**：证据是否真正回答了对应子问题，有无大量无关内容
3. **来源多样性（diversity）**：证据是否来自多个独立来源，有无单一来源垄断

## 输出格式
只返回 JSON 对象（不要其他内容）：
```json
{{
  "sufficiency": 4,
  "relevance": 3,
  "diversity": 4,
  "reasoning": "简要说明各维度评分理由"
}}
```"""

BRIEF_JUDGE_PROMPT = """你是编辑简报评审专家。评估研究写作前的编辑简报是否可执行。

## 研究问题
{question}

## 编辑简报
{brief}

## 评分维度（1-5 分）
1. **清晰性（clarity）**：简报的意图、读者、核心判断是否表述清晰
2. **可执行性（actionability）**：章节职责、必讲要点是否具体到可直接指导写作
3. **证据对齐（evidence_alignment）**：证据映射是否完整，禁止断言是否明确

## 输出格式
只返回 JSON 对象（不要其他内容）：
```json
{{
  "clarity": 4,
  "actionability": 3,
  "evidence_alignment": 4,
  "reasoning": "简要说明各维度评分理由"
}}
```"""


def _default_dimension_scores(dimensions: list[str], reason: str) -> dict:
    """评分不可用时按维度给默认中等分，并标记 parse_failed。"""
    return {dim: 3 for dim in dimensions} | {"reasoning": reason, "parse_failed": True}


def _run_dimension_judge(prompt_name: str, fallback: str, dimensions: list[str],
                          **kwargs) -> dict:
    """通用流程 Judge：拉取 prompt → 调用 LLM → 容错解析并校验维度分数。"""
    llm = create_llm(model=ARTICLE_EVAL_JUDGE_MODEL)
    prompt = get_prompt_from_langfuse(prompt_name, fallback, **kwargs)
    response = invoke_llm(llm, prompt)

    try:
        scores = parse_llm_json(response.content)
    except Exception:
        logger.warning("%s JSON 解析失败，使用默认分数", prompt_name)
        return _default_dimension_scores(dimensions, "评分解析失败，使用默认分数")

    if not isinstance(scores, dict):
        logger.warning("%s 返回的不是 JSON 对象（%s），使用默认分数",
                       prompt_name, type(scores).__name__)
        return _default_dimension_scores(dimensions, "评分格式不是 JSON 对象，使用默认分数")

    for dim in dimensions:
        val = scores.get(dim, 3)
        if isinstance(val, bool) or not isinstance(val, (int, float)) \
                or val < _MIN_SCORE or val > _MAX_SCORE:
            logger.warning("%s 维度 %s 分数 %s 超出范围，修正为 3", prompt_name, dim, val)
            scores[dim] = 3
        else:
            scores[dim] = int(val)

    scores.setdefault("parse_failed", False)
    return scores


def _format_sub_questions(sub_questions: list) -> str:
    """把子问题列表格式化为可读文本。"""
    if not sub_questions:
        return "（无子问题）"
    lines = []
    for i, sq in enumerate(sub_questions, 1):
        if isinstance(sq, dict):
            lines.append(f"{i}. {sq.get('question', '')}")
        else:
            lines.append(f"{i}. {sq}")
    return "\n".join(lines)


def _format_search_results(search_results: list, max_chars: int = 12000) -> str:
    """把搜索结果列表格式化为可读文本（截断控制 token）。"""
    if not search_results:
        return "（无搜索结果）"
    parts = []
    for i, item in enumerate(search_results, 1):
        if not isinstance(item, dict):
            continue
        parts.append(
            f"[{i}] {item.get('title', '')}\nURL: {item.get('url', '')}\n"
            f"{str(item.get('content', ''))[:400]}"
        )
    return "\n\n".join(parts)[:max_chars]


def judge_plan(question: str, sub_questions: list) -> dict:
    """对问题拆解（planner 产物）打分。"""
    return _run_dimension_judge(
        "research-buddy-judge-plan", PLAN_JUDGE_PROMPT, _PLAN_DIMENSIONS,
        question=question,
        sub_questions=_format_sub_questions(sub_questions),
    )


def judge_evidence(question: str, sub_questions: list, search_results: list) -> dict:
    """对搜索证据充分性（searcher/validator 产物）打分。"""
    return _run_dimension_judge(
        "research-buddy-judge-evidence", EVIDENCE_JUDGE_PROMPT, _EVIDENCE_DIMENSIONS,
        question=question,
        sub_questions=_format_sub_questions(sub_questions),
        search_results=_format_search_results(search_results),
    )


def judge_editorial_brief(question: str, brief: dict) -> dict:
    """对编辑简报（editorial_planner 产物）打分。"""
    brief_text = _json.dumps(brief, ensure_ascii=False, indent=2) if brief else "（无编辑简报）"
    return _run_dimension_judge(
        "research-buddy-judge-brief", BRIEF_JUDGE_PROMPT, _BRIEF_DIMENSIONS,
        question=question,
        brief=brief_text,
    )


def judge_full_process(result: dict, expected_points: list[str] | None = None) -> dict:
    """对整个研究流程的中间产物 + 最终结果打分。

    Args:
        result: run_research / run_knowledge_research / run_tracking 返回的 dict，
            需包含 question、sub_questions、search_results、editorial_brief、report。
        expected_points: 可选，用于最终报告打分的预期要点。

    Returns:
        按阶段组织的评分字典：
        {"plan": {...}, "evidence": {...}, "editorial_brief": {...}, "report": {...}}
    """
    question = str(result.get("question", ""))
    sub_questions = result.get("sub_questions", [])
    search_results = result.get("search_results", [])
    brief = result.get("editorial_brief", {})
    report = result.get("report", "")

    process_scores = {
        "plan": judge_plan(question, sub_questions),
        "evidence": judge_evidence(question, sub_questions, search_results),
        "editorial_brief": judge_editorial_brief(question, brief),
    }
    if report:
        process_scores["report"] = judge_report(question, expected_points or [], report)
    return process_scores


# 各阶段分数写入 Langfuse trace 时的 name 前缀
_PROCESS_SCORE_PREFIX = {
    "plan": "plan",
    "evidence": "evidence",
    "editorial_brief": "brief",
    "report": "report",
}


def score_process_trace(trace_id: str, process_scores: dict) -> None:
    """把流程各阶段评分写入 Langfuse trace。

    与 score_trace 不同，这里按「阶段_维度」命名（如 plan_necessity、evidence_sufficiency），
    与报告级 relevance/completeness/accuracy 区分开，便于在 Langfuse 里按阶段筛选。
    """
    if not trace_id:
        return
    langfuse = Langfuse()
    for stage, scores in process_scores.items():
        if not isinstance(scores, dict) or scores.get("parse_failed"):
            continue
        prefix = _PROCESS_SCORE_PREFIX.get(stage, stage)
        for dim, value in scores.items():
            if dim in ("reasoning", "parse_failed"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            langfuse.create_score(
                trace_id=trace_id,
                name=f"{prefix}_{dim}",
                value=float(value),
                comment=str(scores.get("reasoning", ""))[:500],
            )
    langfuse.flush()


def auto_judge_and_score(result: dict, trace_id: str) -> dict:
    """在线自动打分：对整个流程中间产物 + 结果打分并写回 Langfuse trace。

    供 run_research / run_knowledge_research / run_tracking 在流程结束后调用。
    评分写回失败不影响研究结果本身（仅记录 warning）。
    """
    process_scores = judge_full_process(result)
    if trace_id:
        try:
            score_process_trace(trace_id, process_scores)
        except Exception as exc:
            logger.warning("在线打分写回 Langfuse 失败: %s", exc)
    return process_scores
