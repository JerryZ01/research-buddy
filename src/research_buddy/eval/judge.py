"""LLM-as-Judge 自动评分 - 对研究报告进行多维度评估"""

import logging

from langfuse import Langfuse

from research_buddy.utils import create_llm, parse_llm_json, get_prompt_from_langfuse

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


def judge_report(question: str, expected_points: list[str], report: str) -> dict:
    """LLM-as-Judge 评估报告质量

    Args:
        question: 研究问题
        expected_points: 预期要点列表
        report: 实际生成的研究报告

    Returns:
        评分字典 {"relevance": int, "completeness": int, "accuracy": int, "reasoning": str, "parse_failed": bool}
        parse_failed=True 表示解析失败，分数为默认值
    """
    points_text = "\n".join(f"- {p}" for p in expected_points)

    llm = create_llm()

    prompt = get_prompt_from_langfuse(
        "research-buddy-judge", JUDGE_PROMPT,
        question=question,
        expected_points=points_text,
        report=report,
    )
    response = llm.invoke(prompt)

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
