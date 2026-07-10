"""LLM-as-Judge 自动评分 - 对研究报告进行多维度评估"""

import json
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from research_buddy.config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL

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


def judge_report(question: str, expected_points: list[str], report: str) -> dict:
    """LLM-as-Judge 评估报告质量

    Args:
        question: 研究问题
        expected_points: 预期要点列表
        report: 实际生成的研究报告

    Returns:
        评分字典 {"relevance": int, "completeness": int, "accuracy": int, "reasoning": str}
    """
    points_text = "\n".join(f"- {p}" for p in expected_points)

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        temperature=0,
    )

    response = llm.invoke(
        JUDGE_PROMPT.format(
            question=question,
            expected_points=points_text,
            report=report,
        )
    )

    # 解析 JSON
    content = response.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        scores = json.loads(content.strip())
    except json.JSONDecodeError:
        # 解析失败，给默认中等分
        scores = {
            "relevance": 3,
            "completeness": 3,
            "accuracy": 3,
            "reasoning": "评分解析失败，使用默认分数",
        }

    return scores


def score_trace(trace_id: str, scores: dict) -> None:
    """将评分写入 Langfuse trace

    Args:
        trace_id: Langfuse trace ID
        scores: judge_report 返回的评分字典
    """
    langfuse = Langfuse()

    for dimension in ["relevance", "completeness", "accuracy"]:
        if dimension in scores:
            langfuse.create_score(
                trace_id=trace_id,
                name=dimension,
                value=scores[dimension],
                comment=scores.get("reasoning", ""),
            )

    # 总分
    total = scores.get("relevance", 0) + scores.get("completeness", 0) + scores.get("accuracy", 0)
    langfuse.create_score(
        trace_id=trace_id,
        name="total",
        value=total,
        comment=f"总分 {total}/15",
    )
