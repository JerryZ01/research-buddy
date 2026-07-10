"""反思节点 - LLM 自评报告质量，决定是否需要修正"""

import logging

from research_buddy.state import ResearchState
from research_buddy.utils import parse_llm_json, create_llm, get_prompt_from_langfuse

logger = logging.getLogger(__name__)


REFLECTOR_PROMPT = """你是一个研究质量评审专家。请评估以下研究报告的质量，从三个维度打分（1-5 分）：

## 研究问题
{question}

## 子问题
{sub_questions}

## 搜索结果数量
{result_count} 条

## 研究报告
{report}

{user_feedback_section}

## 评分维度
1. **完整性**（1-5）：是否回答了所有子问题，有无遗漏
2. **准确性**（1-5）：论点是否有充分来源支撑，有无凭空推测
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
- supplement_queries 必须是英文搜索词，且简短（2-5 个关键词）
- 如果有用户反馈，优先针对用户反馈的不足生成补充搜索词"""


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

    logger.info("正在反思评估报告质量...")

    llm = create_llm()
    prompt_template = get_prompt_from_langfuse("research-buddy-reflector", REFLECTOR_PROMPT)

    response = llm.invoke(
        prompt_template.format(
            question=question,
            sub_questions=sq_text,
            result_count=len(search_results),
            report=report,
            user_feedback_section=user_feedback_section,
        )
    )

    # 解析 LLM 返回的 JSON
    try:
        evaluation = parse_llm_json(response.content)
    except (ValueError, Exception):
        logger.warning("反思节点解析失败，默认通过")
        return {
            "reflection_pass": True,
            "reflection_feedback": "反思节点解析失败，默认通过",
            "reflection_round": current_round + 1,
        }

    passed = evaluation.get("pass", True)
    feedback = evaluation.get("feedback", "")
    supplement_queries = evaluation.get("supplement_queries", [])
    total_score = evaluation.get("total_score", 0)

    # 验证 total_score 与维度分数之和一致
    dim_sum = evaluation.get("completeness", 0) + evaluation.get("accuracy", 0) + evaluation.get("clarity", 0)
    if total_score != dim_sum and dim_sum > 0:
        logger.debug("total_score (%d) 与维度之和 (%d) 不一致，使用维度之和", total_score, dim_sum)
        total_score = dim_sum

    logger.info("评分: %d/15 → %s", total_score, "✅ 通过" if passed else "⚠️  需要修正")

    # 如果有用户反馈但反思通过了，仍需补充搜索（用户要求优先）
    if user_feedback and passed:
        if supplement_queries:
            passed = False
        else:
            # 有用户反馈但 LLM 没生成补充搜索词，强制不通过
            logger.info("有用户反馈但无补充搜索词，强制不通过")
            passed = False
            supplement_queries = [f"{question} detailed analysis"]

    # 如果未通过，生成补充搜索任务
    gaps = []
    if not passed and supplement_queries:
        for query in supplement_queries:
            gaps.append({
                "question": f"补充搜索：{query}",
                "search_query": query,
            })

    # 合并用户反馈到 reflection_feedback
    if user_feedback:
        feedback = f"[用户要求] {user_feedback}\n[评估反馈] {feedback}"

    result_msg = f"🔄 反思: 评分 {total_score}/15 → {'✅ 通过' if passed else '⚠️ 需要修正'}"
    if not passed and supplement_queries:
        result_msg += f"，补充搜索: {', '.join(supplement_queries[:3])}"

    return {
        "reflection_pass": passed,
        "reflection_feedback": feedback,
        "reflection_round": current_round + 1,
        "validation_gaps": gaps,
        "messages": [result_msg],
    }
