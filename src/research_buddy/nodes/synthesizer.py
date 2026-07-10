"""综合节点 - 将搜索结果综合为结构化报告（支持流式输出）"""

import logging

from research_buddy.state import ResearchState
from research_buddy.utils import create_llm, get_prompt_from_langfuse

logger = logging.getLogger(__name__)


SYNTHESIZER_PROMPT = """你是一个研究综合专家。根据以下研究问题和搜索结果，生成一份结构化的研究报告。

## 研究问题
{question}

## 搜索结果
{search_results}

## 要求
1. 报告结构：概述 → 各子问题分析 → 结论
2. 每个论点都要引用来源（标注 URL）
3. 如果信息有矛盾，明确标注
4. 最后给出整体置信度（高/中/低）
5. 使用中文撰写"""

SYNTHESIZER_INCREMENTAL_PROMPT = """你是一个研究综合专家。现在需要基于已有知识进行增量综合。

## 研究问题
{question}

## 已有知识
{knowledge_context}

## 新搜索结果
{search_results}

## 要求
1. 在已有知识基础上，补充和更新信息
2. 明确标注哪些是新增/更新的信息（用 🆕 标记）
3. 如果新信息与已有知识矛盾，明确标注（用 ⚠️ 标记）
4. 生成完整的更新后报告（不是只写增量部分）
5. 报告结构：概述 → 各子问题分析 → 结论
6. 每个论点都要引用来源
7. 最后给出整体置信度（高/中/低）
8. 使用中文撰写"""

SYNTHESIZER_REFINE_PROMPT = """你是一个研究综合专家。以下是之前生成的研究报告和改进建议，请根据建议改进报告。

## 研究问题
{question}

## 搜索结果（包含补充搜索的新结果）
{search_results}

## 当前报告
{report}

## 改进建议
{feedback}

## 要求
1. 根据改进建议针对性补充和修正
2. 保留原有报告中仍然有效的部分
3. 每个论点都要引用来源（标注 URL）
4. 如果信息有矛盾，明确标注
5. 最后给出整体置信度（高/中/低）
6. 使用中文撰写"""


def synthesizer(state: ResearchState) -> dict:
    """综合节点：流式输出结构化报告

    支持三种模式：
    - 全新模式：正常生成报告
    - 增量模式：基于已有知识，补充更新报告
    - 改进模式：根据反思反馈改进报告
    使用 streaming 模式，报告内容"打字机式"逐步输出到终端。
    """
    question = state["question"]
    search_results = state.get("search_results", [])
    report = state.get("report", "")
    feedback = state.get("reflection_feedback", "")
    is_incremental = state.get("is_incremental", False)
    has_knowledge = state.get("has_knowledge", False)
    knowledge_context = state.get("knowledge_context", "")

    # 格式化搜索结果
    formatted_results = ""
    for i, r in enumerate(search_results, 1):
        formatted_results += f"\n### 结果 {i}（子问题：{r['sub_question']}）\n"
        formatted_results += f"- 标题：{r['title']}\n"
        formatted_results += f"- 来源：{r['url']}\n"
        formatted_results += f"- 内容：{r['content']}\n"
        formatted_results += f"- 相关度：{r['score']}\n"

    llm = create_llm(streaming=True)

    # 选择模式
    if feedback and report:
        # 改进模式（反思后重写）
        prompt_template = get_prompt_from_langfuse("research-buddy-synthesizer-refine", SYNTHESIZER_REFINE_PROMPT)
        prompt = prompt_template.format(
            question=question,
            search_results=formatted_results,
            report=report,
            feedback=feedback,
        )
        mode = "改进"
    elif is_incremental and has_knowledge and knowledge_context:
        # 增量模式也走 Langfuse Prompt 管理
        prompt_template = get_prompt_from_langfuse("research-buddy-synthesizer-incremental", SYNTHESIZER_INCREMENTAL_PROMPT)
        prompt = prompt_template.format(
            question=question,
            knowledge_context=knowledge_context,
            search_results=formatted_results,
        )
        mode = "增量"
    else:
        # 全新模式
        prompt_template = get_prompt_from_langfuse("research-buddy-synthesizer", SYNTHESIZER_PROMPT)
        prompt = prompt_template.format(
            question=question,
            search_results=formatted_results,
        )
        mode = "全新"

    logger.info("正在生成研究报告（%s模式）...", mode)

    # 流式输出
    full_report = ""
    for chunk in llm.stream(prompt):
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_report += content

    print()  # 换行
    logger.info("报告生成完成（%s模式）", mode)

    return {"report": full_report, "messages": [f"📝 报告生成完成（{mode}模式）"]}
