"""规划节点 - 将研究问题拆解为子问题（支持增量模式）"""

import json
from langchain_openai import ChatOpenAI
from research_buddy.config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL
from research_buddy.state import ResearchState


PLANNER_PROMPT = """你是一个研究规划专家。给定一个研究问题，将其拆解为 3-5 个子问题，
并为每个子问题生成英文搜索词。

重要规则：
- search_query 必须是英文
- search_query 应简洁具体，3-6 个关键词
- 不要使用有歧义的缩写，使用完整表达：
  - ❌ "CFA" → ✅ "Chinese Football Association"
  - ❌ "CSL" → ✅ "Chinese Super League"
  - ❌ "PEP" → ✅ "Python enhancement proposal"
- 子问题用中文描述（方便后续综合时理解）

要求：
- 子问题应覆盖原始问题的不同方面
- search_query 必须是英文，简洁、具体
- 返回 JSON 格式

研究问题：{question}

请返回如下 JSON 格式（不要包含其他内容）：
```json
[
  {{"question": "子问题1（中文）", "search_query": "English search query"}},
  {{"question": "子问题2（中文）", "search_query": "English search query"}}
]
```"""

INCREMENTAL_PLANNER_PROMPT = """你是一个研究规划专家。现在需要基于已有知识进行增量研究。

## 已有知识
{knowledge_context}

## 新的研究问题
{question}

## 规划要求
基于已有知识，只规划需要补充搜索的子问题。已有知识中已经覆盖的方面不需要重复搜索。

重要规则：
- 只规划已有知识中缺失或需要更新的子问题（2-3 个即可）
- search_query 必须是英文，应聚焦于最新动态、新变化、补充信息
- 子问题用中文描述
- 返回 JSON 格式

请返回如下 JSON 格式（不要包含其他内容）：
```json
[
  {{"question": "需要补充的子问题1（中文）", "search_query": "English search query"}},
  {{"question": "需要补充的子问题2（中文）", "search_query": "English search query"}}
]
```"""


def _get_prompt() -> str:
    """获取 prompt 模板：优先从 Langfuse 拉取，fallback 到本地"""
    try:
        from research_buddy.eval.prompts import get_prompt
        return get_prompt("research-buddy-planner", PLANNER_PROMPT)
    except ImportError:
        return PLANNER_PROMPT


def planner(state: ResearchState) -> dict:
    """规划节点：拆解研究问题为子问题

    支持两种模式：
    - 全新模式：正常拆解 3-5 个子问题
    - 增量模式：基于已有知识，只规划需要补充的子问题
    """
    question = state["question"]
    has_knowledge = state.get("has_knowledge", False)
    knowledge_context = state.get("knowledge_context", "")
    is_incremental = state.get("is_incremental", False)

    # 选择 prompt 模板
    if is_incremental and has_knowledge and knowledge_context:
        prompt_template = INCREMENTAL_PLANNER_PROMPT
        prompt = prompt_template.format(
            question=question,
            knowledge_context=knowledge_context,
        )
        mode = "增量"
    else:
        prompt_template = _get_prompt()
        prompt = prompt_template.format(question=question)
        mode = "全新"

    print(f"📝 正在规划子问题（{mode}模式）...")

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        temperature=0,
    )

    response = llm.invoke(prompt)

    # 解析 LLM 返回的 JSON
    content = response.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    sub_questions = json.loads(content.strip())

    print(f"📋 规划完成，拆解为 {len(sub_questions)} 个子问题（{mode}模式）")
    for i, sq in enumerate(sub_questions, 1):
        print(f"   {i}. {sq.get('question', '')[:40]}")

    msgs = [f"📝 正在规划子问题（{mode}模式）...",
            f"📋 拆解为 {len(sub_questions)} 个子问题"]
    for i, sq in enumerate(sub_questions, 1):
        msgs.append(f"   {i}. {sq.get('question', '')[:40]}")

    return {"sub_questions": sub_questions, "messages": msgs}
