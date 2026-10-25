"""规划节点 - 将研究问题拆解为子问题（支持增量模式）"""

import json
import logging
import re

from langchain_core.runnables import RunnableConfig

from research_buddy.config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL
from research_buddy.state import ResearchState
from research_buddy.utils import invoke_llm, parse_llm_json, create_llm, get_prompt_from_langfuse

logger = logging.getLogger(__name__)


PLANNER_PROMPT = """你是一个研究规划专家。给定一个研究问题，根据复杂度拆解为 1-6 个必要子问题，
并为每个子问题生成适合研究对象的搜索词。

重要规则：
- 根据问题涉及的国家、地区和信息源自动选择搜索语言
- 中国本地政策、市场、机构和社会议题以中文查询为主，必要时增加英文查询用于国际对照
- 国际技术、论文和全球议题：**以中文查询为主，同时生成英文查询做国际对照**——用户用中文提问时，中文资料应占主导，英文用于补充权威一手来源，不要把所有搜索词都切成英文
- 跨地区问题可同时生成中文和英文查询
- 查询应简洁、具体，不得机械翻译专有名词
- 不要使用有歧义的缩写，使用完整表达：
  - ❌ "CFA" → ✅ "Chinese Football Association"
  - ❌ "CSL" → ✅ "Chinese Super League"
  - ❌ "PEP" → ✅ "Python enhancement proposal"
- 子问题用中文描述（方便后续综合时理解）

要求：
- 子问题应覆盖原始问题的不同方面
- 简单事实问题只保留必要分支，不要为了凑数量重复拆分
- search_query 保留主查询；search_queries 列出实际执行的全部查询
- language 使用 zh、en 或其他 ISO 语言代码，region 使用 CN、GLOBAL 或明确国家/地区代码
- source_preference 描述优先来源，如 official、academic、news、industry
- 返回 JSON 格式

研究问题：{question}

请返回如下 JSON 格式（不要包含其他内容）：
```json
[
  {{"question": "子问题1（中文）", "search_query": "中文主搜索词", "language": "zh", "region": "CN", "source_preference": "official", "search_queries": [{{"query": "中文搜索词", "language": "zh", "region": "CN"}}, {{"query": "English comparison query", "language": "en", "region": "GLOBAL"}}]}},
  {{"question": "子问题2（中文）", "search_query": "中文主搜索词", "language": "zh", "region": "CN", "source_preference": "academic", "search_queries": [{{"query": "中文搜索词", "language": "zh", "region": "CN"}}, {{"query": "English comparison query", "language": "en", "region": "GLOBAL"}}]}}
]
```

注意：中文提问时，每个子问题的 search_query 用中文、language 用 zh，同时在 search_queries 里补充一条英文对照查询（language=en、region=GLOBAL），不要只生成英文查询。"""

INCREMENTAL_PLANNER_PROMPT = """你是一个研究规划专家。现在需要基于已有知识进行增量研究。

## 已有知识
{knowledge_context}

## 新的研究问题
{question}

## 规划要求
基于已有知识，只规划需要补充搜索的子问题。已有知识中已经覆盖的方面不需要重复搜索。

重要规则：
- 只规划已有知识中缺失或需要更新的子问题；如果已有知识足够，可返回空数组
- 搜索语言应匹配信息来源；国内变化优先中文，全球或学术变化优先英文，必要时双语
- 子问题用中文描述
- 返回 JSON 格式

请返回如下 JSON 格式（不要包含其他内容）：
```json
[
  {{"question": "需要补充的子问题1（中文）", "search_query": "主搜索词", "language": "zh", "region": "CN", "source_preference": "official", "search_queries": [{{"query": "中文最新动态", "language": "zh", "region": "CN"}}]}},
  {{"question": "需要补充的子问题2（中文）", "search_query": "English update query", "language": "en", "region": "GLOBAL", "source_preference": "academic", "search_queries": [{{"query": "English update query", "language": "en", "region": "GLOBAL"}}]}}
]
```"""


# 提取原问题里的英文核心术语（排除提问功能词），用于检测模型是否把专有名词
# 漂移成了近义词（如 Harness → runtime）。模型对 "harness" 这类 agent 领域专有名词
# 认知不稳，即使 prompt 明令禁止也会漂移，必须用确定性兜底。
_ENGLISH_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
_TERM_STOPWORDS = {
    "what", "is", "are", "how", "why", "when", "where", "who", "which",
    "the", "a", "an", "of", "to", "for", "with", "about", "write", "article",
    "deep", "analysis", "analyze", "explain", "describe", "compare", "please",
    "me", "my", "and", "or", "on", "in", "can", "you", "do", "does", "did",
    "give", "tell", "vs", "versus",
}


def _core_terms(question: str) -> list[str]:
    """原问题里的英文核心术语（长度≥2，排除提问/功能词）。"""
    terms = [t for t in _ENGLISH_TERM_RE.findall(question or "") if len(t) >= 2]
    return [t for t in terms if t.lower() not in _TERM_STOPWORDS]


def _terms_drifted(question: str, sub_questions: list[dict]) -> bool:
    """判断核心英文术语是否在所有子问题里都丢失（被模型漂移成近义词）。

    仅在「纯英文输入」时启用：用户明确用英文专有名词（如 "Agent Harness"），
    若模型把它偷换成别的英文词（runtime），就是主题漂移。
    中文语境输入（含中文字符）时不检测——模型把英文术语翻译成中文
    （如 "AI Agent" → "人工智能 Agent"）是正常且期望的行为，不能误判。
    """
    terms = _core_terms(question)
    if not terms:
        return False
    # 含中文字符 = 中文语境输入，翻译是正常的，跳过漂移检测
    if any("一" <= ch <= "鿿" for ch in question):
        return False
    haystack = " ".join(
        (str(sq.get("question", "")) or "") + " " + (str(sq.get("search_query", "")) or "")
        for sq in sub_questions
    ).lower()
    return any(term.lower() not in haystack for term in terms)


def planner(state: ResearchState, config: RunnableConfig | None = None) -> dict:
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
        # 增量模式也走 Langfuse Prompt 管理
        prompt = get_prompt_from_langfuse(
            "research-buddy-planner-incremental", INCREMENTAL_PLANNER_PROMPT,
            question=question,
            knowledge_context=knowledge_context,
        )
        mode = "增量"
    else:
        prompt = get_prompt_from_langfuse(
            "research-buddy-planner", PLANNER_PROMPT,
            question=question,
        )
        mode = "全新"

    logger.info("正在规划子问题（%s模式）...", mode)

    llm = create_llm()
    # 传 config：让 graph 级 callbacks（含 Langfuse CallbackHandler）传播到本次调用
    response = invoke_llm(llm, prompt, config=config)

    # 解析 LLM 返回的 JSON（统一用 parse_llm_json，含 try/except）
    try:
        sub_questions = parse_llm_json(response.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("规划节点 JSON 解析失败: %s，返回空列表", e)
        sub_questions = []

    # 稳定 ID 由代码生成，避免后续补搜依赖可能变化的中文问题文本。
    normalized_sub_questions = []
    for index, item in enumerate(sub_questions, 1):
        if not isinstance(item, dict):
            continue
        question_text = str(item.get("question", "")).strip()
        raw_queries = item.get("search_queries", [])
        search_queries = []
        if isinstance(raw_queries, list):
            for query_item in raw_queries:
                if isinstance(query_item, str):
                    query_text = query_item.strip()
                    query_item = {"query": query_text}
                elif isinstance(query_item, dict):
                    query_text = str(query_item.get("query", "")).strip()
                else:
                    continue
                if query_text:
                    search_queries.append({
                        "query": query_text,
                        "language": str(query_item.get("language", item.get("language", ""))).strip() or "auto",
                        "region": str(query_item.get("region", item.get("region", ""))).strip() or "GLOBAL",
                    })
        search_query = str(item.get("search_query", "")).strip()
        if not search_query and search_queries:
            search_query = search_queries[0]["query"]
        if search_query and not search_queries:
            search_queries = [{
                "query": search_query,
                "language": str(item.get("language", "")).strip() or "auto",
                "region": str(item.get("region", "")).strip() or "GLOBAL",
            }]
        if not question_text or not search_query:
            continue
        normalized_sub_questions.append({
            "id": str(item.get("id") or f"sq_{index:02d}"),
            "question": question_text,
            "search_query": search_query,
            "search_queries": search_queries,
            "language": str(item.get("language", search_queries[0].get("language", "auto"))).strip() or "auto",
            "region": str(item.get("region", search_queries[0].get("region", "GLOBAL"))).strip() or "GLOBAL",
            "source_preference": str(item.get("source_preference", "general")).strip() or "general",
        })
    sub_questions = normalized_sub_questions

    # 主题漂移兜底：模型可能把核心英文专有名词（如 Harness）漂移成近义词
    # （runtime）。若所有子问题都丢失了原问题的核心术语，退回单子问题，
    # 保留原问题作为唯一的搜索依据，避免整个研究答非所问。
    if sub_questions and _terms_drifted(question, sub_questions):
        logger.warning("检测到主题漂移（核心术语 %s 丢失），回退为单子问题",
                       ", ".join(_core_terms(question)))
        sub_questions = [{
            "id": "sq_01",
            "question": question,
            "search_query": question,
            "search_queries": [{"query": question, "language": "auto", "region": "GLOBAL"}],
            "language": "auto",
            "region": "GLOBAL",
            "source_preference": "general",
        }]

    logger.info("规划完成，拆解为 %d 个子问题（%s模式）", len(sub_questions), mode)
    for i, sq in enumerate(sub_questions, 1):
        logger.debug("   %d. %s", i, sq.get('question', '')[:40])

    msgs = [f"📝 正在规划子问题（{mode}模式）...",
            f"📋 拆解为 {len(sub_questions)} 个子问题"]
    for i, sq in enumerate(sub_questions, 1):
        msgs.append(f"   {i}. {sq.get('question', '')[:40]}")

    return {"sub_questions": sub_questions, "messages": msgs}