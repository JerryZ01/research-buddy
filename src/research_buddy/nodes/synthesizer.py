"""综合节点 - 将搜索结果综合为可发布的结构化报告（支持流式输出）

报告 = 纯文章正文（无评价性内容、无内嵌 URL）+ 代码生成的文末参考文献。
- 正文引用用 [编号]（如「...引用了来源[1]」），编号来自 synthesizer 构建的 source_table
- 参考文献（## 参考文献）由代码按同一编号表生成，与 LLM 输出无关，100% 准确
- 矛盾/不足/置信度等评价性信息写入 research_notes / confidence（state 字段），不进正文
"""

import logging

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from research_buddy.state import ResearchState
from research_buddy.utils import create_llm, get_prompt_from_langfuse, normalize_url

logger = logging.getLogger(__name__)


# ── Prompt 模板（可发布文章风格） ────────────────────────

SYNTHESIZER_PROMPT = """你是一位资深研究撰稿人。根据研究问题和检索证据，撰写一篇结构完整、可直接发布的研究文章。

## 研究问题
{question}

## 检索证据
{search_results}

## 可用来源编号表
{source_table}

## 撰写要求
1. 文章结构：概述 → 各子问题分析 → 结论
2. 论点引用来源时，在句末用方括号标注编号，如「……这一机制于 1992 年引入[1]」。编号必须来自「可用来源编号表」，一个论点可引用多个编号（如 [1][2]）
3. 只陈述客观事实与基于证据的分析，语气中立、行文流畅，达到可直接发布的质量
4. 不评价检索过程本身：不要在正文中写「信息存在矛盾」「证据不足」「研究局限」「本报告基于……搜索」等元评论
5. 不要在正文中直接写出任何 URL（来源只通过 [编号] 引用）
6. 不要自行添加「参考文献」「来源」「置信度」章节（文末参考文献由系统自动生成）
7. 使用中文撰写"""

SYNTHESIZER_INCREMENTAL_PROMPT = """你是一位资深研究撰稿人。请基于已有知识撰写一篇更新后的完整研究文章（不是只写增量部分），达到可直接发布的质量。

## 研究问题
{question}

## 已有知识
{knowledge_context}

## 新检索证据
{search_results}

## 可用来源编号表
{source_table}

## 撰写要求
1. 在已有知识基础上补充和更新信息，生成完整的更新后文章
2. 客观陈述最新进展与既有结论之间的关系（如「截至 2024 年……」「此前的结论在最新资料中得到确认」），不使用 🆕/⚠️ 等进度标记符号
3. 论点引用来源时，在句末用方括号标注编号（如 [1]），编号必须来自「可用来源编号表」
4. 只陈述客观事实与基于证据的分析，不评价检索过程本身（不写「存在矛盾」「证据不足」「研究局限」等元评论）
5. 不要在正文中直接写出任何 URL
6. 不要自行添加「参考文献」「来源」「置信度」章节
7. 文章结构：概述 → 各子问题分析 → 结论
8. 使用中文撰写"""

SYNTHESIZER_REFINE_PROMPT = """你是一位资深研究撰稿人。以下是之前生成的研究文章和改进建议，请根据建议改进，产出可直接发布的最终稿。

## 研究问题
{question}

## 检索证据（包含补充搜索的新结果）
{search_results}

## 可用来源编号表
{source_table}

## 当前文章
{report}

## 改进建议
{feedback}

## 撰写要求
1. 根据改进建议针对性补充和修正，保留原文中仍然有效的部分
2. 论点引用来源时，在句末用方括号标注编号（如 [1]），编号必须来自「可用来源编号表」，新增内容使用新编号
3. 只陈述客观事实与基于证据的分析，不评价检索过程本身（不写「存在矛盾」「证据不足」「研究局限」等元评论）
4. 不要在正文中直接写出任何 URL
5. 不要自行添加「参考文献」「来源」「置信度」章节
6. 文章结构：概述 → 各子问题分析 → 结论
7. 使用中文撰写"""


# ── 来源编号表 ──────────────────────────────────────────

def build_source_table(state: ResearchState) -> list[dict]:
    """从 search_results + known_source_urls 构建编号引用表。

    顺序：本次检索结果（按收集顺序）在前，历史知识来源在后。
    按 normalize_url 去重；knowledge 来源没有标题，用 URL 充当。
    """
    table: list[dict] = []
    seen: set[str] = set()
    for r in state.get("search_results", []):
        url = r.get("url", "")
        key = normalize_url(url)
        if url and key and key not in seen:
            seen.add(key)
            table.append({
                "index": len(table) + 1,
                "title": r.get("title", "") or url,
                "url": url,
                "source": "search",
            })
    for url in state.get("known_source_urls", []):
        key = normalize_url(url)
        if url and key and key not in seen:
            seen.add(key)
            table.append({
                "index": len(table) + 1,
                "title": url,
                "url": url,
                "source": "knowledge",
            })
    return table


def format_source_table(table: list[dict]) -> str:
    """把编号表格式化为 prompt 可读文本。"""
    if not table:
        return "（无可用来源）"
    return "\n".join(
        f"[{item['index']}] {item.get('title', '')} — {item.get('url', '')}"
        for item in table
    )


def render_references(table: list[dict]) -> str:
    """生成文末参考文献的 Markdown（代码生成，编号与正文 [n] 一一对应）。"""
    if not table:
        return ""
    lines = [
        "",
        "## 参考文献",
        "",
        *[f"{item['index']}. [{item.get('title', item.get('url', ''))}]({item.get('url', '')})"
          for item in table],
        "",
    ]
    return "\n".join(lines)


def compute_confidence(state: ResearchState) -> str:
    """由代码从证据质量确定性计算置信度（高/中/低），不进报告正文。

    规则（按优先级）：
    - 搜索层不可用（无新证据）→ 低
    - 语义证据评估降级（仅机械校验）→ 中
    - 预算耗尽/无新查询/反思轮次用尽 → 中
    - 仍有未解决缺口 → 中
    - 其余（无缺口、无降级、无预算问题）→ 高
    """
    if state.get("search_unavailable"):
        return "低"
    if state.get("evidence_assessment_degraded"):
        return "中"
    if state.get("stop_reason") in {"search_budget_exhausted", "no_new_queries",
                                    "reflection_budget_exhausted"}:
        return "中"
    if state.get("validation_gaps"):
        return "中"
    return "高"


def _build_research_notes(state: ResearchState) -> list[str]:
    """收集不进正文的研究说明（局限/降级/未解决缺口）。"""
    unresolved_gaps = state.get("validation_gaps", [])
    stop_reason = state.get("stop_reason", "")

    notes: list[str] = []
    if state.get("search_unavailable"):
        notes.append("本次检索未获得任何新证据（搜索层不可用），结论缺少来源支撑。")
    if state.get("evidence_assessment_degraded"):
        notes.append("语义证据评估不可用，仅做了来源数/域名/覆盖度的机械校验，未做语义充分性判断。")
    if stop_reason in {"search_budget_exhausted", "no_new_queries", "reflection_budget_exhausted"}:
        notes.append(f"本次研究因 `{stop_reason}` 停止，以下内容仍需进一步验证。")
        notes.extend(
            f"- {gap.get('question', '未解决问题')}：{gap.get('reason', '证据不足')}"
            for gap in unresolved_gaps
        )
    return notes


def synthesizer(state: ResearchState, config: RunnableConfig, *, writer: StreamWriter) -> dict:
    """综合节点：流式输出可发布的研究文章

    支持三种模式：
    - 全新模式：正常生成文章
    - 增量模式：基于已有知识，补充更新文章
    - 改进模式：根据反思反馈重写文章

    使用 streaming 模式，报告内容"打字机式"逐步输出到终端和前端。
    通过 writer 参数将每个 chunk 推送到 SSE 层，实现前端实时显示。

    writer 必须标注为 StreamWriter：LangGraph 只在注解命中
    `(StreamWriter, "StreamWriter", inspect.Parameter.empty)` 白名单时才注入该参数
    （见 langgraph/_internal/_runnable.py 的 KWARGS_CONFIG_KEYS）。标成
    `Callable | None` 会让注入被静默跳过，writer 恒为 None，report_chunk 事件全部消失。
    这里不给默认值，缺少注入时直接 TypeError，而不是退化成无声失效。
    """
    question = state["question"]
    search_results = state.get("search_results", [])
    report = state.get("report", "")
    feedback = state.get("reflection_feedback", "")
    is_incremental = state.get("is_incremental", False)
    has_knowledge = state.get("has_knowledge", False)
    knowledge_context = state.get("knowledge_context", "")
    evidence_assessments = state.get("evidence_assessments", [])
    unresolved_gaps = state.get("validation_gaps", [])
    stop_reason = state.get("stop_reason", "")

    # 编号引用表：正文 [n] 与文末参考文献的单一事实来源
    source_table = build_source_table(state)

    # 格式化搜索结果
    formatted_results = ""
    for i, r in enumerate(search_results, 1):
        formatted_results += f"\n### 结果 {i}（子问题：{r['sub_question']}）\n"
        formatted_results += f"- 标题：{r['title']}\n"
        formatted_results += f"- 来源：{r['url']}\n"
        formatted_results += f"- 内容：{r['content']}\n"
        formatted_results += f"- 相关度：{r['score']}\n"

    if evidence_assessments:
        formatted_results += "\n## 证据覆盖状态（仅供你判断，不要写进正文）\n"
        for assessment in evidence_assessments:
            formatted_results += (
                f"- {assessment.get('sub_question_id', '')}: "
                f"{assessment.get('status', '')}, 覆盖度 {assessment.get('coverage', 0):.0%}, "
                f"有效来源 {assessment.get('valid_results', 0)}，独立域名 {assessment.get('distinct_domains', 0)}\n"
            )
    if unresolved_gaps:
        formatted_results += "\n## 未解决证据缺口（仅供你判断，不要写进正文）\n"
        for gap in unresolved_gaps:
            formatted_results += f"- {gap.get('question', '')}: {gap.get('reason', '信息不足')}\n"
    if state.get("evidence_assessment_degraded"):
        formatted_results += (
            "\n注意：本次语义证据评估不可用，只做了来源数/域名/覆盖度的机械校验。"
            "正文中不得声称结论已交叉验证，但也不要把这条写进正文。\n"
        )
    if state.get("search_unavailable"):
        formatted_results += (
            "\n注意：本次检索没有获得任何新证据（搜索层不可用）。"
            "只能基于上面的已有知识作答，且不要声称有来源支撑。\n"
        )
    if stop_reason in {"search_budget_exhausted", "no_new_queries", "reflection_budget_exhausted"}:
        formatted_results += f"\n研究因 {stop_reason} 停止。正文不得将有限结论描述为已完全验证。\n"

    llm = create_llm(streaming=True)
    prompt_kwargs = {
        "question": question,
        "search_results": formatted_results,
        "source_table": format_source_table(source_table),
    }

    # 选择模式
    if feedback and report:
        # 改进模式（反思后重写）
        prompt = get_prompt_from_langfuse(
            "research-buddy-synthesizer-refine", SYNTHESIZER_REFINE_PROMPT,
            **prompt_kwargs,
            report=report,
            feedback=feedback,
        )
        mode = "改进"
    elif is_incremental and has_knowledge and knowledge_context:
        # 增量模式也走 Langfuse Prompt 管理
        prompt = get_prompt_from_langfuse(
            "research-buddy-synthesizer-incremental", SYNTHESIZER_INCREMENTAL_PROMPT,
            **prompt_kwargs,
            knowledge_context=knowledge_context,
        )
        mode = "增量"
    else:
        # 全新模式
        prompt = get_prompt_from_langfuse(
            "research-buddy-synthesizer", SYNTHESIZER_PROMPT,
            **prompt_kwargs,
        )
        mode = "全新"

    logger.info("正在生成研究文章（%s模式）...", mode)

    # 流式输出正文
    full_report = ""
    for chunk in llm.stream(prompt):
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_report += content
            # 推送 chunk 到 SSE 层，实现前端实时显示
            if writer:
                writer({"type": "report_chunk", "content": content})

    # 文末参考文献由代码确定性生成：编号与正文 [n] 引用一一对应，不依赖 LLM
    references = render_references(source_table)
    if references:
        full_report += references
        if writer:
            writer({"type": "report_chunk", "content": references})

    print()  # 换行
    logger.info("报告生成完成（%s模式）", mode)

    return {
        "report": full_report,
        "confidence": compute_confidence(state),
        "research_notes": _build_research_notes(state),
        "source_table": source_table,
        "messages": [f"📝 报告生成完成（{mode}模式）"],
    }
