"""LangGraph 工作流定义 - Phase 7 定时追踪 + 变化检测 + 智能通知"""

import logging
from contextlib import contextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from research_buddy.state import ResearchState
from research_buddy.nodes.planner import planner
from research_buddy.nodes.searcher import searcher
from research_buddy.nodes.validator import validator
from research_buddy.nodes.synthesizer import synthesizer
from research_buddy.nodes.reflector import reflector
from research_buddy.nodes.knowledge_lookup import knowledge_lookup
from research_buddy.nodes.knowledge_store import knowledge_store
from research_buddy.nodes.diff_analyzer import diff_analyzer
from research_buddy.nodes.change_notifier import change_notifier
from research_buddy.config import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    LANGFUSE_TIMEOUT,
    MAX_REFLECTION_ROUNDS,
    MAX_SEARCH_ROUNDS,
    MAX_TOTAL_QUERIES,
)
from research_buddy.utils import stream_and_accumulate

logger = logging.getLogger(__name__)


def get_langfuse_handler() -> CallbackHandler | None:
    """获取 Langfuse CallbackHandler"""
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            # SDK 默认 5 秒，跨境导出 span 批次经常读超时并丢 trace
            timeout=LANGFUSE_TIMEOUT,
        )
        return CallbackHandler()
    return None


@contextmanager
def _langfuse_run(name: str, question: str, handler: CallbackHandler | None):
    """为一次完整运行开一个 Langfuse 根 span，并把 trace_id 交给调用方。

    评估链路需要一个确定的 trace_id 才能把 LLM-as-Judge 的分数挂到正确的 trace 上。
    事后查询「最近一条 trace」既有竞态，也依赖 Langfuse v2 才有的 get_traces()
    （v3 重写时已删除，装的 4.x 上直接 AttributeError）。这里主动建 span，
    图内所有节点的 span 会通过 OTEL 上下文挂到同一个 trace 下。

    未配置 Langfuse 密钥时 yield 空字符串，调用方据此跳过打分。
    """
    if handler is None:
        yield ""
        return

    client = get_client()
    with client.start_as_current_observation(
        name=name, as_type="span", input={"question": question},
    ):
        yield client.get_current_trace_id() or ""


def should_continue(state: ResearchState) -> str:
    """路由函数：反思后决定继续搜索还是结束（核心图用）"""
    if state.get("reflection_pass", False):
        return "end"

    if (state.get("search_round", 0) >= MAX_SEARCH_ROUNDS
            or state.get("total_queries", 0) >= MAX_TOTAL_QUERIES):
        return "end"

    if state.get("reflection_round", 0) >= MAX_REFLECTION_ROUNDS:
        return "end"

    # 搜索层不可用时补搜必然再失败，只能就现有材料改写报告
    if state.get("search_unavailable", False):
        return "revise_report"

    if not state.get("validation_gaps", []):
        return "revise_report"

    return "search_again"


def should_continue_to_store(state: ResearchState) -> str:
    """路由函数：反思后决定继续搜索还是存入知识库（知识/追踪图用）"""
    if state.get("reflection_pass", False):
        return "knowledge_store"

    if (state.get("search_round", 0) >= MAX_SEARCH_ROUNDS
            or state.get("total_queries", 0) >= MAX_TOTAL_QUERIES):
        return "knowledge_store"

    if state.get("reflection_round", 0) >= MAX_REFLECTION_ROUNDS:
        return "knowledge_store"

    if state.get("search_unavailable", False):
        return "revise_report"

    if not state.get("validation_gaps", []):
        return "revise_report"

    return "search_again"


def route_after_validation(state: ResearchState) -> str:
    """证据评估后决定补搜或生成报告。"""
    gaps = state.get("validation_gaps", [])
    if not gaps:
        return "synthesize"
    if state.get("stop_reason") in {"search_budget_exhausted", "no_new_queries",
                                    "search_unavailable"}:
        return "synthesize"
    return "search_again"


def _add_core_nodes_and_edges(graph: StateGraph) -> None:
    """添加核心研究节点和边（不含 HITL，不含知识层）"""
    graph.add_node("planner", planner)
    graph.add_node("searcher", searcher)
    graph.add_node("validator", validator)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("reflector", reflector)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {"search_again": "searcher", "synthesize": "synthesizer"},
    )
    graph.add_edge("synthesizer", "reflector")

    graph.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "end": END,
            "search_again": "searcher",
            "revise_report": "synthesizer",
        },
    )


# ── 工作流创建 ──────────────────────────────────────────

def create_research_graph() -> StateGraph:
    """创建研究工作流（无知识层，全自动，向后兼容）"""
    graph = StateGraph(ResearchState)
    _add_core_nodes_and_edges(graph)
    return graph.compile()


def create_knowledge_research_graph() -> StateGraph:
    """创建知识研究工作流（带知识层，支持增量研究）

    流程：knowledge_lookup → planner → searcher → validator → synthesizer → reflector → knowledge_store → END
    """
    graph = StateGraph(ResearchState)

    # 添加核心节点
    graph.add_node("planner", planner)
    graph.add_node("searcher", searcher)
    graph.add_node("validator", validator)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("reflector", reflector)

    # 添加知识层节点
    graph.add_node("knowledge_lookup", knowledge_lookup)
    graph.add_node("knowledge_store", knowledge_store)

    # 边：START → knowledge_lookup → planner → searcher → validator → synthesizer → reflector
    graph.add_edge(START, "knowledge_lookup")
    graph.add_edge("knowledge_lookup", "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {"search_again": "searcher", "synthesize": "synthesizer"},
    )
    graph.add_edge("synthesizer", "reflector")

    # 条件边：reflector → knowledge_store（通过） 或 searcher（不通过）
    graph.add_conditional_edges(
        "reflector",
        should_continue_to_store,
        {
            "knowledge_store": "knowledge_store",
            "search_again": "searcher",
            "revise_report": "synthesizer",
        },
    )
    graph.add_edge("knowledge_store", END)

    return graph.compile()


def create_research_graph_with_hitl() -> StateGraph:
    """创建研究工作流（带 HITL，无知识层，向后兼容）"""
    graph = StateGraph(ResearchState)
    _add_core_nodes_and_edges(graph)

    memory = MemorySaver()

    return graph.compile(
        checkpointer=memory,
        interrupt_before=["searcher", "reflector"],
    )


# ── 便捷运行函数 ────────────────────────────────────────

def run_research(question: str) -> dict:
    """运行全自动研究流程（流式输出，实时进度，无知识层）"""
    graph = create_research_graph()
    langfuse_handler = get_langfuse_handler()

    config = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    logger.info("开始研究: %s", question)

    with _langfuse_run("research", question, langfuse_handler) as trace_id:
        result = stream_and_accumulate(graph, {"question": question}, config)
    result.setdefault("question", question)
    result["langfuse_trace_id"] = trace_id

    # 确保 Langfuse 数据刷出
    if langfuse_handler:
        langfuse_handler._langfuse_client.flush()

    return result


def run_knowledge_research(question: str, topic_id: str,
                           is_incremental: bool = True) -> dict:
    """运行知识研究流程（带知识层，支持增量研究）

    Args:
        question: 研究问题
        topic_id: 关联的研究主题 ID
        is_incremental: 是否增量模式（True=只搜索新信息，False=全新研究但保存结果）
    """
    graph = create_knowledge_research_graph()
    langfuse_handler = get_langfuse_handler()

    config = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    mode = "增量" if is_incremental else "全新"
    logger.info("开始%s研究: %s (主题: %s)", mode, question, topic_id)

    with _langfuse_run("knowledge-research", question, langfuse_handler) as trace_id:
        result = stream_and_accumulate(graph, {
            "question": question,
            "topic_id": topic_id,
            "is_incremental": is_incremental,
        }, config)
    result.setdefault("question", question)
    result["langfuse_trace_id"] = trace_id

    # 确保 Langfuse 数据刷出
    if langfuse_handler:
        langfuse_handler._langfuse_client.flush()

    return result


# ── 追踪工作流（Phase 7）─────────────────────────────────

def create_tracking_graph() -> StateGraph:
    """创建追踪工作流

    流程：knowledge_lookup → planner → searcher → validator → synthesizer
          → reflector → knowledge_store → diff_analyzer → change_notifier → END

    与知识研究工作流的区别：
    - 反思通过后先保存报告，再做变化分析
    - 变化分析后决定是否发送通知
    """
    graph = StateGraph(ResearchState)

    # 核心研究节点
    graph.add_node("planner", planner)
    graph.add_node("searcher", searcher)
    graph.add_node("validator", validator)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("reflector", reflector)

    # 知识层节点
    graph.add_node("knowledge_lookup", knowledge_lookup)
    graph.add_node("knowledge_store", knowledge_store)

    # 追踪层节点
    graph.add_node("diff_analyzer", diff_analyzer)
    graph.add_node("change_notifier", change_notifier)

    # 边：research pipeline
    graph.add_edge(START, "knowledge_lookup")
    graph.add_edge("knowledge_lookup", "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {"search_again": "searcher", "synthesize": "synthesizer"},
    )
    graph.add_edge("synthesizer", "reflector")

    # 条件边：reflector → knowledge_store（通过） 或 searcher（不通过）
    graph.add_conditional_edges(
        "reflector",
        should_continue_to_store,
        {
            "knowledge_store": "knowledge_store",
            "search_again": "searcher",
            "revise_report": "synthesizer",
        },
    )

    # 追踪链：knowledge_store → diff_analyzer → change_notifier → END
    graph.add_edge("knowledge_store", "diff_analyzer")
    graph.add_edge("diff_analyzer", "change_notifier")
    graph.add_edge("change_notifier", END)

    return graph.compile()


def run_tracking(topic_id: str, question: str | None = None) -> dict:
    """运行一次追踪任务

    Args:
        topic_id: 要追踪的主题 ID
        question: 可选的自定义研究问题（默认自动生成）
    """
    from research_buddy.knowledge.store import get_knowledge_store

    store = get_knowledge_store()
    topic = store.get_topic(topic_id)
    if not topic:
        raise ValueError(f"主题 {topic_id} 不存在")

    # 自动生成追踪问题
    if not question:
        keywords = topic.get("tracking_keywords", [])
        kw_str = "、".join(keywords[:3]) if keywords else topic["name"]
        question = f"{topic['name']} 最新动态和变化（{kw_str}）"

    graph = create_tracking_graph()
    langfuse_handler = get_langfuse_handler()

    config = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    logger.info("开始追踪: %s (问题: %s)", topic['name'], question)

    with _langfuse_run("tracking", question, langfuse_handler) as trace_id:
        result = stream_and_accumulate(graph, {
            "question": question,
            "topic_id": topic_id,
            "is_incremental": True,
        }, config)
    result.setdefault("question", question)
    result["langfuse_trace_id"] = trace_id

    # 确保 Langfuse 数据刷出
    if langfuse_handler:
        langfuse_handler._langfuse_client.flush()

    return result


def run_research_interactive(question: str) -> dict:
    """运行交互式研究流程（带 HITL，流式输出）"""
    from langgraph.types import Command

    graph = create_research_graph_with_hitl()
    langfuse_handler = get_langfuse_handler()

    config = {"configurable": {"thread_id": "research-1"}}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    logger.info("开始交互式研究: %s", question)

    # 第一次执行：运行到 searcher 前暂停
    for event in graph.stream({"question": question}, config=config):
        pass  # 节点内部已打印进度

    result = dict(graph.get_state(config).values)

    # === 第一次中断：用户确认子问题 ===
    sub_questions = result.get("sub_questions", [])
    print("\n📋 规划的子问题：")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq.get('question', '')}（搜索词：{sq.get('search_query', '')}）")

    print("\n✏️  输入调整后的子问题（JSON 格式），或直接回车确认：")
    user_input = input("> ").strip()

    if user_input:
        import json
        try:
            updated_sqs = json.loads(user_input)
            resume_value = Command(resume={"sub_questions": updated_sqs})
        except json.JSONDecodeError:
            print("⚠️  JSON 解析失败，使用原始子问题")
            resume_value = Command(resume={})
    else:
        resume_value = Command(resume={})

    # 恢复执行
    print("\n🔍 继续搜索和验证...")
    for event in graph.stream(resume_value, config=config):
        pass

    result = dict(graph.get_state(config).values)

    # === 第二次中断：用户查看报告并补充要求 ===
    report = result.get("report", "")
    print("\n📝 当前报告预览：")
    print("-" * 40)
    print(report[:800] + ("..." if len(report) > 800 else ""))
    print("-" * 40)

    print("\n✏️  输入补充要求或改进建议，或直接回车确认：")
    feedback = input("> ").strip()

    resume_value2 = Command(resume={"user_feedback": feedback}) if feedback else Command(resume={})

    # 恢复执行到结束
    print("\n🔄 继续反思和修正...")
    for event in graph.stream(resume_value2, config=config):
        pass

    result = dict(graph.get_state(config).values)

    # 确保 Langfuse 数据刷出
    if langfuse_handler:
        langfuse_handler._langfuse_client.flush()

    return result
