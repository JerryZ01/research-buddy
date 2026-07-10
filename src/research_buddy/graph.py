"""LangGraph 工作流定义 - Phase 7 定时追踪 + 变化检测 + 智能通知"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse import Langfuse
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
    MAX_REFLECTION_ROUNDS,
)


def get_langfuse_handler() -> CallbackHandler | None:
    """获取 Langfuse CallbackHandler"""
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        return CallbackHandler()
    return None


def should_continue(state: ResearchState) -> str:
    """路由函数：反思后决定继续搜索还是结束（核心图用）"""
    if state.get("reflection_pass", False):
        return "end"

    if state.get("reflection_round", 0) >= MAX_REFLECTION_ROUNDS:
        return "end"

    return "search_again"


def should_continue_to_store(state: ResearchState) -> str:
    """路由函数：反思后决定继续搜索还是存入知识库（知识/追踪图用）"""
    if state.get("reflection_pass", False):
        return "knowledge_store"

    if state.get("reflection_round", 0) >= MAX_REFLECTION_ROUNDS:
        return "knowledge_store"

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
    graph.add_edge("validator", "synthesizer")
    graph.add_edge("synthesizer", "reflector")

    graph.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "end": END,
            "search_again": "searcher",
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
    graph.add_edge("validator", "synthesizer")
    graph.add_edge("synthesizer", "reflector")

    # 条件边：reflector → knowledge_store（通过） 或 searcher（不通过）
    graph.add_conditional_edges(
        "reflector",
        should_continue_to_store,
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

def _stream_and_accumulate(graph, input_data, config) -> dict:
    """流式执行图并累积最终状态

    处理 graph.stream() 返回的两种格式：
    - 无 Checkpointer: event 是 dict {node_name: state_update}
    - 有 Checkpointer: event 可能是 tuple (node_name, state_update)

    统一用 graph.get_state() 获取最终状态。
    """
    for event in graph.stream(input_data, config=config):
        # 节点内部已打印进度，这里只消费事件驱动执行
        pass

    # 通过 checkpointer 获取完整最终状态（如果有）
    try:
        return dict(graph.get_state(config).values)
    except (ValueError, AttributeError):
        # 无 checkpointer 的图，手动累积
        result = {}
        for event in graph.stream(input_data, config=config):
            if isinstance(event, dict):
                for node_name, state_update in event.items():
                    if isinstance(state_update, dict):
                        for key, value in state_update.items():
                            if isinstance(value, list) and key in result and isinstance(result[key], list):
                                result[key].extend(value)
                            else:
                                result[key] = value
        return result


def run_research(question: str) -> dict:
    """运行全自动研究流程（流式输出，实时进度，无知识层）"""
    graph = create_research_graph()
    langfuse_handler = get_langfuse_handler()

    config = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    print(f"\n🚀 开始研究: {question}")
    print("=" * 60)

    # 流式执行，节点内部打印进度
    result = {}
    for event in graph.stream({"question": question}, config=config):
        if isinstance(event, dict):
            for node_name, state_update in event.items():
                if isinstance(state_update, dict):
                    for key, value in state_update.items():
                        if isinstance(value, list) and key in result and isinstance(result[key], list):
                            result[key].extend(value)
                        else:
                            result[key] = value

    result.setdefault("question", question)

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
    print(f"\n🚀 开始{mode}研究: {question}")
    print(f"   主题 ID: {topic_id}")
    print("=" * 60)

    # 流式执行
    result = {}
    for event in graph.stream({
        "question": question,
        "topic_id": topic_id,
        "is_incremental": is_incremental,
    }, config=config):
        if isinstance(event, dict):
            for node_name, state_update in event.items():
                if isinstance(state_update, dict):
                    for key, value in state_update.items():
                        if isinstance(value, list) and key in result and isinstance(result[key], list):
                            result[key].extend(value)
                        else:
                            result[key] = value

    result.setdefault("question", question)

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
    graph.add_edge("validator", "synthesizer")
    graph.add_edge("synthesizer", "reflector")

    # 条件边：reflector → knowledge_store（通过） 或 searcher（不通过）
    graph.add_conditional_edges(
        "reflector",
        should_continue_to_store,
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

    print(f"\n⏰ 开始追踪: {topic['name']}")
    print(f"   问题: {question}")
    print("=" * 60)

    # 流式执行
    result = {}
    for event in graph.stream({
        "question": question,
        "topic_id": topic_id,
        "is_incremental": True,
    }, config=config):
        if isinstance(event, dict):
            for node_name, state_update in event.items():
                if isinstance(state_update, dict):
                    for key, value in state_update.items():
                        if isinstance(value, list) and key in result and isinstance(result[key], list):
                            result[key].extend(value)
                        else:
                            result[key] = value

    result.setdefault("question", question)

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

    print(f"\n🚀 开始研究: {question}")
    print("=" * 60)

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
