"""Research Buddy FastAPI 应用 - HTTP 服务 + SSE 流式输出 + Web UI + 知识管理 + 追踪"""

import json
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from research_buddy.graph import (
    create_research_graph,
    create_knowledge_research_graph,
    create_tracking_graph,
    get_langfuse_handler,
    run_tracking,
)
from research_buddy.state import ResearchState
from research_buddy.knowledge.store import get_knowledge_store
from research_buddy.tracking.scheduler import get_scheduler
from research_buddy.utils import stream_and_accumulate

logger = logging.getLogger(__name__)


# ── FastAPI Lifespan ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时启动追踪调度器，停止时关闭"""
    # Startup
    scheduler = get_scheduler()
    scheduler.start()
    yield
    # Shutdown
    scheduler.stop()


app = FastAPI(
    title="Research Buddy",
    description="基于 LangGraph + Langfuse 的深度研究 Agent API",
    version="0.3.0",
    lifespan=lifespan,
)


# ── 请求/响应模型 ──────────────────────────────────────

class ResearchRequest(BaseModel):
    """研究请求"""
    question: str


class KnowledgeResearchRequest(BaseModel):
    """知识研究请求"""
    question: str
    topic_id: str
    is_incremental: bool = True


class TopicCreateRequest(BaseModel):
    """创建主题请求"""
    name: str
    description: str = ""
    tracking_keywords: list[str] | None = None
    tracking_cron: str = ""


class TopicUpdateRequest(BaseModel):
    """更新主题请求"""
    name: str | None = None
    description: str | None = None
    tracking_keywords: list[str] | None = None
    tracking_cron: str | None = None
    tracking_enabled: bool | None = None


class ResearchResponse(BaseModel):
    """研究响应"""
    question: str
    report: str
    sub_questions: list[dict]
    search_results_count: int
    reflection_round: int
    reflection_pass: bool


class KnowledgeResearchResponse(BaseModel):
    """知识研究响应"""
    question: str
    report: str
    topic_id: str
    report_id: str
    is_incremental: bool
    sub_questions: list[dict]
    search_results_count: int
    reflection_round: int
    reflection_pass: bool


# ── 健康检查 ────────────────────────────────────────────

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "research-buddy", "version": "0.3.0"}


# ── 原有研究接口（向后兼容） ────────────────────────────

@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """同步研究接口 - 返回完整报告（无知识层）"""
    graph = create_research_graph()
    langfuse_handler = get_langfuse_handler()

    config = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    result = stream_and_accumulate(graph, {"question": req.question}, config)
    result.setdefault("question", req.question)

    if langfuse_handler:
        langfuse_handler._langfuse_client.flush()

    return ResearchResponse(
        question=result.get("question", req.question),
        report=result.get("report", ""),
        sub_questions=result.get("sub_questions", []),
        search_results_count=len(result.get("search_results", [])),
        reflection_round=result.get("reflection_round", 0),
        reflection_pass=result.get("reflection_pass", False),
    )


@app.post("/research/stream")
async def research_stream(req: ResearchRequest):
    """SSE 流式研究接口（POST）- 适合 API 调用"""
    return EventSourceResponse(_event_generator(req.question), ping=15)


@app.get("/research/stream")
async def research_stream_get(question: str = Query(..., description="研究问题")):
    """SSE 流式研究接口（GET）- 适合浏览器 EventSource"""
    return EventSourceResponse(_event_generator(question), ping=15)


# ── 知识研究接口（Phase 6） ─────────────────────────────

@app.post("/research/knowledge", response_model=KnowledgeResearchResponse)
async def knowledge_research(req: KnowledgeResearchRequest):
    """知识研究接口 - 支持增量研究，结果保存到知识库"""
    graph = create_knowledge_research_graph()
    langfuse_handler = get_langfuse_handler()

    config = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    result = stream_and_accumulate(graph, {
        "question": req.question,
        "topic_id": req.topic_id,
        "is_incremental": req.is_incremental,
    }, config)
    result.setdefault("question", req.question)

    if langfuse_handler:
        langfuse_handler._langfuse_client.flush()

    return KnowledgeResearchResponse(
        question=result.get("question", req.question),
        report=result.get("report", ""),
        topic_id=req.topic_id,
        report_id=result.get("saved_report_id", ""),
        is_incremental=req.is_incremental,
        sub_questions=result.get("sub_questions", []),
        search_results_count=len(result.get("search_results", [])),
        reflection_round=result.get("reflection_round", 0),
        reflection_pass=result.get("reflection_pass", False),
    )


@app.post("/research/knowledge/stream")
async def knowledge_research_stream(req: KnowledgeResearchRequest):
    """知识研究 SSE 流式接口"""
    return EventSourceResponse(
        _event_generator(req.question, topic_id=req.topic_id, is_incremental=req.is_incremental),
        ping=15,
    )


# ── 主题管理接口 ────────────────────────────────────────

@app.post("/topics")
async def create_topic(req: TopicCreateRequest):
    """创建研究主题"""
    store = get_knowledge_store()
    topic = store.create_topic(
        name=req.name,
        description=req.description,
        tracking_keywords=req.tracking_keywords,
        tracking_cron=req.tracking_cron,
    )
    return topic


@app.get("/topics")
async def list_topics():
    """列出所有研究主题"""
    store = get_knowledge_store()
    return store.list_topics()


@app.get("/topics/{topic_id}")
async def get_topic(topic_id: str):
    """获取主题详情"""
    store = get_knowledge_store()
    topic = store.get_topic(topic_id)
    if not topic:
        return JSONResponse(status_code=404, content={"error": "主题不存在"})
    return topic


@app.put("/topics/{topic_id}")
async def update_topic(topic_id: str, req: TopicUpdateRequest):
    """更新主题"""
    store = get_knowledge_store()
    kwargs = {}
    if req.name is not None:
        kwargs["name"] = req.name
    if req.description is not None:
        kwargs["description"] = req.description
    if req.tracking_keywords is not None:
        kwargs["tracking_keywords"] = req.tracking_keywords
    if req.tracking_cron is not None:
        # 清洗 cron 表达式：只取前 5 段，忽略用户可能附加的注释
        cron_parts = req.tracking_cron.strip().split()[:5]
        kwargs["tracking_cron"] = " ".join(cron_parts) if len(cron_parts) == 5 else req.tracking_cron.strip()
    if req.tracking_enabled is not None:
        kwargs["tracking_enabled"] = req.tracking_enabled

    topic = store.update_topic(topic_id, **kwargs)
    if not topic:
        return JSONResponse(status_code=404, content={"error": "主题不存在"})
    return topic


@app.delete("/topics/{topic_id}")
async def delete_topic(topic_id: str):
    """删除主题"""
    store = get_knowledge_store()
    success = store.delete_topic(topic_id)
    if not success:
        return JSONResponse(status_code=404, content={"error": "主题不存在"})
    return {"deleted": True}


# ── 报告接口 ────────────────────────────────────────────

@app.get("/topics/{topic_id}/reports")
async def list_reports(topic_id: str, limit: int = Query(20, description="返回数量")):
    """列出主题下的研究报告"""
    store = get_knowledge_store()
    return store.list_reports(topic_id, limit=limit)


@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """获取单个报告"""
    store = get_knowledge_store()
    report = store.get_report(report_id)
    if not report:
        return JSONResponse(status_code=404, content={"error": "报告不存在"})
    return report


@app.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """删除报告"""
    store = get_knowledge_store()
    success = store.delete_report(report_id)
    if not success:
        return JSONResponse(status_code=404, content={"error": "报告不存在"})
    return {"deleted": True}


# ── 追踪接口（Phase 7）─────────────────────────────────────────

class TrackingRequest(BaseModel):
    """手动触发追踪请求"""
    topic_id: str
    question: str | None = None


@app.post("/tracking/run")
async def tracking_run(req: TrackingRequest):
    """手动触发一次追踪任务"""
    import asyncio
    try:
        result = await asyncio.to_thread(run_tracking, req.topic_id, req.question)
        return {
            "status": "completed",
            "topic_id": req.topic_id,
            "changes_detected": len(result.get("detected_changes", [])),
            "notification_sent": result.get("notification_sent", False),
            "report_id": result.get("saved_report_id", ""),
        }
    except Exception as e:
        logger.error("追踪任务失败: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/tracking/jobs")
async def tracking_jobs():
    """列出所有追踪任务"""
    scheduler = get_scheduler()
    return scheduler.list_jobs()


@app.get("/topics/{topic_id}/tracking-logs")
async def list_tracking_logs(topic_id: str, limit: int = Query(20)):
    """列出主题的追踪记录"""
    store = get_knowledge_store()
    return store.list_tracking_logs(topic_id, limit=limit)


@app.get("/tracking-logs/{log_id}/changes")
async def list_changes(log_id: str):
    """列出追踪记录中的变更条目"""
    store = get_knowledge_store()
    return store.list_changes(log_id)


@app.post("/tracking/test-notification")
async def test_notification():
    """发送测试通知"""
    from research_buddy.tracking.notifier import get_notifier
    notifier = get_notifier()
    sent = notifier.send_test_notification()
    return {"sent": sent}


# ── SSE 事件生成器 ──────────────────────────────────────

def _event_generator(question: str, topic_id: str = "",
                     is_incremental: bool = False):
    """SSE 事件生成器（共用）

    使用 asyncio.Queue + asyncio.to_thread 将同步的 graph.stream()
    放到线程中执行，避免阻塞事件循环，实现真正的 SSE 实时推送。
    """

    async def inner():
        import asyncio

        # 选择图
        if topic_id:
            graph = create_knowledge_research_graph()
            input_data = {
                "question": question,
                "topic_id": topic_id,
                "is_incremental": is_incremental,
            }
        else:
            graph = create_research_graph()
            input_data = {"question": question}

        langfuse_handler = get_langfuse_handler()

        config = {}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        queue: asyncio.Queue = asyncio.Queue()

        def _run_graph():
            """在线程中运行 graph.stream()，将事件推入队列"""
            try:
                queue.put_nowait({
                    "event": "progress",
                    "data": json.dumps({"node": "start", "message": f"开始研究: {question}"}),
                })

                result = {}
                for event in graph.stream(input_data, config=config):
                    if isinstance(event, dict):
                        for node_name, state_update in event.items():
                            if isinstance(state_update, dict):
                                # 推送节点进度（含结构化详情）
                                detail = _extract_detail(node_name, state_update)
                                queue.put_nowait({
                                    "event": "progress",
                                    "data": json.dumps({
                                        "node": node_name,
                                        "summary": _summarize_update(node_name, state_update),
                                        "detail": detail,
                                    }),
                                })

                                # 推送节点内的详细消息
                                messages = state_update.get("messages", [])
                                for msg in messages:
                                    queue.put_nowait({
                                        "event": "message",
                                        "data": json.dumps({"text": msg}),
                                    })

                                for key, value in state_update.items():
                                    if isinstance(value, list) and key in result and isinstance(result[key], list):
                                        result[key].extend(value)
                                    else:
                                        result[key] = value

                result.setdefault("question", question)

                queue.put_nowait({
                    "event": "report",
                    "data": json.dumps({
                        "question": result.get("question", question),
                        "report": result.get("report", ""),
                        "sub_questions": result.get("sub_questions", []),
                        "search_results_count": len(result.get("search_results", [])),
                        "reflection_round": result.get("reflection_round", 0),
                        "reflection_pass": result.get("reflection_pass", False),
                        "topic_id": topic_id,
                        "report_id": result.get("saved_report_id", ""),
                        "is_incremental": is_incremental,
                    }),
                })

                queue.put_nowait({
                    "event": "done",
                    "data": json.dumps({"message": "研究完成"}),
                })

            except Exception as e:
                logger.error("SSE 图执行失败: %s", e)
                queue.put_nowait({
                    "event": "error",
                    "data": json.dumps({"message": str(e)}),
                })

            finally:
                # 哨兵值，通知异步生成器结束
                queue.put_nowait(None)
                if langfuse_handler:
                    langfuse_handler._langfuse_client.flush()

        # 在线程中启动 graph.stream()
        thread_task = asyncio.ensure_future(asyncio.to_thread(_run_graph))

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            # 确保线程任务被清理
            if not thread_task.done():
                thread_task.cancel()

    return inner()


def _extract_detail(node_name: str, state_update: dict) -> dict:
    """从节点状态更新中提取结构化详情，供前端渲染子流程卡片"""
    detail: dict = {}

    if node_name == "planner":
        sqs = state_update.get("sub_questions", [])
        detail["sub_questions"] = [
            {"question": sq.get("question", ""), "search_query": sq.get("search_query", "")}
            for sq in sqs
        ]

    elif node_name == "searcher":
        results = state_update.get("search_results", [])
        detail["results_count"] = len(results)
        detail["results_preview"] = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "score": round(r.get("score", 0), 2)}
            for r in results[:8]
        ]

    elif node_name == "validator":
        gaps = state_update.get("validation_gaps", [])
        detail["gaps_count"] = len(gaps)
        detail["gaps"] = [
            {"question": g.get("question", ""), "search_query": g.get("search_query", "")}
            for g in gaps[:5]
        ]

    elif node_name == "synthesizer":
        report = state_update.get("report", "")
        detail["report_length"] = len(report)

    elif node_name == "reflector":
        detail["reflection_pass"] = state_update.get("reflection_pass", False)
        detail["reflection_round"] = state_update.get("reflection_round", 0)
        detail["reflection_feedback"] = state_update.get("reflection_feedback", "")
        gaps = state_update.get("validation_gaps", [])
        if gaps:
            detail["supplement_queries"] = [g.get("search_query", "") for g in gaps[:3]]

    elif node_name == "knowledge_lookup":
        detail["has_knowledge"] = state_update.get("has_knowledge", False)
        detail["knowledge_context_length"] = len(state_update.get("knowledge_context", ""))

    elif node_name == "knowledge_store":
        detail["saved_report_id"] = state_update.get("saved_report_id", "")

    elif node_name == "diff_analyzer":
        changes = state_update.get("detected_changes", [])
        detail["changes_count"] = len(changes)
        detail["similarity"] = round(state_update.get("similarity", 0), 2)

    elif node_name == "change_notifier":
        detail["notification_sent"] = state_update.get("notification_sent", False)

    return detail


def _summarize_update(node_name: str, state_update: dict) -> str:
    """将节点状态更新摘要为可读字符串"""
    if node_name == "knowledge_lookup":
        has = state_update.get("has_knowledge", False)
        return "找到历史知识" if has else "全新研究"
    elif node_name == "planner":
        count = len(state_update.get("sub_questions", []))
        return f"拆解为 {count} 个子问题"
    elif node_name == "searcher":
        count = len(state_update.get("search_results", []))
        return f"获取 {count} 条搜索结果"
    elif node_name == "validator":
        count = len(state_update.get("validation_gaps", []))
        if count > 0:
            return f"{count} 个子问题信息不足"
        return "搜索结果充足"
    elif node_name == "synthesizer":
        return "报告生成完成"
    elif node_name == "reflector":
        passed = state_update.get("reflection_pass", False)
        round_num = state_update.get("reflection_round", 0)
        return f"第 {round_num} 轮反思：{'通过' if passed else '需要修正'}"
    elif node_name == "knowledge_store":
        return "报告已保存到知识库"
    elif node_name == "diff_analyzer":
        changes = len(state_update.get("detected_changes", []))
        return f"检测到 {changes} 项变化"
    elif node_name == "change_notifier":
        sent = state_update.get("notification_sent", False)
        return "通知已发送" if sent else "无需通知"
    return f"{node_name} 完成"


# 挂载静态文件（Web UI）— 必须放在路由之后
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
