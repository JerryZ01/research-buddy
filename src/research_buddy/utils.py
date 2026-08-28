"""共享工具 — 消除各模块重复的 JSON 解析、LLM 实例化、通知常量等逻辑"""

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from research_buddy.config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_MODEL,
    OPENAI_STRIP_SDK_HEADERS,
)

logger = logging.getLogger(__name__)

# ── 通知常量 ────────────────────────────────────────────

SIGNIFICANCE_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}

CHANGE_TYPE_LABEL = {
    "new_info": "新增",
    "update": "更新",
    "contradiction": "⚠️ 矛盾",
}


# ── JSON 解析 ───────────────────────────────────────────

def parse_llm_json(content: str) -> Any:
    """解析 LLM 返回的 JSON，自动剥离 code-fence

    处理 LLM 常见的返回格式：
    - 纯 JSON
    - ```json ... ```
    - ``` ... ```

    Raises:
        json.JSONDecodeError: 内容无法解析为 JSON
    """
    text = content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


# ── LLM 工厂 ───────────────────────────────────────────

# 当前研究的 token 累计计数器（contextvar，每次研究运行开始时开启）。
# 节点里的 LLM 调用不传 config（Langfuse 的 CallbackHandler 因而也收不到
# 节点内 Generation），所以统一在 create_llm 挂一个记录回调，把每次调用的
# usage 累进这个计数器——不依赖 langchain 的回调传播。
_run_tokens: ContextVar[dict | None] = ContextVar("research_run_token_usage", default=None)


def _normalize_usage(usage: dict | None) -> dict:
    """把各家 LLM 的 usage 字段归一化为 {input_tokens, output_tokens, total_tokens}。"""
    usage = usage or {}
    try:
        return {
            "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
    except (TypeError, ValueError):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


@contextmanager
def track_run_tokens():
    """开启一次研究的 token 累计。

    with track_run_tokens() as usage:
        ... 执行图 ...
    # usage 是 {input_tokens, output_tokens, total_tokens}（始终含这三个键）
    """
    token = _run_tokens.set({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    try:
        yield _run_tokens.get()
    finally:
        _run_tokens.reset(token)


def add_tokens(usage: dict | None) -> None:
    """把一次 LLM 调用的 usage 累进当前研究计数器（无追踪上下文时为空操作）。"""
    counter = _run_tokens.get()
    if counter is None:
        return
    norm = _normalize_usage(usage)
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        counter[key] = counter.get(key, 0) + norm[key]


def get_current_token_usage() -> dict:
    """当前研究的累计 token 用量（无追踪上下文时返回全 0）。"""
    counter = _run_tokens.get()
    return {
        "input_tokens": (counter or {}).get("input_tokens", 0),
        "output_tokens": (counter or {}).get("output_tokens", 0),
        "total_tokens": (counter or {}).get("total_tokens", 0),
    }


class _UsageRecorder(BaseCallbackHandler):
    """挂在每个 LLM 上的回调：把每次调用的 token 用量累计进当前研究计数器。"""

    def on_llm_end(self, response, **kwargs):
        try:
            usage = (response.llm_output or {}).get("token_usage")
            add_tokens(usage)
        except Exception:
            pass


# ── LLM 调用重试（瞬时错误韧性） ─────────────────────────

# 瞬时错误：限流(429)与 5xx（服务端过载/抖动）——重试有意义；
# 400/401/403 等重试无意义，直接抛。
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_LLM_RETRY_SLEEP = (1.5, 3.0)


def _is_transient_error(exc: Exception) -> bool:
    """判断异常是否为可重试的瞬时错误（openai.APIStatusError 等带 status_code）。"""
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            return int(status) in _TRANSIENT_STATUS
        except (TypeError, ValueError):
            return False
    # 兜底：错误消息里带 503/429/5xx 字样也算瞬时
    msg = str(exc)
    return any(token in msg for token in ("503", "429", "502", "504", "temporarily unavailable"))


def invoke_llm(llm, prompt: str, config: dict | None = None,
               max_retries: int = 2) -> Any:
    """调用 LLM，瞬时错误（429/5xx）自动退避重试。

    搜索层已有重试，但 LLM 调用此前是一次性的——一个瞬时 503 就能让
    10 分钟的研究整体失败。这里给非流式调用补上重试。
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(prompt, config=config)
        except Exception as exc:
            last_error = exc
            if not _is_transient_error(exc) or attempt >= max_retries:
                raise
            delay = _LLM_RETRY_SLEEP[min(attempt, len(_LLM_RETRY_SLEEP) - 1)]
            logger.warning("LLM 调用瞬时失败（第 %d/%d 次，%.1fs 后重试）: %s",
                           attempt + 1, max_retries + 1, delay, str(exc)[:120])
            time.sleep(delay)
    raise last_error


def _strip_openai_sdk_headers(request: httpx.Request) -> None:
    """清理 SDK 环境指纹，保留 LangChain 解析响应所需的功能性头。"""
    functional_headers = {"x-stainless-raw-response", "x-stainless-helper-method"}
    for name in list(request.headers):
        normalized = name.lower()
        if normalized.startswith("x-stainless-") and normalized not in functional_headers:
            del request.headers[name]
    request.headers["user-agent"] = "python-httpx"


async def _strip_openai_sdk_headers_async(request: httpx.Request) -> None:
    _strip_openai_sdk_headers(request)


@lru_cache(maxsize=1)
def _compat_http_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    """进程内复用连接池，避免每次节点调用都新建 HTTP 客户端。"""
    timeout = httpx.Timeout(600.0, connect=30.0)
    return (
        httpx.Client(
            timeout=timeout,
            event_hooks={"request": [_strip_openai_sdk_headers]},
        ),
        httpx.AsyncClient(
            timeout=timeout,
            event_hooks={"request": [_strip_openai_sdk_headers_async]},
        ),
    )


def create_llm(streaming: bool = False, max_tokens: int | None = None,
               temperature: float = 0, model: str | None = None) -> ChatOpenAI:
    """创建 ChatOpenAI 实例（统一配置，每次调用自动累计 token 用量）

    Args:
        streaming: 是否启用流式输出（流式时请求 usage 字段，token 统计才完整）
        max_tokens: 单次生成最大 token 数；None 不限制（由提供商默认决定）
        temperature: 采样温度。默认 0（确定性）用于规划/评估/反思等需要稳定的
            调用；写作类调用（synthesizer 出稿）传 WRITER_TEMPERATURE 放开
            随机性，避免多次生成结构雷同。
        model: 可选模型覆盖；Article Eval 可借此使用独立 Judge 模型。

    Returns:
        配置好的 ChatOpenAI 实例
    """
    client_kwargs = {}
    if OPENAI_STRIP_SDK_HEADERS:
        http_client, http_async_client = _compat_http_clients()
        client_kwargs = {
            "http_client": http_client,
            "http_async_client": http_async_client,
        }
    llm = ChatOpenAI(
        model=model or OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        temperature=temperature,
        streaming=streaming,
        stream_usage=streaming,
        max_tokens=max_tokens,
        **client_kwargs,
    )
    # 挂上 usage 记录回调（不依赖 langchain 的回调传播，节点内裸调用也能计数）
    return llm.with_config({"callbacks": [_UsageRecorder()]})


# ── Prompt 获取 ─────────────────────────────────────────

def get_prompt_from_langfuse(name: str, fallback: str, **kwargs) -> str:
    """从 Langfuse Prompt Management 获取并渲染 prompt，失败则用本地 fallback

    Args:
        name: Langfuse 中的 prompt 名称，如 "research-buddy-planner"
        fallback: 本地硬编码的 prompt 模板（Python .format() 语法）
        **kwargs: 要渲染到 prompt 中的变量值

    Returns:
        渲染后的 prompt 字符串
    """
    try:
        from research_buddy.eval.prompts import get_prompt
        return get_prompt(name, fallback, **kwargs)
    except ImportError:
        if kwargs:
            return fallback.format(**kwargs)
        return fallback


# ── 变更摘要 ───────────────────────────────────────────

def summarize_changes(changes: list[dict]) -> str:
    """生成变更摘要文本（统一格式，消除 4 处重复）"""
    if not changes:
        return "无变化"
    parts = []
    for c in changes:
        sig = SIGNIFICANCE_EMOJI.get(c.get("significance", "medium"), "⚪")
        parts.append(f"{sig} {c.get('description', '')}")
    return "\n".join(parts)


# ── URL 规范化 ─────────────────────────────────────────

def normalize_url(url: str) -> str:
    """URL 规范化：统一主机、路径并移除 fragment/常见追踪参数。"""
    if not url:
        return ""
    try:
        parts = urlsplit(url if "://" in url else f"https://{url}")
        host = (parts.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = parts.path.rstrip("/") or "/"
        tracking_prefixes = ("utm_", "ref", "source", "fbclid", "gclid")
        query = urlencode([
            (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith(tracking_prefixes)
        ])
        return urlunsplit(("", host, path, query, ""))
    except ValueError:
        return url.strip().lower().rstrip("/")


# ── 流式累积 ───────────────────────────────────────────

# 覆盖语义的列表字段（不使用 operator.add，便于 HITL 编辑与清空已处理缺口）
# 单一来源：stream_and_accumulate 与 api.py 的 SSE 生成器共用。
# research_notes / source_table / selected_images 每次综合由 synthesizer 全量重建，覆盖而非追加。
_OVERWRITE_LIST_KEYS = frozenset({
    "sub_questions", "validation_gaps", "evidence_assessments",
    "research_notes", "source_table", "selected_images", "core_refs",
    "evidence_ledger", "language_edits", "evidence_edits",
})


def merge_state_update(result: dict, state_update: dict) -> None:
    """把单个节点的 state_update 合并进累积 result（就地修改）。

    全工作流 state merge 的单一来源：
    - 覆盖语义键（_OVERWRITE_LIST_KEYS）直接覆盖；
    - 其余 list 键在已存在同键 list 时 extend，否则覆盖（首现落入覆盖分支）；
    - 标量覆盖。
    """
    for key, value in state_update.items():
        if key in _OVERWRITE_LIST_KEYS:
            result[key] = value
        elif isinstance(value, list) and key in result and isinstance(result[key], list):
            result[key].extend(value)
        else:
            result[key] = value


def stream_and_accumulate(graph, input_data: dict, config: dict | None = None) -> dict:
    """流式执行 LangGraph 图并累积最终状态

    处理 graph.stream() 返回的格式：
    - event 是 dict {node_name: state_update}

    列表字段 extend（追加语义），但 sub_questions、validation_gaps、
    evidence_assessments 用覆盖语义。标量字段 overwrite。
    """
    config = config or {}
    result: dict = {}
    for event in graph.stream(input_data, config=config):
        if isinstance(event, dict):
            for node_name, state_update in event.items():
                if isinstance(state_update, dict):
                    merge_state_update(result, state_update)
    return result
