"""共享工具 — 消除各模块重复的 JSON 解析、LLM 实例化、通知常量等逻辑"""

import json
import logging
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any

from langchain_openai import ChatOpenAI

from research_buddy.config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL

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

def create_llm(streaming: bool = False) -> ChatOpenAI:
    """创建 ChatOpenAI 实例（统一配置）

    Args:
        streaming: 是否启用流式输出

    Returns:
        配置好的 ChatOpenAI 实例
    """
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        temperature=0,
        streaming=streaming,
    )


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
# research_notes / source_table 每次综合由 synthesizer 全量重建，覆盖而非追加。
_OVERWRITE_LIST_KEYS = frozenset({
    "sub_questions", "validation_gaps", "evidence_assessments",
    "research_notes", "source_table",
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
