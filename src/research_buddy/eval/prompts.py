"""Prompt 版本管理 - 从 Langfuse 拉取 prompt 模板，fallback 到本地

核心设计：
- 本地 prompt 常量保持 Python .format() 语法（{variable}）
- 注册到 Langfuse 时转换为 mustache 语法（{{variable}}）
- 拉取时用 prompt.compile(**kwargs) 渲染变量
- Langfuse 不可用时 fallback 到本地 .format(**kwargs)
"""

import logging
from string import Formatter

from langfuse import Langfuse

from research_buddy.config import (
    LANGFUSE_HOST,
    LANGFUSE_PROMPT_CACHE_TTL,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_TIMEOUT,
)

logger = logging.getLogger(__name__)


# ── 语法转换 ─────────────────────────────────────────────

def convert_format_to_mustache(template: str) -> str:
    """将 Python .format() 模板转换为 Langfuse mustache 语法

    转换规则：
    - {variable} → {{variable}}  （格式占位符 → mustache 变量）
    - {{ → { 和 }} → }  （Python 转义的字面花括号 → 字面花括号）

    使用 string.Formatter 正确解析模板，处理 JSON 示例中
    {{ 和 }} 作为转义字面花括号的情况。

    Args:
        template: Python .format() 语法的模板字符串

    Returns:
        Langfuse mustache 语法的模板字符串
    """
    formatter = Formatter()
    result_parts = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        # 将 Python .format() 的双花括号还原为字面花括号
        unescaped = literal_text.replace('{{', '{').replace('}}', '}')
        result_parts.append(unescaped)
        if field_name is not None:
            # {field_name} → {{field_name}}
            result_parts.append('{{' + field_name + '}}')
    return ''.join(result_parts)


# ── Prompt 拉取 ──────────────────────────────────────────

# 客户端复用：SDK 内部按 public key 做单例，但每次 Langfuse() 仍会重跑一遍
# 配置解析。get_prompt 是每个 LLM 节点调用一次的热路径，这里缓存实例。
_client: Langfuse | None = None


def _get_client() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            timeout=LANGFUSE_TIMEOUT,
        )
    return _client


def get_prompt(name: str, local_fallback: str, **kwargs) -> str:
    """从 Langfuse 获取并渲染 prompt，失败则用本地 fallback

    Args:
        name: Langfuse 中的 prompt 名称，如 "research-buddy-planner"
        local_fallback: 本地硬编码的 prompt 模板（Python .format() 语法）
        **kwargs: 要渲染到 prompt 中的变量值

    Returns:
        渲染后的 prompt 字符串
    """
    try:
        langfuse = _get_client()
        # 将本地 fallback 转为 mustache 语法，供 SDK fallback 参数使用
        mustache_fallback = convert_format_to_mustache(local_fallback)
        prompt = langfuse.get_prompt(
            name=name,
            type="text",
            fallback=mustache_fallback,
            # 命中缓存就不走网络；冷取失败时快速退回本地模板，
            # 不要让一个可观测性组件把研究节点堵在这里
            cache_ttl_seconds=LANGFUSE_PROMPT_CACHE_TTL,
            max_retries=1,
            fetch_timeout_seconds=LANGFUSE_TIMEOUT,
        )
        if kwargs:
            return prompt.compile(**kwargs)
        return prompt.compile()
    except Exception:
        # Langfuse 完全不可用，退回本地 .format()
        if kwargs:
            return local_fallback.format(**kwargs)
        return local_fallback


# ── Prompt 注册 ──────────────────────────────────────────

def register_prompts() -> None:
    """将本地 prompt 注册到 Langfuse（应用启动时调用）

    逻辑：
    - 已存在且内容未变 → 跳过
    - 已存在但内容有变 → 创建新版本（带 production 标签）
    - 不存在 → 创建新 prompt（带 production 标签）
    """
    from research_buddy.nodes.planner import PLANNER_PROMPT, INCREMENTAL_PLANNER_PROMPT
    from research_buddy.nodes.synthesizer import SYNTHESIZER_PROMPT, SYNTHESIZER_INCREMENTAL_PROMPT, SYNTHESIZER_REFINE_PROMPT
    from research_buddy.nodes.reflector import REFLECTOR_PROMPT
    from research_buddy.nodes.validator import EVIDENCE_EVALUATOR_PROMPT
    from research_buddy.tracking.diff import DIFF_ANALYZER_PROMPT
    from research_buddy.eval.judge import JUDGE_PROMPT
    from research_buddy.nodes.knowledge_store import KEY_FACTS_PROMPT

    langfuse = _get_client()

    prompts = {
        "research-buddy-planner": PLANNER_PROMPT,
        "research-buddy-planner-incremental": INCREMENTAL_PLANNER_PROMPT,
        "research-buddy-synthesizer": SYNTHESIZER_PROMPT,
        "research-buddy-synthesizer-incremental": SYNTHESIZER_INCREMENTAL_PROMPT,
        "research-buddy-synthesizer-refine": SYNTHESIZER_REFINE_PROMPT,
        "research-buddy-reflector": REFLECTOR_PROMPT,
        "research-buddy-evidence-evaluator": EVIDENCE_EVALUATOR_PROMPT,
        "research-buddy-diff-analyzer": DIFF_ANALYZER_PROMPT,
        "research-buddy-judge": JUDGE_PROMPT,
        "research-buddy-key-facts": KEY_FACTS_PROMPT,
    }

    for name, template in prompts.items():
        mustache_template = convert_format_to_mustache(template)
        try:
            # 先尝试获取已有 prompt
            existing = langfuse.get_prompt(name, type="text")
            # 比较时用 mustache 版本，因为 Langfuse 中存的就是 mustache 语法
            if existing.prompt != mustache_template:
                # 内容有变化，创建新版本
                langfuse.create_prompt(
                    name=name,
                    type="text",
                    prompt=mustache_template,
                    labels=["production"],
                    config={},
                    commit_message="Auto-sync from local constants",
                )
                logger.info("更新 prompt: %s", name)
            else:
                logger.info("prompt 未变化: %s", name)
        except Exception:
            # prompt 不存在，创建新的
            try:
                langfuse.create_prompt(
                    name=name,
                    type="text",
                    prompt=mustache_template,
                    labels=["production"],
                    config={},
                    commit_message="Initial registration from local constants",
                )
                logger.info("注册 prompt: %s", name)
            except Exception as e:
                logger.warning("注册 prompt 失败: %s (%s)", name, e)


if __name__ == "__main__":
    register_prompts()
