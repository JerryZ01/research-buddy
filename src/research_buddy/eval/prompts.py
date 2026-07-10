"""Prompt 版本管理 - 从 Langfuse 拉取 prompt 模板，fallback 到本地"""

import logging

from langfuse import Langfuse

logger = logging.getLogger(__name__)


def get_prompt(name: str, local_fallback: str) -> str:
    """从 Langfuse Prompt Management 获取最新 prompt，失败则用本地硬编码

    Args:
        name: Langfuse 中的 prompt 名称，如 "research-buddy-planner"
        local_fallback: 本地硬编码的 prompt 模板

    Returns:
        prompt 模板字符串
    """
    try:
        langfuse = Langfuse()
        prompt = langfuse.get_prompt(name)
        template = prompt.prompt
        logger.info("使用 Langfuse prompt: %s (v%s)", name, prompt.version)
        return template
    except Exception:
        return local_fallback


def register_prompts() -> None:
    """将本地 prompt 注册到 Langfuse（首次使用时调用）"""
    from research_buddy.nodes.planner import PLANNER_PROMPT, INCREMENTAL_PLANNER_PROMPT
    from research_buddy.nodes.synthesizer import SYNTHESIZER_PROMPT, SYNTHESIZER_INCREMENTAL_PROMPT, SYNTHESIZER_REFINE_PROMPT
    from research_buddy.nodes.reflector import REFLECTOR_PROMPT

    langfuse = Langfuse()

    prompts = {
        "research-buddy-planner": PLANNER_PROMPT,
        "research-buddy-planner-incremental": INCREMENTAL_PLANNER_PROMPT,
        "research-buddy-synthesizer": SYNTHESIZER_PROMPT,
        "research-buddy-synthesizer-incremental": SYNTHESIZER_INCREMENTAL_PROMPT,
        "research-buddy-synthesizer-refine": SYNTHESIZER_REFINE_PROMPT,
        "research-buddy-reflector": REFLECTOR_PROMPT,
    }

    for name, template in prompts.items():
        try:
            # 先尝试获取已有 prompt
            existing = langfuse.get_prompt(name)
            if existing.prompt != template:
                # 内容有变化，创建新版本
                langfuse.create_prompt(
                    name=name,
                    prompt=template,
                    config={},
                )
                logger.info("更新 prompt: %s", name)
            else:
                logger.info("prompt 未变化: %s", name)
        except Exception:
            # prompt 不存在，创建新的
            try:
                langfuse.create_prompt(
                    name=name,
                    prompt=template,
                    config={},
                )
                logger.info("注册 prompt: %s", name)
            except Exception as e:
                logger.warning("注册 prompt 失败: %s (%s)", name, e)


if __name__ == "__main__":
    register_prompts()
