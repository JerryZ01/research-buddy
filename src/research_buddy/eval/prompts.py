"""Prompt 版本管理 - 从 Langfuse 拉取 prompt 模板，fallback 到本地"""

from langfuse import Langfuse


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
        # Langfuse prompt 可能是 text 或 chat 格式
        # text 格式直接用 prompt.prompt
        template = prompt.prompt
        print(f"📌 使用 Langfuse prompt: {name} (v{prompt.version})")
        return template
    except Exception:
        return local_fallback


def register_prompts() -> None:
    """将本地 prompt 注册到 Langfuse（首次使用时调用）"""
    from research_buddy.nodes.planner import PLANNER_PROMPT
    from research_buddy.nodes.synthesizer import SYNTHESIZER_PROMPT
    from research_buddy.nodes.reflector import REFLECTOR_PROMPT

    langfuse = Langfuse()

    prompts = {
        "research-buddy-planner": PLANNER_PROMPT,
        "research-buddy-synthesizer": SYNTHESIZER_PROMPT,
        "research-buddy-reflector": REFLECTOR_PROMPT,
    }

    for name, template in prompts.items():
        try:
            langfuse.create_prompt(
                name=name,
                prompt=template,
                config={},
            )
            print(f"✅ 注册 prompt: {name}")
        except Exception as e:
            # 可能已存在，尝试创建新版本
            try:
                # 获取已有 prompt 然后创建新版本
                existing = langfuse.get_prompt(name)
                if existing.prompt != template:
                    langfuse.create_prompt(
                        name=name,
                        prompt=template,
                        config={},
                    )
                    print(f"✅ 更新 prompt: {name}")
                else:
                    print(f"⏭️  prompt 未变化: {name}")
            except Exception:
                print(f"⚠️  注册 prompt 失败: {name} ({e})")


if __name__ == "__main__":
    register_prompts()
