"""Research Buddy - 基于 LangGraph + Langfuse 的深度研究 Agent"""

import logging


def _setup_logging() -> None:
    """配置日志（仅在未配置时设置默认）"""
    root_logger = logging.getLogger("research_buddy")
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)


_setup_logging()

from research_buddy.graph import create_research_graph, run_research


def main():
    graph = create_research_graph()
    question = input("请输入研究问题: ")
    result = run_research(question)
    print("\n" + "=" * 60)
    print(result.get("report", "未生成报告"))


if __name__ == "__main__":
    main()
