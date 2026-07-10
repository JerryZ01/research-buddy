"""Research Buddy - 基于 LangGraph + Langfuse 的深度研究 Agent"""

from research_buddy.graph import create_research_graph, run_research


def main():
    graph = create_research_graph()
    question = input("请输入研究问题: ")
    result = run_research(question)
    print("\n" + "=" * 60)
    print(result.get("report", "未生成报告"))


if __name__ == "__main__":
    main()
