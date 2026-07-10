"""Phase 1 端到端测试脚本

用法：
    uv run python scripts/test_phase1.py
"""

from research_buddy.graph import run_research


def main():
    question = "LangGraph 和 LangChain 的区别是什么？"
    print(f"🔍 研究问题: {question}")
    print("-" * 60)

    result = run_research(question)

    print("\n" + "=" * 60)
    print("📋 研究报告")
    print("=" * 60)
    print(result.get("report", "未生成报告"))

    # 打印搜索结果统计
    sub_questions = result.get("sub_questions", [])
    search_results = result.get("search_results", [])
    print("\n" + "-" * 60)
    print(f"📊 统计: {len(sub_questions)} 个子问题, {len(search_results)} 条搜索结果")


if __name__ == "__main__":
    main()
