"""Phase 3 交互式测试脚本 - Human-in-the-loop

用法：
    uv run python scripts/test_phase3.py

流程：
1. 输入研究问题
2. 图执行到 searcher 前暂停 → 用户确认/调整子问题
3. 图恢复执行到 reflector 前暂停 → 用户查看报告并补充要求
4. 图恢复执行到结束 → 输出最终报告
"""

from research_buddy.graph import run_research_interactive


def main():
    question = input("🔍 请输入研究问题: ").strip()
    if not question:
        question = "LangGraph 和 LangChain 的区别是什么？"
        print(f"使用默认问题: {question}")

    result = run_research_interactive(question)

    # 打印统计
    sub_questions = result.get("sub_questions", [])
    search_results = result.get("search_results", [])
    reflection_round = result.get("reflection_round", 0)
    reflection_pass = result.get("reflection_pass", False)

    print("\n" + "=" * 60)
    print("📊 执行统计")
    print("=" * 60)
    print(f"子问题数: {len(sub_questions)}")
    print(f"搜索结果数: {len(search_results)}")
    print(f"反思轮次: {reflection_round}")
    print(f"反思通过: {reflection_pass}")

    print("\n" + "=" * 60)
    print("📋 最终研究报告")
    print("=" * 60)
    print(result.get("report", "未生成报告"))


if __name__ == "__main__":
    main()
