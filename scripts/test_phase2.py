"""Phase 2 端到端测试脚本（流式输出版）

用法：
    uv run python scripts/test_phase2.py
"""

from research_buddy.graph import run_research
from research_buddy.config import MAX_REFLECTION_ROUNDS


def main():
    question = "LangGraph 和 LangChain 的区别是什么？"
    print(f"🔍 研究问题: {question}")
    print(f"🔄 最大反思轮次: {MAX_REFLECTION_ROUNDS}")

    result = run_research(question)

    # 打印统计
    sub_questions = result.get("sub_questions", [])
    search_results = result.get("search_results", [])
    reflection_round = result.get("reflection_round", 0)
    reflection_pass = result.get("reflection_pass", False)
    reflection_feedback = result.get("reflection_feedback", "")

    print("\n" + "=" * 60)
    print("📊 执行统计")
    print("=" * 60)
    print(f"子问题数: {len(sub_questions)}")
    print(f"搜索结果数: {len(search_results)}")
    print(f"反思轮次: {reflection_round}")
    print(f"反思通过: {reflection_pass}")
    if reflection_feedback:
        print(f"反思反馈: {reflection_feedback[:200]}")

    print("\n" + "=" * 60)
    print("📋 研究报告")
    print("=" * 60)
    print(result.get("report", "未生成报告"))


if __name__ == "__main__":
    main()
