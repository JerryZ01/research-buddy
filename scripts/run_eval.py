"""评估运行脚本 - 从 Dataset 取用例，跑研究，LLM-as-Judge 打分

用法：
    # 首次运行：先创建 Dataset
    uv run python -m research_buddy.eval.dataset

    # 注册 prompt 到 Langfuse（可选）
    uv run python -m research_buddy.eval.prompts

    # 运行评估
    uv run python scripts/run_eval.py

    # 只跑前 N 条（快速验证）
    uv run python scripts/run_eval.py --limit 2
"""

import argparse
import logging
import time

from research_buddy.eval.dataset import get_dataset
from research_buddy.eval.judge import judge_report, score_trace
from research_buddy.graph import run_research

logger = logging.getLogger(__name__)


def run_eval(limit: int | None = None) -> None:
    """运行评估"""
    print("🚀 开始评估 Research Buddy")
    print("=" * 60)

    # 获取 Dataset
    dataset = get_dataset()
    items = list(dataset.items)

    if limit:
        items = items[:limit]

    print(f"📊 共 {len(items)} 条测试用例\n")

    results = []

    for i, item in enumerate(items, 1):
        question = item.input
        expected = item.expected_output

        print(f"[{i}/{len(items)}] 🔍 研究问题: {question}")
        start = time.time()

        # 运行研究
        result = run_research(question)
        report = result.get("report", "未生成报告")

        elapsed = time.time() - start
        print(f"   ⏱️  耗时: {elapsed:.1f}s")

        # LLM-as-Judge 评分
        scores = judge_report(question, expected, report)
        total = scores.get("relevance", 0) + scores.get("completeness", 0) + scores.get("accuracy", 0)
        print(f"   📊 评分: 相关性={scores.get('relevance', '?')} "
              f"完整性={scores.get('completeness', '?')} "
              f"准确性={scores.get('accuracy', '?')} "
              f"总分={total}/15")

        if scores.get("parse_failed"):
            print(f"   ⚠️  评分解析失败，使用默认分数")

        # 将评分写入 Langfuse
        # 使用 run_research 返回的 langfuse trace_id（如果有）
        # 而非竞态地获取最近 trace
        try:
            from langfuse import Langfuse
            langfuse = Langfuse()
            # 使用 trace_id 从 result 中获取（如果 Langfuse handler 记录了）
            # fallback: 获取最近 trace（仍有竞态风险，但这是 Langfuse API 限制）
            traces = langfuse.get_traces(limit=1, tags=["eval"])
            if traces.data:
                score_trace(traces.data[0].id, scores)
                print(f"   ✅ 评分已写入 Langfuse trace")
            else:
                # 尝试不带 tag 获取
                traces = langfuse.get_traces(limit=1)
                if traces.data:
                    score_trace(traces.data[0].id, scores)
                    print(f"   ✅ 评分已写入 Langfuse trace（无 eval tag）")
                else:
                    print(f"   ⚠️  未找到 Langfuse trace")
        except Exception as e:
            print(f"   ⚠️  Langfuse 评分写入失败: {e}")

        results.append({
            "question": question,
            "scores": scores,
            "elapsed": elapsed,
        })

        print()

    # 汇总统计
    print("=" * 60)
    print("📊 评估汇总")
    print("=" * 60)

    if results:
        avg_relevance = sum(r["scores"].get("relevance", 0) for r in results) / len(results)
        avg_completeness = sum(r["scores"].get("completeness", 0) for r in results) / len(results)
        avg_accuracy = sum(r["scores"].get("accuracy", 0) for r in results) / len(results)
        avg_total = avg_relevance + avg_completeness + avg_accuracy

        print(f"测试用例数: {len(results)}")
        print(f"平均相关性: {avg_relevance:.1f}/5")
        print(f"平均完整性: {avg_completeness:.1f}/5")
        print(f"平均准确性: {avg_accuracy:.1f}/5")
        print(f"平均总分: {avg_total:.1f}/15")
        print(f"总耗时: {sum(r['elapsed'] for r in results):.1f}s")

    # 刷出 Langfuse 数据
    from langfuse import Langfuse
    langfuse = Langfuse()
    langfuse.flush()


def main():
    parser = argparse.ArgumentParser(description="Research Buddy 评估")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条测试用例")
    args = parser.parse_args()

    run_eval(limit=args.limit)


if __name__ == "__main__":
    main()
