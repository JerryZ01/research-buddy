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

        # 运行研究（run_research 会开一个 Langfuse 根 span 并回传 trace_id）
        result = run_research(question)
        report = result.get("report", "未生成报告")
        trace_id = result.get("langfuse_trace_id", "")

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
            print("   ⚠️  评分解析失败，使用默认分数（不计入汇总）")

        # 将评分写入 Langfuse：用本次运行自己的 trace_id，
        # 不再事后查询「最近一条 trace」（有竞态，且 get_traces 在 Langfuse v3+ 已删除）
        if trace_id:
            try:
                score_trace(trace_id, scores)
                print(f"   ✅ 评分已写入 Langfuse trace {trace_id}")
            except Exception as e:
                print(f"   ⚠️  Langfuse 评分写入失败: {e}")
        else:
            print("   ⚠️  未配置 LANGFUSE_* 密钥，跳过评分写入")

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
        # 解析失败的条目分数是占位的 3/3/3，计入均值会把结论粉饰得比实际好
        scored = [r for r in results if not r["scores"].get("parse_failed")]
        skipped = len(results) - len(scored)

        print(f"测试用例数: {len(results)}")
        if skipped:
            print(f"评分解析失败（已排除）: {skipped}")

        if scored:
            avg_relevance = sum(r["scores"].get("relevance", 0) for r in scored) / len(scored)
            avg_completeness = sum(r["scores"].get("completeness", 0) for r in scored) / len(scored)
            avg_accuracy = sum(r["scores"].get("accuracy", 0) for r in scored) / len(scored)
            avg_total = avg_relevance + avg_completeness + avg_accuracy

            print(f"有效评分数: {len(scored)}")
            print(f"平均相关性: {avg_relevance:.1f}/5")
            print(f"平均完整性: {avg_completeness:.1f}/5")
            print(f"平均准确性: {avg_accuracy:.1f}/5")
            print(f"平均总分: {avg_total:.1f}/15")
        else:
            print("⚠️  全部用例评分解析失败，没有可用的汇总结果")

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
