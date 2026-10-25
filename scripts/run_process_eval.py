"""离线流程评估脚本 - 对整个研究流程的中间产物 + 最终结果打分并写回 Langfuse。

用法：
    # 首次运行：先创建 Dataset
    uv run python -m research_buddy.eval.dataset

    # 注册流程 Judge prompt 到 Langfuse（可选，推荐先跑一次）
    uv run python -m research_buddy.eval.prompts

    # 运行流程评估（跑完整研究 + 分阶段打分）
    uv run python scripts/run_process_eval.py

    # 只跑前 N 条（快速验证）
    uv run python scripts/run_process_eval.py --limit 2

与 scripts/run_eval.py 的区别：
    run_eval.py 只对最终报告打 relevance/completeness/accuracy 三维分；
    本脚本额外对 planner 拆解、搜索证据、编辑简报三个中间产物打分，
    并把所有分数按「阶段_维度」写回同一条 Langfuse trace。
"""

import argparse
import logging
import time

from research_buddy.eval.dataset import get_dataset
from research_buddy.eval.judge import judge_full_process, score_process_trace
from research_buddy.graph import run_research
from research_buddy.knowledge.db import get_db

logger = logging.getLogger(__name__)

# 各阶段展示名（用于 CLI 输出）
_STAGE_LABELS = {
    "plan": "拆解",
    "evidence": "证据",
    "editorial_brief": "编辑简报",
    "report": "报告",
}


def _stage_summary(stage: str, scores: dict) -> str:
    if not scores.get("parse_failed", False):
        dims = {k: v for k, v in scores.items() if k not in ("reasoning", "parse_failed")}
        total = sum(dims.values())
        detail = " ".join(f"{k}={v}" for k, v in dims.items())
        return f"{detail} 合计={total}"
    return f"解析失败（{scores.get('reasoning', '')}）"


def run_process_eval(limit: int | None = None) -> None:
    print("🚀 开始 Research Buddy 全流程评估")
    print("=" * 60)

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

        # 运行完整研究（run_research 会开 Langfuse 根 span 并回传 trace_id）
        result = run_research(question)
        trace_id = result.get("langfuse_trace_id", "")
        elapsed = time.time() - start
        print(f"   ⏱️  耗时: {elapsed:.1f}s")

        # 全流程打分
        process_scores = judge_full_process(result, expected)
        for stage, scores in process_scores.items():
            label = _STAGE_LABELS.get(stage, stage)
            print(f"   📊 {label}: {_stage_summary(stage, scores)}")

        # 写回 Langfuse trace
        if trace_id:
            try:
                score_process_trace(trace_id, process_scores)
                print(f"   ✅ 分阶段评分已写入 Langfuse trace {trace_id}")
            except Exception as exc:
                print(f"   ⚠️  Langfuse 评分写入失败: {exc}")
        else:
            print("   ⚠️  未配置 LANGFUSE_* 密钥，跳过评分写入")

        # 落库到本地素材库，前端「流程打分」面板可直接查看
        article_id = result.get("article_id", "")
        if article_id:
            try:
                get_db().update_article_generation(
                    article_id, process_scores=process_scores,
                )
                print(f"   ✅ 评分已落库到素材库 {article_id}")
            except Exception as exc:
                print(f"   ⚠️  评分落库失败: {exc}")

        results.append({
            "question": question,
            "process_scores": process_scores,
            "elapsed": elapsed,
        })
        print()

    # 汇总统计（只汇总解析成功的阶段）
    print("=" * 60)
    print("📊 流程评估汇总")
    print("=" * 60)
    if results:
        print(f"测试用例数: {len(results)}")
        for stage in ("plan", "evidence", "editorial_brief", "report"):
            scored = [r["process_scores"][stage] for r in results
                      if stage in r["process_scores"]
                      and not r["process_scores"][stage].get("parse_failed", False)]
            if not scored:
                print(f"  {_STAGE_LABELS[stage]}: 无有效评分（{len(results) - len(scored)} 条解析失败）")
                continue
            dims = {k for s in scored for k in s if k not in ("reasoning", "parse_failed")}
            for dim in sorted(dims):
                avg = sum(s.get(dim, 0) for s in scored) / len(scored)
                print(f"  {_STAGE_LABELS[stage]}.{dim}: 平均 {avg:.1f}/5")
        print(f"总耗时: {sum(r['elapsed'] for r in results):.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Research Buddy 全流程评估")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条测试用例")
    args = parser.parse_args()
    run_process_eval(limit=args.limit)


if __name__ == "__main__":
    main()
