"""运行可见的文章质量 A/B 回归评测。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from research_buddy.eval.article_quality import (
    default_baseline_rules,
    load_cases,
    run_evaluation,
    write_artifacts,
)


def _read_optional(path: str | None, fallback: str) -> str:
    return Path(path).read_text(encoding="utf-8") if path else fallback


def main() -> int:
    parser = argparse.ArgumentParser(
        description="用冻结证据比较 baseline/candidate 写作规则并生成 HTML 报告",
    )
    parser.add_argument("--cases", default="eval/cases/starter.json",
                        help="冻结证据用例 JSON")
    parser.add_argument("--candidate-rules", required=True,
                        help="候选 WRITING_RULES 文本文件")
    parser.add_argument("--baseline-rules",
                        help="基线规则文件；默认使用当前代码中的 WRITING_RULES")
    parser.add_argument("--baseline-label", default="current-production")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-editorial-brief", action="store_true",
                        help="baseline 写作前生成结构化编辑简报")
    parser.add_argument("--candidate-editorial-brief", action="store_true",
                        help="candidate 写作前生成结构化编辑简报")
    parser.add_argument("--baseline-evidence-editor", action="store_true",
                        help="baseline 写作后运行证据定向编辑")
    parser.add_argument("--candidate-evidence-editor", action="store_true",
                        help="candidate 写作后运行证据定向编辑")
    parser.add_argument("--samples", type=int, default=3,
                        help="每个版本每题采样次数，建议至少 3")
    parser.add_argument("--limit", type=int, help="只跑前 N 题，用于快速检查")
    parser.add_argument("--case-id", action="append", default=[],
                        help="只跑指定用例 ID，可重复")
    parser.add_argument("--skip-judge", action="store_true",
                        help="只生成文章和确定性指标，不运行质量/A-B Judge")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="回归门禁失败时退出码为 2，供 CI 使用")
    parser.add_argument("--output", help="输出目录；默认 eval/results/<时间戳>")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            parser.error("未知 case-id: " + ", ".join(sorted(missing)))
    if args.limit:
        cases = cases[:args.limit]
    baseline_rules = _read_optional(args.baseline_rules, default_baseline_rules())
    candidate_rules = Path(args.candidate_rules).read_text(encoding="utf-8")
    output = args.output or f"eval/results/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"用例: {len(cases)}，每版本采样: {args.samples}")
    print(f"Baseline: {args.baseline_label}，Candidate: {args.candidate_label}")
    if args.skip_judge:
        print("Judge: 已跳过（只生成文章与确定性指标）")

    result = run_evaluation(
        cases,
        baseline_rules=baseline_rules,
        candidate_rules=candidate_rules,
        samples=args.samples,
        run_judges=not args.skip_judge,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        baseline_editorial_brief=args.baseline_editorial_brief,
        candidate_editorial_brief=args.candidate_editorial_brief,
        baseline_evidence_editor=args.baseline_evidence_editor,
        candidate_evidence_editor=args.candidate_evidence_editor,
        progress=lambda message: print(message, flush=True),
    )
    json_path, html_path = write_artifacts(result, output)
    summary = result["summary"]
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    print(f"Baseline Judge: {summary['baseline']['judge_average']}")
    print(f"Candidate Judge: {summary['candidate']['judge_average']}")
    print(f"A/B: {summary['pairwise']['wins']}")
    gate = summary["gate"]
    if not gate.get("conclusive", True):
        status = "确定性检查通过，质量结论不完整" if gate["passed"] else "确定性检查未通过，质量结论不完整"
    else:
        status = "通过" if gate["passed"] else "未通过"
    print("回归门禁: " + status)
    for reason in gate["reasons"]:
        print(f"- {reason}")
    return 2 if args.fail_on_regression and not gate["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
