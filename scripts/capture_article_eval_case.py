"""把真实研究结果的搜索证据固化为文章回归用例。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_buddy.eval.article_quality import append_case, case_from_research_result


def main() -> None:
    parser = argparse.ArgumentParser(description="捕获真实搜索证据，追加到 Article Eval 用例集")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result", help="包含完整 ResearchState 的 JSON 文件")
    source.add_argument("--question", help="主动运行一次完整研究并捕获结果（会调用 LLM/Tavily）")
    parser.add_argument("--id", required=True, help="稳定且唯一的用例 ID")
    parser.add_argument("--category", default="general")
    parser.add_argument("--style", default="tech-blog")
    parser.add_argument("--expected", action="append", default=[], help="预期要点，可重复")
    parser.add_argument("--risk-tag", action="append", default=[], help="已知失败模式，可重复")
    parser.add_argument("--output", default="eval/cases/regression.json")
    args = parser.parse_args()

    if args.result:
        result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    else:
        from research_buddy.graph import run_research
        result = run_research(args.question, style=args.style)
    case = case_from_research_result(
        args.id, result, category=args.category, style=args.style,
        expected_points=args.expected, risk_tags=args.risk_tag,
    )
    append_case(args.output, case)
    print(f"已追加 {args.id}: {args.output}（冻结证据 {len(case['evidence'])} 条）")


if __name__ == "__main__":
    main()
