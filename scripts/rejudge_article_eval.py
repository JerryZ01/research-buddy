"""对已有文章评测产物重新运行质量 Judge，不重复生成文章。"""

from __future__ import annotations

import argparse
import json
import sys

from research_buddy.eval.article_quality import (
    load_cases,
    rejudge_evaluation,
    write_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="复用已有文章重跑质量评分与 A/B 盲评")
    parser.add_argument("--input", required=True, help="已有 results.json")
    parser.add_argument("--cases", default="eval/cases/starter.json", help="冻结证据用例")
    parser.add_argument("--output", required=True, help="新产物目录，不覆盖原结果")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as source:
        result = json.load(source)
    revised = rejudge_evaluation(
        result, load_cases(args.cases), progress=lambda message: print(message, flush=True),
    )
    json_path, html_path = write_artifacts(revised, args.output)
    gate = revised["summary"]["gate"]
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    print("回归门禁: " + ("通过" if gate["passed"] else "未通过"))
    for reason in gate["reasons"]:
        print(f"- {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
