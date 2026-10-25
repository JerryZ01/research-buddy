"""Research Buddy 评估模块"""

from research_buddy.eval.dataset import create_dataset, get_dataset
from research_buddy.eval.judge import (
    auto_judge_and_score,
    judge_editorial_brief,
    judge_evidence,
    judge_full_process,
    judge_plan,
    judge_report,
    score_process_trace,
    score_trace,
)
from research_buddy.eval.prompts import get_prompt, register_prompts

__all__ = [
    "create_dataset",
    "get_dataset",
    "judge_report",
    "judge_plan",
    "judge_evidence",
    "judge_editorial_brief",
    "judge_full_process",
    "auto_judge_and_score",
    "score_trace",
    "score_process_trace",
    "get_prompt",
    "register_prompts",
]
