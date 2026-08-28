"""对明确的模板化表达做受约束局部改写，不改变文章事实载荷。"""

import json
import logging
import re
from collections import Counter

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from research_buddy.config import ENABLE_LANGUAGE_EDITOR
from research_buddy.state import ResearchState
from research_buddy.utils import (
    create_llm,
    get_prompt_from_langfuse,
    invoke_llm,
    parse_llm_json,
)

logger = logging.getLogger(__name__)


LANGUAGE_EDITOR_PROMPT = """你是克制的中文文字编辑。只处理代码已经标出的模板化表达，让句子更直接、自然、具体；保持原有事实、判断强度和作者立场，不润色全文。

## 原始问题
{question}

## 编辑简报
{brief}

## 候选句
{candidates}

## 文章
{report}

只返回 JSON：
{{"edits": [
  {{
    "quote": "从候选句中逐字复制的完整原句",
    "replacement": "信息不变的直接表达",
    "issue_type": "contrast_template|empty_transition|reader_directive|meta_summary|repetitive_opening",
    "reason": "具体说明删掉了什么模板成分"
  }}
]}}

规则：
- quote 必须与某个候选句完全一致，不得自行选择文章其他内容；
- replacement 必须保留 quote 中的数字、引用编号、URL、英文专名、事实、限定词和判断强度；
- 不得新增事实、例子、因果关系、建议、评价、比喻、人物口吻或读者称呼；
- 不得用另一种「不是……而是……」「这意味着」「值得注意的是」「从本质上说」「综上所述」替换原模板；
- 对 contrast_template，如果否定分句只是为核心句搭舞台，删除铺垫并直接陈述核心句；只有否定范围本身是问题答案时才跳过；
- 不修改标题、列表结构、表格、代码、引用编号和参考文献；
- 不合并不相邻的句子，不拆分段落，不返回空 replacement；
- 某句无法在信息不变的前提下改好就跳过；最多 8 处；
- 没有可靠修改时返回 {{"edits": []}}。"""


_SENTENCE_RE = re.compile(r"[^\n。！？!?]{6,}[。！？!?]")
_STYLE_PATTERNS = (
    ("contrast_template", re.compile(r"(?:并|绝)?不(?:是|在于).{0,90}(?:，|,)?而(?:是|在于)")),
    ("empty_transition", re.compile(
        r"(?:值得注意的是|需要指出的是|不可忽视的是|从本质上(?:说|讲|来看)|换句话说|"
        r"归根结底|毋庸置疑)"
    )),
    ("reader_directive", re.compile(r"(?:我们需要认识到|我们必须看到|让我们(?:来看|回到|思考))")),
    ("meta_summary", re.compile(r"(?:综上所述|总而言之|由此可见|这(?:也)?意味着)")),
)
_REFERENCE_HEADING_RE = re.compile(r"(?:^|\n)##\s*参考文献(?:\s|$)")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")
_CITATION_RE = re.compile(r"(?:\[\d+(?:\s*[-,，]\s*\d+)*\]|【\d+】)")
_URL_RE = re.compile(r"https?://[^\s)\]}>]+")
_ENGLISH_TERM_RE = re.compile(r"\b[A-Z][A-Za-z0-9_.+#/-]*\b")
_MARKDOWN_LINE_RE = re.compile(r"\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\|)")


def scan_language_issues(report: str, max_candidates: int = 12) -> list[dict]:
    """确定性扫描高置信模板句；跳过代码、Markdown 结构和参考文献。"""
    reference_match = _REFERENCE_HEADING_RE.search(report)
    body = report[:reference_match.start()] if reference_match else report
    candidates: list[dict] = []
    prose_lines: list[str] = []
    in_code = False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or _MARKDOWN_LINE_RE.match(line) or line.count("|") >= 2:
            continue
        prose_lines.append(line)
        for match in _SENTENCE_RE.finditer(line):
            quote = match.group(0).strip()
            for issue_type, pattern in _STYLE_PATTERNS:
                cue = pattern.search(quote)
                if cue:
                    candidates.append({
                        "quote": quote,
                        "issue_type": issue_type,
                        "cue": cue.group(0),
                    })
                    break
            if len(candidates) >= max_candidates:
                return candidates

    # 三次以上使用同一个显式段首连接词时，只改第二次以后的句子。
    opener_groups: dict[str, list[str]] = {}
    for line in prose_lines:
        match = re.match(r"\s*((?:首先|其次|最后|同时|此外|因此|具体来说)[，,])([^\n]+[。！？!?])", line)
        if match:
            opener_groups.setdefault(match.group(1), []).append(match.group(0).strip())
    for opener, quotes in opener_groups.items():
        if len(quotes) < 3:
            continue
        for quote in quotes[1:]:
            if quote not in {item["quote"] for item in candidates}:
                candidates.append({
                    "quote": quote,
                    "issue_type": "repetitive_opening",
                    "cue": opener,
                })
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _preserved_tokens(text: str) -> dict[str, Counter]:
    return {
        "numbers": Counter(_NUMBER_RE.findall(text)),
        "citations": Counter(_CITATION_RE.findall(text)),
        "urls": Counter(_URL_RE.findall(text)),
        "terms": Counter(_ENGLISH_TERM_RE.findall(text)),
    }


def validate_language_edits(payload, report: str, candidates: list[dict]) -> list[dict]:
    """验证模型只改候选句，并保持可确定检查的事实载荷。"""
    if not isinstance(payload, dict) or not isinstance(payload.get("edits"), list):
        raise ValueError("语言审校输出缺少 edits 列表")
    candidate_by_quote = {item["quote"]: item for item in candidates}
    valid = []
    used_quotes: set[str] = set()
    for item in payload["edits"][:8]:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote", "")).strip()
        replacement = str(item.get("replacement", "")).strip()
        reason = str(item.get("reason", "")).strip()
        issue_type = str(item.get("issue_type", "")).strip()
        candidate = candidate_by_quote.get(quote)
        if (not candidate or quote in used_quotes or quote not in report
                or issue_type != candidate["issue_type"] or not reason):
            continue
        if (not 6 <= len(replacement) <= max(24, int(len(quote) * 1.35))
                or replacement == quote or "\n" in replacement):
            continue
        if _preserved_tokens(quote) != _preserved_tokens(replacement):
            continue
        if any(pattern.search(replacement) for _kind, pattern in _STYLE_PATTERNS):
            continue
        if re.search(r"```|^\s*(?:#|[-*+]\s|\d+[.)]\s|>|\|)", replacement):
            continue
        used_quotes.add(quote)
        valid.append({
            "quote": quote,
            "replacement": replacement,
            "issue_type": issue_type,
            "reason": reason,
        })
    return valid


def apply_language_edits(report: str, edits: list[dict]) -> tuple[str, list[dict]]:
    """应用仍然逐字存在的候选句，并返回真实落地的审计记录。"""
    applied = []
    for edit in edits:
        quote = edit["quote"]
        if quote in report:
            report = report.replace(quote, edit["replacement"], 1)
            applied.append(edit)
    return report, applied


def edit_article_language(state: ResearchState, report: str,
                          config: RunnableConfig | None = None) -> tuple[str, list[dict]]:
    """扫描并局部处理模板化表达；没有候选时不调用模型。"""
    candidates = scan_language_issues(report[:28000])
    if not report or not candidates:
        return report, []
    prompt_kwargs = {
        "question": state.get("question", ""),
        "brief": json.dumps(state.get("editorial_brief", {}), ensure_ascii=False, indent=2),
        "candidates": json.dumps(candidates, ensure_ascii=False, indent=2),
        "report": report[:28000],
    }
    try:
        prompt = (
            LANGUAGE_EDITOR_PROMPT.format(**prompt_kwargs)
            if state.get("eval_use_local_prompts") else
            get_prompt_from_langfuse(
                "research-buddy-language-editor", LANGUAGE_EDITOR_PROMPT, **prompt_kwargs,
            )
        )
        response = invoke_llm(create_llm(), prompt, config=config)
        edits = validate_language_edits(parse_llm_json(response.content), report, candidates)
        return apply_language_edits(report, edits)
    except Exception as exc:
        logger.warning("语言审校失败，保留原稿: %s", exc)
        return report, []


def language_editor(state: ResearchState, config: RunnableConfig | None = None,
                    *, writer: StreamWriter) -> dict:
    """生产语言审校节点；有可靠修改时用重置事件替换前端初稿。"""
    report = state.get("report", "")
    candidate_count = len(scan_language_issues(report[:28000]))
    if not ENABLE_LANGUAGE_EDITOR:
        return {
            "language_edits": [],
            "language_editor_changed": False,
            "language_candidates_count": 0,
            "messages": ["语言审校已关闭，保留当前文章"],
        }
    revised, edits = edit_article_language(state, report, config=config)
    changed = revised != report
    if changed and writer:
        writer({"type": "report_reset"})
        for offset in range(0, len(revised), 2000):
            writer({"type": "report_chunk", "content": revised[offset:offset + 2000]})
    return {
        "report": revised,
        "language_edits": edits,
        "language_editor_changed": changed,
        "language_candidates_count": candidate_count,
        "messages": [
            f"语言审校完成，改写 {len(edits)} 处模板化表达"
            if edits else
            f"语言审校发现 {candidate_count} 处候选，但没有可安全应用的修改"
            if candidate_count else
            "语言审校完成，未发现需要处理的模板化表达"
        ],
    }
