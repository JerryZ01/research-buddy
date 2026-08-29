"""基于冻结证据对初稿做可验证的局部编辑，不整篇重写。"""

import json
import logging
import re
from difflib import SequenceMatcher

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from research_buddy.config import ARTICLE_EDITOR_ROUNDS, ENABLE_ARTICLE_EDITOR
from research_buddy.state import ResearchState
from research_buddy.utils import (
    create_llm,
    get_prompt_from_langfuse,
    invoke_llm,
    parse_llm_json,
)

logger = logging.getLogger(__name__)


EVIDENCE_EDITOR_PROMPT = """你是事实核查兼删冗编辑。请逐句对照给定证据审查初稿，只做可核验的局部修改，不整篇重写，不改变文章主线。

## 问题
{question}

## 可用证据
{evidence}

## 编辑简报
{brief}

## 初稿
{report}

先扫描文章中的每个可验证断言，重点检查：
- 证据没有出现的数字、版本、库名、案例、历史背景、工具行为或实现建议；
- 从团队人数、服务器数量、节点数量等输入直接推出成本、可靠性、故障模式或最佳实践；
- 把「提供某项能力」扩展成「在某个规模才有价值」，或把一般性描述扩展成具体结论；
- 把经验判断、因果关系或可能性写成确定事实；
- 违反简报 scope_exclude 或 claims_to_avoid 的内容；
- 相邻或跨段重复同一个结论、只是替换措辞的句子。
- 假想争论（「有人说……另一边说……」）、连续反问、自问自答、刻意口语收束等不提供信息的修辞；这类内容只能整句删除。
- 正文泄露 E1、E2 等内部证据编号，或出现「E2 说明」之类研究过程表述；改成直接陈述证据支持的事实。

只返回 JSON：
{{"edits": [
  {{
    "quote": "从初稿逐字复制、需要替换的连续原文",
    "replacement": "忠于证据的替换文本",
    "edit_type": "unsupported|overstated|redundant",
    "reason": "为什么原文越界",
    "evidence_ids": ["E1"],
    "support_quotes": [{{"evidence_id": "E1", "quote": "从该证据逐字复制的连续原文"}}]
  }}
]}}

规则：
- quote 必须是初稿中逐字存在的 10-300 字连续片段，不得拼接；
- replacement 非空时，只能使用 support_quotes 直接支持的事实；support_quotes 必须逐字存在于对应证据；
- 完全没有证据且删除后不影响句意衔接时，replacement 可为空，evidence_ids 和 support_quotes 也应为空；
- replacement 放回原段后不能与前后句重复，也不能用「某个机制」「相关原因」等模糊词掩盖证据不足；
- 重复内容优先删除后出现的句子，不要把两句改成另一组重复句；
- 没有问题就返回 {{"edits": []}}；最多 10 处；
- 不编辑标题、参考文献、Markdown 结构或纯粹的文风偏好。"""


GROUNDING_VERIFIER_PROMPT = """你是独立证据复核员。逐项判断候选 replacement 的全部事实是否被给定证据直接支持。只要 replacement 比证据更具体、加入因果关系、成本判断、故障后果或实施建议，就判为 false。不要因原文更差而放宽标准。

## 可用证据
{evidence}

## 待复核编辑
{edits}

只返回 JSON：
{{"verdicts": [{{"index": 0, "supported": true, "reason": "判断依据"}}]}}
每个输入 index 必须恰好出现一次。"""


def _evidence_items(state: ResearchState) -> list[dict]:
    """返回与编辑简报共用 E1/E2 编号的证据列表。"""
    ledger = state.get("evidence_ledger", [])
    if ledger:
        return [
            {
                "title": item.get("title", "未命名来源"),
                "content": item.get("excerpt", ""),
            }
            for item in ledger
        ]
    return state.get("search_results", [])


def _evidence_text(state: ResearchState, max_chars: int = 16000) -> str:
    return "\n\n".join(
        f"E{index} | {item.get('title', '未命名来源')}\n{item.get('content', '')}"
        for index, item in enumerate(_evidence_items(state), 1)
    )[:max_chars]


def validate_edits(payload, report: str, evidence: list[dict]) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("edits"), list):
        raise ValueError("证据编辑器输出缺少 edits 列表")
    evidence_by_id = {
        f"E{index}": str(item.get("content", ""))
        for index, item in enumerate(evidence, 1)
    }
    valid_ids = set(evidence_by_id)
    edits = []
    used_quotes: set[str] = set()
    for item in payload["edits"][:10]:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote", "")).strip()
        replacement = str(item.get("replacement", "")).strip()
        reason = str(item.get("reason", "")).strip()
        edit_type = str(item.get("edit_type", "unsupported")).strip().lower()
        ids = []
        for raw in item.get("evidence_ids", []) if isinstance(item.get("evidence_ids"), list) else []:
            evidence_id = str(raw).strip().upper()
            if evidence_id in valid_ids and evidence_id not in ids:
                ids.append(evidence_id)
        if (not 10 <= len(quote) <= 300 or quote not in report or quote in used_quotes
                or "\n#" in quote or "```" in quote or "http" in quote or "![" in quote):
            continue
        if not reason or edit_type not in {"unsupported", "overstated", "redundant"}:
            continue
        if re.search(r"https?://", replacement) or re.search(r"某个机制|相关原因|某种方式|某些方面", replacement):
            continue

        support_quotes = []
        raw_support = item.get("support_quotes", [])
        for support in raw_support if isinstance(raw_support, list) else []:
            if not isinstance(support, dict):
                continue
            evidence_id = str(support.get("evidence_id", "")).strip().upper()
            support_quote = str(support.get("quote", "")).strip()
            if (evidence_id in valid_ids and 6 <= len(support_quote) <= 240
                    and support_quote in evidence_by_id[evidence_id]):
                support_quotes.append({"evidence_id": evidence_id, "quote": support_quote})
        if replacement:
            if not support_quotes:
                continue
            ids = list(dict.fromkeys(item["evidence_id"] for item in support_quotes))
        elif edit_type not in {"unsupported", "redundant"}:
            continue
        else:
            ids = []
            support_quotes = []
        used_quotes.add(quote)
        edits.append({
            "quote": quote, "replacement": replacement,
            "edit_type": edit_type, "reason": reason,
            "evidence_ids": ids, "support_quotes": support_quotes,
        })
    return edits


def validate_verdicts(payload, edits: list[dict]) -> set[int]:
    """只接受形状完整且明确判定为 supported 的编辑编号。"""
    verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
    if not isinstance(verdicts, list):
        raise ValueError("证据复核缺少 verdicts 列表")
    expected = set(range(len(edits)))
    seen: set[int] = set()
    accepted: set[int] = set()
    for item in verdicts:
        if not isinstance(item, dict) or isinstance(item.get("index"), bool):
            raise ValueError("证据复核 verdict 条目无效")
        index = item.get("index")
        if not isinstance(index, int) or index not in expected or index in seen:
            raise ValueError("证据复核 index 无效或重复")
        if not str(item.get("reason", "")).strip():
            raise ValueError("证据复核缺少 reason")
        seen.add(index)
        if item.get("supported") is True:
            accepted.add(index)
    if seen != expected:
        raise ValueError("证据复核没有覆盖全部编辑")
    return accepted


def verify_replacements(edits: list[dict], state: ResearchState,
                        config: RunnableConfig | None = None) -> list[dict]:
    """用独立调用复核非空替换；纯删除不会引入新事实，可直接保留。"""
    replacements = [(index, edit) for index, edit in enumerate(edits) if edit["replacement"]]
    if not replacements:
        return edits
    verifier_edits = [
        {"index": local_index, "quote": edit["quote"], "replacement": edit["replacement"],
         "support_quotes": edit["support_quotes"]}
        for local_index, (_original_index, edit) in enumerate(replacements)
    ]
    prompt_kwargs = {
        "evidence": _evidence_text(state),
        "edits": json.dumps(verifier_edits, ensure_ascii=False, indent=2),
    }
    prompt = (
        GROUNDING_VERIFIER_PROMPT.format(**prompt_kwargs)
        if state.get("eval_use_local_prompts") else
        get_prompt_from_langfuse(
            "research-buddy-grounding-verifier", GROUNDING_VERIFIER_PROMPT, **prompt_kwargs,
        )
    )
    try:
        response = invoke_llm(create_llm(), prompt, config=config)
        local_accepted = validate_verdicts(
            parse_llm_json(response.content), [edit for _, edit in replacements],
        )
        accepted_original = {replacements[index][0] for index in local_accepted}
        return [
            edit for index, edit in enumerate(edits)
            if not edit["replacement"] or index in accepted_original
        ]
    except Exception as exc:
        logger.warning("替换文本证据复核失败，仅保留纯删除编辑: %s", exc)
        return [edit for edit in edits if not edit["replacement"]]


def apply_evidence_edits(report: str, edits: list[dict]) -> str:
    """只替换验证过且仍存在的第一个精确片段。"""
    revised, _applied = _apply_evidence_edits_with_audit(report, edits)
    return revised


def _apply_evidence_edits_with_audit(report: str, edits: list[dict]) -> tuple[str, list[dict]]:
    applied = []
    for edit in edits:
        quote = edit["quote"]
        if quote in report and _context_safe(report, edit):
            report = report.replace(quote, edit["replacement"], 1)
            applied.append(edit)
    if not applied:
        return report, []
    return _normalize_blank_lines(remove_exact_duplicate_sentences(report)), applied


def _normalize_blank_lines(report: str) -> str:
    parts = re.split(r"(```[\s\S]*?```)", report)
    return "".join(
        part if part.startswith("```") else re.sub(r"\n{3,}", "\n\n", part)
        for part in parts
    )


def _context_safe(report: str, edit: dict) -> bool:
    """拒绝会制造悬空标点或相邻近义重复的局部编辑。"""
    quote = edit["quote"]
    start = report.find(quote)
    if start < 0:
        return False
    replacement = edit["replacement"]
    before = report[max(0, start - 240):start]
    after = report[start + len(quote):start + len(quote) + 240]
    line_start = report.rfind("\n", 0, start) + 1
    line_end = report.find("\n", start + len(quote))
    if line_end < 0:
        line_end = len(report)
    line = report[line_start:line_end]
    if not replacement and re.match(r"\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>)", line):
        return False
    if not replacement and line.count("|") >= 2:
        return False
    if not replacement and re.search(r"[：:]\s*$", before):
        return False
    if not replacement and re.match(r"\s*(?:[，、；：]|但|而|因此|所以|同时)", after):
        return False
    replacement_sentences = [
        sentence.strip() for sentence in re.split(r"[。！？!?；;\n]+", replacement)
        if len(re.sub(r"\W", "", sentence)) >= 14
    ]
    neighbors = [
        sentence.strip() for sentence in re.split(r"[。！？!?；;\n]+", before + "\n" + after)
        if len(re.sub(r"\W", "", sentence)) >= 14
    ]
    for replacement_sentence in replacement_sentences:
        normalized = re.sub(r"\W", "", replacement_sentence)
        for neighbor in neighbors:
            neighbor_normalized = re.sub(r"\W", "", neighbor)
            if SequenceMatcher(None, normalized, neighbor_normalized).ratio() >= 0.68:
                return False
    return True


_SENTENCE_RE = re.compile(r"[^\n。！？]+[。！？]")


def remove_exact_duplicate_sentences(report: str) -> str:
    """删除正文中后出现的完全重复长句，不触碰标题、代码块或参考文献。"""
    body, separator, references = report.partition("\n## 参考文献")
    seen: set[str] = set()
    parts = []
    cursor = 0
    in_code = False
    for match in _SENTENCE_RE.finditer(body):
        prefix = body[cursor:match.start()]
        in_code ^= prefix.count("```") % 2 == 1
        sentence = match.group(0)
        normalized = re.sub(r"\s+", "", sentence)
        parts.append(prefix)
        if in_code or len(normalized) < 14 or normalized not in seen:
            parts.append(sentence)
            if not in_code and len(normalized) >= 14:
                seen.add(normalized)
        cursor = match.end()
    parts.append(body[cursor:])
    cleaned = "".join(parts)
    return cleaned + (separator + references if separator else "")


def edit_article_evidence(state: ResearchState, report: str,
                          config: RunnableConfig | None = None,
                          max_rounds: int | None = None) -> tuple[str, list[dict]]:
    evidence = _evidence_items(state)
    if not report or not evidence:
        return report, []
    rounds = ARTICLE_EDITOR_ROUNDS if max_rounds is None else max(0, max_rounds)
    use_local_prompts = bool(state.get("eval_use_local_prompts"))
    all_edits = []
    revised = report
    for _round in range(rounds):
        try:
            prompt_kwargs = {
                "question": state.get("question", ""),
                "evidence": _evidence_text(state),
                "brief": json.dumps(
                    state.get("editorial_brief", {}), ensure_ascii=False, indent=2,
                ),
                "report": revised[:28000],
            }
            round_prompt = (
                EVIDENCE_EDITOR_PROMPT.format(**prompt_kwargs)
                if use_local_prompts else
                get_prompt_from_langfuse(
                    "research-buddy-evidence-editor", EVIDENCE_EDITOR_PROMPT, **prompt_kwargs,
                )
            )
            response = invoke_llm(create_llm(), round_prompt, config=config)
            edits = validate_edits(
                parse_llm_json(response.content), revised, evidence,
            )
            if not edits:
                break
            edits = verify_replacements(edits, state, config=config)
            if not edits:
                continue
            revised, applied = _apply_evidence_edits_with_audit(revised, edits)
            all_edits.extend(applied)
        except Exception as exc:
            logger.warning("第 %d 轮证据定向编辑失败，保留已完成修改: %s", _round + 1, exc)
            break
    return revised, all_edits


def article_editor(state: ResearchState, config: RunnableConfig | None = None,
                   *, writer: StreamWriter) -> dict:
    """生产事实审校节点：只应用可验证局部修改，失败时保留原稿。"""
    report = state.get("report", "")
    if not ENABLE_ARTICLE_EDITOR or ARTICLE_EDITOR_ROUNDS == 0:
        return {
            "evidence_edits": [],
            "article_editor_changed": False,
            "article_versions": [{
                "stage": "article_editor",
                "reflection_round": int(state.get("reflection_round", 0)),
                "report": report,
                "metadata": {"changed": False, "disabled": True},
            }],
            "messages": ["事实审校已关闭，保留初稿"],
        }

    revised, edits = edit_article_evidence(state, report, config=config)
    changed = revised != report
    if changed and writer:
        writer({"type": "report_reset"})
        for offset in range(0, len(revised), 2000):
            writer({"type": "report_chunk", "content": revised[offset:offset + 2000]})

    return {
        "report": revised,
        "article_versions": [{
            "stage": "article_editor",
            "reflection_round": int(state.get("reflection_round", 0)),
            "report": revised,
            "metadata": {"changed": changed, "edits_count": len(edits)},
        }],
        "evidence_edits": edits,
        "article_editor_changed": changed,
        "messages": [
            f"事实审校完成，应用 {len(edits)} 处可验证修改"
            if edits else
            "事实审校完成，已清理重复表达"
            if changed else
            "事实审校完成，未发现需要修改的断言"
        ],
    }
