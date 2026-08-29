"""写作前编辑简报：先固定文章判断、范围、章节职责与证据映射。"""

import json
import logging

from langchain_core.runnables import RunnableConfig

from research_buddy.state import ResearchState
from research_buddy.utils import (
    create_llm,
    get_prompt_from_langfuse,
    invoke_llm,
    normalize_url,
    parse_llm_json,
)

logger = logging.getLogger(__name__)


EDITORIAL_BRIEF_PROMPT = """你是资深内容编辑。写作者稍后会基于同一批证据写文章；你只负责先制定编辑简报，不写正文。

## 研究问题
{question}

## 写作风格
{style}

## 可用证据
{evidence}

## 任务
先判断问题真正要求的文章类型和边界，再设计一条能自然推进的主线。所有具体事实都必须能追溯到 E1、E2 等证据编号。证据没提供的数字、案例、库名、实现细节和历史背景不得写进简报；必要时放进 claims_to_avoid。

证据编号不是支持关系本身。尤其禁止仅凭「存在控制平面」「团队有三个人」「有两台服务器」就推导资源消耗、维护工时、可靠性、故障后果、学习成本或业务效率。若做决策所需的信息未在证据中出现，把它写入 evidence_gaps，并把结论改成条件式判断，不得用常识补齐。

只返回 JSON 对象：
{{
  "intent": "mechanism|comparison|decision|tutorial|trend|explanation",
  "audience": "具体读者及其已有认知",
  "thesis": "全文唯一核心判断，一句话",
  "scope_include": ["必须回答的范围"],
  "scope_exclude": ["容易跑题但不应展开的内容"],
  "must_cover": [
    {{"point": "必须讲清的要点", "evidence_ids": ["E1"]}}
  ],
  "section_plan": [
    {{"heading": "直接说明讨论对象或判断的标题", "purpose": "本节如何推进主线", "evidence_ids": ["E1", "E2"]}}
  ],
  "claims_to_avoid": ["证据不足、容易被模型凭常识补出的具体断言"],
  "evidence_gaps": ["回答问题仍缺少的事实或用户条件"],
  "ending": "结尾完成哪一步推论，不写仪式化总结"
}}

约束：通常规划 2-4 个核心章节，只有复杂长文才允许 5 个；每节必须承担不同的推进职责，能合并就不要拆节。标题使用直接、克制的陈述或名词短语，禁止比喻、拟人、口号、悬念和为了风格变化而写的反问句。结尾不是独立章节；同一证据可以复用；没有证据支持的章节不要创建；不要输出 Markdown 或解释。"""

_INTENTS = {"mechanism", "comparison", "decision", "tutorial", "trend", "explanation"}


def build_evidence_ledger(state: ResearchState, max_items: int = 30) -> list[dict]:
    """将累计搜索结果收敛为可稳定引用的证据账本。

    账本只做确定性整理，不用 LLM 概括原文，避免在写作前先引入一层
    「摘要幻觉」。同一 URL 多次命中时保留更长的证据片段，并合并关联子问题。
    """
    assessments = {
        str(item.get("sub_question_id", "")): item
        for item in state.get("evidence_assessments", [])
    }
    entries: list[dict] = []
    by_key: dict[str, dict] = {}
    for result in state.get("search_results", []):
        title = str(result.get("title", "")).strip() or "未命名来源"
        url = str(result.get("url", "")).strip()
        content = str(result.get("content", "")).strip()
        if not content:
            continue
        key = normalize_url(url) or f"text:{title}\n{content[:240]}"
        sub_question_id = str(result.get("sub_question_id", "")).strip()
        assessment = assessments.get(sub_question_id, {})
        if key in by_key:
            entry = by_key[key]
            if len(content) > len(entry["excerpt"]):
                entry["excerpt"] = content[:4000]
            if sub_question_id and sub_question_id not in entry["sub_question_ids"]:
                entry["sub_question_ids"].append(sub_question_id)
            entry["score"] = max(entry["score"], float(result.get("score", 0) or 0))
            for contradiction in assessment.get("contradictions", []):
                if contradiction not in entry["contradictions"]:
                    entry["contradictions"].append(contradiction)
            if assessment.get("status") == "insufficient":
                entry["assessment_status"] = "insufficient"
            continue

        entry = {
            "id": "",  # 去重完成后统一编号
            "title": title,
            "url": url,
            "excerpt": content[:4000],
            "score": float(result.get("score", 0) or 0),
            "sub_question_ids": [sub_question_id] if sub_question_id else [],
            "assessment_status": str(assessment.get("status", "unassessed")),
            "contradictions": [
                str(item).strip() for item in assessment.get("contradictions", [])
                if str(item).strip()
            ],
        }
        entries.append(entry)
        by_key[key] = entry
        if len(entries) >= max_items:
            break

    for index, entry in enumerate(entries, 1):
        entry["id"] = f"E{index}"
    return entries


def _evidence_text(ledger: list[dict], max_chars: int = 18000) -> str:
    blocks = []
    for item in ledger:
        meta = [f"验证状态: {item.get('assessment_status', 'unassessed')}"]
        if item.get("contradictions"):
            meta.append("已知冲突: " + "; ".join(item["contradictions"]))
        blocks.append(
            f"{item['id']} | {item.get('title', '未命名来源')}\n"
            f"{' | '.join(meta)}\n{item.get('excerpt', '')}"
        )
    return "\n\n".join(blocks)[:max_chars]


def _strings(value, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def _evidence_ids(value, valid_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        evidence_id = str(item).strip().upper()
        if evidence_id in valid_ids and evidence_id not in result:
            result.append(evidence_id)
    return result


def normalize_editorial_brief(payload, evidence_count: int) -> dict:
    """校验简报形状并删除幻觉证据编号。"""
    if not isinstance(payload, dict):
        raise ValueError("编辑简报不是 JSON 对象")
    valid_ids = {f"E{index}" for index in range(1, evidence_count + 1)}
    thesis = str(payload.get("thesis", "")).strip()
    if not thesis:
        raise ValueError("编辑简报缺少 thesis")

    must_cover = []
    for item in payload.get("must_cover", []) if isinstance(payload.get("must_cover"), list) else []:
        if not isinstance(item, dict) or not str(item.get("point", "")).strip():
            continue
        ids = _evidence_ids(item.get("evidence_ids"), valid_ids)
        if ids:
            must_cover.append({"point": str(item["point"]).strip(), "evidence_ids": ids})

    sections = []
    for item in payload.get("section_plan", []) if isinstance(payload.get("section_plan"), list) else []:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading", "")).strip()
        purpose = str(item.get("purpose", "")).strip()
        ids = _evidence_ids(item.get("evidence_ids"), valid_ids)
        if heading and purpose and ids:
            sections.append({"heading": heading, "purpose": purpose, "evidence_ids": ids})
    if not must_cover or not 2 <= len(sections) <= 5:
        raise ValueError("编辑简报缺少有效 must_cover 或 section_plan")

    intent = str(payload.get("intent", "explanation")).strip().lower()
    return {
        "intent": intent if intent in _INTENTS else "explanation",
        "audience": str(payload.get("audience", "普通读者")).strip() or "普通读者",
        "thesis": thesis,
        "scope_include": _strings(payload.get("scope_include")),
        "scope_exclude": _strings(payload.get("scope_exclude")),
        "must_cover": must_cover[:8],
        "section_plan": sections[:5],
        "claims_to_avoid": _strings(payload.get("claims_to_avoid"), limit=10),
        "evidence_gaps": _strings(payload.get("evidence_gaps"), limit=8),
        "ending": str(payload.get("ending", "")).strip(),
    }


def build_editorial_brief(state: ResearchState, config: RunnableConfig | None = None,
                           use_local_prompt: bool = False,
                           evidence_ledger: list[dict] | None = None) -> dict:
    """调用编辑模型生成结构化简报；失败返回空对象，由写作流程自然降级。"""
    ledger = evidence_ledger if evidence_ledger is not None else build_evidence_ledger(state)
    evidence_count = len(ledger)
    if evidence_count == 0:
        return {}
    kwargs = {
        "question": state.get("question", ""),
        "style": state.get("style", "tech-blog"),
        "evidence": _evidence_text(ledger),
    }
    try:
        prompt = (EDITORIAL_BRIEF_PROMPT.format(**kwargs) if use_local_prompt else
                  get_prompt_from_langfuse(
                      "research-buddy-editorial-brief", EDITORIAL_BRIEF_PROMPT, **kwargs,
                  ))
        response = invoke_llm(create_llm(), prompt, config=config)
        return normalize_editorial_brief(parse_llm_json(response.content), evidence_count)
    except Exception as exc:
        logger.warning("编辑简报生成失败，降级为直接写作: %s", exc)
        return {}


def format_editorial_brief(brief: dict) -> str:
    """用稳定 JSON 注入写作上下文，避免二次解释改变语义。"""
    return json.dumps(brief, ensure_ascii=False, indent=2)


def editorial_planner(state: ResearchState, config: RunnableConfig | None = None) -> dict:
    ledger = build_evidence_ledger(state)
    brief = build_editorial_brief(state, config=config, evidence_ledger=ledger)
    return {
        "evidence_ledger": ledger,
        "editorial_brief": brief,
        "messages": [
            f"证据账本已收敛为 {len(ledger)} 条，并生成写作编辑简报"
            if brief else
            f"证据账本已收敛为 {len(ledger)} 条，编辑简报不可用，降级为直接写作"
        ],
    }
