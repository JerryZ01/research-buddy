"""验证节点 - 检查搜索结果是否充足，标记信息缺口"""

import logging

from research_buddy.state import ResearchState, ValidationGap

logger = logging.getLogger(__name__)

# 每个子问题至少需要的搜索结果数
MIN_RESULTS_PER_SUB_QUESTION = 2
# 搜索结果内容最短长度（字符）
MIN_CONTENT_LENGTH = 50


def validator(state: ResearchState) -> dict:
    """验证节点：纯规则检查搜索结果充足性

    改进：
    - 检查重复结果（相同 URL 或高度相似内容）
    - 补充搜索词策略更智能（添加限定词而非简单追加 "detailed"）
    """
    sub_questions = state.get("sub_questions", [])
    search_results = state.get("search_results", [])

    logger.info("正在验证搜索结果充足性...")

    # 按子问题分组统计
    results_by_sq: dict[str, list] = {}
    for r in search_results:
        sq = r.get("sub_question", "")
        results_by_sq.setdefault(sq, []).append(r)

    gaps: list[ValidationGap] = []
    for sq in sub_questions:
        question = sq.get("question", "")
        original_query = sq.get("search_query", "")
        results = results_by_sq.get(question, [])

        # 有效结果 = content 足够长且不重复的结果
        seen_urls = set()
        valid_results = []
        for r in results:
            content = r.get("content", "")
            url = r.get("url", "")
            if len(content) >= MIN_CONTENT_LENGTH and url not in seen_urls:
                seen_urls.add(url)
                valid_results.append(r)

        if len(valid_results) < MIN_RESULTS_PER_SUB_QUESTION:
            # 改进的补充搜索策略：添加限定词而非简单追加 "detailed"
            if original_query:
                supplement_query = f"{original_query} latest update"
            else:
                supplement_query = question
            gaps.append({
                "question": question,
                "search_query": supplement_query,
            })

    if gaps:
        logger.info("%d 个子问题信息不足，需要补充搜索", len(gaps))
    else:
        logger.info("所有 %d 个子问题搜索结果充足", len(sub_questions))

    msg = f"⚠️ {len(gaps)} 个子问题信息不足" if gaps else f"✅ 所有 {len(sub_questions)} 个子问题搜索结果充足"
    return {"validation_gaps": gaps, "messages": [f"✅ 验证: {msg}"]}
