"""验证节点 - 检查搜索结果是否充足，标记信息缺口"""

from research_buddy.state import ResearchState, ValidationGap

# 每个子问题至少需要的搜索结果数
MIN_RESULTS_PER_SUB_QUESTION = 2
# 搜索结果内容最短长度（字符）
MIN_CONTENT_LENGTH = 50


def validator(state: ResearchState) -> dict:
    """验证节点：纯规则检查搜索结果充足性"""
    sub_questions = state.get("sub_questions", [])
    search_results = state.get("search_results", [])

    print("✅ 正在验证搜索结果充足性...")

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

        # 有效结果 = content 足够长的结果
        valid_results = [
            r for r in results if len(r.get("content", "")) >= MIN_CONTENT_LENGTH
        ]

        if len(valid_results) < MIN_RESULTS_PER_SUB_QUESTION:
            # 补充搜索词用英文，保持简短
            supplement_query = f"{original_query} detailed" if original_query else question
            gaps.append({
                "question": question,
                "search_query": supplement_query,
            })

    if gaps:
        print(f"   ⚠️  {len(gaps)} 个子问题信息不足，需要补充搜索")
    else:
        print(f"   ✅ 所有 {len(sub_questions)} 个子问题搜索结果充足")

    msg = f"⚠️ {len(gaps)} 个子问题信息不足" if gaps else f"✅ 所有 {len(sub_questions)} 个子问题搜索结果充足"
    return {"validation_gaps": gaps, "messages": [f"✅ 验证: {msg}"]}
