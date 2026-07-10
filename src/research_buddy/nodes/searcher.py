"""搜索节点 - 并行搜索各子问题，支持补充搜索"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from research_buddy.state import ResearchState, SearchResult
from research_buddy.tools.search import search


def _search_one(task: dict) -> list[SearchResult]:
    """搜索单个子问题，返回结果列表"""
    query = task.get("search_query", task.get("question", ""))
    if not query:
        return []

    results = search(query)

    return [
        {
            "sub_question": task.get("question", ""),
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in results
    ]


def searcher(state: ResearchState) -> dict:
    """搜索节点：并行搜索各子问题，支持补充搜索

    增量模式下，会过滤掉与已有知识来源 URL 重复的搜索结果。
    """
    sub_questions = list(state.get("sub_questions", []))
    validation_gaps = state.get("validation_gaps", [])
    existing_results = state.get("search_results", [])
    is_incremental = state.get("is_incremental", False)

    # 已搜索过的子问题
    searched_questions = set()
    for r in existing_results:
        searched_questions.add(r.get("sub_question", ""))

    # 增量模式：收集已有知识的来源 URL，用于去重
    known_urls: set[str] = set()
    if is_incremental:
        for url in state.get("known_source_urls", []):
            # 规范化：去掉协议和末尾斜杠
            known_urls.add(url.replace("https://", "").replace("http://", "").rstrip("/"))

    # 合并搜索任务：补充搜索 + 未搜索的原始子问题
    search_tasks = []

    # 1. 补充搜索任务（来自验证/反思阶段）
    for gap in validation_gaps:
        search_tasks.append(gap)

    # 2. 原始子问题中尚未搜索的
    for sq in sub_questions:
        if sq.get("question", "") not in searched_questions:
            search_tasks.append(sq)

    if not search_tasks:
        print("🔍 无需搜索（所有子问题已有结果）")
        return {"search_results": [], "validation_gaps": []}

    total = len(search_tasks)
    print(f"🔍 开始并行搜索 {total} 个子问题...")

    all_results: list[SearchResult] = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=min(total, 4)) as executor:
        futures = {
            executor.submit(_search_one, task): task
            for task in search_tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                query = task.get("search_query", "")
                print(f"   ⚠️  搜索失败: {query[:30]}... ({e})")

            done_count += 1
            print(f"   搜索进度: {done_count}/{total}")

    # 增量模式去重：过滤已知 URL 的搜索结果
    if is_incremental and known_urls:
        before = len(all_results)
        all_results = [r for r in all_results if _normalize_url(r.get("url", "")) not in known_urls]
        deduped = before - len(all_results)
        if deduped > 0:
            print(f"🔍 增量去重：过滤 {deduped} 条与已有知识重复的搜索结果")

    print(f"🔍 搜索完成，共获取 {len(all_results)} 条结果")

    # 清空 validation_gaps（已处理）
    msgs = [
        f"🔍 开始并行搜索 {total} 个子问题...",
        *[f"   搜索进度: {i+1}/{total}" for i in range(done_count)],
        f"🔍 搜索完成，共获取 {len(all_results)} 条结果",
    ]
    return {"search_results": all_results, "validation_gaps": [], "messages": msgs}


def _normalize_url(url: str) -> str:
    """URL 规范化：去掉协议和末尾斜杠，用于去重比较"""
    return url.replace("https://", "").replace("http://", "").rstrip("/")
