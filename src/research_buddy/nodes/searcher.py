"""搜索节点 - 并行搜索各子问题，支持补充搜索"""

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_buddy.state import ResearchState, SearchResult
from research_buddy.tools.search import SearchUnavailableError, search
from research_buddy.utils import normalize_url

logger = logging.getLogger(__name__)


def _search_one(task: dict) -> list[SearchResult]:
    """搜索单个子问题，返回结果列表"""
    query = task.get("search_query", task.get("question", ""))
    if not query:
        return []

    results = search(query, search_depth=task.get("search_depth", "basic"))

    return [
        {
            "sub_question_id": task.get("sub_question_id", ""),
            "sub_question": task.get("question", ""),
            "query": query,
            "language": task.get("language", "auto"),
            "region": task.get("region", "GLOBAL"),
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in results
    ]


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _content_fingerprint(content: str) -> str:
    normalized = re.sub(r"\s+", " ", content.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def searcher(state: ResearchState) -> dict:
    """搜索节点：并行搜索各子问题，支持补充搜索

    增量模式下，会过滤掉与已有知识来源 URL 重复的搜索结果。
    使用统一的 normalize_url 进行 URL 去重。
    """
    sub_questions = list(state.get("sub_questions", []))
    validation_gaps = state.get("validation_gaps", [])
    existing_results = state.get("search_results", [])
    is_incremental = state.get("is_incremental", False)

    searched_sub_question_ids = {
        r.get("sub_question_id", "") for r in existing_results if r.get("sub_question_id")
    }
    searched_questions = {r.get("sub_question", "") for r in existing_results}
    attempted_queries = {
        _normalize_query(item.get("query", ""))
        for item in state.get("search_history", []) if item.get("query")
    }

    # 增量模式：收集已有知识的来源 URL，用于去重（使用统一 normalize_url）
    known_urls: set[str] = set()
    if is_incremental:
        for url in state.get("known_source_urls", []):
            known_urls.add(normalize_url(url))

    # 合并搜索任务：补充搜索 + 未搜索的原始子问题
    search_tasks = []

    # 1. 补充搜索任务（来自验证/反思阶段）
    for gap in validation_gaps:
        search_tasks.append(gap)

    # 2. 原始子问题中尚未搜索的
    for index, sq in enumerate(sub_questions, 1):
        sq_id = sq.get("id", f"sq_{index:02d}")
        if sq_id not in searched_sub_question_ids and sq.get("question", "") not in searched_questions:
            query_specs = sq.get("search_queries") or [{
                "query": sq.get("search_query", ""),
                "language": sq.get("language", "auto"),
                "region": sq.get("region", "GLOBAL"),
            }]
            for query_spec in query_specs:
                if not isinstance(query_spec, dict) or not query_spec.get("query"):
                    continue
                search_tasks.append({
                    **sq,
                    "sub_question_id": sq_id,
                    "search_query": query_spec["query"],
                    "language": query_spec.get("language", sq.get("language", "auto")),
                    "region": query_spec.get("region", sq.get("region", "GLOBAL")),
                    "reason": "initial",
                })

    # 同一轮及历史轮次都不重复执行完全相同的查询。
    unique_tasks = []
    seen_task_queries = set()
    for task in search_tasks:
        query = task.get("search_query", task.get("question", ""))
        query_key = _normalize_query(query)
        if not query_key or query_key in attempted_queries or query_key in seen_task_queries:
            continue
        seen_task_queries.add(query_key)
        if not task.get("sub_question_id"):
            task = {**task, "sub_question_id": task.get("id", "")}
        unique_tasks.append(task)
    search_tasks = unique_tasks
    if state.get("search_round", 0) >= 1:
        search_tasks = [
            {**task, "search_depth": "advanced" if task.get("priority") == "high" else "basic"}
            for task in search_tasks
        ]

    if not search_tasks:
        logger.info("无需搜索（所有子问题已有结果）")
        return {
            "search_results": [],
            "validation_gaps": [],
            "search_round": state.get("search_round", 0) + 1,
            "stop_reason": "no_new_queries",
        }

    total = len(search_tasks)
    logger.info("开始并行搜索 %d 个子问题...", total)

    all_results: list[SearchResult] = []
    done_count = 0
    failures: list[str] = []

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
                failures.append(str(e))
                logger.warning("搜索失败: %s... (%s)", query[:30], e)

            done_count += 1
            logger.debug("搜索进度: %d/%d", done_count, total)

    # 本轮全部失败且整个流程一条证据都没拿到 —— 不能继续往下走，
    # 否则 synthesizer 会凭模型内部知识编一份零来源的报告。
    if failures and len(failures) == total and not existing_results:
        detail = failures[0]
        if state.get("has_knowledge") and state.get("knowledge_context"):
            # 增量/追踪模式还有历史知识可用，降级继续，但把状态标出来供报告披露
            logger.error("搜索层不可用（%s），改为仅基于历史知识作答", detail)
            return {
                "search_results": [],
                "validation_gaps": [],
                "search_round": state.get("search_round", 0) + 1,
                "total_queries": state.get("total_queries", 0) + total,
                "stop_reason": "search_unavailable",
                "search_unavailable": True,
                "messages": [f"⛔ 搜索层不可用（{detail}），仅基于历史知识生成报告"],
            }
        raise SearchUnavailableError(
            f"搜索层不可用，{total} 个查询全部失败且没有任何历史证据: {detail}"
        )

    existing_urls = {normalize_url(r.get("url", "")) for r in existing_results}
    existing_fingerprints = {_content_fingerprint(r.get("content", "")) for r in existing_results}
    if is_incremental:
        existing_urls.update(known_urls)

    unique_results = []
    seen_urls = set(existing_urls)
    seen_fingerprints = set(existing_fingerprints)
    for result in all_results:
        url_key = normalize_url(result.get("url", ""))
        content_key = _content_fingerprint(result.get("content", ""))
        if (url_key and url_key in seen_urls) or (content_key and content_key in seen_fingerprints):
            continue
        if url_key:
            seen_urls.add(url_key)
        if content_key:
            seen_fingerprints.add(content_key)
        unique_results.append(result)
    deduped = len(all_results) - len(unique_results)
    all_results = unique_results
    if deduped:
        logger.info("搜索去重：过滤 %d 条重复结果", deduped)

    logger.info("搜索完成，共获取 %d 条结果", len(all_results))

    # 清空 validation_gaps（已处理）
    all_failed = bool(failures) and len(failures) == total
    msgs = [
        f"🔍 开始并行搜索 {total} 个子问题...",
        f"🔍 搜索完成，共获取 {len(all_results)} 条结果",
    ]
    if failures:
        msgs.append(f"⚠️ {len(failures)}/{total} 个查询失败：{failures[0]}")
    history = [{
        "sub_question_id": task.get("sub_question_id", ""),
        "query": task.get("search_query", task.get("question", "")),
        "reason": task.get("reason", "initial"),
        "search_depth": task.get("search_depth", "basic"),
        "language": task.get("language", "auto"),
        "region": task.get("region", "GLOBAL"),
        "round": state.get("search_round", 0) + 1,
    } for task in search_tasks]
    return {
        "search_results": all_results,
        "validation_gaps": [],
        "search_history": history,
        "search_round": state.get("search_round", 0) + 1,
        "total_queries": state.get("total_queries", 0) + len(search_tasks),
        # 本轮全挂但之前已有证据：不再让路由把预算耗在必然失败的补搜上
        "stop_reason": "search_unavailable" if all_failed else "",
        "search_unavailable": all_failed,
        "messages": msgs,
    }
