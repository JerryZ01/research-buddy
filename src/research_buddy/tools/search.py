"""搜索工具 - Tavily API 搜索（专为 AI Agent 优化）

Tavily 是专为 AI Agent 设计的搜索引擎：
- 中文搜索质量远优于 cn.bing.com
- 返回结构化结果（标题、URL、内容摘要、相关性评分）
- 无需页面抓取，结果已包含清洗后的内容
- 免费 1000 次/月
"""

import logging
import time

from research_buddy.config import MAX_SEARCH_RESULTS, TAVILY_API_KEY

logger = logging.getLogger(__name__)

# 懒初始化 TavilyClient 单例
_tavily_client = None

# 瞬时错误（限流、网络抖动）重试次数与退避间隔
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 3.0)


class SearchUnavailableError(RuntimeError):
    """搜索层不可用：没有 API key，或重试后仍然失败。

    以前这两种情况都只打一行 warning 然后返回 []，于是整条流水线照跑，
    最后由 LLM 凭训练数据编出一份零来源却写着「整体置信度：高」的报告。
    搜索失败必须能和「搜到了但没有结果」区分开。
    """


def _get_tavily_client():
    """获取全局 TavilyClient 实例（懒初始化，避免每次搜索新建 HTTP 会话）"""
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def search(query: str, max_results: int | None = None,
           search_depth: str = "basic") -> list[dict]:
    """使用 Tavily 搜索

    Args:
        query: 搜索查询词（由规划器根据目标来源选择语言）
        max_results: 最大结果数

    Returns:
        结果列表；空列表表示「搜索成功但没有命中」。

    Raises:
        SearchUnavailableError: 未配置 API key，或重试后依然失败。
    """
    limit = max_results or MAX_SEARCH_RESULTS

    if not TAVILY_API_KEY:
        raise SearchUnavailableError("TAVILY_API_KEY 未配置，无法执行搜索")

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            client = _get_tavily_client()

            response = client.search(
                query=query,
                max_results=limit,
                search_depth=search_depth if search_depth in {"basic", "advanced"} else "basic",
                include_raw_content=False,
            )

            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    # Tavily 缺省 score 时默认 0.0（未知相关性），而非误导性的 1.0
                    "score": r.get("score", 0.0),
                })

            logger.info("Tavily 返回 %d 条结果 (query: %s)", len(results), query[:40])
            return results

        except Exception as e:
            last_error = e
            if attempt < _MAX_ATTEMPTS - 1:
                delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
                logger.warning("Tavily 搜索失败（第 %d/%d 次，%.0fs 后重试）: %s",
                               attempt + 1, _MAX_ATTEMPTS, delay, e)
                time.sleep(delay)

    raise SearchUnavailableError(
        f"Tavily 搜索连续 {_MAX_ATTEMPTS} 次失败 (query: {query[:40]}): {last_error}"
    ) from last_error
