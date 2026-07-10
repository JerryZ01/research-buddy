"""搜索工具 - Tavily API 搜索（专为 AI Agent 优化）

Tavily 是专为 AI Agent 设计的搜索引擎：
- 中文搜索质量远优于 cn.bing.com
- 返回结构化结果（标题、URL、内容摘要、相关性评分）
- 无需页面抓取，结果已包含清洗后的内容
- 免费 1000 次/月
"""

import logging

from research_buddy.config import MAX_SEARCH_RESULTS, TAVILY_API_KEY

logger = logging.getLogger(__name__)

# 懒初始化 TavilyClient 单例
_tavily_client = None


def _get_tavily_client():
    """获取全局 TavilyClient 实例（懒初始化，避免每次搜索新建 HTTP 会话）"""
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def search(query: str, max_results: int | None = None) -> list[dict]:
    """使用 Tavily 搜索

    Args:
        query: 搜索查询词（英文效果最佳）
        max_results: 最大结果数
    """
    limit = max_results or MAX_SEARCH_RESULTS

    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY 未配置，无法搜索")
        return []

    try:
        client = _get_tavily_client()

        response = client.search(
            query=query,
            max_results=limit,
            search_depth="basic",  # basic=1 credit, advanced=2 credits
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
        logger.warning("Tavily 搜索失败: %s", e)
        return []
