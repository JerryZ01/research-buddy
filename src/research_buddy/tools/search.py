"""搜索工具 - Tavily API 搜索（专为 AI Agent 优化）

Tavily 是专为 AI Agent 设计的搜索引擎：
- 中文搜索质量远优于 cn.bing.com
- 返回结构化结果（标题、URL、内容摘要、相关性评分）
- 无需页面抓取，结果已包含清洗后的内容
- 免费 1000 次/月
"""

from research_buddy.config import MAX_SEARCH_RESULTS, TAVILY_API_KEY


def search(query: str, max_results: int | None = None, scrape_pages: bool = True) -> list[dict]:
    """使用 Tavily 搜索

    Args:
        query: 搜索查询词（英文效果最佳）
        max_results: 最大结果数
        scrape_pages: 兼容参数，Tavily 已自带内容提取，此参数无效
    """
    limit = max_results or MAX_SEARCH_RESULTS

    if not TAVILY_API_KEY:
        print(f"[search] ⚠️  TAVILY_API_KEY 未配置，无法搜索")
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

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
                "score": r.get("score", 1.0),
            })

        print(f"[search] Tavily 返回 {len(results)} 条结果 (query: {query[:40]})")
        return results

    except Exception as e:
        print(f"[search] Tavily 搜索失败: {e}")
        return []
