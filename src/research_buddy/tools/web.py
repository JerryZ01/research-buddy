"""网页抓取工具 — 抓取网页全文并提取正文，作为高质量写作证据。

现有 Tavily 搜索只返回内容摘要，搜不到好来源时写作依据是断章取义的碎片。
本工具补齐「真正读网页」的能力：给一个 URL，抓取 HTML 并用标准库 html.parser
提取正文文本（去脚本/导航/页脚），供 agentic 研究循环和写作者使用。

零新依赖：不引入 trafilatura/bs4，用标准库 html.parser 实现够用的正文提取。

安全与资源边界：
- 单页抓取超时（默认 15s）、正文大小上限（默认 200KB）；
- 只抓 http/https，用浏览器 UA 降低被防爬拦截的概率；
- 失败返回 None，绝不抛异常阻断研究主流程。
"""

import logging
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx

from research_buddy.config import FETCH_TIMEOUT, FETCH_MAX_BYTES, FETCH_MAX_TEXT_CHARS

logger = logging.getLogger(__name__)

_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
}

# 需要整体丢弃的标签（导航、脚本、样式、页脚等非正文内容）
_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside",
              "iframe", "form", "button", "svg", "canvas", "template"}
# 块级标签：文本后追加换行，保留段落结构
_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "section",
               "article", "main", "blockquote", "pre", "tr", "td", "th", "br", "ul", "ol", "table"}


class _TextExtractor(HTMLParser):
    """从 HTML 提取正文纯文本，去脚本/导航/页脚，保留段落结构。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []
        self._current: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._current.append(text)

    def _flush(self):
        if self._current:
            line = " ".join(self._current)
            if line:
                self._parts.append(line)
        self._current = []

    def text(self) -> str:
        self._flush()
        return "\n".join(self._parts)


def extract_text_from_html(html: str, max_chars: int = FETCH_MAX_TEXT_CHARS) -> str:
    """从 HTML 提取正文文本。提取为空时 fallback 到去除标签后的全文。"""
    if not html:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        text = parser.text()
    except Exception:
        text = ""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # 只有提取结果为空（完全失败）时才退化为去掉标签的全文本；
    # 短文章本来就短，不应因此走 fallback 把 script/nav 等噪声混进来。
    if not text:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _is_http_url(url: str) -> bool:
    try:
        return urlsplit(url).scheme.lower() in {"http", "https"}
    except ValueError:
        return False


def fetch_webpage(url: str) -> dict | None:
    """抓取网页并提取正文。

    Args:
        url: 目标网页 URL（http/https）。

    Returns:
        {"url", "title", "text"}；失败返回 None（不抛异常）。
    """
    if not _is_http_url(url):
        return None
    try:
        with httpx.Client(follow_redirects=True, timeout=FETCH_TIMEOUT) as client:
            resp = client.get(url, headers=_HEADERS)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            if "html" not in content_type and "text" not in content_type:
                # 非 HTML（PDF/图片/二进制等）不尝试提取
                return None
            data = resp.content
            if len(data) > FETCH_MAX_BYTES:
                data = data[:FETCH_MAX_BYTES]
            html = data.decode("utf-8", errors="replace")
            text = extract_text_from_html(html)
            if not text:
                return None
            # 提取 <title>
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else url
            return {"url": url, "title": title, "text": text}
    except Exception as exc:
        logger.info("网页抓取失败，跳过: %s (%s)", url[:80], str(exc)[:100])
        return None
