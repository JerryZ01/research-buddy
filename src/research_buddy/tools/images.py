"""图片选择工具 — 用视觉模型从搜索结果中挑选与文章内容相关的插图

流程：
1. 按子问题聚合候选图 URL（searcher 节点已去重限量）
2. 下载候选图（httpx，超时 + 大小上限 + content-type 校验）
3. 转 base64 发给视觉模型（VISION_MODEL），让它选出相关图片并写 alt 文本
4. 校验模型输出（编号在候选范围内），映射回 {url, alt, sub_question_id}

降级链（任何一步失败都不阻塞出稿）：
- VISION_MODEL / OPENAI_API_KEY 未配置 → 直接返回 []（图片功能整体关闭）
- 单张图片下载失败 → 跳过该图
- 视觉模型调用失败 → 打 WARNING，该子问题无插图
"""

import base64
import hashlib
import json
import logging
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from research_buddy.config import (DATA_DIR, MAX_IMAGES_IN_ARTICLE, OPENAI_API_BASE,
                              OPENAI_API_KEY, VISION_API_BASE, VISION_API_KEY, VISION_MODEL)
from research_buddy.utils import add_tokens, parse_llm_json

logger = logging.getLogger(__name__)

# 候选与选中上限（与 searcher 聚合共用语义）
MAX_CANDIDATES_PER_SUB_QUESTION = 6   # 每个子问题最多提交给视觉模型的候选数
MAX_IMAGES_PER_SUB_QUESTION = 3       # 每个子问题最多选中的插图数
# 全局供图上限：要足够多，让文章模型能按 MAX_IMAGES_IN_ARTICLE 配图
MAX_TOTAL_IMAGES = MAX_IMAGES_IN_ARTICLE
MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024  # 单张图片下载大小上限（3MB）
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024   # 单次视觉调用累计原始字节上限（4MB）
# 插图质量下限：过小的图（logo/图标）和宽高比极端的图（横幅）放进文章很难看
MIN_IMAGE_WIDTH = 400
MIN_IMAGE_HEIGHT = 300
MIN_ASPECT_RATIO = 0.25   # 1:4
MAX_ASPECT_RATIO = 4.0    # 4:1
_DOWNLOAD_TIMEOUT = 10.0
_VISION_TIMEOUT = 60.0
IMAGE_CACHE_DIR = Path(DATA_DIR) / "research-images"
_MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def _cache_image(data: bytes, mime: str) -> str:
    """按内容哈希缓存已验证图片，返回同源静态 URL；失败时返回空串。"""
    extension = _MIME_EXTENSIONS.get(mime.lower())
    if not data or not extension:
        return ""
    try:
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(data).hexdigest()[:24] + extension
        target = IMAGE_CACHE_DIR / filename
        if not target.exists():
            temporary = target.with_name(f".{filename}.{threading.get_ident()}.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
        return f"/media/research-images/{filename}"
    except OSError as exc:
        logger.warning("缓存选中图片失败，继续使用原始 URL: %s", exc)
        return ""


def _image_size(data: bytes) -> tuple[int, int] | None:
    """从图片字节解析宽高（PNG/JPEG/GIF；WebP 布局复杂，返回 None 不拦截）。"""
    if len(data) < 24:
        return None
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", data[16:24])
            return w, h
        if data[:6] in (b"GIF87a", b"GIF89a"):
            w, h = struct.unpack("<HH", data[6:10])
            return w, h
        if data[:2] == b"\xff\xd8":  # JPEG：扫描 SOF 标记取宽高
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                length = struct.unpack(">H", data[i + 2:i + 4])[0]
                i += 2 + length
    except (struct.error, IndexError):
        return None
    return None

SELECT_IMAGES_PROMPT = """你是图片选图专家。下面是一个研究子问题及其候选图片。

## 子问题
{question}

## 候选图片（index 对应上面图片的顺序）
{candidate_list}

## 任务
从候选图中选出适合插入文章的图片（最多 {max_count} 张）。选图标准：
- **相关性**：图中内容应与子问题讨论的事物相关；允许与主题相关但信息量
  较低的配图/装饰图（氛围图），但完全无关、明显凑数的图不选
- **信息量优先**：优先图表、架构图、流程图、截图等有实质内容的图；
  其次才是装饰性配图，两者都可选
- **数量克制**：不要重复选内容相似的图，每个子问题最多 {max_count} 张

为每张选中的图写 alt：**描述图中实际内容**（如「柱状图：2023-2025 年营收对比」、
「架构图：控制面与数据面分离」；装饰图写画面主题，如「云端数据中心夜景」），
中文 15 字以内——它既是插图题注，也是文章模型判断图文是否匹配的依据。

## 输出格式
只返回 JSON（不要包含其他内容）：
```json
{{"images": [{{"index": 1, "alt": "画面内容描述"}}]}}
```
- index 必须是候选列表中的编号
- 没有合适的图片就返回 {{"images": []}}"""


def _vision_base() -> str:
    """视觉 API base：VISION_API_BASE 优先，未配回退 OPENAI_API_BASE。"""
    return (VISION_API_BASE or OPENAI_API_BASE).rstrip("/")


def _vision_key() -> str:
    """视觉 API key：VISION_API_KEY 优先，未配回退 OPENAI_API_KEY。"""
    return VISION_API_KEY or OPENAI_API_KEY


def _is_http_url(url: str) -> bool:
    try:
        scheme = urlsplit(url).scheme.lower()
        return scheme in {"http", "https"}
    except ValueError:
        return False


# 伪装成浏览器请求：很多图床/CDN（i.sstatic.net、ResearchGate 等）会拒绝
# 非浏览器 UA 或没有 Referer 的下载请求（浏览器能正常加载正是因为它带了
# 完整的 UA + Referer）。多套 header 策略轮换重试：有些站要求 Referer
# 同源，有些站反而反感 Referer，最后用最简 UA 兜底。
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_BROWSER_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
}


def _download_headers(url: str) -> list[dict]:
    """候选下载 header 策略：带同源 Referer → 不带 Referer → 最简。"""
    origin = ""
    try:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}/"
    except ValueError:
        pass
    strategies = [{**_BROWSER_HEADERS, "Referer": origin}] if origin else []
    strategies.append(_BROWSER_HEADERS)
    strategies.append({"User-Agent": _BROWSER_UA})
    return strategies


def _download_image(client: httpx.Client, url: str) -> tuple[bytes, str] | None:
    """下载图片，返回 (原始字节, mime)。失败返回 None。

    403/瞬时网络错误用多套浏览器风格 header 重试；仍失败则跳过该图。
    """
    last_reason = ""
    for headers in _download_headers(url):
        try:
            resp = client.get(url, timeout=_DOWNLOAD_TIMEOUT,
                              follow_redirects=True, headers=headers)
            if resp.status_code == 403:
                last_reason = "HTTP 403（防盗链）"
                continue  # 换 header 策略再试
            resp.raise_for_status()
            content = resp.content
            if not content:
                last_reason = "空响应"
                continue
            if len(content) > MAX_DOWNLOAD_BYTES:
                logger.warning("图片超过大小上限 %.1fMB，跳过: %s",
                               MAX_DOWNLOAD_BYTES / 1024 / 1024, url[:60])
                return None
            mime = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if mime and not mime.startswith("image/"):
                logger.warning("非图片 content-type（%s），跳过: %s", mime, url[:60])
                return None
            if not mime:
                mime = "image/jpeg"

            # 质量下限：拒绝小方块图（logo/缩略图，宽高都小）和宽高比极端的横幅。
            # 宽型图表（如 576x152 的折线图）宽度达标就放行——技术文章很需要这类图。
            size = _image_size(content)
            if size:
                width, height = size
                if width < MIN_IMAGE_WIDTH and height < MIN_IMAGE_HEIGHT:
                    logger.warning("图片尺寸过小（%dx%d），跳过: %s", width, height, url[:60])
                    return None
                ratio = width / max(height, 1)
                if ratio < MIN_ASPECT_RATIO or ratio > MAX_ASPECT_RATIO:
                    logger.warning("图片宽高比极端（%.2f），跳过: %s", ratio, url[:60])
                    return None

            return content, mime
        except Exception as exc:
            last_reason = str(exc)[:120]
            # 403/SSL 瞬时错误等：换 header 策略再试
            continue

    logger.warning("图片下载失败，跳过: %s (%s)", url[:60], last_reason)
    return None


def _vision_select(question: str, images: list[tuple[dict, bytes, str]],
                   max_tokens: int = 1024) -> list[dict]:
    """调用视觉模型选图。index 对应 images 列表顺序（即实际下载成功的图）。

    Args:
        question: 子问题文本
        images: [(候选 dict, 原始字节, mime), ...] 下载成功的图
        max_tokens: 输出上限。JSON 截断时由重试逻辑翻倍重试

    Returns:
        [{url, alt, sub_question_id, query}]
    """
    prompt = SELECT_IMAGES_PROMPT.format(
        question=question,
        candidate_list="\n".join(
            f"- [{i}] {c.get('url', '')}" for i, (c, _, _) in enumerate(images, 1)
        ),
        max_count=MAX_IMAGES_PER_SUB_QUESTION,
    )
    content_parts: list[dict] = [{"type": "text", "text": prompt}]
    for _, raw, mime in images:
        b64 = base64.b64encode(raw).decode("ascii")
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": content_parts}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    resp = httpx.post(
        f"{_vision_base()}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {_vision_key()}"},
        timeout=_VISION_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    # 视觉调用也计入本次研究的 token 统计（裸 httpx 调用，Langfuse 看不到，
    # 这里手动累计 + 记录观测）
    usage = data.get("usage") or {}
    add_tokens(usage)
    _record_vision_observation(payload, data)

    text = data["choices"][0]["message"]["content"]
    parsed = parse_llm_json(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"视觉模型输出不是 JSON 对象: {type(parsed).__name__}")

    picked = []
    seen_indexes: set[int] = set()
    for item in parsed.get("images", []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        alt = str(item.get("alt", "")).strip()
        # 编号必须在已下载列表范围内，且不重复；空 alt 视为无效输出——否则视为幻觉丢弃
        if index < 1 or index > len(images) or index in seen_indexes:
            continue
        if not alt:
            continue
        seen_indexes.add(index)
        candidate = images[index - 1][0]
        picked.append({
            "url": candidate.get("url", ""),
            "alt": alt,
            "sub_question_id": candidate.get("sub_question_id", ""),
            "query": candidate.get("query", ""),
        })
        if len(picked) >= MAX_IMAGES_PER_SUB_QUESTION:
            break
    return picked


# 单次请求图片数上限：中转站/DeepSeek 对多图请求有数量限制，超了会 400
_MAX_VISION_BATCH = 4
# 子问题间视觉调用并发上限（避免触发中转站限流）
_MAX_VISION_PARALLEL = 3


def _record_vision_observation(payload: dict, data: dict) -> None:
    """把视觉选图调用记录到 Langfuse（可选，未配置/失败静默跳过）。

    视觉调用走裸 httpx，不走 langchain，默认不在 Langfuse 里；
    这里用 OTEL 上下文挂一个 generation 观测，usage 手动喂入。
    """
    try:
        from research_buddy.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
        if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
            return
        from langfuse import get_client
        usage = data.get("usage") or {}
        choices = data.get("choices") or [{}]
        output_text = (choices[0].get("message") or {}).get("content", "")
        client = get_client()
        with client.start_as_current_observation(
            name="vision-select",
            as_type="generation",
            model=VISION_MODEL,
            input={
                "model": payload.get("model"),
                "images": sum(
                    1 for part in (payload.get("messages") or [{}])[0].get("content", [])
                    if isinstance(part, dict) and part.get("type") == "image_url"
                ),
            },
            output=output_text,
            usage_details={
                "input": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                "output": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                "total": int(usage.get("total_tokens") or 0),
            },
        ):
            pass
    except Exception:
        # 观测是锦上添花，失败不影响选图
        pass


def _vision_select_with_retry(question: str,
                              images: list[tuple[dict, bytes, str]]) -> list[dict]:
    """带自动降级重试的视觉选图。

    - 请求过大（400/413）或限流（429）：减半图片批次重试（每张图最多试 3 次）
    - 输出 JSON 截断/解析失败：把 max_tokens 翻倍到 2048 重试一次
    - 全部失败返回 []（该子问题无插图，不向调用方抛异常）
    """
    batch = images[:_MAX_VISION_BATCH]
    max_tokens = 1024
    for _ in range(3):
        try:
            return _vision_select(question, batch, max_tokens=max_tokens)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {400, 413, 429} and len(batch) > 1:
                logger.warning("视觉请求过大（HTTP %s），图片批次减半重试（%d→%d 张）",
                               exc.response.status_code, len(batch), max(1, len(batch) // 2))
                batch = batch[:max(1, len(batch) // 2)]
                continue
            raise
        except Exception:
            # JSON 截断或解析失败：大概率是 max_tokens 把输出切断了
            if max_tokens < 2048:
                logger.warning("视觉输出解析失败，加大 max_tokens 重试（%d→2048）", max_tokens)
                max_tokens = 2048
                continue
            raise
    return []


def select_images(sub_questions: list[dict],
                  image_candidates: list[dict]) -> list[dict]:
    """从候选图中选图（每子问题一次视觉调用）。

    Args:
        sub_questions: 规划出的子问题 [{id, question, ...}]
        image_candidates: searcher 聚合的候选 [{url, sub_question_id, query}]

    Returns:
        [{url, alt, sub_question_id, query}]；功能未配置或全部失败时返回 []。
    """
    if not VISION_MODEL or not _vision_key():
        return []

    # 子问题 id → 问题文本
    question_by_id = {
        sq.get("id", ""): sq.get("question", "")
        for sq in sub_questions if sq.get("id")
    }

    # 按子问题分组，每子问题最多保留 MAX_CANDIDATES_PER_SUB_QUESTION 张
    by_subq: dict[str, list[dict]] = {}
    for c in image_candidates:
        url = c.get("url", "")
        if not _is_http_url(url):
            continue
        sq_id = c.get("sub_question_id", "") or "general"
        bucket = by_subq.setdefault(sq_id, [])
        if len(bucket) < MAX_CANDIDATES_PER_SUB_QUESTION:
            bucket.append(c)

    if not by_subq:
        return []

    def _select_for_subq(item: tuple[str, list[dict]]) -> list[dict]:
        """单个子问题的选图流程（下载 + 视觉调用），供线程池并行。"""
        sq_id, candidates = item
        question = question_by_id.get(sq_id) or candidates[0].get("query", "")
        if not question:
            return []

        # 下载候选图（并行，最多 4 并发）——httpx.Client 线程安全，
        # pool.map 保持输入顺序，视觉模型的 index 映射不会错位。
        # 只保留下载成功的图，累计字节不超单次调用上限。
        candidates_pool = candidates[:MAX_CANDIDATES_PER_SUB_QUESTION]
        downloaded_all: list[tuple[dict, bytes, str]] = []
        with ThreadPoolExecutor(max_workers=min(4, len(candidates_pool))) as pool:
            for cand, result in pool.map(
                lambda c: (c, _download_image(client, c.get("url", ""))),
                candidates_pool,
            ):
                if result:
                    downloaded_all.append((cand, result[0], result[1]))

        images: list[tuple[dict, bytes, str]] = []
        total = 0
        for item_ in downloaded_all:
            if total + len(item_[1]) > MAX_PAYLOAD_BYTES:
                continue
            total += len(item_[1])
            images.append(item_)
            if len(images) >= MAX_CANDIDATES_PER_SUB_QUESTION:
                break

        if not images:
            return []

        try:
            picked = _vision_select_with_retry(question, images)
            downloaded_by_url = {
                item_[0].get("url", ""): (item_[1], item_[2]) for item_ in images
            }
            for selected_image in picked:
                downloaded = downloaded_by_url.get(selected_image.get("url", ""))
                if downloaded:
                    selected_image["cached_url"] = _cache_image(downloaded[0], downloaded[1])
            if picked:
                logger.info("子问题 %s 选中 %d 张插图", sq_id, len(picked))
            return picked
        except Exception as exc:
            logger.warning("视觉选图失败（子问题 %s），该子问题无插图: %s", sq_id, exc)
            return []

    # 子问题之间并行（≤3 并发，pool.map 保持顺序，截断/去重不受影响）
    selected_groups: list[list[dict]] = []
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=min(_MAX_VISION_PARALLEL, len(by_subq))) as pool:
            for picked in pool.map(_select_for_subq, list(by_subq.items())):
                selected_groups.append(picked)

    # 按子问题轮询取图，避免前几个分支各占 3 张后把后续章节全部截掉。
    selected: list[dict] = []
    for position in range(MAX_IMAGES_PER_SUB_QUESTION):
        for group in selected_groups:
            if position < len(group):
                selected.append(group[position])
                if len(selected) >= MAX_TOTAL_IMAGES:
                    return selected

    return selected


if __name__ == "__main__":
    # 冒烟测试：python -m research_buddy.tools.images
    print("select_images =", select_images([], []))
