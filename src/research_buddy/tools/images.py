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
import json
import logging
from urllib.parse import urlsplit

import httpx

from research_buddy.config import OPENAI_API_BASE, OPENAI_API_KEY, VISION_MODEL
from research_buddy.utils import add_tokens, parse_llm_json

logger = logging.getLogger(__name__)

# 候选与选中上限（与 searcher 聚合共用语义）
MAX_CANDIDATES_PER_SUB_QUESTION = 6   # 每个子问题最多提交给视觉模型的候选数
MAX_IMAGES_PER_SUB_QUESTION = 3       # 每个子问题最多选中的插图数
MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024  # 单张图片下载大小上限（3MB）
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024   # 单次视觉调用累计原始字节上限（4MB）
_DOWNLOAD_TIMEOUT = 10.0
_VISION_TIMEOUT = 60.0

SELECT_IMAGES_PROMPT = """你是图片选图专家。下面是一个研究子问题及其候选图片。

## 子问题
{question}

## 候选图片（index 对应上面图片的顺序）
{candidate_list}

## 任务
选出与子问题内容最相关、最适合插入文章的图片（最多 {max_count} 张），
并为每张选中的图片写一个简短、描述性的 alt 文本（中文，15 字以内，说明画面内容）。

## 输出格式
只返回 JSON（不要包含其他内容）：
```json
{{"images": [{{"index": 1, "alt": "画面内容描述"}}]}}
```
- index 必须是候选列表中的编号
- 没有合适的图片就返回 {{"images": []}}"""


def _is_http_url(url: str) -> bool:
    try:
        scheme = urlsplit(url).scheme.lower()
        return scheme in {"http", "https"}
    except ValueError:
        return False


def _download_image(client: httpx.Client, url: str) -> tuple[bytes, str] | None:
    """下载图片，返回 (原始字节, mime)。失败返回 None。"""
    try:
        resp = client.get(url, timeout=_DOWNLOAD_TIMEOUT,
                          follow_redirects=True,
                          headers={"User-Agent": "research-buddy/0.3"})
        resp.raise_for_status()
        content = resp.content
        if not content:
            logger.warning("图片下载为空: %s", url[:60])
            return None
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
        return content, mime
    except Exception as exc:
        logger.warning("图片下载失败，跳过: %s (%s)", url[:60], exc)
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
        f"{OPENAI_API_BASE.rstrip('/')}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
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
    if not VISION_MODEL or not OPENAI_API_KEY:
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

    selected: list[dict] = []
    with httpx.Client() as client:
        for sq_id, candidates in by_subq.items():
            question = question_by_id.get(sq_id) or candidates[0].get("query", "")
            if not question:
                continue

            # 下载候选图，累计字节不超单次调用上限。
            # 只保留下载成功的图：视觉模型的 index 对应这个顺序，映射不会错位。
            images: list[tuple[dict, bytes, str]] = []
            total = 0
            for c in candidates:
                if total >= MAX_PAYLOAD_BYTES:
                    break
                downloaded = _download_image(client, c.get("url", ""))
                if not downloaded:
                    continue
                raw, mime = downloaded
                total += len(raw)
                images.append((c, raw, mime))
                if len(images) >= MAX_CANDIDATES_PER_SUB_QUESTION:
                    break

            if not images:
                continue

            try:
                picked = _vision_select_with_retry(question, images)
                selected.extend(picked)
                if picked:
                    logger.info("子问题 %s 选中 %d 张插图", sq_id, len(picked))
            except Exception as exc:
                logger.warning("视觉选图失败（子问题 %s），该子问题无插图: %s", sq_id, exc)

    return selected


if __name__ == "__main__":
    # 冒烟测试：python -m research_buddy.tools.images
    print("select_images =", select_images([], []))
