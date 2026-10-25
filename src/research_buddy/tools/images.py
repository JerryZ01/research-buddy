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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from research_buddy.config import (DATA_DIR, MAX_GENERATED_IMAGES,
                              MAX_IMAGES_IN_ARTICLE, OPENAI_API_BASE,
                              OPENAI_API_KEY, VISION_API_BASE, VISION_API_KEY, VISION_MODEL)
from research_buddy.tools.image_gen import generate_image, is_enabled as image_gen_enabled
from research_buddy.utils import add_tokens, parse_llm_json

logger = logging.getLogger(__name__)

# 候选与选中上限（与 searcher 聚合共用语义）
MAX_CANDIDATES_PER_SUB_QUESTION = 10  # 每个子问题最多提交给视觉模型的候选数（放宽，提高选图余地）
MAX_IMAGES_PER_SUB_QUESTION = 4       # 每个子问题最多选中的插图数（放宽，让文章更图文并茂）
# 全局供图上限：要足够多，让文章模型能按 MAX_IMAGES_IN_ARTICLE 配图
MAX_TOTAL_IMAGES = MAX_IMAGES_IN_ARTICLE
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024  # 单张图片下载大小上限（放宽到 5MB，减少高质量图被拒）
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024   # 单次视觉调用累计原始字节上限（放宽到 8MB，让更多候选进入筛选）
# 插图质量下限：过小的图（logo/图标）和宽高比极端的图（横幅）放进文章很难看。
# 高度下限放宽到 250、宽高比上限放宽到 6:1，减少误杀横向图表/长截图，
# 但仍拒绝 200×200 这类小图标。
MIN_IMAGE_WIDTH = 400
MIN_IMAGE_HEIGHT = 250
MIN_ASPECT_RATIO = 0.20   # 1:5
MAX_ASPECT_RATIO = 6.0    # 6:1
_DOWNLOAD_TIMEOUT = 15.0
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
- **无水印（硬门槛，必须逐张做视觉检查，不能凭印象）**：见下方「水印检查方法」
- **相关性**：图中内容应与子问题讨论的事物相关；允许与主题相关但信息量
  较低的配图/装饰图（氛围图），但完全无关、明显凑数的图不选
- **信息量优先**：优先图表、架构图、流程图、截图等有实质内容的图；
  其次才是装饰性配图，两者都可选
- **数量克制**：不要重复选内容相似的图，每个子问题最多 {max_count} 张

## 水印检查方法（每张候选图都要做，不能跳过）
**第一步——抄写**：逐字抄写画面上叠加在图片内容之上的所有文字和标记。
必须检查四个角和边缘。包括但不限于：账号署名（知乎@xxx、公众号xxx、微博@xxx）、
网址/域名、平台名称、©版权、二维码、半透明平铺的重复文字或图案、图库预览字样、
以及任何作者签名。抄写时给出大致位置，如「右下角：知乎 @北方的郎」。
以下不算叠加内容，不要抄进来：图表自带的坐标轴、图例、标题、数据标注；
产品界面自带的品牌 Logo。
确实没有任何叠加内容时，overlay 填空字符串 ""。

**第二步——判定**：只有 overlay 为空字符串时，watermark_status 才能是 "none"；
只要抄到了上面任何一类内容，watermark_status 必须是 "present"，并且该图不能选。

为每张选中的图写 alt：**描述图中实际内容**（如「柱状图：2023-2025 年营收对比」、
「架构图：控制面与数据面分离」；装饰图写画面主题，如「云端数据中心夜景」），
中文 15 字以内——它既是插图题注，也是文章模型判断图文是否匹配的依据。

## 输出格式
只返回 JSON（不要包含其他内容）：
```json
{{"images": [{{"index": 1, "overlay": "", "alt": "画面内容描述", "watermark_status": "none"}}]}}
```
- index 必须是候选列表中的编号
- overlay 是第一步抄写的内容，无水印时为空字符串 ""
- watermark_status 只能是 `none`、`present` 或 `uncertain`；只有 overlay 为空
  且确认无水印时才能写 `none`；`present` 和 `uncertain` 的图片不要输出
- 没有合适的图片就返回 {{"images": []}}"""


# 二次复核：独立的一次一图调用，只问水印，避免和选图/写 alt 抢注意力。
# 实测同一模型在「选图+写 alt+判水印」的复合任务里会把水印图答成 none，
# 而单独问「抄写叠加内容」时能稳定读出署名、网址、素材站标记。
VERIFY_WATERMARK_PROMPT = """你是图片水印审查员。仔细看这张图，注意四个角和边缘。

逐字抄写画面上叠加在图片内容之上的所有文字和标记，包括但不限于：
账号署名（知乎@xxx、公众号xxx、微博@xxx）、网址/域名、平台名称（CSDN/掘金/简书等）、
©版权标识、二维码、半透明平铺的重复文字或图案、图库预览或素材站名称、
作者签名或水印 Logo。

以下不算叠加内容，不要抄进来：图表自带的坐标轴、图例、标题、数据标注；
照片或截图本身的内容文字（如文档正文、界面按钮、聊天记录）；
产品界面自带的品牌 Logo。

没有任何叠加内容时，overlay_text 填空字符串 ""。

只返回 JSON（不要包含其他内容）：
{{"overlay_text": "抄写内容；无则空串"}}"""


def _retry_after(resp: httpx.Response) -> float | None:
    """解析 Retry-After 头（秒）；缺失或无法解析时返回 None。"""
    raw = (resp.headers.get("retry-after") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 30.0))
    except ValueError:
        return None


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
    data = _vision_post(payload, max_attempts=_SELECT_ATTEMPTS)

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
        watermark_status = str(item.get("watermark_status", "")).strip().lower()
        # 模型自报的叠加文字；非空即代表画面上有叠加内容
        overlay = str(item.get("overlay", "") or "").strip()
        # 水印是硬门槛：只有视觉模型明确确认 none 才放行。字段缺失、疑似水印
        # 或返回其他值都 fail-closed，避免旧 prompt/异常输出绕过校验。
        if watermark_status != "none":
            logger.info("视觉选图拒绝水印状态为 %s 的候选图 %s",
                        watermark_status or "missing", index)
            continue
        # 自相矛盾的输出（判 none 但抄出了叠加文字）同样拒绝：
        # 实测这正是水印图漏网时的典型形态。
        if overlay:
            logger.info("视觉选图拒绝自报叠加内容「%s」的候选图 %s",
                        overlay[:60], index)
            continue
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
            "watermark_status": "none",
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
# 全局视觉调用并发闸门：该 endpoint 并发能力很弱（实测 4~6 并发就大量失败），
# 选图调用和水印复核调用共用同一个闸门，避免两者叠加把请求挤挂。
# 拿不到闸门时请求排队而不是失败，代价是可预期的变慢。
_VISION_SEMAPHORE = threading.Semaphore(_MAX_VISION_PARALLEL)
# 水印复核的调用重试次数（瞬时限流/网络抖动重试，避免误丢好图）
_VERIFY_ATTEMPTS = 3
# 复核输出上限：模型要逐字抄写画面所有文字，太小会把输出截断，
# 截断的 JSON 解析失败后会被当成「有水印」丢掉干净图。
_VERIFY_MAX_TOKENS = 1500
# 上游网关故障码：与请求内容无关的偶发失败（实测端到端跑批时出现），
# 整批重试即可，不要缩小批次——缩批会让后面几张图白丢。
_GATEWAY_RETRY_STATUSES = frozenset({502, 503, 504})
# 单次视觉选图的调用重试次数（429 退避重试；批次减半由上层处理）
_SELECT_ATTEMPTS = 3


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


def _vision_post(payload: dict, max_attempts: int = 1) -> dict:
    """发一次视觉请求（带全局并发闸门 + 限流退避），返回响应 JSON。

    所有视觉调用都必须走这里：闸门限制同时在途的请求数，避免选图和复核
    叠加把并发很弱的 endpoint 挤挂。

    重试策略只覆盖**瞬时**故障：429（该端点按时间窗限流，顺序调用也会中招）
    和网络抖动。其他 4xx 立即抛出——400/413 意味着请求本身有问题，
    要交给调用方做语义降级（例如减半图片批次），在这里闷头重试没意义。
    """
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with _VISION_SEMAPHORE:
                resp = httpx.post(
                    f"{_vision_base()}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {_vision_key()}"},
                    timeout=_VISION_TIMEOUT,
                )
                if resp.status_code == 429:
                    last_error = httpx.HTTPStatusError(
                        "429 Too Many Requests", request=resp.request, response=resp)
                    if attempt < max_attempts - 1:
                        delay = _retry_after(resp) or (2.0 * (2 ** attempt))
                        logger.info("视觉调用被限流，%.1fs 后重试（第 %d/%d 次）",
                                    delay, attempt + 1, max_attempts)
                        time.sleep(delay)
                        continue
                    break
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError:
            raise               # 非 429 的 HTTP 错误：交给调用方做语义降级
        except Exception as exc:  # noqa: BLE001 — 网络层瞬时故障，可重试
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(1.0 * (attempt + 1))
    raise last_error if last_error else RuntimeError("视觉请求失败")


def _overlay_detected(overlay_text: str) -> bool:
    """判断复核抄写内容是否代表画面存在叠加水印。

    模型被要求「没有叠加内容时填空字符串」，但仍会返回各种否定表述，
    例如 "none"、"无"、"None"、"没有水印"。这些必须判定为「无水印」，
    否则会把干净图全部误杀。
    """
    text = overlay_text.strip().strip("。.、,，:：;；").lower()
    if not text:
        return False
    return text not in {
        "none", "null", "n/a", "无", "没有", "暂无", "未发现", "无叠加内容",
        "无叠加文字", "没有水印", "未发现水印", "无文字", "空", '""', "''",
    }


def verify_watermark(raw: bytes, mime: str) -> bool:
    """独立复核单张图是否有水印，返回 True 表示**确认无水印**。

    fail-closed：任何异常、解析失败、字段缺失、抄出内容都返回 False（拒绝）。
    与选图调用分开单发一图，专问水印，避免复合任务里模型对水印判断敷衍。
    """
    content_parts: list[dict] = [
        {"type": "text", "text": VERIFY_WATERMARK_PROMPT},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
            },
        },
    ]
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": content_parts}],
        "temperature": 0,
        "max_tokens": _VERIFY_MAX_TOKENS,
    }
    try:
        data = _vision_post(payload, max_attempts=_VERIFY_ATTEMPTS)
        usage = data.get("usage") or {}
        add_tokens(usage)
        _record_vision_observation(payload, data)
        text = data["choices"][0]["message"]["content"]
        parsed = parse_llm_json(text)
        if not isinstance(parsed, dict) or "overlay_text" not in parsed:
            # 输出不合约（例如只回了散文）：保守视为有叠加内容
            logger.warning("水印复核输出不合格式，按有水印处理: %s", str(text)[:80])
            return False
        overlay = str(parsed.get("overlay_text") or "")
        if _overlay_detected(overlay):
            logger.info("水印复核拒绝：抄出叠加内容「%s」", overlay.strip()[:80])
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — 复核失败一律 fail-closed
        logger.warning("水印复核调用失败，按有水印处理（拒绝该图）: %s", str(exc)[:120])
        return False


def _vision_select_with_retry(question: str,
                              images: list[tuple[dict, bytes, str]]) -> list[dict]:
    """带自动降级重试的视觉选图。

    - 上游网关故障（502/503/504）：整批重试，不缩小批次——这类错误与图多少
      无关，实测端到端跑批时偶发出现，重试通常能过
    - 请求过大（400/413）或限流（429）：减半图片批次重试（每张图最多试 3 次）
    - 输出 JSON 截断/解析失败：把 max_tokens 翻倍到 2048 重试一次
    - 全部失败返回 []（该子问题无插图，不向调用方抛异常）
    """
    batch = images[:_MAX_VISION_BATCH]
    max_tokens = 1024
    gateway_retries = 0
    for _ in range(4):
        try:
            return _vision_select(question, batch, max_tokens=max_tokens)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in _GATEWAY_RETRY_STATUSES and gateway_retries < 2:
                gateway_retries += 1
                logger.warning("视觉请求遇到上游故障（HTTP %s），整批重试 %d/2",
                               status, gateway_retries)
                time.sleep(1.5 * gateway_retries)
                continue
            if status in {400, 413, 429} and len(batch) > 1:
                logger.warning("视觉请求过大（HTTP %s），图片批次减半重试（%d→%d 张）",
                               status, len(batch), max(1, len(batch) // 2))
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


def _supplement_with_generated(*, selected_groups: list[list[dict]],
                               group_ids: list[str],
                               plans: dict[str, dict],
                               question_by_id: dict[str, str],
                               skipped: set[str]) -> list[list[dict]]:
    """为 structure / decorative 需求生成示意图（返回分组后的列表）。

    只有 kind=generate 的位置会生图。搜图型需求（real_asset / data_viz）
    即使一张都没搜到也不生成：生成「看起来像官方公告/基准图表」的图等于
    伪造证据，这类位置宁可不配图。

    不触发：kind=skip、生图未配置、已达 MAX_GENERATED_IMAGES 上限。
    """
    if not image_gen_enabled() or MAX_GENERATED_IMAGES <= 0:
        return selected_groups

    # selected_groups 与 group_ids 一一对应；同时覆盖「完全没有候选图、
    # 因此没进 by_subq」但在计划里被要求生图的子问题。
    index_by_id = {group_id: index for index, group_id in enumerate(group_ids)}
    buckets: list[list[dict]] = [list(group) for group in selected_groups]
    for group_id in group_ids:
        if group_id not in index_by_id:  # pragma: no cover - 由构造方式保证
            continue

    extra_ids = [sq_id for sq_id in plans
                 if sq_id not in index_by_id and sq_id not in skipped]
    for sq_id in extra_ids:
        index_by_id[sq_id] = len(buckets)
        buckets.append([])

    tasks: list[tuple[str, str, str]] = []  # (sq_id, prompt, want)
    for sq_id, index in index_by_id.items():
        if sq_id in skipped:
            continue
        plan = plans.get(sq_id) or {}
        kind = plan.get("kind", "search")
        if kind != "generate":
            # 搜图型需求（real_asset / data_viz）即使一张都没搜到，也不生图兜底：
            # 生成一张「看起来像官方公告/基准图表」的图等于伪造证据。
            # 这类位置宁可不配图。
            continue
        want = str(plan.get("want", "") or "")
        tasks.append((sq_id, _build_generation_prompt(
            question_by_id.get(sq_id, ""), want, plan.get("reason", ""),
            need=str(plan.get("need", "") or ""),
        ), want))

    if not tasks:
        return buckets

    tasks = tasks[:MAX_GENERATED_IMAGES]
    logger.info("为 %d 个子问题生成示意图", len(tasks))

    def _generate(item: tuple[str, str, str]) -> dict | None:
        sq_id, prompt, want = item
        made = generate_image(prompt)
        if not made:
            return None
        made["sub_question_id"] = sq_id
        made["query"] = "generated"
        made["kind"] = "generated"  # 不依赖 generate_image 的实现细节
        made["url"] = made.get("cached_url") or made.get("url", "")
        made["alt"] = _generated_alt(question_by_id.get(sq_id, ""), prompt, want)
        return made

    with ThreadPoolExecutor(max_workers=min(len(tasks), 3)) as pool:
        for item, made in zip(tasks, pool.map(_generate, tasks)):
            if made:
                buckets[index_by_id[item[0]]].append(made)
    return buckets


def _build_generation_prompt(question: str, want: str, reason: str,
                             need: str = "structure") -> str:
    """拼出生图提示词。按需求类型走两套不同规范。

    structure（精细流程图/架构图）：要求结构准确、中文标签齐全、层级分明。
    decorative（好看配图）：不要求标签，重在意象与视觉美感。

    实测结论（gpt-image-2.5）：
    - 中文标签渲染准确，是技术图传递信息的主要手段，structure 必须写全；
    - 构图越具体（分层、模块形态、箭头走向、标签文字）细节越丰富；
      只给「一张 XX 示意图」这种抽象描述，模型只能画通用图标拼盘。
    """
    if need == "decorative":
        parts = ["绘制一张用于中文技术文章的配图插画，现代扁平风格，视觉上悦目。"]
        if want:
            parts.append(f"\n【画面意象】\n{want}")
        if question:
            parts.append(f"\n【配图所处语境】\n{question}")
        parts.append(
            "\n【画面规范】\n"
            "- 以简洁的形状、层次和留白构成画面，不要堆砌元素；\n"
            "- 不要求出现任何文字标签，画面本身说明意象即可；\n"
            "- 配色和谐：选一组同色系主色，最多一个对比色点缀，可使用柔和渐变。"
        )
        parts.append(
            "\n【禁止】\n"
            "- 不要画照片写实的人脸、真实产品外观或具体品牌 logo；\n"
            "- 不要出现水印、署名、网址、二维码；\n"
            "- 不要写任何文字。"
        )
        return "\n".join(parts)

    parts = ["绘制一张用于中文技术文章的信息示意图，扁平化矢量插画风格，非写实摄影。"]
    if want:
        parts.append(f"\n【构图要求】\n{want}")
    if question:
        parts.append(f"\n【这张图要帮助读者理解的内容】\n{question}")
    parts.append(
        "\n【画面规范】\n"
        "- 所有模块、节点、箭头都要有明确含义，避免装饰性元素堆砌；\n"
        "- 模块内标注简洁的中文标签（每处 2~6 字），标签直接取自技术术语，"
        "确保渲染清晰、位置居中、不重叠；\n"
        "- 统一线宽、圆角矩形、充足留白，层级分明；\n"
        "- 配色克制：一个主色（如蓝）+ 一个强调色（如橙）+ 中性灰，不使用渐变和高光。"
    )
    parts.append(
        "\n【禁止】\n"
        "- 不要画照片写实的人脸、真实产品外观或具体品牌 logo；\n"
        "- 不要出现水印、署名、网址、二维码；\n"
        "- 不要写大段说明文字或段落，图上的文字只能是短标签。"
    )
    return "\n".join(parts)


def _generated_alt(question: str, prompt: str, want: str = "") -> str:
    """示意图的 alt/题注。优先用 want 的第一句（更贴近画面实际内容）。"""
    source = (want or question or "").strip()
    if not source:
        return "示意图"
    # 取第一个分句，避免把整段构图指令塞进题注
    for separator in ("。", "；", "\n", "，", "、"):
        head = source.split(separator, 1)[0]
        if head:
            source = head
            break
    source = source.strip().rstrip("？?。.")
    if not source:
        return "示意图"
    return f"示意图：{source[:20]}"


def plans_by_sub_question(plans: list[dict]) -> dict[str, dict]:
    """把配图计划列表转成 sub_question_id → plan 索引。"""
    return {
        str(plan.get("sub_question_id", "")): plan
        for plan in (plans or []) if plan.get("sub_question_id")
    }


def _generate_only(sub_questions: list[dict],
                   plans: dict[str, dict]) -> list[dict]:
    """视觉选图不可用时的退化路径：只为规划为 generate 的子问题生图。

    没有计划时不猜——返回空，保持「配图功能未配置就无图」的旧行为。
    """
    if not image_gen_enabled() or MAX_GENERATED_IMAGES <= 0:
        return []
    question_by_id = {
        str(sq.get("id", "")): str(sq.get("question", ""))
        for sq in sub_questions if sq.get("id")
    }
    tasks = [
        (sq_id, plan) for sq_id, plan in plans.items()
        if plan.get("kind") == "generate" and sq_id in question_by_id
    ][:MAX_GENERATED_IMAGES]
    if not tasks:
        return []

    def _generate(item: tuple[str, dict]) -> dict | None:
        sq_id, plan = item
        want = str(plan.get("want", "") or "")
        prompt = _build_generation_prompt(
            question_by_id.get(sq_id, ""), want, plan.get("reason", ""),
            need=str(plan.get("need", "") or ""),
        )
        made = generate_image(prompt)
        if not made:
            return None
        made.update({
            "sub_question_id": sq_id,
            "query": "generated",
            "kind": "generated",
            "url": made.get("cached_url") or made.get("url", ""),
            "alt": _generated_alt(question_by_id.get(sq_id, ""), prompt, want),
        })
        return made

    selected: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), 3)) as pool:
        for made in pool.map(_generate, tasks):
            if made:
                selected.append(made)
    if selected:
        logger.info("视觉选图未配置，直接生成 %d 张示意图", len(selected))
    return selected


def select_images(sub_questions: list[dict],
                  image_candidates: list[dict],
                  image_plans: list[dict] | None = None) -> list[dict]:
    """从候选图中选图（每子问题一次视觉调用），并为结构类需求生成示意图。

    Args:
        sub_questions: 规划出的子问题 [{id, question, ...}]
        image_candidates: searcher 聚合的候选 [{url, sub_question_id, query}]
        image_plans: 配图计划 [{sub_question_id, need, kind, want}]。为空时
            退化为纯搜图行为（与旧版完全一致）。
            kind=search  → 走视觉选图；选不到就不配图（不生成，防伪造）
            kind=generate → 走生图（structure/decorative）
            kind=skip    → 不做任何配图

    Returns:
        [{url, alt, sub_question_id, query, kind}]；功能未配置或全部失败时返回 []。
        kind 取值：search（搜图命中）或 generated（生图）。
    """
    plans = plans_by_sub_question(image_plans or [])
    if not VISION_MODEL or not _vision_key():
        # 视觉选图关闭时仍可生图（生图不依赖视觉模型），只是全部走生成分支。
        return _generate_only(sub_questions, plans)

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

    # 规划为「直接生图」的子问题不需要下载候选图，先挑出来并行处理。
    generate_queue = [sq_id for sq_id, plan in plans.items()
                      if plan.get("kind") == "generate" and sq_id in question_by_id]
    skipped = {sq_id for sq_id, plan in plans.items() if plan.get("kind") == "skip"}

    if not by_subq and not generate_queue:
        return []

    def _select_for_subq(item: tuple[str, list[dict]]) -> list[dict]:
        """单个子问题的选图流程（下载 + 视觉调用），供线程池并行。"""
        sq_id, candidates = item
        if sq_id in skipped:
            return []
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
            # 二次复核：入选图逐张独立再问一次水印。选图调用要在同一请求里
            # 兼顾相关性、alt 和水印，实测会敷衍水印判断；复核单图单问才可靠。
            # 缓存也放在复核之后——没通过复核的图不进缓存，避免污染复用。
            verified: list[dict] = []
            for selected_image in picked:
                downloaded = downloaded_by_url.get(selected_image.get("url", ""))
                if not downloaded:
                    continue
                raw, mime = downloaded
                if not verify_watermark(raw, mime):
                    logger.info("二次复核未通过，丢弃候选图 %s",
                                selected_image.get("url", "")[:60])
                    continue
                selected_image["cached_url"] = _cache_image(raw, mime)
                selected_image["kind"] = "search"
                verified.append(selected_image)
            if verified:
                logger.info("子问题 %s 选中 %d 张插图", sq_id, len(verified))
            return verified
        except Exception as exc:
            logger.warning("视觉选图失败（子问题 %s），该子问题无插图: %s", sq_id, exc)
            return []

    # 子问题之间并行（≤3 并发，pool.map 保持顺序，截断/去重不受影响）
    selected_groups: list[list[dict]] = []
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=min(_MAX_VISION_PARALLEL, len(by_subq))) as pool:
            for picked in pool.map(_select_for_subq, list(by_subq.items())):
                selected_groups.append(picked)

    # 兜底：搜图没结果、或该子问题被规划为生图的，补生成示意图。
    # 生图数量受 MAX_GENERATED_IMAGES 硬上限约束，避免把一次研究拖慢到不可接受。
    generated_groups = _supplement_with_generated(
        selected_groups=list(selected_groups),
        group_ids=list(by_subq.keys()),
        plans=plans,
        question_by_id=question_by_id,
        skipped=skipped,
    )

    # 按子问题轮询取图，避免前几个分支各占 3 张后把后续章节全部截掉。
    selected: list[dict] = []
    for position in range(MAX_IMAGES_PER_SUB_QUESTION):
        for group in generated_groups:
            if position < len(group):
                selected.append(group[position])
                if len(selected) >= MAX_TOTAL_IMAGES:
                    return selected

    return selected


if __name__ == "__main__":
    # 冒烟测试：python -m research_buddy.tools.images
    print("select_images =", select_images([], []))
