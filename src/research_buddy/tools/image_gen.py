"""生图工具 — 为搜图覆盖不到的语义位置生成「示意性」插图

定位（与搜图的分工）：
- 搜图负责「事实性图像」：人物、产品、新闻现场、真实数据图表、界面截图。
  这类内容必须是真实素材，生成出来就是编造。
- 生图负责「解释性图像」：机制、流程、架构、对比关系、概念示意。
  这类内容本来就该由作者画出来，生成比搜来的图更贴切。
  生成的图一律标注为「示意图」，不冒充事实。

降级链（任何一步失败都不阻塞出稿）：
- IMAGE_GEN_MODEL / 密钥未配置 → 返回空串（生图整体关闭）
- HTTP 报错（401/404/400/5xx）或超时 → 打 WARNING 返回空串，该处无图
- 返回数据既不是 b64_json 也不是可用 URL → 返回空串

接口：OpenAI 兼容 `POST {base}/images/generations`，
请求体 {model, prompt, n=1, size?, response_format?}。
不同中转站对 response_format / size 的支持不一，这里做能力自适应：
先按配置发；遇到 400 且错误信息提到这些参数，去掉对应字段重试一次。
"""

import base64
import hashlib
import logging
import threading
import time
from pathlib import Path

import httpx

from research_buddy.config import (
    DATA_DIR,
    IMAGE_GEN_API_BASE,
    IMAGE_GEN_API_KEY,
    IMAGE_GEN_MODEL,
    IMAGE_GEN_SIZE,
    IMAGE_GEN_TIMEOUT,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

IMAGE_CACHE_DIR = Path(DATA_DIR) / "research-images"

_MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
# OpenAI 兼容接口的 response_format 常见两种返回；有的站忽略该字段直接给 URL。
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def is_enabled() -> bool:
    """生图是否可用（模型 + 密钥都配了）。"""
    return bool(IMAGE_GEN_MODEL and _api_key())


def _api_base() -> str:
    """生图 API base：IMAGE_GEN_API_BASE 优先，未配回退 OPENAI_API_BASE。"""
    return (IMAGE_GEN_API_BASE or OPENAI_API_BASE).rstrip("/")


def _api_key() -> str:
    """生图 API key：IMAGE_GEN_API_KEY 优先，未配回退 OPENAI_API_KEY。"""
    return IMAGE_GEN_API_KEY or OPENAI_API_KEY


def _cache_image_bytes(data: bytes) -> str:
    """按内容哈希缓存生成的图片，返回同源静态 URL；失败返回空串。

    与 tools/images.py 的缓存目录、URL 前缀保持一致，前端只需要一套静态路由。
    """
    if not data:
        return ""
    try:
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(data).hexdigest()[:24] + ".png"
        target = IMAGE_CACHE_DIR / filename
        if not target.exists():
            temporary = target.with_name(f".{filename}.{threading.get_ident()}.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
        return f"/media/research-images/{filename}"
    except OSError as exc:
        logger.warning("缓存生成图片失败: %s", exc)
        return ""


def _decode_image(payload: dict) -> bytes:
    """从接口返回体里取出图片字节，兼容 b64_json 与 url 两种返回。"""
    items = payload.get("data") or []
    if not isinstance(items, list) or not items:
        return b""
    first = items[0]
    if not isinstance(first, dict):
        return b""

    encoded = first.get("b64_json") or first.get("b64") or ""
    if isinstance(encoded, str) and encoded.strip():
        try:
            return base64.b64decode(encoded)
        except (ValueError, TypeError) as exc:
            logger.warning("生成图片 base64 解码失败: %s", exc)
            return b""

    url = first.get("url") or ""
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        try:
            resp = httpx.get(url, timeout=IMAGE_GEN_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001 - 兜底，任何失败都只要 None
            logger.warning("下载生成图片失败: %s", str(exc)[:120])
            return b""
    return b""


def _post(prompt: str, *, size: str | None, with_format: bool,
          signal_timeout: float) -> httpx.Response:
    """发一次生图请求。参数可裁剪，供能力自适应重试复用。"""
    body: dict = {"model": IMAGE_GEN_MODEL, "prompt": prompt, "n": 1}
    if size:
        body["size"] = size
    if with_format:
        body["response_format"] = "b64_json"
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    return httpx.post(
        f"{_api_base()}/images/generations",
        json=body,
        headers=headers,
        timeout=signal_timeout,
    )


def _error_mentions(resp: httpx.Response, *keywords: str) -> bool:
    """400 类错误是否与给定参数有关（用于判断该裁剪哪个字段）。"""
    try:
        text = resp.text or ""
    except Exception:  # noqa: BLE001
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def generate_image(prompt: str) -> dict | None:
    """生成一张示意图并缓存到本地。

    Args:
        prompt: 生图提示词。调用方负责确保它是「示意图」性质的描述，
            不含真实人物姓名、真实数据等会诱导模型编造事实的内容。

    Returns:
        {"cached_url", "alt", "kind": "generated", "prompt"}；失败返回 None。
    """
    prompt = (prompt or "").strip()
    if not prompt or not is_enabled():
        return None

    size = IMAGE_GEN_SIZE or None
    # 第一轮：size + response_format 都带；第二轮：去掉 response_format；
    # 第三轮：连 size 也去掉（部分中转站只接受 model + prompt + n）。
    attempts: list[tuple[str | None, bool]] = [
        (size, True),
        (size, False),
        (None, False),
    ]

    started = time.monotonic()
    last_reason = ""
    for index, (attempt_size, with_format) in enumerate(attempts):
        try:
            resp = _post(prompt, size=attempt_size, with_format=with_format,
                         signal_timeout=IMAGE_GEN_TIMEOUT)
        except httpx.TimeoutException:
            last_reason = f"超时（{IMAGE_GEN_TIMEOUT:.0f}s）"
            # 超时通常是服务端仍在生成，重试价值低且会翻倍等待，直接放弃。
            break
        except Exception as exc:  # noqa: BLE001 - 网络/连接错误
            last_reason = str(exc)[:120]
            if index + 1 < len(attempts):
                continue
            break

        if resp.status_code < 400:
            try:
                payload = resp.json()
            except ValueError:
                last_reason = "返回体不是 JSON"
                break
            data = _decode_image(payload if isinstance(payload, dict) else {})
            if not data:
                last_reason = "返回体没有可用图片"
                break
            cached = _cache_image_bytes(data)
            if not cached:
                last_reason = "缓存写入失败"
                break
            logger.info("生成示意图成功（%.1fs，%d KB）",
                        time.monotonic() - started, len(data) // 1024)
            return {
                "cached_url": cached,
                "url": cached,
                "alt": "",
                "kind": "generated",
                "prompt": prompt,
            }

        last_reason = f"HTTP {resp.status_code}: {resp.text[:120]}"
        # 只有「参数不被接受」这一类 400 才值得裁剪重试；鉴权/额度问题直接放弃。
        if resp.status_code == 400 and index + 1 < len(attempts):
            if with_format and _error_mentions(resp, "response_format", "format"):
                continue
            if attempt_size and _error_mentions(resp, "size", "resolution", "dimension"):
                continue
        if resp.status_code in _RETRYABLE_STATUS and index + 1 < len(attempts):
            time.sleep(min(2.0 * (index + 1), 6.0))
            continue
        break

    logger.warning("生成示意图失败，该处不配图: %s", last_reason)
    return None


if __name__ == "__main__":
    # 冒烟测试：python -m research_buddy.tools.image_gen
    print("enabled =", is_enabled())
    print("model =", IMAGE_GEN_MODEL or "(未配置)")
