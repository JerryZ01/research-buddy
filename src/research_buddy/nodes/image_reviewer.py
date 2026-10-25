"""配图终审节点 — 文章写完后逐图复核是否该保留

为什么需要这一层：
选图阶段只能看到「单张图 + 子问题」，看不到**图最终落在哪一段、和正文说了什么**。
实测暴露的问题：
- 图文数字矛盾：正文写「得分 65.7%」，插入的图表其实画的是别的数字，
  读者一眼看穿，但选图阶段无从判断；
- 张冠李戴：图被用到和它无关的段落（占位符顺序和正文节奏错位）；
- alt 泛化：`alt` 由视觉模型按子问题生成，跨段落使用时可能不再贴切；
- 冗余：同一子问题选 4 张同类图表，信息重复。

设计原则（决定「正向收益」能不能成立）：

1. **只做减法，不做加法**。终审判定只有 keep / drop / fix_alt 三种，
   绝不新增图、不重新搜图——那会让这一层变成第二个选图器，成本不可控。

2. **不确定就保留**。这是最关键的一条。删图是不可逆的净损失（读者少一张图），
   留一张弱图只是轻微减分。因此：
   - 视觉模型不可用 / 调用失败 → 整层跳过，全部保留；
   - 单图复核失败 → 该图保留；
   - 判定置信度低 → 保留。
   宁可漏删，不可误删。

3. **矛盾时删图，不改正文**。图片是外部素材，正文的证据链由更早的节点负责。
   为了迁就一张来源不明的图去改正文数字，是本末倒置，还可能引入事实错误。

4. **总预算兜底**。即使全部通过，也受 MAX_IMAGES_IN_ARTICLE 约束。

5. **失败不影响出稿**。整层任何异常都只能降级为「全部保留」，绝不能抛错。
"""

import base64
import logging
import re

from langchain_core.runnables import RunnableConfig

from research_buddy.config import MAX_IMAGES_IN_ARTICLE, VISION_MODEL
from research_buddy.state import ResearchState
from research_buddy.tools.images import (
    _download_image,
    _is_http_url,
    _vision_post,
    _vision_key,
)
from research_buddy.utils import parse_llm_json

logger = logging.getLogger(__name__)

IMAGE_REVIEW_PROMPT = """你是文章配图终审编辑。下面是一篇文章的正文片段，以及这段里插入的一张图片。

## 正文片段（图片就插在这里）
{context}

## 图片的题注（写作时给的说明）
{alt}

## 这张图原本服务于哪个子问题
{sub_question}

## 任务

先仔细看图，再回答三个问题：

1. **相关性**：这张图和这段正文讨论的内容是否对得上？如果正文在讲 A，图却在展示 B，就是 `mismatch`。
2. **数字一致性**：如果正文里出现了具体数字（分数、百分比、参数量、价格），
   图里是否也有数字？两者是否一致？
   - 图里没有数字，或正文没有数字 → `numbers_ok`
   - 图里的数字和正文一致 → `numbers_ok`
   - 图里的数字和正文**明显不一致** → `numbers_conflict`
   - 图里数字看不清、或无法确认 → `numbers_unclear`
3. **题注准确性**：题注描述的内容和图上实际画的是否相符？不符就给出更准确的题注（20 字内）。

## 判定标准

- 只有**明确**的错配或**明确**的数字冲突才判 `drop`。看不懂、看不清、拿不准，一律 `keep`。
- 图片是示意图（题注带「示意图」）时，只要它表达的机制/结构与正文一致就算通过，
  不要因为画得不精确而判 drop。

## 输出格式
只返回 JSON（不要其他内容）：
{{"relevance": "match|mismatch", "numbers": "numbers_ok|numbers_conflict|numbers_unclear", "alt_ok": true/false, "better_alt": "更准确的题注，alt_ok 为 true 时留空", "reason": "一句话说明"}}"""

# 判定为要删的来源：只有这两种硬信号才允许删图。
_DROP_RELEVANCE = "mismatch"
_DROP_NUMBERS = "numbers_conflict"

# 正文取图前后各多少字符作为上下文（够看清这段在讲什么，又不会撑爆 prompt）
_CONTEXT_WINDOW = 700


def _review_enabled() -> bool:
    """终审需要读图能力；视觉模型未配置时整层跳过。"""
    return bool(VISION_MODEL and _vision_key())


def _image_markdown_regex() -> re.Pattern[str]:
    return re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def _extract_context(report: str, start: int, end: int) -> str:
    """取图片前后一段正文，去掉其中的其它图片标记，避免干扰判断。"""
    left = max(0, start - _CONTEXT_WINDOW)
    right = min(len(report), end + _CONTEXT_WINDOW)
    snippet = report[left:right]
    snippet = _image_markdown_regex().sub("", snippet)
    snippet = re.sub(r"\n{3,}", "\n\n", snippet).strip()
    return snippet


def _image_index(report: str, images: list[dict]) -> list[tuple[int, int, int, str]]:
    """在正文里定位每张已插入图片：返回 [(池下标, 起点, 终点, url)]。

    通过 URL 末段匹配，兼容 cached_url / 原 URL 两种写法。
    """
    by_tail: dict[str, int] = {}
    for index, image in enumerate(images):
        for key in (image.get("url", ""), image.get("cached_url", "")):
            key = str(key or "").strip()
            if key:
                by_tail[key.rsplit("/", 1)[-1].split("?")[0]] = index

    found: list[tuple[int, int, int, str]] = []
    for match in _image_markdown_regex().finditer(report):
        url = match.group(2).strip()
        tail = url.rsplit("/", 1)[-1].split("?")[0]
        index = by_tail.get(tail)
        if index is None:
            continue
        found.append((index, match.start(), match.end(), url))
    return found


def _load_image_bytes(image: dict) -> tuple[bytes, str] | None:
    """取图片字节：站内缓存读本地文件，外链走下载。失败返回 None。"""
    from pathlib import Path

    from research_buddy.config import DATA_DIR

    cached = str(image.get("cached_url") or "").strip()
    if cached.startswith("/media/"):
        path = Path(DATA_DIR) / cached.replace("/media/", "", 1)
        if path.exists():
            suffix = path.suffix.lower()
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webp": "image/webp", ".gif": "image/gif"}.get(suffix, "image/png")
            try:
                return path.read_bytes(), mime
            except OSError:
                return None

    url = str(image.get("url") or "").strip()
    if not _is_http_url(url):
        return None
    try:
        import httpx
        with httpx.Client(timeout=20, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"}) as client:
            return _download_image(client, url)
    except Exception:  # noqa: BLE001 — 下载失败只是跳过该图复核
        return None


def _review_one(image: dict, context: str) -> dict:
    """复核单张图。返回 {action, better_alt, reason}；任何异常都返回 keep。"""
    payload_bytes = _load_image_bytes(image)
    if not payload_bytes:
        return {"action": "keep", "reason": "图片无法读取，跳过复核"}

    raw, mime = payload_bytes
    b64 = base64.b64encode(raw).decode("ascii")
    prompt = IMAGE_REVIEW_PROMPT.format(
        context=context,
        alt=str(image.get("alt", "")).strip() or "（无题注）",
        sub_question=str(image.get("sub_question") or "（未记录）"),
    )
    try:
        data = _vision_post({
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
            "temperature": 0,
            "max_tokens": 400,
        }, max_attempts=2)
        parsed = parse_llm_json(data["choices"][0]["message"]["content"])
    except Exception as exc:  # noqa: BLE001 — 复核失败一律保留
        logger.warning("配图终审调用失败，保留该图: %s", str(exc)[:120])
        return {"action": "keep", "reason": "复核失败，保留"}

    if not isinstance(parsed, dict):
        return {"action": "keep", "reason": "输出不可解析，保留"}

    relevance = str(parsed.get("relevance", "")).strip().lower()
    numbers = str(parsed.get("numbers", "")).strip().lower()
    reason = str(parsed.get("reason", "")).strip()[:160]

    # 数字冲突优先：这是最需要拦的问题，且改文字风险高于删图。
    if numbers == _DROP_NUMBERS:
        return {"action": "drop", "reason": f"图文数字冲突：{reason}"}
    if relevance == _DROP_RELEVANCE:
        return {"action": "drop", "reason": f"图文不匹配：{reason}"}

    better_alt = str(parsed.get("better_alt", "") or "").strip()
    if parsed.get("alt_ok") is False and better_alt and len(better_alt) <= 40:
        return {"action": "fix_alt", "better_alt": better_alt, "reason": reason}
    return {"action": "keep", "reason": reason or "通过"}


def image_reviewer(state: ResearchState,
                   config: RunnableConfig | None = None) -> dict:
    """LangGraph 节点：文章写完后逐图终审。

    只做减法：drop 掉明确错配/数字冲突的图，必要时修正题注，其余全部保留。
    任何失败都降级为「全部保留」，绝不阻断出稿。
    """
    try:
        return _review(state)
    except Exception as exc:  # noqa: BLE001 — 终审是增强环节，绝不能拖垮出稿
        logger.warning("配图终审异常，本次全部保留: %s", str(exc)[:160])
        return {"messages": ["配图终审异常，保留全部图片"]}


def _review(state: ResearchState) -> dict:
    report = str(state.get("report") or "")
    images = list(state.get("selected_images") or [])

    if not report or not images:
        return {"messages": ["配图终审跳过（无正文或无图）"]}
    if not _review_enabled():
        return {"messages": ["配图终审跳过（视觉模型未配置）"]}

    # 反思回环时本轮会再次进入：上一轮已审过的图（删掉的、改过题注的、
    # 通过的）不再重审。重审既无收益（同一张图 + 同一段正文），又会让每轮
    # 反思多花 N 次视觉调用，而视觉接口并发很弱，会把出稿明显拖慢。
    already = state.get("image_review") or {}
    reviewed_urls = {
        str(item.get("url", "")) for item in (already.get("dropped") or [])
    }
    reviewed_urls.update(str(key) for key in (already.get("alt_fixes") or {}))
    reviewed_urls.update(str(url) for url in (already.get("passed") or []))
    reviewed_urls.discard("")

    if reviewed_urls and str(state.get("reflection_round", 0)):
        logger.info("反思第 %s 轮：跳过已审过的 %d 张图",
                    state.get("reflection_round"), len(reviewed_urls))

    located = _image_index(report, images)
    located = [item for item in located if item[3] not in reviewed_urls]
    if not located:
        # 没有待审的图，但历史记录必须原样写回：下一步 synthesizer 会复用
        # selected_images，而 image_review 是它判断「哪些图已被删」的唯一依据。
        result: dict = {"messages": ["配图终审跳过（正文中没有待审的图）"]}
        if already:
            result["image_review"] = already
        return result

    dropped: list[dict] = []
    alt_fixes: dict[str, str] = {}
    passed_urls: list[str] = []

    # 逐张复核。串行执行：与选图共用视觉闸门，且张数本来就不多（≤10）。
    for index, start, end, url in located:
        image = images[index]
        context = _extract_context(report, start, end)
        try:
            verdict = _review_one(image, context)
        except Exception as exc:  # noqa: BLE001 — 单张失败不影响其它图
            logger.warning("单张配图复核失败，保留该图: %s", str(exc)[:120])
            passed_urls.append(url)
            continue
        if verdict["action"] == "drop":
            dropped.append({"url": url, "reason": verdict.get("reason", "")})
            logger.info("配图终审删除一张图：%s（%s）", url[:70],
                        verdict.get("reason", "")[:60])
        elif verdict["action"] == "fix_alt":
            alt_fixes[url] = verdict["better_alt"]
            logger.info("配图终审修正题注：%s → %s", url[:60],
                        verdict["better_alt"][:40])
        else:
            passed_urls.append(url)

    # 删除按偏移量从后往前做，避免前面的删除让后面的偏移量失效。
    dropped_urls = {item["url"] for item in dropped}
    new_report = report
    for _index, start, end, url in sorted(located, key=lambda item: item[1], reverse=True):
        if url in dropped_urls:
            new_report = new_report[:start] + new_report[end:]

    for url, new_alt in alt_fixes.items():
        new_report = re.sub(
            r"!\[[^\]]*\]\(" + re.escape(url) + r"\)",
            f"![{new_alt}]({url})",
            new_report,
        )
    if dropped or alt_fixes:
        new_report = re.sub(r"\n{3,}", "\n\n", new_report).strip() + "\n"

    # 只把还在正文里的图写回状态，避免图片列表与正文脱节。
    # 注意要合并历史已删的图（反思回环时上一轮删掉的图本轮不再出现于 located），
    # 否则下一轮 synthesizer 复用 selected_images 时会把已删的图放回正文。
    dead_tails = {url.rsplit("/", 1)[-1] for url in dropped_urls}
    dead_tails.update(
        str(item.get("url", "")).rsplit("/", 1)[-1]
        for item in (already.get("dropped") or [])
    )
    surviving = [
        image for image in images
        if str(image.get("cached_url") or image.get("url") or "").rsplit("/", 1)[-1]
        not in dead_tails
    ]

    summary = []
    if dropped:
        summary.append(f"删除 {len(dropped)} 张")
    if alt_fixes:
        summary.append(f"修正题注 {len(alt_fixes)} 处")

    updated_review = {
        "dropped": list(already.get("dropped") or []) + dropped,
        "alt_fixes": {**(already.get("alt_fixes") or {}), **alt_fixes},
        "passed": list(already.get("passed") or []) + passed_urls,
        "reviewed": int(already.get("reviewed", 0)) + len(located),
    }

    if not dropped and not alt_fixes:
        # 本轮没有删改，但有历史记录时要同步（否则下一轮图会回来）。
        if dead_tails:
            return {
                "selected_images": surviving[:MAX_IMAGES_IN_ARTICLE],
                "image_review": updated_review,
                "messages": [f"配图终审通过（{len(located)} 张）"],
            }
        return {
            "image_review": updated_review,
            "messages": [f"配图终审通过（{len(located)} 张）"],
        }

    return {
        "report": new_report,
        "selected_images": surviving[:MAX_IMAGES_IN_ARTICLE],
        "image_review": updated_review,
        "messages": ["🖼️ 配图终审：" + "、".join(summary)],
    }
