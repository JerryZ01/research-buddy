#!/usr/bin/env python
"""审计并清除图片缓存中的水印图。

背景：早期选图 prompt 用「watermark_status 是不是 none」这种封闭式分类问
模型，实测会被橡皮图章式地答成 none——带「知乎 @北方的郎」署名的图就这么
混进了缓存。改成「先逐字抄写画面叠加内容再判定」后检出率大幅改善，这个脚本
用同一套证据优先问法复查存量缓存。

判定分三层，只对**硬信号**动手：
- 硬信号水印（删）：作者署名/账号（知乎@、公众号、微博@、掘金@、博客园@）、
  网址域名、素材库或版权标记（©、®、版权归、图库名）。这些必然是叠加标识。
- 疑似叠加（报告，不删）：抄出了内容但匹配不上硬信号。实测包含大量误报——
  AWS 架构图的层级标签、PPT 页标题、图表免责声明都会被抄出来。
- 干净：抄不出任何叠加内容。

三种结果分得很清楚：
- 干净：复核抄不出任何叠加内容 → 保留
- 硬信号水印：→ 默认删除
- 调用失败：重试后仍失败 → **不删**（无法判断），报告出来人工决定

用法：
    uv run python scripts/audit_image_watermarks.py             # 审计并删除硬信号水印
    uv run python scripts/audit_image_watermarks.py --dry-run   # 只报告不删除
    uv run python scripts/audit_image_watermarks.py --purge-articles   # 同时清理文章引用
    uv run python scripts/audit_image_watermarks.py --workers 1 # 调并发（端点限流严）

退出码：0 正常；1 未配置视觉模型；2 存在调用失败（结果不完整，未处理失败的图）。
"""

import argparse
import base64
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_buddy.config import (  # noqa: E402
    OPENAI_API_KEY, VISION_API_KEY, VISION_MODEL,
)
from research_buddy.utils import parse_llm_json  # noqa: E402
from research_buddy.tools.images import (  # noqa: E402
    IMAGE_CACHE_DIR, VERIFY_WATERMARK_PROMPT, _overlay_detected, _vision_post,
)

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
}
_ATTEMPTS = 3

CLEAN, DIRTY, SUSPECT, ERROR = "clean", "dirty", "suspect", "error"

# 硬信号：命中任一即判为叠加水印（作者标识/来源域名/素材库与版权标记）
_HARD_SIGNAL = re.compile(
    r"知乎|公众号|微博|小红书|掘金|CSDN|博客园|简书|豆瓣|B站|bilibili"
    r"|https?://|www\.|\.com|\.cn|\.ai|/articles/|/blog/"
    r"|版权所有|版权归|©|®|素材|图库|视觉中国|图虫|Shutterstock|Getty|veerb"
    r"|@[A-Za-z一-鿿]",
    re.IGNORECASE,
)
# 营销/引流短语：中转站和素材站的水印常带这些，配合重复出现即为平铺水印
_PROMO_SIGNAL = re.compile(
    r"注册就送|扫码|加微信|免费领取|点击领取|限时|福利|最全最便宜"
)
# 同一段文字在画面里重复出现多少次算平铺水印
_REPEAT_THRESHOLD = 3


def _hard_signal(overlay: str) -> bool:
    """抄出的内容是否命中硬水印信号。"""
    if _HARD_SIGNAL.search(overlay):
        return True
    # 平铺水印：营销短语在抄写结果里反复出现（模型会把重复水印逐条抄出）
    if _PROMO_SIGNAL.search(overlay):
        return True
    return False


def _audit_one(path: Path) -> tuple[Path, str, str]:
    """复查单张图，返回 (路径, 结论, 抄出的内容或错误)。"""
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
    if not mime:
        return path, ERROR, "未知扩展名"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return path, ERROR, f"读取失败: {exc}"

    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": VERIFY_WATERMARK_PROMPT},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 512,
    }
    last_error = ""
    for _ in range(_ATTEMPTS):
        try:
            data = _vision_post(payload, max_attempts=_ATTEMPTS)
            text = data["choices"][0]["message"]["content"]
            parsed = parse_llm_json(text)
            if not isinstance(parsed, dict) or "overlay_text" not in parsed:
                # 输出不合约：无法确认干净，但也不是抄出了水印 → 报错交人工
                return path, ERROR, f"输出不合格式: {str(text)[:60]}"
            overlay = str(parsed.get("overlay_text") or "").strip()
            if not _overlay_detected(overlay):
                return path, CLEAN, ""
            if _hard_signal(overlay):
                return path, DIRTY, overlay[:120]
            return path, SUSPECT, overlay[:120]
        except Exception as exc:  # noqa: BLE001 — 重试后仍失败就报告
            last_error = str(exc)[:100]
    return path, ERROR, last_error


def _purge_article_references(cache_names: set[str]) -> int:
    """从已生成文章里删掉引用这些缓存图的 Markdown 行，返回改动文章数。"""
    import sqlite3

    from research_buddy.config import DATA_DIR
    db_path = Path(DATA_DIR) / "research_buddy.db"
    if not db_path.exists():
        print(f"数据库不存在，跳过文章清理: {db_path}")
        return 0

    # Markdown 图片语法：![alt](/media/research-images/xxx.ext)
    pattern = re.compile(r"!\[[^\]]*\]\([^)]*/media/research-images/([^)/]+)[^)]*\)")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    changed = 0
    try:
        # article_generations.report 是当前稿；article_versions.report 是阶段快照
        for table in ("article_generations", "article_versions"):
            rows = connection.execute(
                f"select id, report from {table} where report like '%/media/research-images/%'"
            ).fetchall()
            for row in rows:
                report = row["report"] or ""

                def drop(match: re.Match) -> str:
                    return "" if match.group(1) in cache_names else match.group(0)

                cleaned = pattern.sub(drop, report)
                # 顺带清掉引用被删图片的裸链接行，避免留下空行堆
                if cleaned != report:
                    connection.execute(
                        f"update {table} set report = ? where id = ?", (cleaned, row["id"])
                    )
                    changed += 1
        connection.commit()
    finally:
        connection.close()
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只报告，不删除")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发数（端点限流很严，默认 1）")
    parser.add_argument("--cache-dir", default=None, help="覆盖缓存目录")
    parser.add_argument("--purge-articles", action="store_true",
                        help="删图前先清理已生成文章里对它们的引用（推荐）")
    args = parser.parse_args()

    if not VISION_MODEL or not (VISION_API_KEY or OPENAI_API_KEY):
        print("VISION_MODEL / VISION_API_KEY 未配置，无法审计", file=sys.stderr)
        return 1

    cache_dir = Path(args.cache_dir) if args.cache_dir else IMAGE_CACHE_DIR
    if not cache_dir.is_dir():
        print(f"缓存目录不存在: {cache_dir}")
        return 0

    files = sorted(p for p in cache_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in _MIME_BY_SUFFIX)
    if not files:
        print(f"缓存目录没有图片: {cache_dir}")
        return 0

    print(f"审计 {len(files)} 张图片（{cache_dir}），并发 {args.workers}...")
    dirty: list[tuple[Path, str]] = []
    suspects: list[tuple[Path, str]] = []
    errors: list[tuple[Path, str]] = []
    clean = 0
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (path, verdict, note) in enumerate(
                pool.map(_audit_one, files), start=1):
            if verdict == DIRTY:
                dirty.append((path, note))
            elif verdict == SUSPECT:
                suspects.append((path, note))
            elif verdict == ERROR:
                errors.append((path, note))
            else:
                clean += 1
            if index % 25 == 0 or index == len(files):
                print(f"  {index}/{len(files)}  干净 {clean}  硬信号水印 {len(dirty)}  "
                      f"疑似 {len(suspects)}  失败 {len(errors)}  "
                      f"用时 {time.monotonic() - start:.0f}s")

    if dirty:
        print(f"\n硬信号水印 {len(dirty)} 张（将删除）：")
        for path, note in dirty:
            print(f"  {path.name}\t{note}")
    if suspects:
        print(f"\n疑似叠加、未命中硬信号 {len(suspects)} 张（保留，供人工复核）：")
        for path, note in suspects:
            print(f"  {path.name}\t{note}")
    if errors:
        print(f"\n调用失败、未判定 {len(errors)} 张（建议稍后重跑）：")
        for path, note in errors:
            print(f"  {path.name}\t{note}")

    print(f"\n汇总：干净 {clean}，硬信号水印 {len(dirty)}，"
          f"疑似 {len(suspects)}，失败 {len(errors)}")

    if args.dry_run:
        print("--dry-run：未改动任何文件")
        return 2 if errors else 0

    if dirty and args.purge_articles:
        names = {path.name for path, _ in dirty}
        changed = _purge_article_references(names)
        print(f"已清理 {changed} 篇文章中对这些图片的引用")

    removed = 0
    for path, _note in dirty:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            print(f"删除失败 {path.name}: {exc}", file=sys.stderr)
    print(f"已删除 {removed} 张水印图，缓存剩余 {len(files) - removed} 张")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
