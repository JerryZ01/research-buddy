"""配图终审节点测试。

核心保证（决定这一层是正向收益还是负资产）：
1. **不确定就保留** —— 视觉失败/解析失败/低置信一律 keep，绝不误删
2. **只做减法** —— 不新增图、不重新搜图
3. **矛盾删图不改字** —— 图文数字冲突时删图，正文保持原样
4. **失败不阻断出稿** —— 任何异常都降级为全部保留
"""

import pytest

from research_buddy.nodes import image_reviewer


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(image_reviewer, "_review_enabled", lambda: True)


def _state(report: str, images: list[dict]) -> dict:
    return {"report": report, "selected_images": images}


REPORT_TEMPLATE = """# 标题

Terminal-Bench 得分 65.7%，超过竞品。

![基准对比图](https://img.example/chart.png)

后面还有一段正文继续说明数据来源和测试口径。

![架构示意图](https://img.example/arch.png)

结尾段落。
"""

IMAGES = [
    {"url": "https://img.example/chart.png", "alt": "基准对比图",
     "sub_question_id": "sq_01", "kind": "search"},
    {"url": "https://img.example/arch.png", "alt": "架构示意图",
     "sub_question_id": "sq_02", "kind": "generated"},
]


def _mock_verdict(monkeypatch, verdict_for: dict):
    """按 url（或 cached_url）返回预设裁定。"""
    def fake_review(image, context):
        for key in (image.get("url"), image.get("cached_url")):
            if key and key in verdict_for:
                return verdict_for[key]
        return {"action": "keep", "reason": "通过"}
    monkeypatch.setattr(image_reviewer, "_review_one", fake_review)


class TestKeepOnUncertainty:
    """不确定必须保留——这是整层的安全底线。"""

    def test_vision_disabled_keeps_all(self, monkeypatch):
        monkeypatch.setattr(image_reviewer, "_review_enabled", lambda: False)
        result = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        assert "report" not in result  # 不改正文
        assert "selected_images" not in result  # 不改图片列表

    def test_review_exception_keeps_image(self, monkeypatch):
        """单图复核抛错 → 该图保留，其余照常处理，绝不误删。"""
        def boom(image, context):
            raise RuntimeError("视觉服务挂了")
        monkeypatch.setattr(image_reviewer, "_review_one", boom)
        result = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        # 两张都复核失败 → 都保留，正文不变
        assert "report" not in result
        assert "通过" in result["messages"][0]

    def test_whole_node_exception_degrades(self, monkeypatch):
        """整层异常也不能拖垮出稿。"""
        monkeypatch.setattr(image_reviewer, "_image_index",
                            lambda report, images: (_ for _ in ()).throw(
                                RuntimeError("意外崩溃")))
        result = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        assert "异常" in result["messages"][0]
        assert "report" not in result

    def test_parse_failure_keeps_image(self, monkeypatch):
        """视觉输出不可解析 → keep（在 _review_one 内部处理）。"""
        monkeypatch.setattr(image_reviewer, "_load_image_bytes",
                            lambda image: (b"\x89PNG\r\n\x1a\n", "image/png"))
        monkeypatch.setattr(image_reviewer, "_vision_post",
                            lambda payload, max_attempts=1: {
                                "choices": [{"message": {"content": "不是JSON"}}]})
        verdict = image_reviewer._review_one(IMAGES[0], "上下文")
        assert verdict["action"] == "keep"

    def test_vision_call_failure_keeps_image(self, monkeypatch):
        monkeypatch.setattr(image_reviewer, "_load_image_bytes",
                            lambda image: (b"\x89PNG\r\n\x1a\n", "image/png"))

        def boom(payload, max_attempts=1):
            raise RuntimeError("502")
        monkeypatch.setattr(image_reviewer, "_vision_post", boom)
        verdict = image_reviewer._review_one(IMAGES[0], "上下文")
        assert verdict["action"] == "keep"

    def test_unreadable_image_keeps(self, monkeypatch):
        monkeypatch.setattr(image_reviewer, "_load_image_bytes", lambda image: None)
        verdict = image_reviewer._review_one(IMAGES[0], "上下文")
        assert verdict["action"] == "keep"

    def test_numbers_unclear_keeps(self, monkeypatch):
        monkeypatch.setattr(image_reviewer, "_load_image_bytes",
                            lambda image: (b"\x89PNG\r\n\x1a\n", "image/png"))
        monkeypatch.setattr(image_reviewer, "_vision_post",
                            lambda payload, max_attempts=1: {"choices": [{"message": {
                                "content": '{"relevance":"match","numbers":"numbers_unclear",'
                                           '"alt_ok":true,"reason":"数字看不清"}'}}]})
        assert image_reviewer._review_one(IMAGES[0], "上下文")["action"] == "keep"


class TestDropOnlyOnHardSignals:
    """只有明确的错配/数字冲突才删。"""

    def test_numbers_conflict_drops(self, monkeypatch):
        monkeypatch.setattr(image_reviewer, "_load_image_bytes",
                            lambda image: (b"\x89PNG\r\n\x1a\n", "image/png"))
        monkeypatch.setattr(image_reviewer, "_vision_post",
                            lambda payload, max_attempts=1: {"choices": [{"message": {
                                "content": '{"relevance":"match","numbers":"numbers_conflict",'
                                           '"alt_ok":true,"reason":"图上是 42%，正文写 65.7%"}'}}]})
        verdict = image_reviewer._review_one(IMAGES[0], "上下文")
        assert verdict["action"] == "drop"
        assert "数字冲突" in verdict["reason"]

    def test_mismatch_drops(self, monkeypatch):
        monkeypatch.setattr(image_reviewer, "_load_image_bytes",
                            lambda image: (b"\x89PNG\r\n\x1a\n", "image/png"))
        monkeypatch.setattr(image_reviewer, "_vision_post",
                            lambda payload, max_attempts=1: {"choices": [{"message": {
                                "content": '{"relevance":"mismatch","numbers":"numbers_ok",'
                                           '"alt_ok":true,"reason":"正文讲架构，图是价格表"}'}}]})
        assert image_reviewer._review_one(IMAGES[0], "上下文")["action"] == "drop"

    def test_fix_alt(self, monkeypatch):
        monkeypatch.setattr(image_reviewer, "_load_image_bytes",
                            lambda image: (b"\x89PNG\r\n\x1a\n", "image/png"))
        monkeypatch.setattr(image_reviewer, "_vision_post",
                            lambda payload, max_attempts=1: {"choices": [{"message": {
                                "content": '{"relevance":"match","numbers":"numbers_ok",'
                                           '"alt_ok":false,"better_alt":"六大基准横向对比",'
                                           '"reason":"原题注不够具体"}'}}]})
        verdict = image_reviewer._review_one(IMAGES[0], "上下文")
        assert verdict["action"] == "fix_alt"
        assert verdict["better_alt"] == "六大基准横向对比"

    def test_overlong_better_alt_ignored(self, monkeypatch):
        """更准确的题注太长（>40 字）不采用，保持原题注。"""
        monkeypatch.setattr(image_reviewer, "_load_image_bytes",
                            lambda image: (b"\x89PNG\r\n\x1a\n", "image/png"))
        long_alt = "很长的题注" * 20
        monkeypatch.setattr(image_reviewer, "_vision_post",
                            lambda payload, max_attempts=1: {"choices": [{"message": {
                                "content": '{"relevance":"match","numbers":"numbers_ok",'
                                           '"alt_ok":false,"better_alt":"' + long_alt + '",'
                                           '"reason":"x"}'}}]})
        assert image_reviewer._review_one(IMAGES[0], "上下文")["action"] == "keep"


class TestReportEditing:
    def test_drop_removes_image_from_report(self, monkeypatch):
        _mock_verdict(monkeypatch, {
            "https://img.example/chart.png": {
                "action": "drop", "reason": "图文数字冲突"},
        })
        result = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        assert "chart.png" not in result["report"]
        assert "arch.png" in result["report"]
        assert "Terminal-Bench 得分 65.7%" in result["report"]  # 正文不动
        urls = [i["url"] for i in result["selected_images"]]
        assert "https://img.example/chart.png" not in urls
        assert "https://img.example/arch.png" in urls

    def test_fix_alt_rewrites_caption(self, monkeypatch):
        _mock_verdict(monkeypatch, {
            "https://img.example/arch.png": {
                "action": "fix_alt", "better_alt": "MoE 稀疏激活结构",
                "reason": "原题注不准"},
        })
        result = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        assert "![MoE 稀疏激活结构](https://img.example/arch.png)" in result["report"]
        assert len(result["selected_images"]) == 2  # 一张都没删

    def test_all_keep_leaves_report_untouched(self, monkeypatch):
        _mock_verdict(monkeypatch, {})
        result = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        assert "report" not in result
        assert "通过" in result["messages"][0]

    def test_drop_all_images(self, monkeypatch):
        _mock_verdict(monkeypatch, {
            "https://img.example/chart.png": {"action": "drop", "reason": "错配"},
            "https://img.example/arch.png": {"action": "drop", "reason": "错配"},
        })
        result = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        assert "chart.png" not in result["report"]
        assert "arch.png" not in result["report"]
        assert result["selected_images"] == []


class TestSkipConditions:
    def test_no_report(self):
        assert "跳过" in image_reviewer.image_reviewer(
            {"report": "", "selected_images": IMAGES})["messages"][0]

    def test_no_images(self):
        assert "跳过" in image_reviewer.image_reviewer(
            {"report": REPORT_TEMPLATE, "selected_images": []})["messages"][0]

    def test_no_image_in_report(self):
        """图片列表有图，但正文里一张都没插入（写作者没用）。"""
        report = "# 标题\n\n没有任何图片的正文。\n"
        assert "跳过" in image_reviewer.image_reviewer(
            _state(report, IMAGES))["messages"][0]


class TestReflectionLoopDedup:
    """反思回环时不应重复复核同一张图（视觉接口并发弱，重审只会拖慢出稿）。"""

    def test_passed_image_not_reviewed_again(self, monkeypatch):
        calls = []

        def counting(image, context):
            calls.append(image["url"])
            return {"action": "keep", "reason": "通过"}

        monkeypatch.setattr(image_reviewer, "_review_one", counting)

        # 第一轮：两张都通过
        first = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        assert len(calls) == 2

        # 第二轮（反思回环）：首批图已审过 → 不再调视觉
        calls.clear()
        second = image_reviewer.image_reviewer({
            "report": REPORT_TEMPLATE,
            "selected_images": IMAGES,
            "image_review": first["image_review"],
            "reflection_round": 1,
        })
        assert calls == []
        assert "跳过" in second["messages"][0]

    def test_dropped_image_not_reviewed_again(self, monkeypatch):
        calls = []

        def mixed(image, context):
            calls.append(image["url"])
            if image["url"].endswith("chart.png"):
                return {"action": "drop", "reason": "数字冲突"}
            return {"action": "keep", "reason": "通过"}

        monkeypatch.setattr(image_reviewer, "_review_one", mixed)
        first = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        assert len(first["image_review"]["dropped"]) == 1

        # 第二轮：删掉的那张已不在正文里，剩下的已审过 → 不再复核
        calls.clear()
        second = image_reviewer.image_reviewer({
            "report": first["report"],
            "selected_images": first["selected_images"],
            "image_review": first["image_review"],
            "reflection_round": 1,
        })
        assert calls == []
        # 历史删除记录必须保留，否则 selected_images 会被下一轮重新放回来
        assert len(second["image_review"]["dropped"]) == 1


class TestSurvivingImages:
    def test_dropped_image_removed_from_selected(self, monkeypatch):
        _mock_verdict(monkeypatch, {
            "https://img.example/chart.png": {"action": "drop", "reason": "错配"},
        })
        result = image_reviewer.image_reviewer(_state(REPORT_TEMPLATE, IMAGES))
        assert [i["url"] for i in result["selected_images"]] == [
            "https://img.example/arch.png"]

    def test_cached_url_matches(self, monkeypatch):
        images = [
            {"cached_url": "/media/research-images/a.png", "alt": "x", "url": ""},
            {"cached_url": "/media/research-images/b.png", "alt": "y", "url": ""},
        ]
        report = ("正文\n\n![x](/media/research-images/a.png)\n\n更多\n\n"
                  "![y](/media/research-images/b.png)\n")
        _mock_verdict(monkeypatch, {
            "/media/research-images/a.png": {"action": "drop", "reason": "错配"},
        })
        result = image_reviewer.image_reviewer({"report": report,
                                                "selected_images": images})
        assert len(result["selected_images"]) == 1
        assert result["selected_images"][0]["cached_url"].endswith("b.png")


class TestImageIndexing:
    def test_locates_by_url_tail(self):
        located = image_reviewer._image_index(REPORT_TEMPLATE, IMAGES)
        assert len(located) == 2
        assert [item[0] for item in located] == [0, 1]

    def test_matches_cached_url(self):
        images = [{"cached_url": "/media/research-images/abc.png", "alt": "x"}]
        report = "正文\n\n![x](/media/research-images/abc.png)\n"
        located = image_reviewer._image_index(report, images)
        assert len(located) == 1

    def test_ignores_query_string(self):
        images = [{"url": "https://img.example/a.png"}]
        report = "![x](https://img.example/a.png?w=800)"
        assert len(image_reviewer._image_index(report, images)) == 1

    def test_context_excludes_other_images(self):
        located = image_reviewer._image_index(REPORT_TEMPLATE, IMAGES)
        _i, start, end, _url = located[0]
        context = image_reviewer._extract_context(REPORT_TEMPLATE, start, end)
        assert "arch.png" not in context
        assert "Terminal-Bench" in context
