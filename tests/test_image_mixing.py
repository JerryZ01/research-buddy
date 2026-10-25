"""混合配图（搜图 + 生图）测试。

覆盖三块新增能力：
1. tools/image_gen.py — 生图客户端的降级与返回体兼容
2. nodes/image_planner.py — 配图意图的校验与降级
3. tools/images.py — select_images 的分流与生图兜底
"""

import base64

from importlib import import_module

import pytest

from research_buddy.nodes import image_planner as planner_module
from research_buddy.tools import image_gen as gen_module
from research_buddy.tools import images as images_module

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b"\x00\x00\x04\x00\x00\x00\x03\x00\x08\x02\x00\x00\x00"
)


@pytest.fixture(autouse=True)
def _enable_image_gen(monkeypatch):
    """默认让生图可用；需要关闭的用例自行覆盖。"""
    monkeypatch.setattr(gen_module, "IMAGE_GEN_MODEL", "test-image-model")
    monkeypatch.setattr(gen_module, "_api_key", lambda: "test-key")
    monkeypatch.setattr(images_module, "image_gen_enabled", lambda: True)


class TestImageGen:
    def test_disabled_without_model(self, monkeypatch):
        monkeypatch.setattr(gen_module, "IMAGE_GEN_MODEL", "")
        assert gen_module.is_enabled() is False
        assert gen_module.generate_image("示意图") is None

    def test_empty_prompt_returns_none(self):
        assert gen_module.generate_image("   ") is None

    def test_decodes_b64_json(self):
        payload = {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode()}]}
        assert gen_module._decode_image(payload) == PNG_BYTES

    def test_decodes_nothing_for_unknown_shape(self):
        assert gen_module._decode_image({}) == b""
        assert gen_module._decode_image({"data": []}) == b""
        assert gen_module._decode_image({"data": [{"nope": 1}]}) == b""

    def test_generate_success_is_cached(self, monkeypatch):
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode()}]}

        monkeypatch.setattr(gen_module, "_post", lambda *a, **k: FakeResponse())
        monkeypatch.setattr(gen_module, "_cache_image_bytes",
                            lambda data: "/media/research-images/fake.png")
        result = gen_module.generate_image("示意图：缓存机制")
        assert result is not None
        assert result["kind"] == "generated"
        assert result["cached_url"] == "/media/research-images/fake.png"

    def test_generate_failure_returns_none(self, monkeypatch):
        class FakeResponse:
            status_code = 401
            text = "unauthorized"

        monkeypatch.setattr(gen_module, "_post", lambda *a, **k: FakeResponse())
        assert gen_module.generate_image("示意图") is None

    def test_generate_survives_timeout(self, monkeypatch):
        import httpx

        def boom(*args, **kwargs):
            raise httpx.TimeoutException("timeout")

        monkeypatch.setattr(gen_module, "_post", boom)
        assert gen_module.generate_image("示意图") is None


class TestImagePlanner:
    def test_normalize_rejects_unknown_ids_and_kinds(self):
        sub_questions = [{"id": "sq_01", "question": "问题一"}]
        payload = {"plans": [
            {"id": "sq_01", "kind": "generate", "want": "流程图"},
            {"id": "sq_99", "kind": "generate", "want": "不存在"},
            {"id": "sq_01", "kind": "weird", "want": "重复+非法"},
        ]}
        plans = planner_module.normalize_plans(payload, sub_questions)
        assert len(plans) == 1
        assert plans[0]["sub_question_id"] == "sq_01"
        assert plans[0]["kind"] == "generate"

    def test_generate_degrades_to_skip_when_disabled(self, monkeypatch):
        """生图未配置时，structure/decorative 无处安放 → skip（不硬塞搜图）。"""
        monkeypatch.setattr(planner_module, "image_gen_enabled", lambda: False)
        plans = planner_module.normalize_plans(
            {"plans": [{"id": "sq_01", "need": "structure", "want": "示意图"}]},
            [{"id": "sq_01", "question": "?"}],
        )
        assert plans[0]["kind"] == "skip"

    def test_normalize_tolerates_garbage(self):
        assert planner_module.normalize_plans(None, []) == []
        assert planner_module.normalize_plans({"plans": "nope"}, []) == []
        assert planner_module.normalize_plans({"plans": [1, 2]}, []) == []

    def test_plans_empty_without_sub_questions(self):
        assert planner_module.build_image_plans({}) == []


class TestSelectImagesRouting:
    def _sub_questions(self):
        return [
            {"id": "sq_01", "question": "什么是缓存穿透？"},
            {"id": "sq_02", "question": "某公司 2024 年营收"},
        ]

    def test_generate_plan_uses_image_gen(self, monkeypatch):
        calls = []

        def fake_generate(prompt):
            calls.append(prompt)
            return {"cached_url": "/media/research-images/gen.png", "alt": ""}

        monkeypatch.setattr(images_module, "generate_image", fake_generate)
        monkeypatch.setattr(images_module, "VISION_MODEL", "")  # 视觉选图关闭
        plans = [{"sub_question_id": "sq_01", "kind": "generate",
                  "want": "缓存命中与未命中的两条路径", "reason": "机制"}]

        picked = images_module.select_images(self._sub_questions(), [], plans)

        assert len(picked) == 1
        assert picked[0]["kind"] == "generated"
        assert picked[0]["sub_question_id"] == "sq_01"
        assert picked[0]["url"] == "/media/research-images/gen.png"
        assert calls and "示意图" in calls[0]

    def test_skip_plan_produces_nothing(self, monkeypatch):
        monkeypatch.setattr(images_module, "VISION_MODEL", "")
        monkeypatch.setattr(images_module, "generate_image",
                            lambda prompt: pytest.fail("skip 不该触发生图"))
        plans = [{"sub_question_id": "sq_01", "kind": "skip"}]
        assert images_module.select_images(self._sub_questions(), [], plans) == []

    def test_real_asset_plan_never_falls_back_to_generate(self, monkeypatch):
        """real_asset 搜不到就空着——生成「看起来像官方页面」的图等于伪造证据。"""
        monkeypatch.setattr(images_module, "VISION_MODEL", "vision-model")
        monkeypatch.setattr(images_module, "_vision_key", lambda: "key")
        monkeypatch.setattr(images_module, "_vision_select_with_retry",
                            lambda question, images: [])
        monkeypatch.setattr(images_module, "_download_image",
                            lambda client, url: (PNG_BYTES, "image/png"))
        monkeypatch.setattr(images_module, "generate_image",
                            lambda prompt: pytest.fail("真实素材需求不得生图兜底"))

        candidates = [{"url": "https://img.example/a.png", "sub_question_id": "sq_01"}]
        plans = [{"sub_question_id": "sq_01", "need": "real_asset", "kind": "search",
                  "never_generate": True, "want": "官方公告页"}]

        assert images_module.select_images(self._sub_questions(), candidates, plans) == []

    def test_no_plans_keeps_legacy_behavior(self, monkeypatch):
        """不传 image_plans 时行为与改动前一致：只搜图，不生成。"""
        monkeypatch.setattr(images_module, "VISION_MODEL", "vision-model")
        monkeypatch.setattr(images_module, "_vision_key", lambda: "key")
        monkeypatch.setattr(images_module, "_download_image",
                            lambda client, url: (PNG_BYTES, "image/png"))
        monkeypatch.setattr(images_module, "verify_watermark", lambda raw, mime: True)
        monkeypatch.setattr(images_module, "_cache_image",
                            lambda raw, mime: "/media/research-images/a.png")
        monkeypatch.setattr(images_module, "_vision_select_with_retry",
                            lambda question, images: [{
                                "url": "https://img.example/a.png",
                                "alt": "架构图",
                                "sub_question_id": "sq_01",
                            }])
        monkeypatch.setattr(images_module, "generate_image",
                            lambda prompt: pytest.fail("无计划时不该生图"))

        candidates = [{"url": "https://img.example/a.png", "sub_question_id": "sq_01"}]
        picked = images_module.select_images(self._sub_questions(), candidates)

        assert len(picked) == 1
        assert picked[0]["kind"] == "search"


class TestGeneratedImageLabeling:
    def test_alt_marks_generated(self):
        syn = import_module("research_buddy.nodes.synthesizer")

        labeled = syn._render_image_alt({"alt": "缓存路径", "kind": "generated"})
        assert labeled.startswith("示意图")

        untouched = syn._render_image_alt({"alt": "架构图", "kind": "search"})
        assert untouched == "架构图"

    def test_prompt_section_marks_generated_images(self):
        syn = import_module("research_buddy.nodes.synthesizer")

        assert "示意图" in syn._image_kind_label({"kind": "generated"})
        assert syn._image_kind_label({"kind": "search"}) == ""
