"""配图规划质量回归测试。

锁定核心设计约束：
1. **图源由 need 推导，LLM 无权决定** —— 消除「要真实素材却去生成」的矛盾
2. **搜图型需求（real_asset/data_viz）搜不到就不配图，绝不生图兜底** —— 防伪造证据
3. **structure 与 decorative 走两套不同 prompt** —— 结构图要标签，配图要美感
4. **真实人名/数值不得进 want** —— 提示词层面的约束
"""

import pytest

from research_buddy.nodes import image_planner
from research_buddy.tools import images as images_module


@pytest.fixture(autouse=True)
def _gen_enabled(monkeypatch):
    monkeypatch.setattr(image_planner, "image_gen_enabled", lambda: True)


def _subs(*ids):
    return [{"id": i, "question": f"问题 {i}"} for i in ids]


class TestNeedToKindMapping:
    """need → kind 是固定映射表，这是全模块唯一的图源判断来源。"""

    def test_mapping_is_complete(self):
        assert image_planner.NEED_TO_KIND == {
            "real_asset": "search",
            "data_viz": "search",
            "structure": "generate",
            "decorative": "generate",
            "none": "skip",
        }

    @pytest.mark.parametrize("need,expected", [
        ("real_asset", "search"),
        ("data_viz", "search"),
        ("structure", "generate"),
        ("decorative", "generate"),
        ("none", "skip"),
    ])
    def test_kind_derived_from_need(self, need, expected):
        assert image_planner.kind_for_need(need) == expected

    def test_unknown_need_is_skip(self):
        assert image_planner.kind_for_need("") == "skip"
        assert image_planner.kind_for_need("bogus") == "skip"


class TestLLMCannotChooseKind:
    """LLM 即使输出矛盾的 kind，也必须被 need 覆盖。"""

    def test_llm_kind_field_ignored(self):
        plans = image_planner.normalize_plans(
            {"plans": [{
                "id": "sq_01",
                "need": "real_asset",      # 需要真实素材
                "kind": "generate",        # LLM 却说要生成（矛盾）
                "want": "官方公告页",
            }]},
            _subs("sq_01"),
        )
        assert plans[0]["kind"] == "search"
        assert plans[0]["never_generate"] is True

    def test_real_asset_cannot_generate(self):
        plans = image_planner.normalize_plans(
            {"plans": [{"id": "sq_01", "need": "real_asset"}]}, _subs("sq_01"),
        )
        assert plans[0]["kind"] == "search"
        assert plans[0]["never_generate"] is True

    def test_data_viz_cannot_generate(self):
        plans = image_planner.normalize_plans(
            {"plans": [{"id": "sq_01", "need": "data_viz"}]}, _subs("sq_01"),
        )
        assert plans[0]["kind"] == "search"
        assert plans[0]["never_generate"] is True

    def test_structure_can_generate(self):
        plans = image_planner.normalize_plans(
            {"plans": [{"id": "sq_01", "need": "structure"}]}, _subs("sq_01"),
        )
        assert plans[0]["kind"] == "generate"
        assert plans[0]["never_generate"] is False

    def test_decorative_can_generate(self):
        plans = image_planner.normalize_plans(
            {"plans": [{"id": "sq_01", "need": "decorative"}]}, _subs("sq_01"),
        )
        assert plans[0]["kind"] == "generate"

    def test_legacy_kind_format_still_accepted(self):
        """兼容旧格式：LLM 只给 kind 时映射到对应 need。"""
        plans = image_planner.normalize_plans(
            {"plans": [{"id": "sq_01", "kind": "search"}]}, _subs("sq_01"),
        )
        assert plans[0]["need"] == "real_asset"
        assert plans[0]["kind"] == "search"


class TestGenerateUnavailable:
    def test_generate_need_skipped_without_image_gen(self, monkeypatch):
        monkeypatch.setattr(image_planner, "image_gen_enabled", lambda: False)
        plans = image_planner.normalize_plans(
            {"plans": [{"id": "sq_01", "need": "structure"}]}, _subs("sq_01"),
        )
        assert plans[0]["kind"] == "skip"


class TestNoFabricationFallback:
    """搜图型需求搜不到时，绝不能生图兜底。"""

    def test_search_plan_never_generates_when_empty(self, monkeypatch):
        monkeypatch.setattr(images_module, "VISION_MODEL", "vision-model")
        monkeypatch.setattr(images_module, "_vision_key", lambda: "key")
        monkeypatch.setattr(images_module, "_download_image",
                            lambda client, url: (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "image/png"))
        monkeypatch.setattr(images_module, "_vision_select_with_retry",
                            lambda question, images: [])
        monkeypatch.setattr(images_module, "generate_image",
                            lambda prompt: pytest.fail("搜图型需求不得生图兜底"))

        plans = [{
            "sub_question_id": "sq_01", "need": "real_asset",
            "kind": "search", "never_generate": True, "want": "官方公告页",
        }]
        picked = images_module.select_images(
            _subs("sq_01"),
            [{"url": "https://img.example/a.png", "sub_question_id": "sq_01"}],
            plans,
        )
        assert picked == []

    def test_data_viz_never_generates(self, monkeypatch):
        monkeypatch.setattr(images_module, "VISION_MODEL", "vision-model")
        monkeypatch.setattr(images_module, "_vision_key", lambda: "key")
        monkeypatch.setattr(images_module, "_download_image",
                            lambda client, url: (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "image/png"))
        monkeypatch.setattr(images_module, "_vision_select_with_retry",
                            lambda question, images: [])
        monkeypatch.setattr(images_module, "generate_image",
                            lambda prompt: pytest.fail("数据图表不得生成"))

        plans = [{
            "sub_question_id": "sq_01", "need": "data_viz",
            "kind": "search", "never_generate": True, "want": "基准成绩柱状图",
        }]
        assert images_module.select_images(
            _subs("sq_01"),
            [{"url": "https://img.example/a.png", "sub_question_id": "sq_01"}],
            plans,
        ) == []


class TestStructurePrompt:
    def test_structure_prompt_includes_want(self):
        want = "环形流程图。中央六边形标注『策略模型』。"
        prompt = images_module._build_generation_prompt("问题", want, "机制", need="structure")
        assert want in prompt

    def test_structure_prompt_requires_labels(self):
        prompt = images_module._build_generation_prompt("问题", "构图", "理由", need="structure")
        assert "中文标签" in prompt
        assert "不要出现任何中文或英文文字标签" not in prompt

    def test_structure_prompt_forbids_watermark(self):
        prompt = images_module._build_generation_prompt("问题", "构图", "理由", need="structure")
        assert "水印" in prompt
        assert "写实" in prompt


class TestDecorativePrompt:
    def test_decorative_prompt_includes_want(self):
        want = "层叠的半透明山峦表示知识积累"
        prompt = images_module._build_generation_prompt("问题", want, "配图", need="decorative")
        assert want in prompt

    def test_decorative_prompt_does_not_demand_labels(self):
        prompt = images_module._build_generation_prompt("问题", "意象", "配图", need="decorative")
        assert "不要求出现任何文字标签" in prompt
        # 不应复用 structure 的标签规范
        assert "模块内标注简洁的中文标签" not in prompt

    def test_decorative_allows_gradient(self):
        prompt = images_module._build_generation_prompt("问题", "意象", "", need="decorative")
        assert "渐变" in prompt


class TestGeneratedAlt:
    def test_alt_uses_want_first_clause(self):
        alt = images_module._generated_alt(
            "子问题文本", "prompt", "环形流程图。中央六边形标注策略模型",
        )
        assert alt == "示意图：环形流程图"

    def test_alt_falls_back_to_question(self):
        assert images_module._generated_alt("什么是缓存穿透", "prompt", "").startswith("示意图：")

    def test_alt_handles_empty(self):
        assert images_module._generated_alt("", "prompt", "") == "示意图"
