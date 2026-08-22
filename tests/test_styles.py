"""写作风格预设测试。"""

from research_buddy.styles import DEFAULT_STYLE, STYLES, resolve_style, get_style_section


def test_default_style_is_tech_blog():
    assert DEFAULT_STYLE == "tech-blog"
    assert "tech-blog" in STYLES


def test_resolve_style_valid_and_fallback():
    assert resolve_style("essay") == "essay"
    assert resolve_style("report") == "report"
    # 未知/缺失 → 回退默认
    assert resolve_style("nonexistent") == DEFAULT_STYLE
    assert resolve_style(None) == DEFAULT_STYLE
    assert resolve_style("") == DEFAULT_STYLE


def test_get_style_section_returns_style_prompt():
    assert get_style_section("essay") == STYLES["essay"]["prompt"]
    assert "犀利" in get_style_section("essay")
    # 未知风格回退默认
    assert get_style_section("bad") == STYLES[DEFAULT_STYLE]["prompt"]


def test_all_styles_have_label_and_prompt():
    for sid, meta in STYLES.items():
        assert meta["label"]
        assert meta["prompt"]
        assert sid in {"tech-blog", "report", "essay", "popular"}
