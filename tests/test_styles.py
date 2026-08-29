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


def test_colloquialism_level_differs_by_style():
    """口语化按风格分级：研报极低、锐评/科普高，且都带「不硬凑」原则。"""
    assert "口语化：极低" in get_style_section("report")
    assert "口语化：轻度" in get_style_section("tech-blog")
    assert "口语化：高" in get_style_section("essay")
    assert "口语化：高" in get_style_section("popular")
    for sid in STYLES:
        assert "口语化要自然" in get_style_section(sid)
        assert "不要每段硬塞" in get_style_section(sid)


def test_plainness_level_differs_by_style():
    """朴实度按风格分级：科普最高、研报中、且都带「朴实原则」防华丽造词。"""
    assert "朴实度：最高" in get_style_section("popular")
    assert "朴实度：高" in get_style_section("essay")
    assert "朴实度：中高" in get_style_section("tech-blog")
    assert "朴实度：中" in get_style_section("report")
    for sid in STYLES:
        t = get_style_section(sid)
        assert "朴实原则" in t
        assert "逻辑严密" in t
        assert "不要生造词" in t or "生造词" in t


def test_popular_style_prefers_examples_over_frequent_analogies():
    prompt = get_style_section("popular")
    assert "优先用具体例子和分步解释" in prompt
    assert "才少量使用贴切类比" in prompt
    assert "多用类比" not in prompt
