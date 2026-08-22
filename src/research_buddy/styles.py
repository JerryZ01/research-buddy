"""写作风格预设 — 前端可选，注入 synthesizer prompt 的文风要求

核心质量要求（主线/深度/可信度/归因节制/技术图解/因题而异）在
synthesizer 的三个 prompt 里共享；这里只定义每种风格的「文风」差异层。
"""

DEFAULT_STYLE = "tech-blog"

STYLES: dict[str, dict] = {
    "tech-blog": {
        "label": "专业技术博客",
        "prompt": (
            "专业、克制、自信的技术博客文风：长短句交错，避免排比句堆叠；"
            "用词直白准确，不用「赋能/底层逻辑/闭环/颗粒度」等套话；"
            "直接陈述立场，不用「在我看来」式弱化语气；"
            "重点章节写透，次要章节可以一笔带过。"
        ),
    },
    "report": {
        "label": "深度研究报告",
        "prompt": (
            "行业研究机构风格的正式报告（类似券商研报 / Gartner 报告）："
            "数据与证据密度高，每个论断尽量给出数字或出处；小节标题正式规范；"
            "语气客观严谨，下结论谨慎但明确；"
            "适合政策、产业、趋势类话题，读者是决策者与研究者。"
        ),
    },
    "essay": {
        "label": "观点锐评",
        "prompt": (
            "立场鲜明、敢下判断的评论文风，可以适度犀利；"
            "口语化连接词自然出现（「但问题在于」「说白了」「关键在于」）；"
            "允许适度主观表达；短句多、节奏快；不端着学术腔。"
        ),
    },
    "popular": {
        "label": "通俗科普",
        "prompt": (
            "面向普通读者的科普文风：多用类比和生活中的例子解释概念；"
            "尽量少术语，出现术语必解释；段落短、节奏轻快；语气亲切但不轻浮。"
        ),
    },
}


def resolve_style(style: str | None) -> str:
    """校验风格 id；未知或缺失回退默认。"""
    return style if style in STYLES else DEFAULT_STYLE


def get_style_section(style: str | None) -> str:
    """取风格文风要求文本（注入 synthesizer prompt 的 {style_section}）。"""
    return STYLES[resolve_style(style)]["prompt"]


def style_labels() -> dict[str, str]:
    """前端下拉选项：{id: label}。"""
    return {sid: meta["label"] for sid, meta in STYLES.items()}
