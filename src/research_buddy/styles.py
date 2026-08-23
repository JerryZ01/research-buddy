"""写作风格预设 — 前端可选，注入 synthesizer prompt 的文风要求

核心质量要求（主线/深度/可信度/归因节制/技术图解/因题而异）在
synthesizer 的三个 prompt 里共享；这里只定义每种风格的「文风」差异层，
其中「口语化」是去 AI 味的重要信号——按风格分级，但只作引导不作硬性要求。
"""

DEFAULT_STYLE = "tech-blog"

# 口语化的通用使用原则（附加到每种风格，防止硬凑）
_COLLOQUIAL_GUARD = (
    "口语化要自然：只用在真会这么说的位置（承接转折、强调观点、拉近距离），"
    "不要每段硬塞、不要为了口语打断专业论述的流畅性；"
    "整篇保持语言流畅，突兀的口语词比没有更糟。"
)


def _style(flavor: str, colloquial: str) -> str:
    """拼接风格文风 + 口语化档位说明 + 通用原则。"""
    return f"{flavor}\n口语化：{colloquial}\n{_COLLOQUIAL_GUARD}"


STYLES: dict[str, dict] = {
    "tech-blog": {
        "label": "专业技术博客",
        "prompt": _style(
            "专业、克制、自信的技术博客文风：长短句交错，避免排比句堆叠；"
            "用词直白准确，不用「赋能/底层逻辑/闭环/颗粒度」等套话；"
            "直接陈述立场，不用「在我看来」式弱化语气；"
            "重点章节写透，次要章节可以一笔带过。",
            "轻度。允许自然的口语化过渡（「说白了」「关键在」「但问题是」），"
            "全文两三处即可，保持专业底线的同时更像一个真实作者在说话。",
        ),
    },
    "report": {
        "label": "深度研究报告",
        "prompt": _style(
            "行业研究机构风格的正式报告（类似券商研报 / Gartner 报告）："
            "数据与证据密度高，每个论断尽量给出数字或出处；小节标题正式规范；"
            "语气客观严谨，下结论谨慎但明确；"
            "适合政策、产业、趋势类话题，读者是决策者与研究者。",
            "极低。保持正式研报语体，尽量避免口语词；"
            "仅在极少数需要强调处用「关键在」「实质是」这类书面化的强调语。",
        ),
    },
    "essay": {
        "label": "观点锐评",
        "prompt": _style(
            "立场鲜明、敢下判断的评论文风，可以适度犀利；"
            "口语化连接词自然出现（「但问题在于」「说白了」「就这么简单」）；"
            "允许适度主观表达；短句多、节奏快；不端着学术腔。",
            "高。自然地用口语短句把观点说透（「你可能会说」「问题就出在这」），"
            "敢用大白话，但保持犀利和节奏，不啰嗦。",
        ),
    },
    "popular": {
        "label": "通俗科普",
        "prompt": _style(
            "面向普通读者的科普文风：多用类比和生活中的例子解释概念；"
            "尽量少术语，出现术语必解释；段落短、节奏轻快；语气亲切但不轻浮。",
            "高。用聊天式的口吻（「你可能听说过」「简单说」「别担心，其实不难」），"
            "多用「你」拉近距离，像朋友把一件事讲清楚；信息必须准确，不因口语而含糊。",
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
