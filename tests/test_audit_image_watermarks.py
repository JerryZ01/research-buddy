"""存量图片审计脚本的水印判定测试。

脚本不在包内（scripts/ 不是模块），用 importlib 按路径加载。
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_image_watermarks.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_image_watermarks", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("overlay", [
    "知乎 @北方的郎",
    "公众号：JavaGuide",
    "公众号：JavaGuide\n网站：javaguide.cn",
    "CSDN @人工智能AI技术",
    "掘金技术社区@机器之心",
    "by xiaolinnote.com@小林面试笔记",
    "https://blog.csdn.net/pony1001",
    "https://martinfowler.com/articles/harness-engineering.html",
    "“蒋麟霖”版权所有",
    "Blockchain Council®",
    "知乎@算法小陈",
    "KAWEEI®",
    "Site: 视觉中国 图库",
])
def test_hard_signal_detects_attribution_watermarks(audit, overlay):
    """作者署名/账号/网址/版权/图库标记 → 硬信号水印，可删。"""
    assert audit._hard_signal(overlay) is True


@pytest.mark.parametrize("overlay", [
    "AWS Cloud; AWS",                 # 架构图分层标签
    "Web Layer; Orchestration Layer; S3 Bucket",
    "RAG",                            # 大标题
    "OpenAI",
    "打开终端，输入 claude 命令",       # 界面/标注文字
    "Charts provided herein are for informational purposes only",
    "INTRODUCING GPT-6\nASTRA",
    "Two windows: chat above, prompt below",
])
def test_hard_signal_ignores_in_image_content(audit, overlay):
    """图表标题/图层标签/界面文字 → 不是水印，保留。

    实测这些是模型从正常架构图、PPT 页、界面截图里抄出来的内容，
    早期版本把它们全部判成水印，64 张里约 54 张是这种误报。
    """
    assert audit._hard_signal(overlay) is False


def test_hard_signal_ignores_percent_template_placeholders(audit):
    """模板占位符里的 % 不能被当成信号，且不应误伤。"""
    assert audit._hard_signal("A% %B% C%") is False
