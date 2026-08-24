"""AI 问题润色端点测试（全离线，mock LLM）。

- 成功：纯文本行格式返回多条候选（[轻度]/[深度] 标记）
- LLM 失败：返回单条原问题候选（辅助功能不阻塞主流程）
- 垃圾输出 / 空输入：降级或 400
"""

from fastapi.testclient import TestClient

from research_buddy import api

client = TestClient(api.app)


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def invoke(self, _prompt, **_kwargs):
        return _Resp(
            "[轻度] 2025年国内新能源车市场整体情况如何？\n"
            "[深度] 2025年国内新能源车市场中，主流增程式与纯电动车型在续航、补能效率与用车成本上的差异如何？\n"
            "[深度] 2025年国内新能源车市场中，不同价格区间车型的市场份额与增长趋势如何？\n"
        )


def test_refine_question_returns_multiple_candidates(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(api, "create_llm", lambda **kw: fake)
    monkeypatch.setattr(api, "invoke_llm", lambda llm, prompt, **kw: llm.invoke(prompt))
    res = client.post("/research/refine-question", json={"question": "国内新能源车啥情况"})
    assert res.status_code == 200
    data = res.json()
    assert len(data["candidates"]) == 3
    # 主体必须保留：所有候选都围绕「新能源车」
    for c in data["candidates"]:
        assert "新能源车" in c["refined_question"]
    styles = {c["style"] for c in data["candidates"]}
    assert styles == {"轻度", "深度"}
    assert data["candidates"][0]["style"] == "轻度"


def test_parse_candidates_accepts_bare_lines():
    """无 [标记] 的行（如模型漏了前缀）也能解析成轻度候选，且跳过无效短行。"""
    parsed = api._parse_refine_candidates(
        "1. 新能源车整体情况如何？\n\n废话\n[深度] 新能源车续航对比如何？\n"
    )
    assert len(parsed) == 2
    assert parsed[0].refined_question == "新能源车整体情况如何？"
    assert parsed[0].style == "轻度"
    assert parsed[1].refined_question == "新能源车续航对比如何？"
    assert parsed[1].style == "深度"


def test_refine_question_falls_back_to_original_on_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(api, "create_llm", lambda **kw: object())
    monkeypatch.setattr(api, "invoke_llm", boom)
    res = client.post("/research/refine-question", json={"question": "什么是GIL"})
    assert res.status_code == 200
    data = res.json()
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["refined_question"] == "什么是GIL"
    assert data["candidates"][0]["style"] == "原样"


def test_refine_question_falls_back_when_llm_returns_garbage(monkeypatch):
    class _GarbageLLM:
        def invoke(self, _prompt, **_kwargs):
            return _Resp("嗯嗯，好的呢")

    monkeypatch.setattr(api, "create_llm", lambda **kw: _GarbageLLM())
    monkeypatch.setattr(api, "invoke_llm", lambda llm, prompt, **kw: llm.invoke(prompt))
    res = client.post("/research/refine-question", json={"question": "什么是GIL"})
    assert res.status_code == 200
    assert res.json()["candidates"][0]["refined_question"] == "什么是GIL"


def test_refine_question_empty_rejected():
    res = client.post("/research/refine-question", json={"question": "   "})
    assert res.status_code == 400
