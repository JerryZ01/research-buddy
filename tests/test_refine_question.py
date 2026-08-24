"""AI 问题润色端点测试（全离线，mock LLM）。

- 成功：返回润色后的问题 + 意图 + 润色点
- LLM 失败：原样返回输入（辅助功能不阻塞主流程）
- 空输入：400
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
            '{"refined_question": "中国新能源汽车市场2025年的发展现状与未来趋势分析", '
            '"intent": "趋势分析", "tips": ["补充了时间范围", "明确了分析对象"]}'
        )


def test_refine_question_success(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(api, "create_llm", lambda **kw: fake)
    monkeypatch.setattr(api, "invoke_llm", lambda llm, prompt, **kw: llm.invoke(prompt))
    res = client.post("/research/refine-question", json={"question": "国内新能源车啥情况"})
    assert res.status_code == 200
    data = res.json()
    assert "新能源汽车市场" in data["refined_question"]
    assert data["intent"] == "趋势分析"
    assert len(data["tips"]) == 2


def test_refine_question_falls_back_to_input_on_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(api, "create_llm", lambda **kw: object())
    monkeypatch.setattr(api, "invoke_llm", boom)
    res = client.post("/research/refine-question", json={"question": "什么是GIL"})
    assert res.status_code == 200
    assert res.json()["refined_question"] == "什么是GIL"
    assert res.json()["intent"] == ""


def test_refine_question_falls_back_when_llm_returns_garbage(monkeypatch):
    class _GarbageLLM:
        def invoke(self, _prompt, **_kwargs):
            return _Resp("不是JSON")

    monkeypatch.setattr(api, "create_llm", lambda **kw: _GarbageLLM())
    monkeypatch.setattr(api, "invoke_llm", lambda llm, prompt, **kw: llm.invoke(prompt))
    res = client.post("/research/refine-question", json={"question": "什么是GIL"})
    assert res.status_code == 200
    assert res.json()["refined_question"] == "什么是GIL"


def test_refine_question_empty_rejected():
    res = client.post("/research/refine-question", json={"question": "   "})
    assert res.status_code == 400
