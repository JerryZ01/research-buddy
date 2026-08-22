"""评估链路测试：judge 输出容错 + Dataset 幂等。

两个原始缺陷：
- judge 只在 parse 抛错时兜底，模型返回 JSON 数组时 parse 成功但后面
  scores.get() 直接 AttributeError，整个评估跑崩。
- dataset 用 create_dataset() 的返回值取 .items 做幂等检查，
  那个对象没有 items，异常被 except 吞掉，于是每跑一次重复插一整份用例。
"""

import json
from importlib import import_module

dataset_module = import_module("research_buddy.eval.dataset")
judge_module = import_module("research_buddy.eval.judge")


class _Response:
    def __init__(self, content: str):
        self.content = content


class _LLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _prompt):
        return _Response(self.content)


def _patch_judge(monkeypatch, content: str):
    monkeypatch.setattr(judge_module, "create_llm", lambda **_: _LLM(content))
    monkeypatch.setattr(judge_module, "get_prompt_from_langfuse", lambda *_, **__: "prompt")


def test_judge_accepts_valid_scores(monkeypatch):
    _patch_judge(monkeypatch, json.dumps({
        "relevance": 4, "completeness": 3, "accuracy": 5, "reasoning": "ok",
    }))
    scores = judge_module.judge_report("q", ["要点"], "报告")
    assert (scores["relevance"], scores["completeness"], scores["accuracy"]) == (4, 3, 5)
    assert scores["parse_failed"] is False


def test_judge_survives_json_array_output(monkeypatch):
    """合法 JSON 但不是对象 —— 以前会 AttributeError 打断整轮评估。"""
    _patch_judge(monkeypatch, "```json\n[1, 2, 3]\n```")
    scores = judge_module.judge_report("q", ["要点"], "报告")
    assert scores["parse_failed"] is True
    assert scores["relevance"] == 3


def test_judge_survives_unparseable_output(monkeypatch):
    _patch_judge(monkeypatch, "这不是 JSON")
    scores = judge_module.judge_report("q", ["要点"], "报告")
    assert scores["parse_failed"] is True


def test_judge_clamps_out_of_range_and_boolean_scores(monkeypatch):
    _patch_judge(monkeypatch, json.dumps({
        "relevance": 9, "completeness": True, "accuracy": "五", "reasoning": "",
    }))
    scores = judge_module.judge_report("q", ["要点"], "报告")
    assert scores["relevance"] == 3
    assert scores["completeness"] == 3
    assert scores["accuracy"] == 3


class _Item:
    def __init__(self, item_id: str):
        self.id = item_id


class _DatasetClient:
    def __init__(self, items: list[_Item]):
        self.items = items


class _FakeLangfuse:
    def __init__(self, existing: list[str], created: list[dict]):
        self._existing = existing
        self._created = created

    def create_dataset(self, **_kwargs):
        # 真实 SDK 这里返回 API 的 Dataset 模型，它没有 items 属性
        return object()

    def get_dataset(self, _name: str) -> _DatasetClient:
        return _DatasetClient([_Item(i) for i in self._existing])

    def create_dataset_item(self, **kwargs):
        self._created.append(kwargs)


def _run_create_dataset(monkeypatch, existing: list[str]) -> list[dict]:
    created: list[dict] = []
    monkeypatch.setattr(dataset_module, "Langfuse", lambda: _FakeLangfuse(existing, created))
    dataset_module.create_dataset()
    return created


def test_dataset_items_use_stable_unique_ids(monkeypatch):
    created = _run_create_dataset(monkeypatch, existing=[])
    ids = [item["id"] for item in created]
    assert len(created) == len(dataset_module.TEST_CASES)
    assert len(set(ids)) == len(ids)
    assert ids == [dataset_module._item_id(i) for i in range(len(dataset_module.TEST_CASES))]


def test_dataset_upserts_instead_of_appending(monkeypatch):
    """重复运行不会让条目变多：id 固定，SDK 侧是 upsert。"""
    first = _run_create_dataset(monkeypatch, existing=[])
    second = _run_create_dataset(monkeypatch, existing=[item["id"] for item in first])
    assert [item["id"] for item in first] == [item["id"] for item in second]


def test_dataset_reads_items_from_get_dataset(monkeypatch):
    """幂等检查必须走 get_dataset()，因为只有它返回带 items 的 DatasetClient。"""
    calls = []

    class _Tracking(_FakeLangfuse):
        def get_dataset(self, name):
            calls.append(name)
            return super().get_dataset(name)

    monkeypatch.setattr(dataset_module, "Langfuse", lambda: _Tracking([], []))
    dataset_module.create_dataset()
    assert calls == [dataset_module.DATASET_NAME]
