"""SSE 事件协议端到端测试（全离线）。

跑真实的 create_research_graph()，只把 LLM 和搜索换成假实现，
验证前端依赖的事件流真的成立：
- report_chunk 必须出现（stream_mode 用 list + writer 注入都得对）
- 节点 progress / message / report 事件顺序完整
- 搜索层不可用时是 error 事件，而不是一份零来源的报告
"""

import json
from importlib import import_module

import pytest
from fastapi.testclient import TestClient

from research_buddy import api

planner_module = import_module("research_buddy.nodes.planner")
searcher_module = import_module("research_buddy.nodes.searcher")
validator_module = import_module("research_buddy.nodes.validator")
synthesizer_module = import_module("research_buddy.nodes.synthesizer")
reflector_module = import_module("research_buddy.nodes.reflector")

_URL_A = "https://a.example/one"
_URL_B = "https://b.example/two"

_PLAN = [{
    "id": "sq_01", "question": "子问题", "search_query": "query",
    "language": "zh", "region": "CN", "source_preference": "official",
}]

_EVALUATION = {
    "completeness": 5, "accuracy": 5, "clarity": 5,
    "total_score": 15, "pass": True, "feedback": "", "supplement_queries": [],
}

_REPORT_PIECES = (
    "# 研究报告\n\n## 概述\n",
    "结论一[1]。\n",
    "结论二[2]。\n",
)


class _Response:
    def __init__(self, content: str):
        self.content = content


class _InvokeLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _prompt, **_kwargs):
        return _Response(self.content)


class _Chunk:
    def __init__(self, content: str):
        self.content = content


class _StreamLLM:
    def stream(self, _prompt, **_kwargs):
        for piece in _REPORT_PIECES:
            yield _Chunk(piece)


def _fake_search(_query, **_kwargs):
    return [
        {"title": "来源一", "url": _URL_A, "content": "有效证据内容" * 20, "score": 0.9},
        {"title": "来源二", "url": _URL_B, "content": "另一份证据内容" * 20, "score": 0.8},
    ]


@pytest.fixture
def offline_graph(monkeypatch):
    """把整张图的外部依赖换成假实现，图结构和状态流转保持真实。"""
    monkeypatch.setattr(api, "get_langfuse_handler", lambda: None)

    for module in (planner_module, synthesizer_module, reflector_module):
        monkeypatch.setattr(module, "get_prompt_from_langfuse",
                            lambda *_args, **_kwargs: "prompt")

    monkeypatch.setattr(planner_module, "create_llm",
                        lambda **_: _InvokeLLM(json.dumps(_PLAN)))
    monkeypatch.setattr(synthesizer_module, "create_llm", lambda **_: _StreamLLM())
    monkeypatch.setattr(reflector_module, "create_llm",
                        lambda **_: _InvokeLLM(json.dumps(_EVALUATION)))
    monkeypatch.setattr(validator_module, "_llm_assess", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(searcher_module, "search", _fake_search)


def _collect_sse(body: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 [(event, data), ...]。"""
    events = []
    name = None
    for line in body.splitlines():
        if line.startswith("event:"):
            name = line[len("event:"):].strip()
        elif line.startswith("data:") and name:
            payload = line[len("data:"):].strip()
            try:
                events.append((name, json.loads(payload)))
            except json.JSONDecodeError:
                events.append((name, {"raw": payload}))
            name = None
    return events


def _run_stream() -> list[tuple[str, dict]]:
    client = TestClient(api.app)
    response = client.get("/research/stream", params={"question": "测试研究问题"})
    assert response.status_code == 200
    return _collect_sse(response.text)


def test_stream_emits_report_chunks(offline_graph):
    events = _run_stream()
    chunks = [data["chunk"] for name, data in events if name == "report_chunk"]
    assert chunks, "没有 report_chunk 事件：writer 注入或 stream_mode 又坏了"
    assert "".join(chunks).startswith("# 研究报告")


def test_stream_emits_node_progress_in_graph_order(offline_graph):
    events = _run_stream()
    nodes = [data.get("node") for name, data in events if name == "progress"]
    assert nodes[0] == "start"
    for node in ("planner", "searcher", "validator", "synthesizer", "reflector"):
        assert node in nodes, f"缺少 {node} 的 progress 事件"
    assert nodes.index("planner") < nodes.index("searcher") < nodes.index("validator")
    assert nodes.index("validator") < nodes.index("synthesizer") < nodes.index("reflector")


def test_stream_emits_messages_and_final_report(offline_graph):
    events = _run_stream()
    assert any(name == "message" and data.get("text") for name, data in events)

    reports = [data for name, data in events if name == "report"]
    assert len(reports) == 1
    assert reports[0]["reflection_pass"] is True
    assert reports[0]["search_results_count"] == 2
    assert _URL_A in reports[0]["report"]
    assert [name for name, _ in events][-1] == "done"


def test_stream_progress_carries_structured_detail(offline_graph):
    events = _run_stream()
    planner_detail = next(
        data["detail"] for name, data in events
        if name == "progress" and data.get("node") == "planner"
    )
    assert planner_detail["sub_questions"][0]["question"] == "子问题"

    searcher_detail = next(
        data["detail"] for name, data in events
        if name == "progress" and data.get("node") == "searcher"
    )
    assert searcher_detail["results_count"] == 2


def test_report_payload_carries_quality_fields(offline_graph):
    """前端报告区的指标卡依赖这些字段，缺一个就只能显示「—」。"""
    events = _run_stream()
    report = next(data for name, data in events if name == "report")
    for key in ("reflection_score", "stop_reason",
                "evidence_assessment_degraded", "search_unavailable",
                "confidence", "research_notes", "token_usage"):
        assert key in report, f"report 事件缺少 {key}"
    assert report["reflection_score"] == 15
    assert report["evidence_assessment_degraded"] is True
    # 语义评估降级 → 代码计算置信度为「中」，研究说明里披露降级
    assert report["confidence"] == "中"
    assert any("语义证据评估不可用" in n for n in report["research_notes"])
    # token 统计：mock LLM 不产生 usage，因此是全 0 的 dict，但字段必须在
    assert set(report["token_usage"]) == {"input_tokens", "output_tokens", "total_tokens"}
    # done 事件也带 token_usage
    done = next(data for name, data in events if name == "done")
    assert "token_usage" in done


def test_validator_detail_carries_branch_coverage(offline_graph):
    events = _run_stream()
    detail = next(
        data["detail"] for name, data in events
        if name == "progress" and data.get("node") == "validator"
    )
    assert detail["branch_total"] == 1
    assert detail["branch_sufficient"] == 1
    assert detail["avg_coverage"] > 0
    assert detail["assessment_degraded"] is True


def test_reflector_detail_carries_score(offline_graph):
    events = _run_stream()
    detail = next(
        data["detail"] for name, data in events
        if name == "progress" and data.get("node") == "reflector"
    )
    assert detail["reflection_score"] == 15


def test_search_failure_becomes_error_event_not_a_report(offline_graph, monkeypatch):
    from research_buddy.tools.search import SearchUnavailableError

    def _boom(_query, **_kwargs):
        raise SearchUnavailableError("TAVILY_API_KEY 未配置，无法执行搜索")

    monkeypatch.setattr(searcher_module, "search", _boom)

    events = _run_stream()
    errors = [data["message"] for name, data in events if name == "error"]
    assert errors, "搜索层全挂时必须发 error 事件"
    assert "TAVILY_API_KEY" in errors[0]
    # 关键：不能在零证据的情况下还产出一份报告
    assert not [data for name, data in events if name == "report"]


def test_stream_start_event_carries_run_id_and_recoverable(offline_graph):
    """SSE 首个 progress 事件带 run_id；结束后 /research/run/{id} 可取回结果。"""
    from research_buddy import api as api_module
    events = _run_stream()
    start = next(data for name, data in events if name == "progress" and data.get("node") == "start")
    run_id = start.get("run_id")
    assert run_id, "start 事件必须携带 run_id（断线恢复用）"

    client = TestClient(api_module.app)
    resp = client.get(f"/research/run/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["result"]["report"]
    assert data["result"]["question"] == "测试研究问题"


def test_research_run_unknown_id_returns_404(offline_graph):
    from research_buddy import api as api_module
    client = TestClient(api_module.app)
    assert client.get("/research/run/nonexistent").status_code == 404
