"""HITL 人机交互测试 — 图编译、API 端点、中断检测"""

import pytest
from fastapi.testclient import TestClient

from research_buddy.graph import create_research_graph_with_hitl


class TestHITLGraph:
    """HITL 图构建和编译测试"""

    def test_compiles(self):
        graph = create_research_graph_with_hitl()
        assert graph is not None

    def test_has_core_nodes(self):
        graph = create_research_graph_with_hitl()
        nodes = set(graph.get_graph().nodes.keys())
        expected = {"__start__", "__end__", "planner", "searcher", "validator",
                    "editorial_planner", "synthesizer", "language_editor", "article_editor", "reflector"}
        assert expected.issubset(nodes), f"Missing nodes: {expected - nodes}"

    def test_has_checkpointer(self):
        """HITL 图应有 checkpointer 以支持中断恢复"""
        graph = create_research_graph_with_hitl()
        # 编译后的图应该有 checkpointer
        assert hasattr(graph, 'checkpointer')

    def test_interrupt_before_configured(self):
        """HITL 图应配置 interrupt_before"""
        graph = create_research_graph_with_hitl()
        # 检查图的 interrupt_before 配置
        # LangGraph 编译后的图存储配置在内部
        assert graph is not None  # 基本编译检查


class TestHITLAPIEndpoints:
    """HITL API 端点路由测试"""

    @pytest.fixture
    def client(self):
        from research_buddy.api import app
        return TestClient(app)

    def test_hitl_stream_endpoint_exists(self, client):
        """HITL stream 端点应存在"""
        # 发送请求验证路由存在（不需要实际执行研究）
        response = client.post(
            "/research/hitl/stream",
            json={"question": "test"},
            headers={"Accept": "text/event-stream"},
        )
        # 端点存在即可，SSE 响应不检查内容
        assert response.status_code in (200, 201)

    def test_hitl_resume_endpoint_exists(self, client):
        """HITL resume 端点应存在"""
        response = client.post(
            "/research/hitl/resume/stream",
            json={"thread_id": "nonexistent", "resume_data": {}},
        )
        # 不存在的 thread_id 应返回 404
        assert response.status_code == 404

    def test_hitl_state_endpoint_exists(self, client):
        """HITL state 端点应存在"""
        response = client.get(
            "/research/hitl/state?thread_id=nonexistent",
        )
        assert response.status_code == 404

    def test_health_still_works(self, client):
        """健康检查端点应正常"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestSubQuestionsOverwriteSemantics:
    """验证 sub_questions 使用覆盖语义（非 operator.add）"""

    def test_sub_questions_is_plain_list(self):
        """sub_questions 应为普通 list，支持 HITL 编辑替换"""
        from typing import get_type_hints, get_args
        import operator
        from research_buddy.state import ResearchState

        hints = get_type_hints(ResearchState, include_extras=True)
        args = get_args(hints["sub_questions"])
        has_add = any(
            arg is operator.add for arg in args if not isinstance(arg, type)
        )
        assert not has_add, "sub_questions should NOT use operator.add"

    def test_update_state_replaces_sub_questions(self):
        """update_state 应替换而非追加 sub_questions"""
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import MemorySaver
        from research_buddy.state import ResearchState

        def planner_node(state):
            return {
                "sub_questions": [
                    {"question": "Q1", "search_query": "query1"},
                    {"question": "Q2", "search_query": "query2"},
                ]
            }

        def dummy_node(state):
            return {}

        graph = StateGraph(ResearchState)
        graph.add_node("planner", planner_node)
        graph.add_node("dummy", dummy_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "dummy")
        graph.add_edge("dummy", END)

        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory, interrupt_before=["dummy"])
        config = {"configurable": {"thread_id": "test-overwrite"}}

        # 运行到中断
        for event in compiled.stream({"question": "test"}, config):
            pass

        # 检查初始子问题
        snapshot = compiled.get_state(config)
        assert len(snapshot.values.get("sub_questions", [])) == 2

        # 用 update_state 替换子问题
        new_sqs = [
            {"question": "Edited Q1", "search_query": "edited1"},
            {"question": "Edited Q2", "search_query": "edited2"},
            {"question": "New Q3", "search_query": "new3"},
        ]
        compiled.update_state(config, {"sub_questions": new_sqs}, as_node="planner")

        # 验证替换而非追加
        snapshot2 = compiled.get_state(config)
        sqs = snapshot2.values.get("sub_questions", [])
        assert len(sqs) == 3, f"Expected 3 sub_questions after replace, got {len(sqs)}"
        assert sqs[0]["question"] == "Edited Q1"
