"""图构建单元测试 — 验证所有图能正常编译和节点拓扑"""

import pytest
from langgraph.graph import START, END

from research_buddy.graph import (
    create_research_graph,
    create_knowledge_research_graph,
    create_tracking_graph,
    create_research_graph_with_hitl,
)


class TestResearchGraph:
    def test_compiles(self):
        graph = create_research_graph()
        assert graph is not None

    def test_has_core_nodes(self):
        graph = create_research_graph()
        nodes = set(graph.get_graph().nodes.keys())
        expected = {"__start__", "__end__", "planner", "searcher", "validator",
                    "synthesizer", "reflector"}
        assert expected.issubset(nodes), f"Missing nodes: {expected - nodes}"


class TestKnowledgeResearchGraph:
    def test_compiles(self):
        graph = create_knowledge_research_graph()
        assert graph is not None

    def test_has_knowledge_nodes(self):
        graph = create_knowledge_research_graph()
        nodes = set(graph.get_graph().nodes.keys())
        expected = {"__start__", "__end__", "planner", "searcher", "validator",
                    "synthesizer", "reflector", "knowledge_lookup", "knowledge_store"}
        assert expected.issubset(nodes), f"Missing nodes: {expected - nodes}"


class TestTrackingGraph:
    def test_compiles(self):
        graph = create_tracking_graph()
        assert graph is not None

    def test_has_tracking_nodes(self):
        graph = create_tracking_graph()
        nodes = set(graph.get_graph().nodes.keys())
        expected = {"__start__", "__end__", "planner", "searcher", "validator",
                    "synthesizer", "reflector", "knowledge_lookup", "knowledge_store",
                    "diff_analyzer", "change_notifier"}
        assert expected.issubset(nodes), f"Missing nodes: {expected - nodes}"


class TestHITLGraph:
    def test_compiles(self):
        graph = create_research_graph_with_hitl()
        assert graph is not None

    def test_has_core_nodes(self):
        graph = create_research_graph_with_hitl()
        nodes = set(graph.get_graph().nodes.keys())
        expected = {"__start__", "__end__", "planner", "searcher", "validator",
                    "synthesizer", "reflector"}
        assert expected.issubset(nodes), f"Missing nodes: {expected - nodes}"
