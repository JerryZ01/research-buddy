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

    def test_validator_routes_to_search_or_synthesis(self):
        edges = create_research_graph().get_graph().edges
        edge_keys = {(edge.source, edge.target, edge.data) for edge in edges}
        assert ("validator", "searcher", "search_again") in edge_keys
        assert ("validator", "synthesizer", "synthesize") in edge_keys


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

    def test_reflector_routes_to_store_or_searcher(self):
        graph = create_knowledge_research_graph()
        edges = graph.get_graph().edges
        edge_keys = {(edge.source, edge.target, edge.data) for edge in edges}
        assert any(
            edge.source == "reflector" and edge.target == "knowledge_store" and edge.conditional
            for edge in edges
        )
        assert ("reflector", "searcher", "search_again") in edge_keys


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

    def test_reflector_routes_to_tracking_chain_or_searcher(self):
        graph = create_tracking_graph()
        edges = graph.get_graph().edges
        edge_keys = {(edge.source, edge.target, edge.data) for edge in edges}
        assert any(
            edge.source == "reflector" and edge.target == "knowledge_store" and edge.conditional
            for edge in edges
        )
        assert ("reflector", "searcher", "search_again") in edge_keys
        assert ("knowledge_store", "diff_analyzer", None) in edge_keys


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
