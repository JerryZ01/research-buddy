"""核心路由逻辑单元测试"""

import pytest
from research_buddy.graph import should_continue, should_continue_to_store, route_after_validation
from research_buddy.state import ResearchState


class TestShouldContinue:
    """测试核心图路由函数"""

    def test_reflection_pass_returns_end(self):
        state = {"reflection_pass": True, "reflection_round": 0}
        assert should_continue(state) == "end"

    def test_max_rounds_returns_end(self):
        state = {"reflection_pass": False, "reflection_round": 99}
        assert should_continue(state) == "end"

    def test_not_pass_and_not_max_rounds_returns_search_again(self):
        state = {"reflection_pass": False, "reflection_round": 0, "validation_gaps": [{"search_query": "q"}]}
        assert should_continue(state) == "search_again"

    def test_report_only_issue_returns_revise(self):
        state = {"reflection_pass": False, "reflection_round": 0, "validation_gaps": []}
        assert should_continue(state) == "revise_report"

    def test_pass_takes_priority_over_rounds(self):
        """反思通过时，即使轮次未达上限也结束"""
        state = {"reflection_pass": True, "reflection_round": 1}
        assert should_continue(state) == "end"


class TestShouldContinueToStore:
    """测试知识/追踪图路由函数"""

    def test_reflection_pass_returns_knowledge_store(self):
        state = {"reflection_pass": True, "reflection_round": 0}
        assert should_continue_to_store(state) == "knowledge_store"

    def test_max_rounds_returns_knowledge_store(self):
        state = {"reflection_pass": False, "reflection_round": 99}
        assert should_continue_to_store(state) == "knowledge_store"

    def test_not_pass_and_not_max_rounds_returns_search_again(self):
        state = {"reflection_pass": False, "reflection_round": 0, "validation_gaps": [{"search_query": "q"}]}
        assert should_continue_to_store(state) == "search_again"

    def test_report_only_issue_returns_revise(self):
        state = {"reflection_pass": False, "reflection_round": 0, "validation_gaps": []}
        assert should_continue_to_store(state) == "revise_report"

    def test_pass_takes_priority_over_rounds(self):
        """反思通过时，即使未达到上限也走知识存储"""
        state = {"reflection_pass": True, "reflection_round": 0}
        assert should_continue_to_store(state) == "knowledge_store"


class TestRouteAfterValidation:
    def test_sufficient_evidence_synthesizes(self):
        assert route_after_validation({"validation_gaps": []}) == "synthesize"

    def test_gaps_trigger_search(self):
        assert route_after_validation({"validation_gaps": [{"search_query": "q"}]}) == "search_again"

    def test_budget_exhaustion_synthesizes_limited_report(self):
        state = {"validation_gaps": [{"search_query": "q"}], "stop_reason": "search_budget_exhausted"}
        assert route_after_validation(state) == "synthesize"
