"""State 定义单元测试"""

import operator
from typing import get_type_hints, get_args

from research_buddy.state import ResearchState, SubQuestion, SearchResult, ValidationGap


class TestSubQuestion:
    def test_has_required_fields(self):
        sq: SubQuestion = {"question": "测试问题", "search_query": "test query"}
        assert sq["question"] == "测试问题"
        assert sq["search_query"] == "test query"


class TestSearchResult:
    def test_has_required_fields(self):
        sr: SearchResult = {
            "sub_question": "Q1",
            "title": "Title",
            "url": "https://example.com",
            "content": "Content",
            "score": 0.9,
        }
        assert sr["sub_question"] == "Q1"
        assert sr["score"] == 0.9


class TestValidationGap:
    def test_has_required_fields(self):
        vg: ValidationGap = {
            "question": "需要补充的问题",
            "search_query": "supplement query",
        }
        assert vg["question"] == "需要补充的问题"


class TestResearchState:
    def test_list_fields_use_operator_add(self):
        """列表字段应使用 Annotated[list, operator.add] 实现追加语义"""
        hints = get_type_hints(ResearchState, include_extras=True)

        list_fields = [
            "sub_questions",
            "search_results",
            "validation_gaps",
            "key_facts",
            "detected_changes",
            "messages",
        ]

        for field in list_fields:
            assert field in hints, f"{field} missing from ResearchState"
            args = get_args(hints[field])
            # Annotated[list, operator.add] → args = (list, operator.add)
            has_add = any(
                arg is operator.add for arg in args if not isinstance(arg, type)
            )
            assert has_add, f"{field} should use operator.add for append semantics"
