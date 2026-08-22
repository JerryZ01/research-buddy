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
        """追加语义的列表字段应使用 Annotated[list, operator.add]"""
        hints = get_type_hints(ResearchState, include_extras=True)

        # 这些字段用 operator.add（追加语义）
        append_fields = [
            "search_results",
            "search_history",
            "key_facts",
            "detected_changes",
            "messages",
        ]

        for field in append_fields:
            assert field in hints, f"{field} missing from ResearchState"
            args = get_args(hints[field])
            # Annotated[list, operator.add] → args = (list, operator.add)
            has_add = any(
                arg is operator.add for arg in args if not isinstance(arg, type)
            )
            assert has_add, f"{field} should use operator.add for append semantics"

    def test_sub_questions_uses_overwrite_semantics(self):
        """sub_questions 应使用覆盖语义（普通 list），支持 HITL 编辑替换"""
        hints = get_type_hints(ResearchState, include_extras=True)
        assert "sub_questions" in hints
        args = get_args(hints["sub_questions"])
        # 不应包含 operator.add
        has_add = any(
            arg is operator.add for arg in args if not isinstance(arg, type)
        )
        assert not has_add, "sub_questions should NOT use operator.add (needs overwrite for HITL)"

    def test_validation_gaps_uses_overwrite_semantics(self):
        """validation_gaps 应使用覆盖语义，支持搜索节点清空已处理缺口"""
        hints = get_type_hints(ResearchState, include_extras=True)
        assert "validation_gaps" in hints
        args = get_args(hints["validation_gaps"])
        has_add = any(
            arg is operator.add for arg in args if not isinstance(arg, type)
        )
        assert not has_add, "validation_gaps should NOT use operator.add (needs clearing after search)"

    def test_evidence_assessments_uses_overwrite_semantics(self):
        hints = get_type_hints(ResearchState, include_extras=True)
        args = get_args(hints["evidence_assessments"])
        assert not any(arg is operator.add for arg in args if not isinstance(arg, type))
