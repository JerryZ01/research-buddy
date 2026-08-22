"""Langfuse Prompt 管理测试 — 语法转换、渲染、fallback"""

import pytest

from research_buddy.eval.prompts import convert_format_to_mustache, get_prompt
from research_buddy.utils import get_prompt_from_langfuse


# ── 语法转换测试 ────────────────────────────────────────

class TestConvertFormatToMustache:
    """Python .format() → Langfuse mustache 语法转换"""

    def test_simple_variable(self):
        assert convert_format_to_mustache("Hello {name}!") == "Hello {{name}}!"

    def test_multiple_variables(self):
        result = convert_format_to_mustache("{greeting} {name}!")
        assert result == "{{greeting}} {{name}}!"

    def test_escaped_braces_unescaped(self):
        """Python {{ }} 字面花括号应还原为 { }"""
        assert convert_format_to_mustache('{{"key": "value"}}') == '{"key": "value"}'

    def test_mixed_variable_and_json(self):
        """真实 prompt 模式：变量 + JSON 示例中的 {{ }}"""
        template = 'Question: {question}\n```json\n{{"answer": "x"}}\n```'
        result = convert_format_to_mustache(template)
        assert result == 'Question: {{question}}\n```json\n{"answer": "x"}\n```'

    def test_planner_prompt(self):
        """PLANNER_PROMPT 转换后变量用 {{}}，JSON 示例用 {}"""
        from research_buddy.nodes.planner import PLANNER_PROMPT
        mustache = convert_format_to_mustache(PLANNER_PROMPT)
        assert "{{question}}" in mustache
        # JSON 示例中的字面花括号应为单层
        assert '{{"question"' not in mustache
        assert '{"question"' in mustache

    def test_reflector_prompt(self):
        """REFLECTOR_PROMPT 含 5 个变量 + JSON 示例"""
        from research_buddy.nodes.reflector import REFLECTOR_PROMPT
        mustache = convert_format_to_mustache(REFLECTOR_PROMPT)
        for var in ["question", "sub_questions", "result_count", "source_index", "evidence_status", "report", "user_feedback_section"]:
            assert "{{" + var + "}}" in mustache, f"Missing {{{{{var}}}}}"
        # JSON 示例中的字面花括号应为单层（换行格式）
        assert '"completeness"' in mustache
        # 不应出现三层花括号（转换错误）
        assert '{{{' not in mustache

    def test_no_variables(self):
        assert convert_format_to_mustache("No variables here.") == "No variables here."

    def test_empty_string(self):
        assert convert_format_to_mustache("") == ""

    def test_diff_analyzer_prompt(self):
        """DIFF_ANALYZER_PROMPT 含 2 个变量 + JSON 示例"""
        from research_buddy.tracking.diff import DIFF_ANALYZER_PROMPT
        mustache = convert_format_to_mustache(DIFF_ANALYZER_PROMPT)
        assert "{{old_report}}" in mustache
        assert "{{new_report}}" in mustache
        assert '{"type"' in mustache

    def test_judge_prompt(self):
        """JUDGE_PROMPT 含 3 个变量 + JSON 示例"""
        from research_buddy.eval.judge import JUDGE_PROMPT
        mustache = convert_format_to_mustache(JUDGE_PROMPT)
        assert "{{question}}" in mustache
        assert "{{expected_points}}" in mustache
        assert "{{report}}" in mustache
        # JSON 示例中的字面花括号应为单层（换行格式）
        assert '"relevance"' in mustache
        assert '{{{' not in mustache


# ── Fallback 渲染测试 ───────────────────────────────────

class TestGetPromptFallback:
    """Langfuse 不可用时 fallback 到本地 .format()"""

    def test_simple_fallback(self):
        """简单变量渲染"""
        result = get_prompt_from_langfuse(
            "test-nonexistent-prompt", "Hello {person}!", person="World"
        )
        assert result == "Hello World!"

    def test_multiple_variables_fallback(self):
        """多变量渲染"""
        result = get_prompt_from_langfuse(
            "test-nonexistent-prompt",
            "{greeting} {person}!",
            greeting="Hi",
            person="Alice",
        )
        assert result == "Hi Alice!"

    def test_fallback_with_json(self):
        """含 JSON 示例的模板渲染"""
        template = 'Q: {question}\n```json\n{{"a": 1}}\n```'
        result = get_prompt_from_langfuse(
            "test-nonexistent-prompt", template, question="What?"
        )
        assert "What?" in result
        assert '{"a": 1}' in result

    def test_no_kwargs(self):
        """无变量时返回原始模板"""
        template = "No variables here."
        result = get_prompt_from_langfuse("test-nonexistent-prompt", template)
        assert result == "No variables here."

    def test_integer_variable(self):
        """整数变量渲染（如 result_count）"""
        result = get_prompt_from_langfuse(
            "test-nonexistent-prompt",
            "Found {count} results",
            count=42,
        )
        assert result == "Found 42 results"

    def test_empty_string_variable(self):
        """空字符串变量（如 user_feedback_section）"""
        result = get_prompt_from_langfuse(
            "test-nonexistent-prompt",
            "Q: {question}\n{extra}",
            question="Hello",
            extra="",
        )
        assert result == "Q: Hello\n"
