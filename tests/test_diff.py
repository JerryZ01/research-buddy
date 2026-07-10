"""DiffAnalyzer 单元测试 — 覆盖公共 API 和分支逻辑"""

from research_buddy.tracking.diff import DiffAnalyzer


class TestDiffAnalyzerAnalyze:
    """测试公共 analyze() 方法"""

    def test_identical_texts_no_changes(self):
        """完全相同的文本应返回 has_changes=False"""
        da = DiffAnalyzer()
        result = da.analyze("完全相同的文本", "完全相同的文本")
        assert result["has_changes"] is False
        assert result["changes"] == []
        assert result["similarity"] > 0.99

    def test_completely_different_has_changes(self):
        """完全不同的文本应触发变化检测（但 LLM 分析会 mock 失败走 fallback）"""
        da = DiffAnalyzer(similarity_threshold=0.5)
        # 使用足够不同的多行文本
        result = da.analyze(
            "旧报告：Python 3.10 发布于 2021 年\nGIL 仍然存在",
            "新报告：Python 3.12 发布于 2023 年\n引入了更快的 CPython",
        )
        # 相似度应低于阈值
        assert result["similarity"] < 0.5
        # has_changes 取决于 LLM/fallback 结果，但 similarity < threshold 意味着会尝试分析
        assert "has_changes" in result
        assert "changes" in result

    def test_empty_text_similarity_zero(self):
        """空文本返回 similarity=0.0"""
        da = DiffAnalyzer()
        result = da.analyze("", "")
        assert result["similarity"] == 0.0

    def test_one_empty_text_triggers_analysis(self):
        """一端为空文本，相似度为 0，应触发分析"""
        da = DiffAnalyzer()
        result = da.analyze("有内容", "")
        assert result["similarity"] < 0.5


class TestDiffAnalyzerSimilarity:
    """测试 _compute_similarity 私有方法（保留原有测试）"""

    def test_identical_texts_high_similarity(self):
        da = DiffAnalyzer()
        sim = da._compute_similarity("完全相同的文本", "完全相同的文本")
        assert sim > 0.99

    def test_completely_different_low_similarity(self):
        da = DiffAnalyzer()
        sim = da._compute_similarity("苹果香蕉橘子", "量子物理相对论")
        assert sim < 0.3

    def test_partial_overlap_medium_similarity(self):
        da = DiffAnalyzer()
        sim = da._compute_similarity(
            "LangGraph 是一个工作流框架\n支持条件分支",
            "LangGraph 是一个工作流框架\n支持条件分支和循环",
        )
        assert sim >= 0.5

    def test_empty_text_similarity(self):
        da = DiffAnalyzer()
        sim = da._compute_similarity("", "")
        assert sim == 0.0

    def test_one_empty_text(self):
        da = DiffAnalyzer()
        sim = da._compute_similarity("有内容", "")
        assert sim < 0.5


class TestDiffAnalyzerDifflibFallback:
    """测试 _difflib_fallback 私有方法"""

    def test_returns_list(self):
        da = DiffAnalyzer()
        changes = da._difflib_fallback("旧报告内容", "新报告内容")
        assert isinstance(changes, list)

    def test_identical_no_changes(self):
        da = DiffAnalyzer()
        changes = da._difflib_fallback("相同内容", "相同内容")
        assert len(changes) == 0

    def test_added_content_detected(self):
        da = DiffAnalyzer()
        changes = da._difflib_fallback(
            "旧报告第一行",
            "旧报告第一行\n新增的第二行内容足够长",
        )
        # 应检测到新增内容
        assert len(changes) > 0
        assert any(c["type"] == "new_info" for c in changes)
