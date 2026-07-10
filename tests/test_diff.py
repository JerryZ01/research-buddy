"""DiffAnalyzer 单元测试 — 不依赖 LLM 的纯规则逻辑"""

from research_buddy.tracking.diff import DiffAnalyzer


class TestDiffAnalyzerSimilarity:
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
        # 使用多行文本，SequenceMatcher 按行比较
        sim = da._compute_similarity(
            "LangGraph 是一个工作流框架\n支持条件分支",
            "LangGraph 是一个工作流框架\n支持条件分支和循环",
        )
        assert sim >= 0.5  # 有重叠行，相似度应较高

    def test_empty_text_similarity(self):
        da = DiffAnalyzer()
        # 空文本返回 0.0（无法比较，视为无相似度）
        sim = da._compute_similarity("", "")
        assert sim == 0.0

    def test_one_empty_text(self):
        da = DiffAnalyzer()
        sim = da._compute_similarity("有内容", "")
        assert sim < 0.5


class TestDiffAnalyzerDifflibFallback:
    def test_returns_list(self):
        da = DiffAnalyzer()
        changes = da._difflib_fallback("旧报告内容", "新报告内容")
        assert isinstance(changes, list)

    def test_identical_no_changes(self):
        da = DiffAnalyzer()
        changes = da._difflib_fallback("相同内容", "相同内容")
        # 相同文本应该没有或极少变化
        assert len(changes) == 0
