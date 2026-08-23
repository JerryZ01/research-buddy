"""变化检测 — LLM 语义 diff + difflib 文本 diff

核心思路：
1. 先用 difflib 快速筛选是否有文本变化
2. 如果有变化，用 LLM 做语义分析，识别有意义的信息变更
3. 输出结构化的变更列表（类型、描述、重要性）
"""

import difflib
import logging

from langchain_core.runnables import RunnableConfig

from research_buddy.utils import invoke_llm, parse_llm_json, create_llm, get_prompt_from_langfuse

logger = logging.getLogger(__name__)


DIFF_ANALYZER_PROMPT = """你是一个信息变化分析专家。请对比以下两份研究报告，识别有意义的信息变化。

## 旧报告（之前的研究结果）
{old_report}

## 新报告（最新的研究结果）
{new_report}

## 分析要求
1. 只关注事实性信息的变化（数据更新、新政策、新事件等），忽略文字表述差异
2. 对每个变化标注类型：
   - new_info: 全新信息（旧报告中没有的）
   - update: 已有信息的更新（数据变化、进展等）
   - contradiction: 新信息与旧信息矛盾
3. 对每个变化评估重要性：high（重大政策/数据变化）、medium（一般更新）、low（细节补充）
4. 忽略纯文字表述差异，只关注信息内容的变化

请返回如下 JSON 格式（不要包含其他内容）：
```json
[
  {{"type": "update", "description": "新能源渗透率从35%提升至40%", "old_content": "35%", "new_content": "40%", "significance": "high"}},
  {{"type": "new_info", "description": "以旧换新补贴政策延续至2026年", "old_content": "", "new_content": "以旧换新补贴政策延续至2026年", "significance": "medium"}}
]
```

如果没有有意义的变化，返回空数组：```json [] ```"""


class DiffAnalyzer:
    """变化分析器

    两层检测：
    1. difflib 快速筛选（纯文本差异）
    2. LLM 语义分析（理解信息含义的变化）
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: 文本相似度阈值，高于此值认为无显著变化
        """
        self.similarity_threshold = similarity_threshold

    def analyze(self, old_report: str, new_report: str,
                context: str = "") -> dict:
        """分析两份报告之间的变化

        Args:
            old_report: 旧报告
            new_report: 新报告
            context: 上下文（主题名称等）

        Returns: {
            "has_changes": bool,
            "similarity": float,
            "changes": [{"type", "description", "old_content", "new_content", "significance"}]
        }
        """
        # 第一层：difflib 快速筛选
        similarity = self._compute_similarity(old_report, new_report)

        if similarity >= self.similarity_threshold:
            return {
                "has_changes": False,
                "similarity": similarity,
                "changes": [],
            }

        # 第二层：LLM 语义分析
        changes = self._llm_analyze(old_report, new_report, context)

        return {
            "has_changes": len(changes) > 0,
            "similarity": similarity,
            "changes": changes,
        }

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度（0-1）"""
        if not text1 or not text2:
            return 0.0

        # 按行分割，用 SequenceMatcher 计算相似度
        lines1 = text1.strip().splitlines()
        lines2 = text2.strip().splitlines()

        matcher = difflib.SequenceMatcher(None, lines1, lines2)
        return matcher.ratio()

    def _llm_analyze(self, old_report: str, new_report: str,
                     context: str) -> list[dict]:
        """使用 LLM 做语义变化分析"""
        # 截断过长的报告（避免超出 token 限制）
        max_chars = 3000
        old_truncated = old_report[:max_chars] + ("..." if len(old_report) > max_chars else "")
        new_truncated = new_report[:max_chars] + ("..." if len(new_report) > max_chars else "")

        prompt = get_prompt_from_langfuse(
            "research-buddy-diff-analyzer", DIFF_ANALYZER_PROMPT,
            old_report=old_truncated,
            new_report=new_truncated,
        )

        try:
            llm = create_llm()
            response = invoke_llm(llm, prompt, config=config)

            # 使用统一的 parse_llm_json
            changes = parse_llm_json(response.content)

            # 验证格式
            if not isinstance(changes, list):
                return []

            valid_changes = []
            for c in changes:
                if isinstance(c, dict) and "description" in c:
                    valid_changes.append({
                        "type": c.get("type", "new_info"),
                        "description": c.get("description", ""),
                        "old_content": c.get("old_content", ""),
                        "new_content": c.get("new_content", ""),
                        "significance": c.get("significance", "medium"),
                    })

            return valid_changes

        except Exception as e:
            logger.warning("LLM 变化分析失败: %s", e)
            # fallback: 用 difflib 生成简单差异
            return self._difflib_fallback(old_report, new_report)

    def _difflib_fallback(self, old_report: str, new_report: str) -> list[dict]:
        """difflib fallback：当 LLM 分析失败时，用文本差异做简单检测"""
        changes = []
        old_lines = set(l.strip() for l in old_report.splitlines() if l.strip())
        new_lines = set(l.strip() for l in new_report.splitlines() if l.strip())

        # 新增的行
        added = new_lines - old_lines
        for line in list(added)[:5]:
            if len(line) > 10:  # 忽略短行
                changes.append({
                    "type": "new_info",
                    "description": f"新增内容: {line[:80]}",
                    "old_content": "",
                    "new_content": line,
                    "significance": "medium",
                })

        # 移除的行
        removed = old_lines - new_lines
        for line in list(removed)[:3]:
            if len(line) > 10:
                changes.append({
                    "type": "update",
                    "description": f"移除内容: {line[:80]}",
                    "old_content": line,
                    "new_content": "",
                    "significance": "low",
                })

        return changes
