"""统一知识层接口 — 整合 SQLite + ChromaDB，提供高层 API

上层代码只需调用 KnowledgeStore，不用关心底层是 db 还是 vector。
"""

from research_buddy.knowledge.db import Database, get_db
from research_buddy.knowledge.vector import VectorStore, get_vector_store


class KnowledgeStore:
    """统一知识层

    职责：
    - 保存研究报告（SQLite 存元数据，ChromaDB 存向量）
    - 查询历史知识（向量检索相关 chunk + 结构化查表）
    - 提取关键事实并存储
    """

    def __init__(self, db: Database | None = None, vector: VectorStore | None = None):
        self.db = db or get_db()
        self.vector = vector or get_vector_store()

    # ── 知识保存 ────────────────────────────────────────

    def save_report(self, topic_id: str, question: str, report: str,
                    confidence: str = "", sources: list | None = None,
                    search_results_count: int = 0, reflection_rounds: int = 0,
                    is_incremental: bool = False, parent_report_id: str = "",
                    key_facts: list[str] | None = None) -> dict:
        """保存研究报告到知识库

        同时写入 SQLite（元数据）和 ChromaDB（向量索引）
        """
        # 写 SQLite
        report_record = self.db.create_report(
            topic_id=topic_id,
            question=question,
            report=report,
            confidence=confidence,
            sources=sources,
            search_results_count=search_results_count,
            reflection_rounds=reflection_rounds,
            is_incremental=is_incremental,
            parent_report_id=parent_report_id,
            key_facts=key_facts,
        )

        # 写 ChromaDB
        report_id = report_record["id"]
        self.vector.add_report(report_id, topic_id, report)
        if key_facts:
            self.vector.add_facts(report_id, topic_id, key_facts)

        return report_record

    def delete_report(self, report_id: str) -> bool:
        """删除报告（同时清理 SQLite 和 ChromaDB）"""
        report = self.db.get_report(report_id)
        if not report:
            return False
        self.vector.delete_report(report_id)
        self.vector.delete_facts(report_id)
        return self.db.delete_report(report_id)

    # ── 知识查询 ────────────────────────────────────────

    def lookup(self, query: str, topic_id: str | None = None,
               n_chunks: int = 5, n_facts: int = 10) -> dict:
        """查询历史知识

        Returns: {
            "chunks": [{"text", "report_id", "distance"}],
            "facts": [{"text", "report_id", "distance"}],
            "latest_report": dict | None,
            "has_knowledge": bool,
        }
        """
        chunks = self.vector.search_reports(query, topic_id=topic_id, n_results=n_chunks)
        facts = self.vector.search_facts(query, topic_id=topic_id, n_results=n_facts)

        latest_report = None
        if topic_id:
            latest_report = self.db.get_latest_report(topic_id)

        has_knowledge = bool(chunks) or bool(facts) or bool(latest_report)

        return {
            "chunks": chunks,
            "facts": facts,
            "latest_report": latest_report,
            "has_knowledge": has_knowledge,
        }

    def get_knowledge_summary(self, topic_id: str) -> str:
        """获取主题的知识摘要（用于增量研究的上下文）

        格式化为 LLM 可读的文本，包含最新报告和关键事实。
        """
        latest = self.db.get_latest_report(topic_id)
        if not latest:
            return "该主题暂无历史研究记录。"

        parts = []
        parts.append(f"## 最近研究（{latest['created_at']}）")
        parts.append(f"问题：{latest['question']}")
        parts.append(f"置信度：{latest['confidence']}")

        if latest.get("key_facts"):
            parts.append("\n### 关键事实")
            for i, fact in enumerate(latest["key_facts"], 1):
                parts.append(f"  {i}. {fact}")

        if latest.get("sources"):
            parts.append(f"\n### 来源（{len(latest['sources'])} 条）")
            for src in latest["sources"][:5]:
                title = src.get("title", "")
                url = src.get("url", "")
                parts.append(f"  - {title}: {url}")

        # 报告摘要（取前 500 字）
        report_text = latest.get("report", "")
        if report_text:
            summary = report_text[:500] + ("..." if len(report_text) > 500 else "")
            parts.append(f"\n### 报告摘要\n{summary}")

        return "\n".join(parts)

    # ── 主题管理 ────────────────────────────────────────

    def create_topic(self, name: str, description: str = "",
                     tracking_keywords: list[str] | None = None) -> dict:
        return self.db.create_topic(name, description, tracking_keywords)

    def get_topic(self, topic_id: str) -> dict | None:
        return self.db.get_topic(topic_id)

    def list_topics(self) -> list[dict]:
        return self.db.list_topics()

    def update_topic(self, topic_id: str, **kwargs) -> dict | None:
        return self.db.update_topic(topic_id, **kwargs)

    def delete_topic(self, topic_id: str) -> bool:
        # 也需要清理 ChromaDB
        reports = self.db.list_reports(topic_id)
        for r in reports:
            self.vector.delete_report(r["id"])
            self.vector.delete_facts(r["id"])
        return self.db.delete_topic(topic_id)

    def list_reports(self, topic_id: str, limit: int = 20) -> list[dict]:
        return self.db.list_reports(topic_id, limit)

    # ── 追踪相关 ────────────────────────────────────────

    def create_tracking_log(self, topic_id: str, status: str = "running") -> dict:
        return self.db.create_tracking_log(topic_id, status)

    def update_tracking_log(self, log_id: str, **kwargs) -> dict | None:
        return self.db.update_tracking_log(log_id, **kwargs)

    def list_tracking_logs(self, topic_id: str, limit: int = 20) -> list[dict]:
        return self.db.list_tracking_logs(topic_id, limit)

    def create_change(self, tracking_log_id: str, change_type: str,
                      description: str, **kwargs) -> dict:
        return self.db.create_change(tracking_log_id, change_type, description, **kwargs)

    def list_changes(self, tracking_log_id: str) -> list[dict]:
        return self.db.list_changes(tracking_log_id)


# ── 全局单例 ────────────────────────────────────────────

_knowledge_store: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    """获取全局 KnowledgeStore 实例（懒初始化）"""
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = KnowledgeStore()
    return _knowledge_store
