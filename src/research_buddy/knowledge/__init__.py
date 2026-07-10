"""知识层 — 持久化研究报告、来源、关键事实

上层代码应只使用 KnowledgeStore 和 get_knowledge_store，
不要直接访问 Database/VectorStore 底层实现。
"""

from research_buddy.knowledge.store import KnowledgeStore, get_knowledge_store

__all__ = [
    "KnowledgeStore",
    "get_knowledge_store",
]
