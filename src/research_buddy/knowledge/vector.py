"""ChromaDB 向量存储 — 研究报告和关键事实的语义检索"""

from pathlib import Path

from research_buddy.config import DATA_DIR

# 多语言 embedding 模型（支持中英文语义检索，基于 ONNX，无需 torch）
_MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _get_embedding_function():
    """获取 embedding 函数：优先用 ONNX 多语言模型，fallback 到默认"""
    try:
        from chromadb.utils import embedding_functions
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_MULTILINGUAL_MODEL,
            device="cpu",  # 强制 CPU，避免拉入 CUDA 依赖
        )
    except Exception:
        # fallback: ChromaDB 默认的 all-MiniLM-L6-v2（英文）
        return None


class VectorStore:
    """ChromaDB 向量存储

    将研究报告按 chunk 存储，支持语义检索。
    使用 paraphrase-multilingual-MiniLM-L12-v2 embedding（多语言，支持中文）。

    集合：
    - report_chunks: 研究报告的文本分块
    - key_facts: 关键事实条目
    """

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or str(Path(DATA_DIR) / "chroma_db")
        self._client = None
        self._report_collection = None
        self._facts_collection = None

    @property
    def client(self):
        """懒初始化 ChromaDB 客户端"""
        if self._client is None:
            import chromadb
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def report_collection(self):
        """报告分块集合"""
        if self._report_collection is None:
            ef = _get_embedding_function()
            kwargs = {"metadata": {"hnsw:space": "cosine"}}
            if ef:
                kwargs["embedding_function"] = ef
            self._report_collection = self.client.get_or_create_collection(
                name="report_chunks",
                **kwargs,
            )
        return self._report_collection

    @property
    def facts_collection(self):
        """关键事实集合"""
        if self._facts_collection is None:
            ef = _get_embedding_function()
            kwargs = {"metadata": {"hnsw:space": "cosine"}}
            if ef:
                kwargs["embedding_function"] = ef
            self._facts_collection = self.client.get_or_create_collection(
                name="key_facts",
                **kwargs,
            )
        return self._facts_collection

    # ── 报告分块 ────────────────────────────────────────

    def add_report(self, report_id: str, topic_id: str, text: str,
                   chunk_size: int = 500, chunk_overlap: int = 100) -> int:
        """将报告文本分块后存入 ChromaDB

        Returns: 存入的 chunk 数量
        """
        chunks = self._split_text(text, chunk_size, chunk_overlap)
        if not chunks:
            return 0

        ids = [f"{report_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"report_id": report_id, "topic_id": topic_id, "chunk_index": i}
            for i in range(len(chunks))
        ]

        self.report_collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def search_reports(self, query: str, topic_id: str | None = None,
                       n_results: int = 5, max_distance: float = 1.2) -> list[dict]:
        """语义检索相关报告分块

        Args:
            query: 查询文本
            topic_id: 限定主题（可选）
            n_results: 返回结果数
            max_distance: 最大余弦距离阈值（过滤不相关结果）

        Returns: [{"text", "report_id", "topic_id", "distance"}]
        """
        where_filter = None
        if topic_id:
            where_filter = {"topic_id": topic_id}

        # 确保 n_results 不超过集合大小
        count = self.report_collection.count()
        if count == 0:
            return []
        n_results = min(n_results, count)

        results = self.report_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        output = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 0.0
            # 过滤距离过远（不相关）的结果
            if dist > max_distance:
                continue
            output.append({
                "text": doc,
                "report_id": meta.get("report_id", ""),
                "topic_id": meta.get("topic_id", ""),
                "distance": dist,
            })
        return output

    def delete_report(self, report_id: str) -> None:
        """删除某报告的所有分块"""
        # ChromaDB 需要 where filter 来获取 ids
        try:
            self.report_collection.delete(
                where={"report_id": report_id},
            )
        except Exception:
            pass  # 可能不存在

    # ── 关键事实 ────────────────────────────────────────

    def add_facts(self, report_id: str, topic_id: str, facts: list[str]) -> int:
        """将关键事实存入 ChromaDB

        Returns: 存入的事实数量
        """
        if not facts:
            return 0

        ids = [f"fact_{report_id}_{i}" for i in range(len(facts))]
        metadatas = [
            {"report_id": report_id, "topic_id": topic_id, "fact_index": i}
            for i in range(len(facts))
        ]

        self.facts_collection.upsert(
            ids=ids,
            documents=facts,
            metadatas=metadatas,
        )
        return len(facts)

    def search_facts(self, query: str, topic_id: str | None = None,
                     n_results: int = 10, max_distance: float = 1.2) -> list[dict]:
        """语义检索相关关键事实

        Args:
            max_distance: 最大余弦距离阈值（过滤不相关结果）

        Returns: [{"text", "report_id", "topic_id", "distance"}]
        """
        where_filter = None
        if topic_id:
            where_filter = {"topic_id": topic_id}

        count = self.facts_collection.count()
        if count == 0:
            return []
        n_results = min(n_results, count)

        results = self.facts_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        output = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 0.0
            if dist > max_distance:
                continue
            output.append({
                "text": doc,
                "report_id": meta.get("report_id", ""),
                "topic_id": meta.get("topic_id", ""),
                "distance": dist,
            })
        return output

    def delete_facts(self, report_id: str) -> None:
        """删除某报告的所有关键事实"""
        try:
            self.facts_collection.delete(
                where={"report_id": report_id},
            )
        except Exception:
            pass

    # ── 辅助方法 ────────────────────────────────────────

    @staticmethod
    def _split_text(text: str, chunk_size: int = 500,
                    chunk_overlap: int = 100) -> list[str]:
        """将文本按字符数分块

        简单实现：按段落分割，段落内按 chunk_size 切分。
        """
        if not text or not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # 如果单个段落超长，强制切分
                if len(para) > chunk_size:
                    for i in range(0, len(para), chunk_size - chunk_overlap):
                        chunk = para[i:i + chunk_size]
                        if chunk.strip():
                            chunks.append(chunk)
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return [c for c in chunks if c.strip()]


# ── 全局单例 ────────────────────────────────────────────

_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """获取全局 VectorStore 实例（懒初始化）"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
