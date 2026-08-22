"""ChromaDB 向量存储 — 研究报告和关键事实的语义检索"""

import logging
from pathlib import Path

from research_buddy.config import (
    DATA_DIR,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

# 各后端的默认模型
_DEFAULT_MODELS = {
    "default": "all-MiniLM-L6-v2",                          # ChromaDB 内置（ONNX，英文为主）
    "sentence-transformers": "paraphrase-multilingual-MiniLM-L12-v2",  # 本地多语言
    "openai": "text-embedding-3-small",                      # 远程 API
}

# 记忆化：embedding 模型只解析/加载一次。
# 两个 collection 各自调用会把同一个本地模型加载两遍，占两份内存。
_resolved: tuple[object | None, str, str] | None = None


class EmbeddingBackendMismatch(RuntimeError):
    """已有向量与当前 embedding 后端不一致 —— 混用会让检索结果变成噪声。"""


def _build_sentence_transformers(model_name: str):
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device="cpu",  # 强制 CPU，避免拉入 CUDA 依赖
    )


def _build_openai(model_name: str):
    if not OPENAI_API_KEY:
        raise RuntimeError("EMBEDDING_BACKEND=openai 需要配置 OPENAI_API_KEY")
    from chromadb.utils import embedding_functions
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        api_base=OPENAI_API_BASE,
        model_name=model_name,
    )


def resolve_embedding_function() -> tuple[object | None, str, str]:
    """解析 embedding 后端，返回 (embedding_function, backend, model_name)。

    embedding_function 为 None 表示用 ChromaDB 内置默认模型。

    请求的后端不可用时会打 WARNING 并降级到 default —— 之前是静默 except 返回 None，
    于是「多语言中文检索」这个卖点在 sentence-transformers 没安装的环境里
    完全没有生效，日志里也看不出任何异常。
    """
    global _resolved
    if _resolved is not None:
        return _resolved

    backend = EMBEDDING_BACKEND or "default"
    if backend not in _DEFAULT_MODELS:
        logger.warning(
            "未知的 EMBEDDING_BACKEND=%s，降级到 default。可选值：%s",
            backend, "、".join(_DEFAULT_MODELS),
        )
        backend = "default"

    model_name = EMBEDDING_MODEL or _DEFAULT_MODELS[backend]
    ef = None

    if backend == "sentence-transformers":
        try:
            ef = _build_sentence_transformers(model_name)
        except Exception as exc:
            logger.warning(
                "sentence-transformers 后端不可用（%s），降级到 default（%s，英文为主，"
                "中文检索质量有限）。安装可选依赖：uv sync --extra multilingual",
                exc, _DEFAULT_MODELS["default"],
            )
            backend, model_name = "default", _DEFAULT_MODELS["default"]
    elif backend == "openai":
        try:
            ef = _build_openai(model_name)
        except Exception as exc:
            logger.warning(
                "openai embedding 后端不可用（%s），降级到 default（%s）。"
                "注意中转站不一定提供 /embeddings 接口",
                exc, _DEFAULT_MODELS["default"],
            )
            backend, model_name = "default", _DEFAULT_MODELS["default"]

    logger.info("向量后端=%s 模型=%s", backend, model_name)
    if backend == "default":
        logger.info("default 后端以英文语料训练，中文语义检索召回率偏低；"
                    "需要中文可设 EMBEDDING_BACKEND=sentence-transformers 或 openai")

    _resolved = (ef, backend, model_name)
    return _resolved


def describe_embedding_backend() -> str:
    """给启动日志用的一行描述。"""
    _, backend, model_name = resolve_embedding_function()
    return f"{backend} / {model_name}"


def _reset_embedding_cache() -> None:
    """仅供测试：清掉记忆化结果，让配置改动生效。"""
    global _resolved
    _resolved = None


class VectorStore:
    """ChromaDB 向量存储

    将研究报告按 chunk 存储，支持语义检索。
    embedding 后端由 EMBEDDING_BACKEND 决定，实际生效的模型会写入 collection
    metadata；后续启动如果配置变了但已有向量来自别的模型，会直接拒绝混用
    （两个 MiniLM 都是 384 维，混用不会报维度错，只会让结果悄悄失去意义）。

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

    def _open_collection(self, name: str):
        """打开 collection，并校验已有向量与当前 embedding 模型是否一致。"""
        ef, backend, model_name = resolve_embedding_function()

        kwargs = {
            "metadata": {
                "hnsw:space": "cosine",
                "embedding_backend": backend,
                "embedding_model": model_name,
            },
        }
        if ef:
            kwargs["embedding_function"] = ef

        collection = self.client.get_or_create_collection(name=name, **kwargs)

        # get_or_create_collection 对已存在的 collection 会忽略传入的 metadata，
        # 所以要读回来比对。
        existing = collection.metadata or {}
        recorded = existing.get("embedding_model")

        if recorded is None:
            # 旧库没有标记。历史上 sentence-transformers 从未真正安装成功，
            # 已有向量必然来自当时生效的模型，这里补上标记供以后校验。
            logger.warning(
                "collection %s 没有 embedding 模型标记（旧版本创建），"
                "按当前配置 %s 打标；如果历史向量来自别的模型请清空 %s 后重建",
                name, model_name, self.persist_dir,
            )
            try:
                collection.modify(metadata={**existing, **kwargs["metadata"]})
            except Exception as exc:
                logger.warning("写入 collection %s 的 embedding 标记失败: %s", name, exc)
        elif recorded != model_name:
            raise EmbeddingBackendMismatch(
                f"collection {name} 的已有向量来自 {recorded}，当前配置是 {model_name}。"
                f"混用不同 embedding 模型会让检索结果失去意义。"
                f"请把 EMBEDDING_BACKEND/EMBEDDING_MODEL 改回原模型，"
                f"或删除 {self.persist_dir} 后重建向量库。"
            )

        return collection

    @property
    def report_collection(self):
        """报告分块集合"""
        if self._report_collection is None:
            self._report_collection = self._open_collection("report_chunks")
        return self._report_collection

    @property
    def facts_collection(self):
        """关键事实集合"""
        if self._facts_collection is None:
            self._facts_collection = self._open_collection("key_facts")
        return self._facts_collection

    # ── 报告分块 ────────────────────────────────────────

    def add_report(self, report_id: str, topic_id: str, text: str,
                   chunk_size: int = 500, chunk_overlap: int = 100) -> int:
        """将报告文本分块后存入 ChromaDB

        先清理该报告的旧分块，避免报告变短时旧分块残留。

        Returns: 存入的 chunk 数量
        """
        # 清理旧分块（upsert 不会删除多余的旧 chunk）
        self.delete_report(report_id)

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
                       n_results: int = 5, max_distance: float = 0.5) -> list[dict]:
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
        try:
            self.report_collection.delete(
                where={"report_id": report_id},
            )
        except Exception:
            logger.debug("删除报告分块失败（可能不存在）: %s", report_id, exc_info=True)

    # ── 关键事实 ────────────────────────────────────────

    def add_facts(self, report_id: str, topic_id: str, facts: list[str]) -> int:
        """将关键事实存入 ChromaDB

        先清理该报告的旧事实，避免事实数量变化时旧事实残留。

        Returns: 存入的事实数量
        """
        if not facts:
            return 0

        # 清理旧事实
        self.delete_facts(report_id)

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
                     n_results: int = 10, max_distance: float = 0.5) -> list[dict]:
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
            logger.debug("删除关键事实失败（可能不存在）: %s", report_id, exc_info=True)

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
