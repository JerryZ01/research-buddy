"""向量 embedding 后端解析与混用保护测试。

原始缺陷：sentence-transformers 既没进依赖也没安装，
_get_embedding_function() 的 except 静默返回 None，
于是「多语言中文检索」永远退化成英文 all-MiniLM-L6-v2，日志里毫无痕迹。
"""

import logging

import pytest

from research_buddy.knowledge import vector as vector_module


@pytest.fixture(autouse=True)
def _clean_cache():
    vector_module._reset_embedding_cache()
    yield
    vector_module._reset_embedding_cache()


def _set_backend(monkeypatch, backend: str, model: str = ""):
    monkeypatch.setattr(vector_module, "EMBEDDING_BACKEND", backend)
    monkeypatch.setattr(vector_module, "EMBEDDING_MODEL", model)


def test_default_backend_uses_builtin_model(monkeypatch):
    _set_backend(monkeypatch, "default")
    ef, backend, model = vector_module.resolve_embedding_function()
    assert ef is None
    assert (backend, model) == ("default", "all-MiniLM-L6-v2")


def test_unknown_backend_warns_and_falls_back(monkeypatch, caplog):
    _set_backend(monkeypatch, "bogus")
    with caplog.at_level(logging.WARNING):
        _, backend, model = vector_module.resolve_embedding_function()
    assert (backend, model) == ("default", "all-MiniLM-L6-v2")
    assert "未知的 EMBEDDING_BACKEND" in caplog.text


def test_unavailable_backend_warns_instead_of_silent_fallback(monkeypatch, caplog):
    _set_backend(monkeypatch, "sentence-transformers")
    monkeypatch.setattr(vector_module, "_build_sentence_transformers",
                        lambda _model: (_ for _ in ()).throw(ImportError("no sentence_transformers")))
    with caplog.at_level(logging.WARNING):
        ef, backend, model = vector_module.resolve_embedding_function()
    assert ef is None
    assert (backend, model) == ("default", "all-MiniLM-L6-v2")
    assert "sentence-transformers 后端不可用" in caplog.text
    assert "multilingual" in caplog.text


def test_working_backend_is_used_and_memoized(monkeypatch):
    calls = []
    sentinel = object()

    def _build(model_name):
        calls.append(model_name)
        return sentinel

    _set_backend(monkeypatch, "sentence-transformers")
    monkeypatch.setattr(vector_module, "_build_sentence_transformers", _build)

    first = vector_module.resolve_embedding_function()
    second = vector_module.resolve_embedding_function()

    assert first[0] is sentinel
    assert first[1] == "sentence-transformers"
    assert first is second
    # 记忆化：模型只加载一次，两个 collection 不会各加载一份
    assert calls == ["paraphrase-multilingual-MiniLM-L12-v2"]


def test_explicit_model_override(monkeypatch):
    _set_backend(monkeypatch, "openai", model="text-embedding-3-large")
    monkeypatch.setattr(vector_module, "_build_openai", lambda _model: object())
    _, backend, model = vector_module.resolve_embedding_function()
    assert (backend, model) == ("openai", "text-embedding-3-large")


def test_describe_backend_is_loggable(monkeypatch):
    _set_backend(monkeypatch, "default")
    assert vector_module.describe_embedding_backend() == "default / all-MiniLM-L6-v2"


class _FakeCollection:
    def __init__(self, metadata: dict | None):
        self.metadata = metadata

    def modify(self, metadata):
        self.metadata = metadata


class _FakeClient:
    def __init__(self, collection: _FakeCollection):
        self._collection = collection
        self.kwargs: dict = {}

    def get_or_create_collection(self, name: str, **kwargs):
        self.kwargs = {"name": name, **kwargs}
        return self._collection


def _store_with(collection: _FakeCollection) -> vector_module.VectorStore:
    store = vector_module.VectorStore(persist_dir="/tmp/does-not-matter")
    store._client = _FakeClient(collection)
    return store


def test_new_collection_records_embedding_model(monkeypatch):
    _set_backend(monkeypatch, "default")
    collection = _FakeCollection(metadata={
        "hnsw:space": "cosine",
        "embedding_backend": "default",
        "embedding_model": "all-MiniLM-L6-v2",
    })
    store = _store_with(collection)
    store._open_collection("report_chunks")
    assert store._client.kwargs["metadata"]["embedding_model"] == "all-MiniLM-L6-v2"


def test_legacy_collection_gets_stamped(monkeypatch, caplog):
    _set_backend(monkeypatch, "default")
    collection = _FakeCollection(metadata={"hnsw:space": "cosine"})
    store = _store_with(collection)
    with caplog.at_level(logging.WARNING):
        store._open_collection("report_chunks")
    assert collection.metadata["embedding_model"] == "all-MiniLM-L6-v2"
    assert "没有 embedding 模型标记" in caplog.text


def test_model_mismatch_is_refused(monkeypatch):
    _set_backend(monkeypatch, "default")
    collection = _FakeCollection(metadata={
        "embedding_backend": "sentence-transformers",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    })
    store = _store_with(collection)
    with pytest.raises(vector_module.EmbeddingBackendMismatch) as excinfo:
        store._open_collection("report_chunks")
    assert "paraphrase-multilingual-MiniLM-L12-v2" in str(excinfo.value)
    assert "all-MiniLM-L6-v2" in str(excinfo.value)
