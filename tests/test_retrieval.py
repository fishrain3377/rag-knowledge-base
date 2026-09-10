"""检索链路测试：切分 + FAISS 向量库增删查 + 持久化往返。

说明：测试使用确定性假向量（基于字符编码），不下载 embedding 模型、
不依赖网络，保证 CI 稳定快速。真实模型链路见集成测试/手工验证。
"""

import numpy as np
import pytest

from document_loader import Document
from text_splitter.splitter import TextSplitter
from vector_store.faiss_store import FaissStore


class FakeEmbedding:
    """确定性假 embedding：将字符映射为固定维度的词袋向量并归一化。

    维度默认 256：16 维冲突率过高，无法区分相近中文文本；256 维下
    不同文本的字符维度分布几乎必然不同，检索行为接近真实模型。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text: str):
        vec = np.zeros(self.dim, dtype="float32")
        for ch in text:
            vec[ord(ch) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm else vec.tolist()


# ---------------------------------------------------------------------- #
# 切分器
# ---------------------------------------------------------------------- #
def test_splitter_splits_long_text():
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    text = "这是一段用于测试切分逻辑的较长的中文文本。" * 20
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 60 for c in chunks)  # chunk_size + 重叠余量


def test_splitter_preserves_metadata():
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    docs = [Document(page_content="苹果。香蕉。橘子。" * 10, metadata={"source": "a.md"})]
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= 2
    assert all(c.metadata["source"] == "a.md" for c in chunks)
    # chunk_id 唯一且连续
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))


def test_splitter_short_text_unchanged():
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text("短文本")
    assert chunks == ["短文本"]


# ---------------------------------------------------------------------- #
# FAISS 向量库
# ---------------------------------------------------------------------- #
def test_store_add_search_roundtrip(tmp_path):
    emb = FakeEmbedding(256)
    store = FaissStore(dimension=256, index_dir=str(tmp_path / "idx"))
    docs = [
        Document(page_content="苹果是一种常见的水果", metadata={"source": "a.md"}),
        Document(page_content="Python 是一种编程语言", metadata={"source": "b.md"}),
        Document(page_content="香蕉富含钾元素", metadata={"source": "c.md"}),
    ]
    store.add_documents(docs, emb.embed_documents([d.page_content for d in docs]))
    assert len(store) == 3

    results = store.search(emb.embed_query("苹果"), top_k=1)
    assert len(results) == 1
    assert "苹果" in results[0].document.page_content
    assert results[0].score > 0


def test_store_save_load_roundtrip(tmp_path):
    emb = FakeEmbedding(256)
    index_dir = tmp_path / "idx"
    store = FaissStore(dimension=256, index_dir=str(index_dir))
    docs = [Document(page_content="深度学习是机器学习的分支", metadata={"source": "x.md", "page": 1})]
    store.add_documents(docs, emb.embed_documents([d.page_content for d in docs]))
    saved = store.save()
    assert (saved / "vectors.index").exists()
    assert (saved / "documents.json").exists()

    loaded = FaissStore.load(index_dir)
    assert len(loaded) == 1
    results = loaded.search(emb.embed_query("深度学习"), top_k=1)
    assert len(results) == 1
    assert results[0].document.metadata["page"] == 1  # 元信息完整保留


def test_store_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaissStore.load(tmp_path / "not_exist")


def test_store_search_empty_returns_empty(tmp_path):
    store = FaissStore(dimension=16, index_dir=str(tmp_path / "empty"))
    assert store.search([0.0] * 16, top_k=5) == []


def test_store_dimension_mismatch_raises(tmp_path):
    store = FaissStore(dimension=16)
    with pytest.raises(ValueError):
        store.add_documents(
            [Document(page_content="t", metadata={})],
            [[0.0] * 8],  # 维度不匹配
        )


def test_store_count_mismatch_raises(tmp_path):
    store = FaissStore(dimension=16)
    with pytest.raises(ValueError):
        store.add_documents(
            [Document(page_content="a", metadata={}), Document(page_content="b", metadata={})],
            [[0.0] * 16],  # 数量不匹配
        )
