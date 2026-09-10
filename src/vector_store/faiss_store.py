"""FAISS 向量库封装：向量索引 + 原文/元数据的持久化。

设计约定：
- 向量保存在 ``vectors.index``（FAISS 二进制索引）。
- 原文与元数据保存在 ``documents.json``，与向量按下标一一对应。
- 使用 ``IndexFlatIP``（内积），配合归一化向量等价于余弦相似度，保证暴力检索的召回上限。
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import faiss
import numpy as np

from document_loader import Document


class SearchResult:
    """一次检索命中的结果：命中的文档块 + 相似度得分。"""

    def __init__(self, document: Document, score: float):
        self.document = document
        self.score = score

    def __repr__(self) -> str:  # pragma: no cover
        return f"SearchResult(score={self.score:.4f}, source={self.document.metadata.get('source')})"


class FaissStore:
    """基于 FAISS 的向量库。

    Args:
        dimension: 向量维度，须与 embedding 模型输出维度一致。
        index_dir: 索引持久化目录；为 None 时仅驻留内存。
    """

    def __init__(self, dimension: int, index_dir: Optional[Union[str, Path]] = None):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Document] = []
        self.index_dir = Path(index_dir) if index_dir else None

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """批量写入文档与对应向量，二者顺序必须一致。"""
        if len(documents) != len(embeddings):
            raise ValueError(f"documents 与 embeddings 数量不一致: {len(documents)} != {len(embeddings)}")
        if not documents:
            return

        matrix = np.asarray(embeddings, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(f"向量维度错误: 期望 {self.dimension} 维，实际矩阵形状 {matrix.shape}")

        self.index.add(matrix)
        self.documents.extend(documents)

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """按查询向量召回 top_k 个最相似文档块，返回按得分降序的结果。"""
        if self.index.ntotal == 0:
            return []

        vector = np.asarray([query_embedding], dtype="float32").reshape(1, -1)
        k = min(max(top_k, 1), self.index.ntotal)
        scores, indices = self.index.search(vector, k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS 用 -1 表示无结果
                continue
            results.append(SearchResult(document=self.documents[int(idx)], score=float(score)))
        return results

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def save(self, index_dir: Optional[Union[str, Path]] = None) -> Path:
        """将索引与原文写入磁盘，返回写入目录。"""
        target = Path(index_dir) if index_dir else self.index_dir
        if target is None:
            raise ValueError("未指定保存目录：请传入 index_dir 或构造时提供 index_dir")

        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target / "vectors.index"))
        with open(target / "documents.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    for d in self.documents
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
        self.index_dir = target
        return target

    @classmethod
    def load(cls, index_dir: Union[str, Path]) -> "FaissStore":
        """从磁盘加载向量库；目录中缺少任一文件时抛出 FileNotFoundError。"""
        index_dir = Path(index_dir)
        index_file = index_dir / "vectors.index"
        docs_file = index_dir / "documents.json"
        if not index_file.exists() or not docs_file.exists():
            raise FileNotFoundError(f"索引目录不完整: {index_dir}（需要 vectors.index 与 documents.json）")

        index = faiss.read_index(str(index_file))
        with open(docs_file, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        store = cls(dimension=index.d, index_dir=index_dir)
        store.index = index
        store.documents = [
            Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs
        ]
        return store

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.index.ntotal
