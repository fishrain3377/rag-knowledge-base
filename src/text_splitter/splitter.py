"""文本切分器：将长文本切成适合向量化的块（chunk），支持重叠。"""

from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import Document

# 中文场景下更合理的切分边界：优先按段落/换行/句号等语义边界切分
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]


class TextSplitter:
    """基于 LangChain RecursiveCharacterTextSplitter 的封装。

    Args:
        chunk_size: 单个块的最大字符数。
        chunk_overlap: 相邻块之间的重叠字符数，用于保持上下文连贯。
        separators: 切分优先级从高到低的分隔符列表。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators if separators is not None else DEFAULT_SEPARATORS,
        )

    def split_text(self, text: str) -> List[str]:
        """将单段文本切分为字符串块列表。"""
        return self._splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将 Document 列表切分为更小的 Document 块，保留并扩充元信息。

        每个块都会继承原文档的 metadata，并新增 ``chunk_id`` 便于溯源。
        """
        chunks: List[Document] = []
        for doc in documents:
            for chunk_text in self._splitter.split_text(doc.page_content):
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={**doc.metadata, "chunk_id": len(chunks)},
                    )
                )
        return chunks
