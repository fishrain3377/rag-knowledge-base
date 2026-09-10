"""PDF 文档加载器：基于 pypdf 逐页解析。"""

from pathlib import Path
from typing import List, Union

from pypdf import PdfReader

from . import Document


class PDFLoader:
    """读取 PDF 文件，按页生成 Document 列表。

    用法::

        docs = PDFLoader("doc.pdf").load()
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def load(self) -> List[Document]:
        """解析 PDF，返回逐页的 Document 列表（空白页会被跳过）。"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {self.file_path}")

        reader = PdfReader(str(self.file_path))
        documents: List[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(self.file_path), "page": page_num},
                )
            )
        return documents
