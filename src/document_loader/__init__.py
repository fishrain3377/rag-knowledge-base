"""文档加载器模块：将 PDF / Markdown 等原始文档解析为统一的 Document 对象。"""

from dataclasses import dataclass, field
from typing import Any

# 注意：Document 必须在本模块其余导入之前定义，
# 避免子模块（pdf_loader / md_loader）反向导入时出现循环导入。


@dataclass
class Document:
    """统一的文档单元：正文 + 元信息。

    Attributes:
        page_content: 文档正文。
        metadata: 来源、页码、标题等元信息，随检索结果一起返回，用于引用溯源。
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


from .md_loader import MarkdownLoader
from .pdf_loader import PDFLoader

__all__ = ["Document", "MarkdownLoader", "PDFLoader"]
