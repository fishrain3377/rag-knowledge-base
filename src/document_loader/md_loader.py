"""Markdown 文档加载器：读取 .md 文件，并按标题层级切分为语义块。"""

import re
from pathlib import Path
from typing import List, Tuple, Union

from . import Document


class MarkdownLoader:
    """读取 Markdown 文件，按标题（# ~ ####）切分为多个语义块。

    标题会拼入正文内容，保证切片后仍携带上下文；标题本身同时记录在 metadata。
    """

    _HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def load(self) -> List[Document]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Markdown 文件不存在: {self.file_path}")

        text = self.file_path.read_text(encoding="utf-8")
        sections = self._split_by_headings(text)
        documents = []
        for idx, (heading, body) in enumerate(sections):
            content = f"{heading}\n{body}" if heading else body
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(self.file_path),
                        "section": idx,
                        "heading": heading or "",
                    },
                )
            )
        return documents

    @classmethod
    def _split_by_headings(cls, text: str) -> List[Tuple[str, str]]:
        """按 Markdown 标题切分，返回 [(标题, 正文)] 列表；无标题时返回单块。"""
        matches = list(cls._HEADING_RE.finditer(text))
        if not matches:
            return [("", text.strip())]

        sections: List[Tuple[str, str]] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            heading = match.group(2).strip()
            body = text[start:end].strip()
            sections.append((heading, body))
        return sections
