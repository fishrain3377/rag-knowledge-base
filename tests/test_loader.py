"""文档加载器测试：Markdown 加载、PDF 加载、文件缺失报错。"""

import pytest

from document_loader import MarkdownLoader, PDFLoader

MD_SAMPLE = """# 项目介绍

RAG 私有知识库系统，支持文档上传与问答。

## 技术栈

使用 Python、LangChain、FAISS 构建。
"""


@pytest.fixture
def md_file(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text(MD_SAMPLE, encoding="utf-8")
    return path


def make_minimal_pdf(path, text="Hello RAG Test"):
    """手工构造一个最小合法 PDF（含文本流与正确的 xref 偏移），避免测试依赖生成器库。"""
    content_stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content_stream)).encode() + b" >>\nstream\n"
        + content_stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_pos = len(output)
    output += f"xref\n0 {len(objects) + 1}\n".encode()
    output += b"0000000000 65535 f \n"
    for off in offsets:
        output += f"{off:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()

    path.write_bytes(output)
    return path


# ---------------------------------------------------------------------- #
# Markdown 加载
# ---------------------------------------------------------------------- #
def test_md_loader_loads_sections(md_file):
    docs = MarkdownLoader(md_file).load()
    assert len(docs) == 2  # 两个标题 => 两个语义块
    assert "RAG 私有知识库系统" in docs[0].page_content
    assert docs[0].metadata["heading"] == "项目介绍"
    assert "source" in docs[0].metadata


def test_md_loader_keeps_heading_in_content(md_file):
    docs = MarkdownLoader(md_file).load()
    assert docs[0].page_content.startswith("项目介绍")


def test_md_loader_plain_text_no_heading(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("没有标题的纯文本内容。", encoding="utf-8")
    docs = MarkdownLoader(path).load()
    assert len(docs) == 1
    assert "纯文本" in docs[0].page_content


# ---------------------------------------------------------------------- #
# PDF 加载
# ---------------------------------------------------------------------- #
def test_pdf_loader_extracts_text(tmp_path):
    pdf = make_minimal_pdf(tmp_path / "sample.pdf", "Hello RAG Test")
    docs = PDFLoader(pdf).load()
    assert len(docs) == 1
    assert "Hello RAG Test" in docs[0].page_content
    assert docs[0].metadata["page"] == 1


# ---------------------------------------------------------------------- #
# 异常路径
# ---------------------------------------------------------------------- #
def test_loader_raises_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        MarkdownLoader(tmp_path / "nope.md").load()
    with pytest.raises(FileNotFoundError):
        PDFLoader(tmp_path / "nope.pdf").load()
