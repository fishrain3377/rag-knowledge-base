"""FastAPI 接口服务：文档上传入库 + 知识问答。

启动方式：
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
    # 或
    python src/api_server.py
"""

import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from config import load_config
from document_loader import Document, MarkdownLoader, PDFLoader
from embedding.embedding_model import EmbeddingModel
from rag_chain import RAGChain
from text_splitter.splitter import TextSplitter
from vector_store.faiss_store import FaissStore

# 支持的文件类型 -> 对应的加载器
LOADERS = {
    ".pdf": PDFLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
}

app = FastAPI(title="RAG 私有知识库", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


def _init_components() -> dict:
    """加载配置并初始化全部组件（embedding / 向量库 / 切分器 / RAG 链路）。"""
    config = load_config()

    embedding = EmbeddingModel(
        model_name=config["embedding"]["model_name"],
        device=config["embedding"].get("device", "cpu"),
        normalize=config["embedding"].get("normalize", True),
    )

    index_dir = Path(config["vector_store"]["index_dir"])
    if (index_dir / "vectors.index").exists():
        vector_store = FaissStore.load(index_dir)
    else:
        vector_store = FaissStore(dimension=embedding.dimension, index_dir=str(index_dir))

    splitter = TextSplitter(
        chunk_size=config["splitter"]["chunk_size"],
        chunk_overlap=config["splitter"]["chunk_overlap"],
        separators=config["splitter"].get("separators"),
    )

    rag_chain = RAGChain(embedding, vector_store, config["llm"])
    return {"config": config, "embedding": embedding, "store": vector_store, "splitter": splitter, "chain": rag_chain}


# 服务启动时初始化一次（FastAPI 在 import 时执行模块级代码）
components = _init_components()
vector_store = components["store"]
rag_chain = components["chain"]
text_splitter = components["splitter"]
embedding_model = components["embedding"]


@app.get("/health")
def health() -> dict:
    """健康检查：返回服务状态与当前知识库文档块数量。"""
    return {"status": "ok", "documents": len(vector_store)}


@app.post("/upload")
def upload(file: UploadFile = File(...)) -> dict:  # noqa: B008 - FastAPI 依赖注入的标准写法
    """上传 PDF / Markdown 文档，解析、切分、向量化后写入知识库。"""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    loader_cls = LOADERS.get(suffix)
    if loader_cls is None:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {suffix or '无扩展名'}，仅支持 {sorted(LOADERS)}")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        documents: list[Document] = loader_cls(tmp_path).load()
        # 用用户上传的原始文件名替换临时文件路径，保证引用来源可读
        for doc in documents:
            doc.metadata["source"] = filename
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="文档解析后内容为空，请检查文件是否可读/含文本")

        embeddings = embedding_model.embed_documents([c.page_content for c in chunks])
        vector_store.add_documents(chunks, embeddings)
        vector_store.save()

        return {"filename": filename, "chunks": len(chunks), "total_documents": len(vector_store)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    """基于知识库回答问题，返回答案与引用来源。"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        return rag_chain.answer(question, top_k=req.top_k)
    except RuntimeError as exc:
        # 例如未配置 LLM API Key 等可预期的运行期错误，返回 503 并给出可操作提示
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def main() -> None:
    """命令行入口：python -m api_server 或 rag-api。"""
    config = components["config"]
    uvicorn.run(
        "api_server:app",
        host=config["api"].get("host", "0.0.0.0"),
        port=int(config["api"].get("port", 8000)),
        reload=False,
    )


if __name__ == "__main__":
    main()
