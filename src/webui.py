"""Streamlit 可视化界面：上传文档 + 对话问答。

启动方式：
    streamlit run src/webui.py
    # 或
    python -m streamlit run src/webui.py
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from config import load_config
from document_loader import Document, MarkdownLoader, PDFLoader
from embedding.embedding_model import EmbeddingModel
from rag_chain import RAGChain
from text_splitter.splitter import TextSplitter
from vector_store.faiss_store import FaissStore

LOADERS = {
    ".pdf": PDFLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
}

st.set_page_config(page_title="RAG 私有知识库", page_icon="📚", layout="wide")


@st.cache_resource
def init_components() -> Dict:
    """初始化全部组件（Streamlit 缓存为单例）。"""
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
    return {"config": config, "store": vector_store, "splitter": splitter, "chain": rag_chain}


def handle_upload(uploaded_file) -> Optional[Dict]:
    """处理单次上传：解析 -> 切分 -> 向量化 -> 入库。"""
    filename = uploaded_file.name or ""
    suffix = Path(filename).suffix.lower()
    loader_cls = LOADERS.get(suffix)
    if loader_cls is None:
        st.error(f"不支持的格式: {suffix or '无扩展名'}，仅支持 {sorted(LOADERS)}")
        return None

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        documents: List[Document] = loader_cls(tmp_path).load()
        # 用用户上传的原始文件名替换临时文件路径，保证引用来源可读
        for doc in documents:
            doc.metadata["source"] = filename
        chunks = components["splitter"].split_documents(documents)
        if not chunks:
            st.error("文档解析后内容为空，请检查文件是否可读/含文本")
            return None

        embeddings = components["embedding"].embed_documents([c.page_content for c in chunks])
        components["store"].add_documents(chunks, embeddings)
        components["store"].save()
        return {"filename": filename, "chunks": len(chunks)}
    except Exception as exc:  # noqa: BLE001
        st.error(f"处理失败: {exc}")
        return None
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------- #
# 页面
# ---------------------------------------------------------------------- #
components = init_components()

with st.sidebar:
    st.title("📚 RAG 知识库")
    st.caption("上传 PDF / Markdown，构建本地私有知识库")

    uploaded = st.file_uploader(
        "上传文档", type=["pdf", "md", "markdown"], accept_multiple_files=True
    )
    if uploaded:
        for f in uploaded:
            result = handle_upload(f)
            if result:
                st.success(f"已入库：{result['filename']}（{result['chunks']} 个片段）")

    st.divider()
    st.markdown(f"**知识库文档块数**：{len(components['store'])}")
    st.caption("Embedding: " + components["config"]["embedding"]["model_name"])
    st.caption("LLM: " + components["config"]["llm"].get("model", ""))

# 对话区
st.header("💬 知识问答")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"引用来源（{len(msg['sources'])} 条）"):
                for i, src in enumerate(msg["sources"], start=1):
                    meta = src.get("metadata", {})
                    loc = f"{meta.get('source', '未知来源')}"
                    if meta.get("page"):
                        loc += f" 第{meta['page']}页"
                    st.markdown(f"**[{i}] {loc}**（得分 {src.get('score', 0):.3f}）")
                    st.markdown(src.get("content", ""))

question = st.chat_input("输入你的问题…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答…"):
            try:
                result = components["chain"].answer(question, top_k=components["config"]["vector_store"].get("top_k", 5))
            except Exception as exc:  # noqa: BLE001
                st.error(f"问答失败: {exc}")
                result = None
        if result:
            st.markdown(result["answer"])
            sources = result.get("sources", [])
            if sources:
                with st.expander(f"引用来源（{len(sources)} 条）"):
                    for i, src in enumerate(sources, start=1):
                        meta = src.get("metadata", {})
                        loc = f"{meta.get('source', '未知来源')}"
                        if meta.get("page"):
                            loc += f" 第{meta['page']}页"
                        st.markdown(f"**[{i}] {loc}**（得分 {src.get('score', 0):.3f}）")
                        st.markdown(src.get("content", ""))
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": sources})


def main() -> None:
    """命令行入口：python -m webui（实际由 streamlit 驱动）。"""
    pass


if __name__ == "__main__":
    main()
