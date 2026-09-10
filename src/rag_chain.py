"""RAG 问答链路：问题向量化 -> 向量召回 -> 构建 Prompt -> LLM 生成 -> 返回答案与引用。"""


from langchain_openai import ChatOpenAI

from embedding.embedding_model import EmbeddingModel
from vector_store.faiss_store import FaissStore, SearchResult

SYSTEM_PROMPT = """你是私有知识库问答助手。请严格基于检索到的资料片段回答用户问题。

回答要求：
1. 只依据提供的资料作答，不得编造资料中不存在的信息；
2. 资料不足或与问题无关时，明确回答“根据现有资料无法回答”；
3. 引用资料时标注来源，格式如 [来源: 文件名 第X页] 或 [来源: 文件名]；
4. 回答使用与用户问题相同的语言。"""


class RAGChain:
    """端到端 RAG 链路，组合 embedding 模型、向量库与 LLM。

    Args:
        embedding_model: 向量化模型实例。
        vector_store: 已入库的向量库实例。
        llm_config: 包含 model / api_key / base_url / temperature / max_tokens 的字典。
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FaissStore,
        llm_config: dict,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        api_key = (llm_config.get("api_key") or "").strip()
        if not api_key:
            # 未配置 key 时允许服务启动（文档上传 / 检索仍可用），调用问答时才报错
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model=llm_config.get("model", "deepseek-chat"),
                api_key=api_key,
                base_url=llm_config.get("base_url"),
                temperature=llm_config.get("temperature", 0.3),
                max_tokens=llm_config.get("max_tokens", 1024),
                timeout=60,
            )

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        """将问题向量化并在向量库中召回最相关的文档块。"""
        query_vec = self.embedding_model.embed_query(question)
        return self.vector_store.search(query_vec, top_k=top_k)

    # ------------------------------------------------------------------ #
    # Prompt 构建
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_source(result: SearchResult) -> str:
        meta = result.document.metadata
        source = meta.get("source", "未知来源")
        page = meta.get("page")
        return f"{source} 第{page}页" if page else str(source)

    def _build_prompt(self, question: str, results: list[SearchResult]) -> str:
        context_parts = [
            f"[片段{i}] 来源: {self._format_source(r)}\n{r.document.page_content}"
            for i, r in enumerate(results, start=1)
        ]
        context = "\n\n".join(context_parts)
        return (
            f"以下是检索到的相关资料片段：\n\n{context}\n\n"
            f"用户问题：{question}\n\n"
            f"请根据上述资料给出回答。"
        )

    # ------------------------------------------------------------------ #
    # 端到端问答
    # ------------------------------------------------------------------ #
    def answer(self, question: str, top_k: int = 5) -> dict:
        """执行完整 RAG 问答，返回 {question, answer, sources}。

        sources 为检索命中的原文片段列表，供前端展示引用来源。
        """
        results = self.retrieve(question, top_k=top_k)
        sources = [
            {
                "content": r.document.page_content,
                "score": r.score,
                "metadata": r.document.metadata,
            }
            for r in results
        ]

        if not results:
            answer_text = "知识库中暂无相关文档，请先上传资料后再提问。"
        else:
            if self.llm is None:
                raise RuntimeError(
                    "未配置 LLM API Key：请设置环境变量 OPENAI_API_KEY "
                    "（或修改 configs/config.yaml 的 llm.api_key）后重启服务"
                )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_prompt(question, results)},
            ]
            response = self.llm.invoke(messages)
            answer_text = response.content

        return {"question": question, "answer": answer_text, "sources": sources}
