"""Embedding 模型封装：将文本转为稠密向量，模型可替换。"""


from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """基于 Sentence-Transformers 的向量化封装。

    通过配置文件中的 ``model_name`` 可无缝替换任意 sentence-transformers 模型
    （如 bge / m3e / text-embedding 系列等），满足模块化设计要求。

    Args:
        model_name: 模型名或本地模型目录。
        device: 推理设备，cpu 或 cuda。
        normalize: 是否对向量做 L2 归一化。开启后 FAISS 内积检索等价于余弦相似度。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文本。"""
        vectors = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询。"""
        vector = self.model.encode(
            [text],
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    @property
    def dimension(self) -> int:
        """模型输出向量维度。"""
        # sentence-transformers >= 4.0 重命名为 get_embedding_dimension，旧版保留原名
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(getter())
