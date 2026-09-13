from pymilvus import connections, Collection
from langchain_community.embeddings import DashScopeEmbeddings
from config.settings import settings
from tools.registry import register_tool

@register_tool(name="milvus_rag", category="knowledge", description="Milvus向量知识库检索")
class MilvusRAGTool:
    def __init__(self):
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
        self.collection = Collection("product_knowledge")
        self.embeddings = DashScopeEmbeddings(
            api_key=settings.ALIYUN_API_KEY,
            model="text-embedding-v2"
        )

    def search(self, query: str, top_k: int = 3) -> str:
        """检索知识库"""
        query_vector = self.embeddings.embed_query(query)
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["content"]
        )
        return "\n".join([hit.entity.get("content") for hit in results[0]])
