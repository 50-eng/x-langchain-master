import os
# 必须放在最前面，在导入任何hf相关包之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HF_HUB_HTTP_CLIENT"] = "httpx"
# 针对httpx关闭ssl校验
import httpx
httpx._config.DEFAULT_CA_BUNDLE = None

import json
from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient  # 新增
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------- 配置区域 ----------------------
QDRANT_PATH = r"./qdrant_anti_fraud_store"
COLLECTION_NAME = "campus_anti_fraud"
# 修改为你anti‑fraud.txt的真实路径
ANTI_FRAUD_FILE = r"F:\x-langchain-master\x-langchain-master\anti‑fraud.txt"

# ========== 修正：使用 cache_folder 而不是 cache_dir ==========
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_folder=r"F:\x-langchain-master\.venv\hf_cache"
)


def build_knowledge_base_from_json(file_path: str):
    """
    读取anti‑fraud.txt json反诈知识库，写入本地磁盘Qdrant向量库
    ✅仅第一次运行打开注释，构建完成后注释，防止重复入库
    """
    raw = Path(file_path).read_text(encoding="utf-8")
    data = json.loads(raw)
    points = data["points"]
    texts = []
    metadatas = []
    for p in points:
        payload = p["payload"]
        text_block = f"{payload['title']}\n{payload['content']}"
        texts.append(text_block)
        metadatas.append({
            "category": payload["category"],
            "source": payload["source"],
            "title": payload["title"]
        })
    vector_store = QdrantVectorStore.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME
    )
    print(f"✅Qdrant磁盘向量库构建完成，共 {len(texts)} 条知识库条目")
    return vector_store


def get_vector_store():
    """读取磁盘上已经建好的Qdrant向量库【修复】"""
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    return vector_store


def search_knowledge(query: str, top_k: int = 3) -> str:
    """RAG检索接口，返回相关文档文本"""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


if __name__ == "__main__":
    # ==========【首次运行取消下面注释，构建完立刻注释！！】==========
    # build_knowledge_base_from_json(ANTI_FRAUD_FILE)

    # 测试检索
    test_queries = [
        "什么是刷单返利诈骗？",
        "被骗之后应该怎么做？",
        "帮信罪是什么？"
    ]
    for q in test_queries:
        print("=" * 60)
        print(f"用户查询：{q}")
        res = search_knowledge(q, top_k=3)
        print(f"检索返回：\n{res}")
