import os
# 必须放在最前面，在导入任何hf相关包之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HF_HUB_HTTP_CLIENT"] = "httpx"
# 针对httpx关闭ssl校验
import httpx
httpx._config.DEFAULT_CA_BUNDLE = None

from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------- 配置区域 ----------------------
QDRANT_PATH = r"./qdrant_anti_fraud_store"
COLLECTION_NAME = "campus_anti_fraud"
# 传入你的普通txt文档路径，后续替换不同电商txt就改这里
TXT_FILE = r"F:\x-langchain-master\x-langchain-master\economy.txt"

# ========== 修正：cache_folder，修复路径少反斜杠bug ==========
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_folder=r"F:\x-langchain-master\.venv\hf_cache"
)

# 全局只初始化一次客户端
client = QdrantClient(path=QDRANT_PATH)


def build_knowledge_base_from_txt(file_path: str):
    """
    仅传入普通txt文件；force_recreate=True强制drop同名collection，真正覆写旧向量
    更换不同电商txt调用此函数，会覆盖当前COLLECTION_NAME集合内全部旧数据
    """
    raw_text = Path(file_path).read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "，"]
    )
    chunks = splitter.split_text(raw_text)

    vector_store = QdrantVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True   # 关键：强制删除同名集合，重建覆盖
    )
    print(f"✅Qdrant磁盘向量库(txt)构建完成，切片总数：{len(chunks)}")
    return vector_store


def get_vector_store():
    """读取磁盘上已经建好的Qdrant向量库"""
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
    # ==========【首次构建txt向量库：取消下面注释运行1次，构建完务必注释！！】==========
    # build_knowledge_base_from_txt(TXT_FILE)

    # 测试检索，替换成你自己的问题
    test_queries = [
        "七天无理由退货商品完好是什么标准？",
        "退货可以寄到付件吗？",
        "商品有质量问题超过7天还能售后吗？"
    ]
    for q in test_queries:
        print("=" * 60)
        print(f"用户查询：{q}")
        res = search_knowledge(q, top_k=3)
        print(f"检索返回：\n{res}")

    # 关闭客户端，消除退出析构警告
    client.close()
