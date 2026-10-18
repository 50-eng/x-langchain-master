import os
# 必须放在最前面，在导入任何hf相关包之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HF_HUB_HTTP_CLIENT"] = "httpx"
import httpx
httpx._config.DEFAULT_CA_BUNDLE = None
from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------- 配置区域 ----------------------
QDRANT_PATH = r"./qdrant_anti_fraud_store"
COLLECTION_NAME = "campus_anti_fraud"
# bge‑small‑zh‑v1.5 输出向量维度 = 512
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_folder=r"F:\x-langchain-master\.venv\hf_cache"
)


def build_knowledge_base_from_txt(file_path: str):
    """
    【模拟前端传入txt文件】
    传入txt绝对路径，会覆盖原有向量库，重新构建；
    :param file_path: txt文件完整路径（模拟前端上传后得到的文件路径）
    """
    file_p = Path(file_path)
    if not file_p.exists():
        raise FileNotFoundError(f"txt文件不存在：{file_path}")

    client = QdrantClient(path=QDRANT_PATH)

    # 存在就删除集合，实现覆盖效果
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # 手动新建集合
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE
        )
    )

    raw_text = file_p.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", "Q：", "A：", "。"]
    )
    chunks = splitter.split_text(raw_text)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    vector_store.add_texts(chunks)

    print(f"✅向量库已覆盖重建 | 文件:{file_p.name} | 切片总数：{len(chunks)}")
    client.close()
    return True


def search_knowledge(query: str, top_k: int = 3) -> str:
    """查询知识库"""
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    content = "\n\n".join([doc.page_content for doc in docs])
    client.close()
    return content


if __name__ == "__main__":
    # ======================测试示例======================
    # 场景1：加载 economy.txt（原来的文档）
    print("=====测试1：加载 economy.txt =====")
    txt1 = r"F:\x-langchain-master\x-langchain-master\economy.txt"
    build_knowledge_base_from_txt(txt1)
    res1 = search_knowledge("经济相关问题")
    print("检索结果：\n", res1, "\n")

    # # 场景2：模拟前端传入另外一份纯文本 student.txt（自己新建这个txt）
    # print("=====测试2：覆盖加载 student.txt =====")
    # txt2 = r"F:\x-langchain-master\x-langchain-master\student.txt"
    # build_knowledge_base_from_txt(txt2)
    # res2 = search_knowledge("学生相关问题")
    # print("检索结果：\n", res2, "\n")

    # # 场景3：模拟前端传入 fraud.txt，再次覆盖向量库
    # print("=====测试3：覆盖加载 fraud.txt =====")
    # txt3 = r"F:\x-langchain-master\x-langchain-master\anti‑fraud.txt"
    # build_knowledge_base_from_txt(txt3)
    # res3 = search_knowledge("诈骗相关问题")
    # print("检索结果：\n", res3)
