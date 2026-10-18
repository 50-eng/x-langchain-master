from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ollama run qwen2.5:3b

# 配置ollama，使用本地 qwen2.5:3b
llm = OllamaLLM(
    model="qwen2.5:3b",
    base_url="http://127.0.0.1:11434",
    temperature=0.1
)

# RAG提示词模板：严格基于参考上下文回答，不知道就说不知道
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识问答助手，请严格基于下面的参考上下文回答用户问题。
如果参考上下文没有相关信息，直接回答“知识库中未查询到相关信息”，不要编造内容。
回答简洁易懂。

参考上下文：
{context}
"""),
    ("human", "用户问题：{question}")
])

# 组装链
chain = prompt_template | llm


def ollama_rag_answer(user_query: str, context_text: str) -> str:
    """
    :param user_query: 用户原始问题
    :param context_text: 向量库检索得到的上下文文本
    :return: ollama大模型生成回答
    """
    result = chain.invoke({
        "context": context_text,
        "question": user_query
    })
    return result
