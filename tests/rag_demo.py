# 导入向量检索模块，同目录下odrant_test.py
from odrant_test import search_knowledge

# 导入ollama大模型调用函数
from llm_ollama import ollama_rag_answer


def simulate_frontend_query(user_question: str):
    """
    模拟前端传入问题的接口函数
    1.调用qdrant检索知识库
    2.将检索上下文和用户问题传入ollama大模型
    3.返回大模型最终输出
    """
    # 1、向量数据库检索知识片段
    retrieve_context = search_knowledge(query=user_question, top_k=3)

    # 2、传给ollama，得到大模型回答
    llm_response = ollama_rag_answer(
        user_query=user_question,
        context_text=retrieve_context
    )

    return {
        "user_question": user_question,
        "retrieve_context": retrieve_context,
        "llm_answer": llm_response
    }


if __name__ == "__main__":
    # 模拟前端传过来的一批测试问题
    test_questions = [
        "七天无理由退货商品完好是什么标准？",
        "退货可以寄到付件吗？",
        "商品有质量问题超过7天还能售后吗？"
    ]

    for question in test_questions:
        print("="*80)
        output = simulate_frontend_query(question)
        print(f"【用户提问】：{output['user_question']}")
        print(f"\n【向量库检索上下文】：\n{output['retrieve_context']}")
        print(f"\n【Ollama(qwen2.5:3b)大模型最终回答】：\n{output['llm_answer']}")
