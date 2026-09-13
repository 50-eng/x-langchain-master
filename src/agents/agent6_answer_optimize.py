from langchain_core.prompts import ChatPromptTemplate
from models.providers import create_chat_model
from .state import PipelineState

class AnswerOptimizeAgent:
    def __init__(self):
        self.llm = create_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是客服答复优化专家。请基于所有查询结果和业务信息，生成友好、专业、清晰的客服答复。
要求：口语化、易懂、符合电商客服语气，不暴露技术细节。
            """),
            ("user", """
用户原始问题：{user_query}
数据库查询结果：{db_result}
知识库结果：{rag_result}
联网结果：{web_result}
规划校验结果：{plan_result}
事实校验结果：{fact_result}
            """)
        ])
        self.chain = self.prompt | self.llm

    def run(self, state: PipelineState) -> PipelineState:
        result = self.chain.invoke({
            "user_query": state["user_query"],
            "db_result": state["db_query_result"],
            "rag_result": state["rag_result"],
            "web_result": state["web_search_result"],
            "plan_result": state["plan_verify_result"],
            "fact_result": state["fact_check_result"]
        })
        state["draft_reply"] = result.content
        return state
