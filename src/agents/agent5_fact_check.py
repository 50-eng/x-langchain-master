from langchain_core.prompts import ChatPromptTemplate
from models.providers import create_chat_model
from .state import PipelineState

class FactCheckAgent:
    def __init__(self):
        self.llm = create_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是事实校验专家。对比写操作前后的数据，确认修改是否正确执行。
输出JSON：{"pass": true/false, "issue": "...", "need_retry": true/false}
只输出JSON。
            """),
            ("user", "写操作结果：{write_result}\n查询结果：{query_result}")
        ])
        self.chain = self.prompt | self.llm

    def run(self, state: PipelineState) -> PipelineState:
        # 无写操作则直接通过
        if not state.get("db_write_result"):
            state["fact_check_pass"] = True
            state["fact_check_result"] = {"pass": True, "reason": "无写操作"}
            return state

        result = self.chain.invoke({
            "write_result": state["db_write_result"],
            "query_result": state["db_query_result"]
        })
        import json
        try:
            check_result = json.loads(result.content)
            state["fact_check_pass"] = check_result.get("pass", False)
        except:
            check_result = {"pass": False, "issue": "校验解析失败"}
            state["fact_check_pass"] = False

        state["fact_check_result"] = check_result
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state
