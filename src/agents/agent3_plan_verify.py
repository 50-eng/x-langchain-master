from langchain_core.prompts import ChatPromptTemplate
from models.providers import create_chat_model
from .state import PipelineState

class PlanVerifyAgent:
    def __init__(self):
        self.llm = create_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是规划校验专家。校验操作方案是否符合业务规则：
1. 已发货订单不可修改地址
2. 已退款订单不可再次退款
3. 修改数量不可超过库存上限
4. 所有写操作必须有明确的订单号
输出JSON：{"valid": true/false, "reason": "...", "suggestion": "..."}
只输出JSON。
            """),
            ("user", "操作清单：{operation_plan}")
        ])
        self.chain = self.prompt | self.llm

    def run(self, state: PipelineState) -> PipelineState:
        result = self.chain.invoke({"operation_plan": state["operation_plan"]})
        import json
        try:
            verify_result = json.loads(result.content)
            state["plan_valid"] = verify_result.get("valid", False)
        except:
            verify_result = {"valid": False, "reason": "校验解析失败"}
            state["plan_valid"] = False
        state["plan_verify_result"] = verify_result
        return state
