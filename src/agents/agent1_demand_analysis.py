from langchain_core.prompts import ChatPromptTemplate
from models.providers import create_chat_model
from .state import PipelineState

class DemandAnalysisAgent:
    def __init__(self):
        self.llm = create_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是需求解析专家。请根据用户输入、识别到的意图和槽位，输出结构化的需求报告。
输出格式为JSON，包含：
- demand_type: 需求类型（查询/修改地址/修改手机号/修改数量/退款/商品咨询）
- required_params: 已获取的参数
- missing_params: 缺失的必填参数
- business_priority: 优先级
- description: 需求自然语言描述
只输出JSON，不要多余文字。
            """),
            ("user", "用户输入：{user_query}\n意图：{intent}\n槽位：{slots}")
        ])
        self.chain = self.prompt | self.llm

    def run(self, state: PipelineState) -> PipelineState:
        result = self.chain.invoke({
            "user_query": state["user_query"],
            "intent": state["intent"],
            "slots": state["slots"]
        })
        import json
        try:
            demand_report = json.loads(result.content)
        except:
            demand_report = {"raw": result.content}
        state["demand_report"] = demand_report
        return state
