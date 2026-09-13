from langchain_core.prompts import ChatPromptTemplate
from models.providers import create_chat_model
from .state import PipelineState

class PlanAgent:
    def __init__(self):
        self.llm = create_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是操作规划专家。根据需求报告，制定执行操作清单。
操作类型包括：
- mysql_query: 查询MySQL数据
- mysql_write: 修改MySQL数据
- milvus_rag: 检索向量知识库
- web_search: 联网搜索
输出JSON数组，每个操作包含：op_type, table/collection, params, description。
只输出JSON。
            """),
            ("user", "需求报告：{demand_report}")
        ])
        self.chain = self.prompt | self.llm

    def run(self, state: PipelineState) -> PipelineState:
        result = self.chain.invoke({"demand_report": state["demand_report"]})
        import json
        try:
            operation_plan = json.loads(result.content)
        except:
            operation_plan = []
        state["operation_plan"] = operation_plan
        return state
