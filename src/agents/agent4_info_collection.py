from tools.mysql_tool import MySQLTool
from tools.milvus_rag_tool import MilvusRAGTool
from tools.web_search_tool import WebSearchTool
from .state import PipelineState

class InfoCollectionAgent:
    def __init__(self):
        self.mysql_tool = MySQLTool()
        self.rag_tool = MilvusRAGTool()
        self.web_tool = WebSearchTool()

    def run(self, state: PipelineState) -> PipelineState:
        # 规划不通过则跳过执行
        if not state.get("plan_valid"):
            return state

        plan = state["operation_plan"]
        db_query = {}
        db_write = {}
        rag_text = ""
        web_text = ""

        for op in plan:
            op_type = op.get("op_type")
            params = op.get("params", {})

            if op_type == "mysql_query":
                result = self.mysql_tool.query(op.get("table"), params)
                db_query[op.get("table")] = result
            elif op_type == "mysql_write":
                result = self.mysql_tool.write(op.get("table"), params)
                db_write[op.get("table")] = result
            elif op_type == "milvus_rag":
                rag_text = self.rag_tool.search(params.get("query", ""))
            elif op_type == "web_search":
                web_text = self.web_tool.search(params.get("query", ""))

        state["db_query_result"] = db_query
        state["db_write_result"] = db_write
        state["rag_result"] = rag_text
        state["web_search_result"] = web_text
        return state
