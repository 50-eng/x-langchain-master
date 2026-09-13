from langchain_core.prompts import ChatPromptTemplate
from models.providers import create_chat_model
from .state import PipelineState

class ComplianceAgent:
    def __init__(self):
        self.llm = create_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是合规审核专家。检查答复内容是否符合规则：
1. 不得泄露用户隐私信息
2. 不得做出超出政策的承诺
3. 不得出现违禁词汇
4. 退款、修改等操作必须说明前提条件
输出JSON：{"pass": true/false, "risk": "...", "suggestion": "..."}
只输出JSON。
            """),
            ("user", "答复初稿：{draft}")
        ])
        self.chain = self.prompt | self.llm

    def run(self, state: PipelineState) -> PipelineState:
        result = self.chain.invoke({"draft": state["draft_reply"]})
        import json
        try:
            comp_result = json.loads(result.content)
            state["compliance_pass"] = comp_result.get("pass", False)
        except:
            comp_result = {"pass": False, "risk": "审核解析失败"}
            state["compliance_pass"] = False
        state["compliance_result"] = comp_result
        return state
