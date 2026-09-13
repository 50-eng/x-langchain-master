from .state import PipelineState

class OutputDispatchAgent:
    def run(self, state: PipelineState) -> PipelineState:
        # 合规不通过则返回通用致歉
        if not state.get("compliance_pass"):
            final = "抱歉，您的请求暂时无法处理，请联系人工客服。"
        elif not state.get("plan_valid"):
            final = f"抱歉，{state['plan_verify_result'].get('reason', '操作不支持')}。"
        elif not state.get("fact_check_pass") and state.get("retry_count", 0) >= 2:
            final = "系统处理异常，请稍后重试或联系人工客服。"
        else:
            final = state["draft_reply"]

        state["final_reply"] = final
        return state
