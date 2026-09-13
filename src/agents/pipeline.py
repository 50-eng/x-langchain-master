from langgraph.graph import StateGraph, END
from .state import PipelineState
from .agent1_demand_analysis import DemandAnalysisAgent
from .agent2_plan import PlanAgent
from .agent3_plan_verify import PlanVerifyAgent
from .agent4_info_collection import InfoCollectionAgent
from .agent5_fact_check import FactCheckAgent
from .agent6_answer_optimize import AnswerOptimizeAgent
from .agent7_compliance import ComplianceAgent
from .agent8_output_dispatch import OutputDispatchAgent

def build_pipeline():
    """构建八大专家串行流水线"""
    workflow = StateGraph(PipelineState)

    # 实例化所有Agent
    agent1 = DemandAnalysisAgent()
    agent2 = PlanAgent()
    agent3 = PlanVerifyAgent()
    agent4 = InfoCollectionAgent()
    agent5 = FactCheckAgent()
    agent6 = AnswerOptimizeAgent()
    agent7 = ComplianceAgent()
    agent8 = OutputDispatchAgent()

    # 添加节点
    workflow.add_node("demand_analysis", agent1.run)
    workflow.add_node("plan", agent2.run)
    workflow.add_node("plan_verify", agent3.run)
    workflow.add_node("info_collection", agent4.run)
    workflow.add_node("fact_check", agent5.run)
    workflow.add_node("answer_optimize", agent6.run)
    workflow.add_node("compliance", agent7.run)
    workflow.add_node("output_dispatch", agent8.run)

    # 设置入口
    workflow.set_entry_point("demand_analysis")

    # 串行边
    workflow.add_edge("demand_analysis", "plan")
    workflow.add_edge("plan", "plan_verify")

    # 规划校验分支：不通过直接跳输出分发
    def plan_router(state: PipelineState):
        return "output_dispatch" if not state.get("plan_valid") else "info_collection"
    workflow.add_conditional_edges("plan_verify", plan_router)

    workflow.add_edge("info_collection", "fact_check")

    # 事实校验分支：失败且重试次数<2则回退到规划
    def fact_router(state: PipelineState):
        if state.get("fact_check_pass"):
            return "answer_optimize"
        elif state.get("retry_count", 0) < 2:
            return "plan"  # 回退重新规划
        else:
            return "output_dispatch"
    workflow.add_conditional_edges("fact_check", fact_router)

    workflow.add_edge("answer_optimize", "compliance")
    workflow.add_edge("compliance", "output_dispatch")
    workflow.add_edge("output_dispatch", END)

    return workflow.compile()
