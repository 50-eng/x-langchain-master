from typing import TypedDict, List, Dict, Optional

class PipelineState(TypedDict):
    """八大专家流水线全局状态"""
    # 输入层
    conversation_id: str
    user_query: str
    intent: str
    slots: Dict[str, str]
    chat_history: List[Dict]

    # Agent1 输出
    demand_report: Optional[Dict] = None  # 需求解析报告

    # Agent2 输出
    operation_plan: Optional[List[Dict]] = None  # 操作清单

    # Agent3 输出
    plan_verify_result: Optional[Dict] = None  # 校验结果
    plan_valid: Optional[bool] = None

    # Agent4 输出
    db_query_result: Optional[Dict] = None  # 数据库查询结果
    db_write_result: Optional[Dict] = None  # 数据库写操作结果
    rag_result: Optional[str] = None        # 向量知识库检索结果
    web_search_result: Optional[str] = None # 联网搜索结果

    # Agent5 输出
    fact_check_result: Optional[Dict] = None
    fact_check_pass: Optional[bool] = None
    retry_count: int = 0

    # Agent6 输出
    draft_reply: Optional[str] = None  # 答复初稿

    # Agent7 输出
    compliance_result: Optional[Dict] = None
    compliance_pass: Optional[bool] = None

    # Agent8 输出
    final_reply: Optional[str] = None  # 最终返回给Rasa的结果
