# llm_agent/tools - LLM 可调用工具
from llm_agent.tools.predict_tool import PredictTool
from llm_agent.tools.kg_query_tool import KGQueryTool
from llm_agent.tools.case_search_tool import CaseSearchTool

# 工具注册表（供 agent.py 使用）
TOOL_REGISTRY = {
    "predict_risk": PredictTool,
    "query_kg_impact": KGQueryTool,
    "search_similar_cases": CaseSearchTool,
}

# LLM Function Calling 工具定义列表
TOOL_DEFINITIONS = [
    PredictTool.TOOL_DEFINITION,
    KGQueryTool.TOOL_DEFINITION,
    CaseSearchTool.TOOL_DEFINITION,
]
