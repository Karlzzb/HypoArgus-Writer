"""运行单元名册：全部可独立配置 LLM 参数的执行体的唯一权威清单。

「运行单元」定义见 CONTEXT.md：5 个 LangGraph 主节点 + 2 个业务子智能体。
图装配、LLM 配置、事件渲染与任务管理都从这里同源引用，禁止各自另立名单。
"""

MAIN_NODES: tuple[str, ...] = (
    "framework_orchestrator",
    "reference_orchestrator",
    "writing_orchestrator",
    "citation_validator",
    "human_review_gate",
)
"""5 个 LangGraph 主节点，顺序即主路径接线顺序。"""

SUBAGENT_UNITS: tuple[str, ...] = (
    "search_agent",
    "rewriter_loop",
)
"""2 个业务子智能体。"""

RUNTIME_UNITS: tuple[str, ...] = MAIN_NODES + SUBAGENT_UNITS
"""全部 7 个运行单元。"""
