"""运行单元名册：全部可独立配置 LLM 参数的执行体的唯一权威清单。

「运行单元」定义见 CONTEXT.md：6 个 LangGraph 主节点 + 2 个业务子智能体。
图装配、LLM 配置、事件渲染与任务管理都从这里同源引用，禁止各自另立名单。
"""

MAIN_NODES: tuple[str, ...] = (
    "framework_orchestrator",
    "reference_orchestrator",
    "chapter_drafter",
    "writing_orchestrator",
    "document_reviewer",
    "human_review_gate",
)
"""6 个 LangGraph 主节点，顺序即主路径接线顺序。

chapter_drafter 是首写阶段经 Send 并行扇出的单章首写节点；
writing_orchestrator 保留修订与终审回退的串行自环。"""

PIPELINE_CHAPTER_DRAFTER_NODE = "pipeline_chapter_drafter"
"""内部图节点：检索后管线首写分支。

不是独立运行单元；事件、状态和产物对外均归一为 chapter_drafter。
"""

INTERNAL_GRAPH_NODES: tuple[str, ...] = (PIPELINE_CHAPTER_DRAFTER_NODE,)
"""非运行单元的内部 LangGraph 节点清单。"""

GRAPH_UPDATE_NODES: tuple[str, ...] = MAIN_NODES + INTERNAL_GRAPH_NODES
"""会在 LangGraph updates/debug 流中出现并需要服务层消费的节点。"""

GRAPH_NODE_UNIT_ALIASES: dict[str, str] = {
    PIPELINE_CHAPTER_DRAFTER_NODE: "chapter_drafter",
}
"""内部图节点名到对外运行单元名的映射。"""


def graph_node_unit(node: str) -> str:
    """把 LangGraph 节点名归一为对外运行单元名。"""
    return GRAPH_NODE_UNIT_ALIASES.get(node, node)

SUBAGENT_UNITS: tuple[str, ...] = (
    "search_agent",
    "rewriter_loop",
    "chapter_reviewer",
)
"""3 个业务子智能体（chapter_reviewer 为章级评审，ADR-0006）。"""

RUNTIME_UNITS: tuple[str, ...] = MAIN_NODES + SUBAGENT_UNITS
"""全部 9 个运行单元。"""
