"""LangGraph 刚性流水线图骨架：5 个主节点的接线。

framework_orchestrator（论证框架生成）、reference_orchestrator（检索调度）、
writing_orchestrator（串行写作总控）已接入真实业务逻辑，后两者经黑盒适配层
调用子智能体（本期打桩）；citation_validator 与 human_review_gate 仍为占位实现：
仅通过统一封装层做一次 LLM 调用、推进状态机枚举、记录当前节点生效的
LLM 配置元数据，真实业务逻辑在后续 issue 填充。

流水线：framework_orchestrator → reference_orchestrator → writing_orchestrator
→ citation_validator → human_review_gate。
本期无真实人工中断，human_review_gate 占位实现停在 AWAIT_USER_REVIEW；
FINISHED / ERROR_FAILED 由后续迭代路由 issue 启用。
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from framework_orchestrator import make_framework_orchestrator_node
from llm_client import LLMFactory, default_llm_factory
from reference_orchestrator import make_reference_orchestrator_node
from state import WorkflowStatus, WritingAgentState
from subagents import Subagent, make_stub_rewriter_loop, make_stub_search_agent
from writing_orchestrator import make_writing_orchestrator_node

PG_DSN_ENV = "HYPOARGUS_PG_DSN"

# 主节点名 → 进入该节点后状态机应处的枚举值。
NODE_STATUS: dict[str, WorkflowStatus] = {
    "framework_orchestrator": WorkflowStatus.FRAMEWORK_BUILDING,
    "reference_orchestrator": WorkflowStatus.REFERENCE_FETCHING,
    "writing_orchestrator": WorkflowStatus.ARTICLE_WRITING,
    "citation_validator": WorkflowStatus.CITATION_CHECKING,
    "human_review_gate": WorkflowStatus.AWAIT_USER_REVIEW,
}

MAIN_NODES: tuple[str, ...] = tuple(NODE_STATUS)


def _make_placeholder_node(unit: str, llm_factory: LLMFactory):
    """构造占位节点：推进状态机并经统一封装层空跑一次 LLM 调用。"""

    def node(state: WritingAgentState) -> WritingAgentState:
        llm = llm_factory(unit)
        llm.invoke(
            [
                {"role": "system", "content": f"占位实现，无实际业务：{unit}"},
                {"role": "user", "content": state.get("user_intent", "")},
            ]
        )
        return WritingAgentState(
            status=NODE_STATUS[unit],
            current_node_llm_config={"unit": unit, **llm.metadata},
        )

    return node


def build_graph(
    llm_factory: LLMFactory = default_llm_factory,
    checkpointer: BaseCheckpointSaver | None = None,
    search_agent: Subagent | None = None,
    rewriter_loop: Subagent | None = None,
) -> CompiledStateGraph:
    """构建并编译刚性流水线。

    llm_factory 是注入确定性假 LLM 的测试接缝；
    search_agent / rewriter_loop 未注入时使用本期打桩适配器。
    """
    builder = StateGraph(WritingAgentState)
    builder.add_node(MAIN_NODES[0], make_framework_orchestrator_node(llm_factory))
    builder.add_node(
        MAIN_NODES[1],
        make_reference_orchestrator_node(search_agent or make_stub_search_agent()),
    )
    builder.add_node(
        MAIN_NODES[2],
        make_writing_orchestrator_node(rewriter_loop or make_stub_rewriter_loop()),
    )
    for unit in MAIN_NODES[3:]:
        builder.add_node(unit, _make_placeholder_node(unit, llm_factory))

    builder.add_edge(START, MAIN_NODES[0])
    for upstream, downstream in zip(MAIN_NODES, MAIN_NODES[1:]):
        builder.add_edge(upstream, downstream)
    builder.add_edge(MAIN_NODES[-1], END)

    return builder.compile(checkpointer=checkpointer)


@contextmanager
def postgres_checkpointer(dsn: str | None = None) -> Iterator[PostgresSaver]:
    """按环境变量连接串创建 Postgres 存档器，并确保建表完成。"""
    dsn = dsn or os.environ.get(PG_DSN_ENV, "")
    if not dsn:
        raise ValueError(f"缺少 Postgres 连接串：请设置环境变量 {PG_DSN_ENV}")
    with PostgresSaver.from_conn_string(dsn) as saver:
        saver.setup()
        yield saver
