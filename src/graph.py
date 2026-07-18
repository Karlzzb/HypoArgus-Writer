"""LangGraph 迭代闭环图：5 个主节点的接线与条件路由。

主流程：framework_orchestrator（论证框架生成）→ reference_orchestrator（检索调度）
→ writing_orchestrator（串行写作总控）→ citation_validator（引文终审门禁）
→ human_review_gate（人工中断点与迭代路由）。

闭环路由：
- citation_validator 终审失败且未超重试上限时，定向回退 writing_orchestrator
  只重写不合格章节；通过或超限（携未决引文警告）进入 human_review_gate。
- human_review_gate 经 LangGraph interrupt 真实中断等待人工；恢复后定稿走
  FINISHED 收束，修订指令回到 writing_orchestrator，再经 citation_validator
  增量核查回到中断点，无限循环直至定稿。
- human_review_gate 是全流程唯一安全汇点：机器环节失败若干次后都塌缩到这里。
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

import observability
from assembler_config import AssemblerConfig
from citation_validator import make_citation_validator_node
from framework_orchestrator import make_framework_orchestrator_node
from human_review_gate import make_human_review_gate_node
from llm_client import LLMFactory, default_llm_factory
from reference_orchestrator import make_reference_orchestrator_node
from state import WorkflowStatus, WritingAgentState
from subagents import Subagent, make_stub_rewriter_loop, make_stub_search_agent
from writing_orchestrator import make_writing_orchestrator_node

PG_DSN_ENV = "HYPOARGUS_PG_DSN"

# 主节点名 → 主路径上进入该节点后状态机所处的枚举值。
# citation_validator 与 human_review_gate 会按路由结果改写状态
# （见各节点实现），此处记录的是其主路径值。
NODE_STATUS: dict[str, WorkflowStatus] = {
    "framework_orchestrator": WorkflowStatus.FRAMEWORK_BUILDING,
    "reference_orchestrator": WorkflowStatus.REFERENCE_FETCHING,
    "writing_orchestrator": WorkflowStatus.ARTICLE_WRITING,
    "citation_validator": WorkflowStatus.CITATION_CHECKING,
    "human_review_gate": WorkflowStatus.AWAIT_USER_REVIEW,
}

MAIN_NODES: tuple[str, ...] = tuple(NODE_STATUS)


def route_after_citation_validator(state: WritingAgentState) -> str:
    """终审后的路由：失败且未超限定向回退写作，通过或超限进入人工中断点。

    判定信号是 citation_validator 显式写入的状态机值：CITATION_CHECKING
    表示还有重试预算、回退重写；AWAIT_USER_REVIEW 表示通过或超限交人工。
    """
    if state.get("status") == WorkflowStatus.CITATION_CHECKING:
        return "writing_orchestrator"
    return "human_review_gate"


def route_after_human_review_gate(state: WritingAgentState) -> str:
    """人工中断点恢复后的路由：定稿收束，修订指令回到写作节点。"""
    if state.get("status") == WorkflowStatus.FINISHED:
        return END
    return "writing_orchestrator"


def build_graph(
    llm_factory: LLMFactory = default_llm_factory,
    checkpointer: BaseCheckpointSaver | None = None,
    search_agent: Subagent | None = None,
    rewriter_loop: Subagent | None = None,
    citation_max_retries: int | None = None,
    assembler_config: AssemblerConfig | None = None,
) -> CompiledStateGraph:
    """构建并编译迭代闭环图。

    llm_factory 是注入确定性假 LLM 的测试接缝；
    search_agent / rewriter_loop 未注入时使用本期打桩适配器；
    citation_max_retries 未注入时按环境变量 CITATION_MAX_RETRIES（缺省 2）；
    assembler_config 未注入时各节点执行期按环境变量读取装配配置。
    人工中断点依赖存档器恢复，生产运行必须传入 checkpointer。
    Langfuse 启用时节点函数与子智能体适配层被包进运行单元 span。
    """
    effective_search_agent = observability.wrap_subagent(
        search_agent or make_stub_search_agent()
    )
    effective_rewriter_loop = observability.wrap_subagent(
        rewriter_loop or make_stub_rewriter_loop()
    )

    node_functions = {
        "framework_orchestrator": make_framework_orchestrator_node(
            llm_factory, assembler_config=assembler_config
        ),
        "reference_orchestrator": make_reference_orchestrator_node(
            effective_search_agent, assembler_config
        ),
        "writing_orchestrator": make_writing_orchestrator_node(
            effective_rewriter_loop, effective_search_agent, assembler_config
        ),
        "citation_validator": make_citation_validator_node(
            llm_factory, citation_max_retries, assembler_config
        ),
        "human_review_gate": make_human_review_gate_node(
            llm_factory, assembler_config
        ),
    }

    builder = StateGraph(WritingAgentState)
    for name, node_fn in node_functions.items():
        builder.add_node(name, observability.traced_node(name, node_fn))

    builder.add_edge(START, "framework_orchestrator")
    builder.add_edge("framework_orchestrator", "reference_orchestrator")
    builder.add_edge("reference_orchestrator", "writing_orchestrator")
    builder.add_edge("writing_orchestrator", "citation_validator")
    builder.add_conditional_edges(
        "citation_validator",
        route_after_citation_validator,
        ["writing_orchestrator", "human_review_gate"],
    )
    builder.add_conditional_edges(
        "human_review_gate",
        route_after_human_review_gate,
        ["writing_orchestrator", END],
    )

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
