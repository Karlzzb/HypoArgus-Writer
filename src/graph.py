"""LangGraph 迭代闭环图：5 个主节点的接线与条件路由。

主流程：framework_orchestrator（论证框架生成）→ reference_orchestrator（检索调度）
→ writing_orchestrator（串行写作总控）→ citation_validator（引文终审门禁）
→ human_review_gate（人工中断点与迭代路由）。
writing_orchestrator 是图内自环：每个超步只处理一章、章级产物落 checkpoint，
条件边判定还有未完成章即回到自身写下一章，全部完成才前进终审。

闭环路由：
- citation_validator 终审失败且未超重试上限时，定向回退 writing_orchestrator
  只重写不合格章节；通过或超限（携未决引文警告）进入 human_review_gate。
- human_review_gate 经 LangGraph interrupt 真实中断等待人工；恢复后定稿走
  FINISHED 收束，修订指令回到 writing_orchestrator，再经 citation_validator
  增量核查回到中断点，无限循环直至定稿。
- human_review_gate 是全流程唯一安全汇点：机器环节失败若干次后都塌缩到这里。
"""

import enum
import os
from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg import Connection
from psycopg.rows import dict_row
from pydantic import BaseModel

from agents.contracts import Subagent
from agents.rewriter_loop import make_rewriter_loop
from agents.search_agent import make_stub_search_agent
from assembly.assembler_config import AssemblerConfig
from domain import state as domain_state
from domain.state import WorkflowStatus, WritingAgentState
from domain.units import MAIN_NODES
from llm import observability
from llm.llm_client import LLMFactory, default_llm_factory
from nodes.citation_validator import ValidatorConfig, make_citation_validator_node
from nodes.framework_orchestrator import make_framework_orchestrator_node
from nodes.human_review_gate import make_human_review_gate_node
from nodes.reference_orchestrator import make_reference_orchestrator_node
from nodes.writing_orchestrator import (
    make_writing_orchestrator_node,
    next_writing_step,
)

PG_DSN_ENV = "HYPOARGUS_PG_DSN"

# 存档序列化的类型允许清单：domain.state 内定义的全部状态模型与枚举。
# 从模块自动收集而非手工罗列，新增状态模型时自动纳入，避免清单漂移。
CHECKPOINT_MSGPACK_TYPES: tuple[type, ...] = tuple(
    obj
    for obj in vars(domain_state).values()
    if isinstance(obj, type)
    and obj.__module__ == domain_state.__name__
    and issubclass(obj, (BaseModel, enum.Enum))
)


def checkpoint_serializer() -> JsonPlusSerializer:
    """存档序列化器：把 domain.state 各类型显式注册进 msgpack 允许清单。

    LangGraph 对未注册类型的反序列化会告警并将在未来版本阻断；
    显式注册后严格模式（LANGGRAPH_STRICT_MSGPACK=true）下往返依然成立。
    """
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_MSGPACK_TYPES)

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

assert tuple(NODE_STATUS) == MAIN_NODES, "NODE_STATUS 必须与运行单元名册的主节点一致"


def route_after_writing_orchestrator(state: WritingAgentState) -> str:
    """写作自环路由：还有未完成章回到自身（下一超步写下一章），全部完成前进终审。

    判定与节点单章选取共用 next_writing_step 单一事实源（纯数据推导，
    不依赖 status），保证两处逻辑严格一致、不死循环不漏章。
    """
    if next_writing_step(state) is not None:
        return "writing_orchestrator"
    return "citation_validator"


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
    search_agent 未注入时使用本期打桩适配器；rewriter_loop 未注入时使用
    真实现工厂（make_rewriter_loop：构图时读取一次写作环境配置并按单元名
    取 LLM），打桩仅在显式注入处使用；
    citation_max_retries 未注入时按环境变量 CITATION_MAX_RETRIES（缺省 2）；
    assembler_config 未注入时各节点执行期按环境变量读取装配配置。
    人工中断点依赖存档器恢复，生产运行必须传入 checkpointer。
    Langfuse 启用时节点函数与子智能体适配层被包进运行单元 span。
    """
    effective_search_agent = observability.wrap_subagent(
        search_agent or make_stub_search_agent()
    )
    effective_rewriter_loop = observability.wrap_subagent(
        rewriter_loop or make_rewriter_loop(llm_factory)
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
            llm_factory,
            ValidatorConfig(max_retries=citation_max_retries)
            if citation_max_retries is not None
            else None,
            assembler_config,
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
    builder.add_conditional_edges(
        "writing_orchestrator",
        route_after_writing_orchestrator,
        ["writing_orchestrator", "citation_validator"],
    )
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
    """按环境变量连接串创建 Postgres 存档器，并确保建表完成。

    不走 from_conn_string 而手工建连接：该便捷入口不接受序列化器参数，
    这里必须注入注册了 domain.state 类型的 checkpoint_serializer。
    """
    dsn = dsn or os.environ.get(PG_DSN_ENV, "")
    if not dsn:
        raise ValueError(f"缺少 Postgres 连接串：请设置环境变量 {PG_DSN_ENV}")
    with Connection.connect(
        dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        saver = PostgresSaver(conn, serde=checkpoint_serializer())
        saver.setup()
        yield saver
