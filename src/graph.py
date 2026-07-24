"""LangGraph 迭代闭环图：6 个主节点的接线与条件路由。

主流程：framework_orchestrator（论证框架生成）→ reference_orchestrator（检索并行扇出）
→ chapter_drafter（首写并行扇出）→ document_reviewer（篇级终审门禁）
→ human_review_gate（人工中断点与迭代路由）。
检索阶段：framework_orchestrator 后的条件边为每个待检索章节各发一个 Send，
reference_orchestrator 各分支并行检索一章（existing_materials_digest 只反映
扇出前的既有引文库），素材经 citation_library 合并 reducer 汇入主状态并
跨章按 URL 去重、按超步落 checkpoint，全部分支完成后汇合进入首写扇出。
首写阶段：检索汇合节点 reference_join 后的条件边为每个未写章节各发一个 Send，
chapter_drafter 各分支并行写一章（前文承接用框架生成的规划摘要链），
产物经 chapter_drafts 合并 reducer 汇入主状态、按超步落 checkpoint，
全部分支完成后汇合前进终审。两段并行带 checkpointer 时某分支失败，
已完成分支的写入被保留，resume 只重跑未完成分支（ADR-0001 约束 1）。
writing_orchestrator 将人工修订与终审回退按章节 Send 扇出：每个分支只回写
目标章产物，汇合后才进入终审。其串行自环仅保留给旧 checkpoint 的恢复与
防御性兜底（draft 分支正常路径不再可达）。

闭环路由：
- document_reviewer 终审失败且未超重试上限时，定向回退 writing_orchestrator
  只重写不合格章节；通过或超限（携未决引文警告）进入 human_review_gate。
- human_review_gate 经 LangGraph interrupt 真实中断等待人工；恢复后定稿走
  FINISHED 收束，修订指令回到 writing_orchestrator，再经 document_reviewer
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
from langgraph.types import Send
from psycopg import Connection
from psycopg.rows import dict_row
from pydantic import BaseModel

from agents.chapter_reviewer import make_chapter_reviewer
from agents.contracts import Subagent
from agents.rewriter_loop import make_rewriter_loop
from agents.search_agent import make_search_agent
from assembly.assembler_config import AssemblerConfig
from domain import state as domain_state
from domain.state import WorkflowStatus, WritingAgentState
from domain.units import MAIN_NODES
from llm import observability
from llm.llm_client import LLMFactory, default_llm_factory
from nodes.chapter_drafter import draft_send_payloads, make_chapter_drafter_node
from nodes.document_reviewer import ReviewerConfig, make_document_reviewer_node
from nodes.framework_orchestrator import make_framework_orchestrator_node
from nodes.human_review_gate import make_human_review_gate_node
from nodes.reference_orchestrator import (
    make_reference_orchestrator_node,
    reference_send_payloads,
)
from nodes.writing_orchestrator import (
    DIRECTIVE_CHAPTER_ID_KEY,
    directive_send_payloads,
    make_writing_orchestrator_node,
    next_writing_step,
    revision_send_payloads,
)

PG_DSN_ENV = "HYPOARGUS_PG_DSN"

# 检查点序列化的类型允许清单：domain.state 内定义的全部状态模型与枚举。
# 从模块自动收集而非手工罗列，新增状态模型时自动纳入，避免清单漂移。
CHECKPOINT_MSGPACK_TYPES: tuple[type, ...] = tuple(
    obj
    for obj in vars(domain_state).values()
    if isinstance(obj, type)
    and obj.__module__ == domain_state.__name__
    and issubclass(obj, (BaseModel, enum.Enum))
)


def checkpoint_serializer() -> JsonPlusSerializer:
    """检查点序列化器：把 domain.state 各类型显式注册进 msgpack 允许清单。

    LangGraph 对未注册类型的反序列化会告警并将在未来版本阻断；
    显式注册后严格模式（LANGGRAPH_STRICT_MSGPACK=true）下往返依然成立。
    """
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_MSGPACK_TYPES)

# 主节点名 → 主路径上进入该节点后状态机所处的枚举值。
# document_reviewer 与 human_review_gate 会按路由结果改写状态
# （见各节点实现），此处记录的是其主路径值。
NODE_STATUS: dict[str, WorkflowStatus] = {
    "framework_orchestrator": WorkflowStatus.FRAMEWORK_BUILDING,
    "reference_orchestrator": WorkflowStatus.REFERENCE_FETCHING,
    "chapter_drafter": WorkflowStatus.ARTICLE_WRITING,
    "writing_orchestrator": WorkflowStatus.ARTICLE_WRITING,
    "document_reviewer": WorkflowStatus.CITATION_CHECKING,
    "human_review_gate": WorkflowStatus.AWAIT_USER_REVIEW,
}

assert tuple(NODE_STATUS) == MAIN_NODES, "NODE_STATUS 必须与运行单元名册的主节点一致"


def route_after_framework_orchestrator(state: WritingAgentState) -> str | list[Send]:
    """框架完成后的路由：为每个待检索章节各发一个 Send 并行检索。

    Send 载荷（目标章 id + 装配所需状态切片）与选章判定收敛于
    reference_orchestrator.reference_send_payloads 单一事实源；
    没有任何待检索章节（全部章节无假说、或恢复续跑时素材已齐）时
    落回首写扇出路由，不经过检索节点。
    """
    payloads = reference_send_payloads(state)
    if not payloads:
        return route_after_reference_join(state)
    return [Send("reference_orchestrator", payload) for payload in payloads]


def reference_join(state: WritingAgentState) -> None:
    """检索并行分支的汇合点：无操作节点，不写任何状态。

    LangGraph 对同名节点并行任务的条件边是逐任务求值的，且每次求值只见
    该任务自身的写入；首写扇出必须基于全部检索分支合并后的完整引文库，
    故经本节点的静态边先汇合（静态边按节点名去重激活，目标只跑一次），
    下一超步再从这里的条件边做首写扇出。
    """
    return None


def revision_join(state: WritingAgentState) -> None:
    """人工修订并行分支的汇合点：无操作节点，不写任何状态。

    与检索的汇合点同理，条件边是在各 Send 分支的局部状态上分别求值。
    先经静态边汇合，确保终审只会看见所有已完成分支经 reducer 合并后的草稿。
    """
    return None


def route_after_reference_join(state: WritingAgentState) -> str | list[Send]:
    """检索汇合后的路由：为每个未写章节各发一个 Send 并行首写。

    Send 载荷（目标章 id + 装配所需状态切片）与选章判定收敛于
    chapter_drafter.draft_send_payloads 单一事实源；全部章节已有草稿
    （恢复续跑等场景）时直接前进终审。
    """
    payloads = draft_send_payloads(state)
    if not payloads:
        return "document_reviewer"
    return [Send("chapter_drafter", payload) for payload in payloads]


def route_after_writing_orchestrator(state: WritingAgentState) -> str | list[Send]:
    """写作后的路由：未完成的人工修订按章扇出，其余路径保留单章自环。

    人工修订的 Send 分支全部完成后，``revised_chapter_ids`` 已由 reducer
    合并；据此不再重发已完成章。终审回退与旧 checkpoint 的直调路径仍由
    ``next_writing_step`` 决定是否继续。
    """
    if isinstance(state.get(DIRECTIVE_CHAPTER_ID_KEY), str):
        # 条件边按每个 Send 分支的输入状态求值；不能在此重看全局待办并继续
        # 分发，否则分支会把自己重复发送。先经静态汇合节点，待所有分支的
        # reducer 写入合并后再统一前进终审。
        return "revision_join"
    directive_payloads = directive_send_payloads(state)
    if directive_payloads:
        return [Send("writing_orchestrator", payload) for payload in directive_payloads]
    if next_writing_step(state) is not None:
        return "writing_orchestrator"
    return "document_reviewer"


def route_after_document_reviewer(state: WritingAgentState) -> str | list[Send]:
    """终审后的路由：失败且未超限并行回退重写，通过或超限进入人工中断点。

    判定信号是 document_reviewer 显式写入的状态机值：CITATION_CHECKING
    表示还有重试预算与待修订失败章——为每个本轮未修订的失败章各发一个 Send
    到 writing_orchestrator 并行回退（对齐首写扇出，回退各章数据独立、
    只依赖本轮前 state + 共享 citation_report），全分支完成后汇合经
    route_after_writing_orchestrator 前进终审；AWAIT_USER_REVIEW 表示通过
    或超限交人工。无待修订载荷时回落串行 writing_orchestrator（防御，
    正常路径不达——CITATION_CHECKING 必伴非空 failed_chapter_ids）。
    """
    if state.get("status") == WorkflowStatus.CITATION_CHECKING:
        payloads = revision_send_payloads(state)
        if payloads:
            return [Send("writing_orchestrator", payload) for payload in payloads]
        return "writing_orchestrator"
    return "human_review_gate"


def route_after_human_review_gate(state: WritingAgentState) -> str | list[Send]:
    """人工中断点恢复后的路由：定稿收束，修订指令按章节并行扇出。"""
    if state.get("status") == WorkflowStatus.FINISHED:
        return END
    directive_payloads = directive_send_payloads(state)
    if directive_payloads:
        return [Send("writing_orchestrator", payload) for payload in directive_payloads]
    return "writing_orchestrator"


def build_graph(
    llm_factory: LLMFactory = default_llm_factory,
    checkpointer: BaseCheckpointSaver | None = None,
    search_agent: Subagent | None = None,
    rewriter_loop: Subagent | None = None,
    chapter_reviewer: Subagent | None = None,
    document_review_max_retries: int | None = None,
    assembler_config: AssemblerConfig | None = None,
) -> CompiledStateGraph:
    """构建并编译迭代闭环图。

    llm_factory 是确定性假 LLM 的测试注入点；
    search_agent 未注入时使用真实现工厂（make_search_agent：检索引擎
    无状态一次性调用，构造零环境依赖、首次检索才触碰引擎配置）；
    rewriter_loop 未注入时使用真实现工厂（make_rewriter_loop：构图时读取
    一次写作环境配置并按单元名取 LLM）；打桩仅在显式注入处使用；
    document_review_max_retries 未注入时按环境变量 DOCUMENT_REVIEW_MAX_RETRIES（缺省 2）；
    assembler_config 未注入时各节点执行期按环境变量读取装配配置。
    人工中断点依赖存档器恢复，生产运行必须传入 checkpointer。
    Langfuse 启用时节点函数与子智能体适配层被包进运行单元 span。
    """
    effective_search_agent = observability.wrap_subagent(
        search_agent or make_search_agent()
    )
    effective_rewriter_loop = observability.wrap_subagent(
        rewriter_loop or make_rewriter_loop(llm_factory)
    )
    # chapter_reviewer 未注入时用真实现工厂（ADR-0006）：模型保持 plus（回落全局配置）。
    # 首写路径经章级写→评→重写循环消费其修订说明（ADR-0006 T3）；
    # 修订/终审回退分支的评审消费留 T3b。
    effective_chapter_reviewer = observability.wrap_subagent(
        chapter_reviewer or make_chapter_reviewer(llm_factory)
    )

    node_functions = {
        "framework_orchestrator": make_framework_orchestrator_node(
            llm_factory, assembler_config=assembler_config
        ),
        "reference_orchestrator": make_reference_orchestrator_node(
            effective_search_agent, assembler_config
        ),
        "chapter_drafter": make_chapter_drafter_node(
            effective_rewriter_loop, effective_chapter_reviewer, assembler_config
        ),
        "writing_orchestrator": make_writing_orchestrator_node(
            effective_rewriter_loop,
            effective_search_agent,
            assembler_config,
            effective_chapter_reviewer,
        ),
        "document_reviewer": make_document_reviewer_node(
            llm_factory,
            ReviewerConfig(max_retries=document_review_max_retries)
            if document_review_max_retries is not None
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
    builder.add_conditional_edges(
        "framework_orchestrator",
        route_after_framework_orchestrator,
        ["reference_orchestrator", "chapter_drafter", "document_reviewer"],
    )
    builder.add_node("reference_join", reference_join)
    builder.add_edge("reference_orchestrator", "reference_join")
    builder.add_conditional_edges(
        "reference_join",
        route_after_reference_join,
        ["chapter_drafter", "document_reviewer"],
    )
    builder.add_edge("chapter_drafter", "document_reviewer")
    builder.add_node("revision_join", revision_join)
    builder.add_edge("revision_join", "document_reviewer")
    builder.add_conditional_edges(
        "writing_orchestrator",
        route_after_writing_orchestrator,
        ["writing_orchestrator", "revision_join", "document_reviewer"],
    )
    builder.add_conditional_edges(
        "document_reviewer",
        route_after_document_reviewer,
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
    """按环境变量连接串创建 Postgres 检查点保存器，并确保建表完成。

    不走 from_conn_string 而手工建连接：该便捷入口不接受序列化器参数，
    这里必须注入注册了 domain.state 类型的 checkpoint_serializer；
    连接参数与 PostgresSaver.from_conn_string 内部保持一致。
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
