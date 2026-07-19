"""graph_event_stream 翻译器测试：合成流块单测 + 真图集成测。

第一层用手工构造的 (mode, chunk) 序列验证翻译规则与父子链路；
第二层用 build_graph + FakeLLM + InMemorySaver 实跑
stream_mode=["updates","debug"]，覆盖首跑到中断、恢复定稿与子智能体钩子。
"""

import json
import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from service.event_envelope import EventEnvelope
from graph import build_graph
from service.graph_event_stream import GraphRunEmitter
from llm.llm_client import FakeLLM
from domain.events import SUBAGENT_END, SUBAGENT_PROGRESS, SUBAGENT_START
from domain.state import WorkflowStatus, initial_state
from agents.rewriter_loop import make_stub_rewriter_loop
from agents.search_agent import make_stub_search_agent
from tests.llm_response_plans import FIRST_PASS_RESPONSES, FRAMEWORK_KEYED_RESPONSES


def _make_emitter(
    events: list[EventEnvelope], thread_id: str = "t-1"
) -> GraphRunEmitter:
    """构造把事件收进列表的翻译器。"""
    return GraphRunEmitter(
        publish=events.append,
        trace_id="trace-1",
        session_id="session-1",
        thread_id=thread_id,
    )


def _task_chunk(node: str, step: int) -> dict:
    """按实测形态构造 debug task 块。"""
    return {
        "type": "task",
        "step": step,
        "timestamp": "2026-07-18T00:00:00+00:00",
        "payload": {
            "id": f"tid-{node}",
            "name": node,
            "input": {},
            "triggers": (f"branch:to:{node}",),
        },
    }


def _task_result_chunk(node: str, step: int, interrupts: list | None = None) -> dict:
    """按实测形态构造 debug task_result 块。"""
    return {
        "type": "task_result",
        "step": step,
        "timestamp": "2026-07-18T00:00:01+00:00",
        "payload": {
            "id": f"tid-{node}",
            "name": node,
            "error": None,
            "result": {},
            "interrupts": interrupts or [],
        },
    }


class _FakeInterrupt:
    """模拟 langgraph Interrupt：只需 value 属性。"""

    def __init__(self, value: dict) -> None:
        self.value = value


def test_合成流块_节点启停与父子链路闭合():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    root_id = emitter.emit_root(
        type="progress", unit="graph", payload={"phase": "任务创建"}
    )

    emitter.handle("debug", _task_chunk("framework_orchestrator", 1))
    emitter.handle(
        "updates",
        {
            "framework_orchestrator": {
                "status": WorkflowStatus.FRAMEWORK_BUILDING,
                "outline": ["ch1", "ch2"],
                "current_node_llm_config": {"unit": "framework_orchestrator"},
            }
        },
    )
    emitter.handle("debug", _task_result_chunk("framework_orchestrator", 1))
    # checkpoint 类块必须被忽略。
    emitter.handle("debug", {"type": "checkpoint", "step": 1, "payload": {}})

    types = [event.type for event in events]
    assert types == [
        "progress",
        "node_start",
        "llm_config_used",
        "state_snapshot",
        "progress",
        "node_end",
    ]
    node_start = events[1]
    node_end = events[-1]
    # 父子链路：node_start 挂根事件，派生事件与 node_end 挂 node_start。
    assert node_start.parent_id == root_id
    assert node_end.parent_id == node_start.event_id
    assert all(
        event.parent_id == node_start.event_id for event in events[2:5]
    )
    assert emitter.last_status is WorkflowStatus.FRAMEWORK_BUILDING


def test_合成流块_state_snapshot只含元数据且状态为纯字符串():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    emitter.emit_root(type="progress", unit="graph", payload={})
    emitter.handle("debug", _task_chunk("writing_orchestrator", 3))
    emitter.handle(
        "updates",
        {
            "writing_orchestrator": {
                "status": WorkflowStatus.ARTICLE_WRITING,
                "chapter_drafts": [object(), object()],
            }
        },
    )

    snapshot = next(event for event in events if event.type == "state_snapshot")
    assert set(snapshot.payload) == {
        "status",
        "iteration_round",
        "chapter_total",
        "chapters_completed",
        "material_count",
        "citation_retry_count",
        "citation_warning_count",
    }
    assert snapshot.payload["status"] == "ARTICLE_WRITING"
    assert snapshot.payload["chapters_completed"] == 2
    # 状态机值必须是纯字符串（不带枚举类名前缀）。
    assert "WorkflowStatus" not in json.dumps(snapshot.payload)


def test_合成流块_终审失败触发branch_taken回退与loop_iteration():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    emitter.emit_root(type="progress", unit="graph", payload={})
    emitter.handle("debug", _task_chunk("citation_validator", 4))
    emitter.handle(
        "updates",
        {
            "citation_validator": {
                "status": WorkflowStatus.CITATION_CHECKING,
                "citation_retry_count": 1,
            }
        },
    )

    branch = next(event for event in events if event.type == "branch_taken")
    assert branch.payload == {
        "from": "citation_validator",
        "to": "writing_orchestrator",
        "reason": "引文终审失败，定向回退重写",
    }
    loop = next(event for event in events if event.type == "loop_iteration")
    assert loop.payload == {"loop": "citation_retry", "round": 1}
    assert branch.unit == loop.unit == "citation_validator"


def test_合成流块_终审通过branch_taken去人工中断点且无loop_iteration():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    emitter.emit_root(type="progress", unit="graph", payload={})
    emitter.handle("debug", _task_chunk("citation_validator", 4))
    emitter.handle(
        "updates",
        {"citation_validator": {"status": WorkflowStatus.AWAIT_USER_REVIEW}},
    )

    branch = next(event for event in events if event.type == "branch_taken")
    assert branch.payload["to"] == "human_review_gate"
    assert not [event for event in events if event.type == "loop_iteration"]


def test_合成流块_人工恢复修订触发branch_taken与revision循环():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    emitter.emit_root(type="gate_resumed", unit="human_review_gate", payload={})
    emitter.handle("debug", _task_chunk("human_review_gate", 5))
    emitter.handle(
        "updates",
        {
            "human_review_gate": {
                "status": WorkflowStatus.AWAIT_USER_REVIEW,
                "iteration_round": 2,
            }
        },
    )

    branch = next(event for event in events if event.type == "branch_taken")
    assert branch.payload["to"] == "writing_orchestrator"
    loop = next(event for event in events if event.type == "loop_iteration")
    assert loop.payload == {"loop": "revision", "round": 2}


def test_合成流块_中断转gate_blocked并记录载荷():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    emitter.emit_root(type="progress", unit="graph", payload={})
    emitter.handle("debug", _task_chunk("human_review_gate", 5))
    payload = {"iteration_round": 0, "chapter_ids": ["ch1"], "citation_warnings": []}
    emitter.handle("updates", {"__interrupt__": (_FakeInterrupt(payload),)})

    gate = next(event for event in events if event.type == "gate_blocked")
    assert gate.unit == "human_review_gate"
    assert gate.payload == payload
    node_start = next(event for event in events if event.type == "node_start")
    assert gate.parent_id == node_start.event_id
    assert emitter.interrupt_payload == payload


def test_合成流块_异常转node_error归属当前节点():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    emitter.emit_root(type="progress", unit="graph", payload={})
    emitter.handle("debug", _task_chunk("reference_orchestrator", 2))
    emitter.handle_error(RuntimeError("检索超时"))

    error = next(event for event in events if event.type == "node_error")
    assert error.unit == "reference_orchestrator"
    assert error.payload == {"error_type": "RuntimeError", "message": "检索超时"}
    node_start = next(event for event in events if event.type == "node_start")
    assert error.parent_id == node_start.event_id


def test_合成流块_无节点在执行时node_error归属graph并挂根事件():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    root_id = emitter.emit_root(type="progress", unit="graph", payload={})
    emitter.handle_error(ValueError("配置缺失"))

    error = events[-1]
    assert error.type == "node_error"
    assert error.unit == "graph"
    assert error.parent_id == root_id


def test_合成流块_seed后快照沿用检查点计数():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    emitter.seed(
        {
            "status": WorkflowStatus.AWAIT_USER_REVIEW,
            "outline": ["ch1", "ch2"],
            "chapter_drafts": ["d1", "d2"],
            "citation_library": ["m1", "m2"],
            "iteration_round": 1,
        }
    )
    emitter.emit_root(type="gate_resumed", unit="human_review_gate", payload={})
    emitter.handle("debug", _task_chunk("human_review_gate", 5))
    emitter.handle(
        "updates",
        {"human_review_gate": {"status": WorkflowStatus.FINISHED}},
    )

    snapshot = next(event for event in events if event.type == "state_snapshot")
    # 恢复场景：本次 updates 未携带的字段由 seed 的检查点值补齐。
    assert snapshot.payload["chapter_total"] == 2
    assert snapshot.payload["chapters_completed"] == 2
    assert snapshot.payload["material_count"] == 2
    assert snapshot.payload["iteration_round"] == 1
    assert snapshot.payload["status"] == "FINISHED"
    branch = next(event for event in events if event.type == "branch_taken")
    assert branch.payload["to"] == "END"


def test_合成流块_子智能体progress事件挂最近一次subagent_start():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    emitter.emit_root(type="progress", unit="graph", payload={})
    # 先有当前主节点，subagent_start 才能挂到该节点的 node_start 之下。
    emitter.handle("debug", _task_chunk("writing_orchestrator", 3))
    hook = emitter.make_subagent_hook()

    start_payload = {"unit": "rewriter_loop", "chapter_id": "ch-1", "mode": "draft"}
    progress_payload = {
        "unit": "rewriter_loop",
        "chapter_id": "ch-1",
        "mode": "draft",
        "step": "llm_call_start",
    }
    hook(SUBAGENT_START, start_payload)
    hook(SUBAGENT_PROGRESS, progress_payload)
    hook(SUBAGENT_END, {"unit": "rewriter_loop", "chapter_id": "ch-1", "mode": "draft"})

    start = next(event for event in events if event.type == "subagent_start")
    end = next(event for event in events if event.type == "subagent_end")
    # progress 信封复用既有 progress 类型，按 unit 区分子智能体来源。
    progress = next(
        event
        for event in events
        if event.type == "progress" and event.unit == "rewriter_loop"
    )
    assert progress.parent_id == start.event_id
    # 载荷原样透传（含 step 等子智能体自带上下文），且是独立副本。
    assert progress.payload == progress_payload
    assert progress.payload is not progress_payload
    # 启停配对链路不受 progress 分支影响。
    assert end.parent_id == start.event_id
    node_start = next(event for event in events if event.type == "node_start")
    assert start.parent_id == node_start.event_id


def test_合成流块_子智能体progress无前置start时挂根事件():
    events: list[EventEnvelope] = []
    emitter = _make_emitter(events)
    root_id = emitter.emit_root(type="progress", unit="graph", payload={})
    hook = emitter.make_subagent_hook()

    hook(
        SUBAGENT_PROGRESS,
        {"unit": "rewriter_loop", "chapter_id": None, "mode": None, "step": "warmup"},
    )

    progress = events[-1]
    assert progress.type == "progress"
    assert progress.unit == "rewriter_loop"
    assert progress.parent_id == root_id
    assert progress.payload["step"] == "warmup"


def _run_first_pass(events: list[EventEnvelope]):
    """真图首跑到人工中断点：返回（graph, config, emitter）。"""
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    emitter = GraphRunEmitter(
        publish=events.append,
        trace_id="trace-int",
        session_id="session-int",
        thread_id="thread-int",
    )
    graph = build_graph(
        llm_factory=lambda unit: fake,
        checkpointer=InMemorySaver(),
        search_agent=make_stub_search_agent(emitter.make_subagent_hook()),
        rewriter_loop=make_stub_rewriter_loop(emitter.make_subagent_hook()),
    )
    config: RunnableConfig = {"configurable": {"thread_id": f"evt-{uuid.uuid4()}"}}
    emitter.emit_root(
        type="progress", unit="graph", payload={"phase": "任务创建"}
    )
    for mode, chunk in graph.stream(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-int"),
        config,
        stream_mode=["updates", "debug"],
    ):
        emitter.handle(mode, chunk)
    return graph, config, emitter


def test_真图集成_首跑到中断点事件齐全且字段完整():
    events: list[EventEnvelope] = []
    _, _, emitter = _run_first_pass(events)

    types = {event.type for event in events}
    assert {
        "node_start",
        "node_end",
        "state_snapshot",
        "llm_config_used",
        "progress",
        "branch_taken",
        "gate_blocked",
    } <= types

    # 每条事件字段齐全：event_id 全局唯一，trace/session/thread 正确，ts 非空。
    assert len({event.event_id for event in events}) == len(events)
    for event in events:
        assert event.trace_id == "trace-int"
        assert event.session_id == "session-int"
        assert event.thread_id == "thread-int"
        assert event.ts

    # 停在人工中断点：中断载荷可取且只含元数据。
    assert emitter.interrupt_payload is not None
    assert emitter.interrupt_payload["chapter_ids"] == ["ch1", "ch2"]
    assert emitter.last_status is WorkflowStatus.AWAIT_USER_REVIEW

    # 快照绝不含正文：全部事件载荷里都不出现打桩正文特征串。
    all_payloads = json.dumps(
        [event.payload for event in events], ensure_ascii=False, default=str
    )
    assert "打桩正文" not in all_payloads
    assert "打桩摘要" not in all_payloads


def test_真图集成_node链路闭合与branch_taken指向中断点():
    events: list[EventEnvelope] = []
    _, _, _ = _run_first_pass(events)

    root_id = events[0].event_id
    node_starts = {
        event.unit: event for event in events if event.type == "node_start"
    }
    assert set(node_starts) == {
        "framework_orchestrator",
        "reference_orchestrator",
        "writing_orchestrator",
        "citation_validator",
        "human_review_gate",
    }
    for event in node_starts.values():
        assert event.parent_id == root_id
    for event in events:
        if event.type in {"node_end", "state_snapshot", "llm_config_used", "progress"}:
            if event.parent_id is None:
                continue  # 根 progress 事件本身。
            assert event.parent_id == node_starts[event.unit].event_id

    branch = next(event for event in events if event.type == "branch_taken")
    assert branch.unit == "citation_validator"
    assert branch.payload["to"] == "human_review_gate"


def test_真图集成_子智能体事件成对且挂当前节点():
    events: list[EventEnvelope] = []
    _run_first_pass(events)

    node_starts = {
        event.unit: event for event in events if event.type == "node_start"
    }
    starts = [event for event in events if event.type == "subagent_start"]
    ends = [event for event in events if event.type == "subagent_end"]
    # 2 章：search_agent 每章一次、rewriter_loop 每章一次，成对出现。
    assert len(starts) == len(ends) == 4
    assert {event.unit for event in starts} == {"search_agent", "rewriter_loop"}

    # subagent_start 挂当前主节点的 node_start。
    expected_parent = {
        "search_agent": node_starts["reference_orchestrator"].event_id,
        "rewriter_loop": node_starts["writing_orchestrator"].event_id,
    }
    for event in starts:
        assert event.parent_id == expected_parent[event.unit]

    # subagent_end 挂对应的 subagent_start：按单元名配对顺序逐一对应。
    start_ids = {event.event_id for event in starts}
    for event in ends:
        assert event.parent_id in start_ids

    # 启停载荷携带任务上下文：章节 id 可判定时必带，mode 仅改写任务有。
    for event in starts + ends:
        assert event.payload["chapter_id"] in {"ch1", "ch2"}
        if event.unit == "rewriter_loop":
            assert event.payload["mode"] == "draft"
        else:
            assert event.payload["mode"] is None


def test_真图集成_恢复定稿第二个emitter产出END分支():
    first_events: list[EventEnvelope] = []
    graph, config, _ = _run_first_pass(first_events)

    resume_events: list[EventEnvelope] = []
    emitter = GraphRunEmitter(
        publish=resume_events.append,
        trace_id="trace-int",
        session_id="session-int",
        thread_id="thread-int",
    )
    emitter.seed(graph.get_state(config).values)
    root_id = emitter.emit_root(
        type="gate_resumed",
        unit="human_review_gate",
        payload={"action": "finalize"},
    )
    for mode, chunk in graph.stream(
        Command(resume={"action": "finalize"}), config, stream_mode=["updates", "debug"]
    ):
        emitter.handle(mode, chunk)

    branch = next(
        event for event in resume_events if event.type == "branch_taken"
    )
    assert branch.unit == "human_review_gate"
    assert branch.payload["to"] == "END"
    assert emitter.last_status is WorkflowStatus.FINISHED
    assert emitter.interrupt_payload is None

    # 恢复运行的 node_start 仍挂本次根事件；快照由 seed 补齐章节计数。
    node_start = next(
        event for event in resume_events if event.type == "node_start"
    )
    assert node_start.unit == "human_review_gate"
    assert node_start.parent_id == root_id
    snapshot = next(
        event for event in resume_events if event.type == "state_snapshot"
    )
    assert snapshot.payload["chapter_total"] == 2
    assert snapshot.payload["chapters_completed"] == 2
    assert snapshot.payload["status"] == "FINISHED"
