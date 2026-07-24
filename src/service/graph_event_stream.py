"""LangGraph 原生事件流到事件信封的翻译器。

服务层用 ``graph.stream(..., stream_mode=["updates", "debug"])`` 驱动一次图运行，
本模块把产出的 (mode, chunk) 流块翻译为 EventEnvelope 并通过回调发布。

实测（langgraph 1.2.8）的流块形态：
- debug 块是 dict，含 type / step / timestamp / payload 四键；
  type=="task" 时 payload 含 id / name / input / triggers，
  type=="task_result" 时 payload 含 id / name / error / result / interrupts，
  type=="checkpoint" 的块忽略。
- updates 块是 ``{节点名: 部分状态}``，中断时是 ``{"__interrupt__": (Interrupt, ...)}``。
- 串行路径的到达顺序恒为：task → updates（该节点）→ task_result，
  因此节点内派生事件的父 id 总能取到该节点的 node_start。
- 并行扇出（检索阶段的 reference_orchestrator 与首写阶段的 chapter_drafter）
  时同名节点有多个任务交错到达：node_start/node_end 按 LangGraph task id
  配对，子智能体成对事件按（单元名, chapter_id）配对；updates 块无任务
  标识，其派生事件的父 id 退化为该节点最近一次 node_start
  （视觉归组可接受，链路不丢）。

父子链路规则：根事件（服务层经 emit_root 发出）→ node_start →
节点派生事件（node_end / state_snapshot / llm_config_used / branch_taken /
loop_iteration / progress / gate_blocked / node_error / subagent_start）→
subagent_end 与子智能体内部进度（progress）挂在对应 subagent_start 之下。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from service.event_envelope import EventEnvelope, new_envelope
from domain.units import GRAPH_UPDATE_NODES, graph_node_unit
from domain.state import WorkflowStatus, status_text
from domain.events import (
    CONTENT_DELTA,
    SUBAGENT_END,
    SUBAGENT_PROGRESS,
    SUBAGENT_START,
    EventHook,
)

# 条件路由节点 → 按状态机值判定的路由去向与理由。
_BRANCH_RULES: dict[str, dict[WorkflowStatus, tuple[str, str]]] = {
    "document_reviewer": {
        WorkflowStatus.CITATION_CHECKING: (
            "writing_orchestrator",
            "篇级终审失败，定向回退重写",
        ),
    },
    "human_review_gate": {
        WorkflowStatus.FINISHED: ("END", "定稿收束"),
        WorkflowStatus.AWAIT_USER_REVIEW: (
            "writing_orchestrator",
            "执行修订指令",
        ),
    },
}


def make_standalone_subagent_hook(
    *,
    publish: Callable[[EventEnvelope], None],
    trace_id: str,
    session_id: str,
    thread_id: str = "",
) -> EventHook:
    """独立子智能体调用（不经图运行）的事件钩子：成对事件入信封、进度挂 start 之下。

    供独立检索接口使用：没有图任务块，不代发 node_start，subagent_start
    即本次调用的根事件（parent_id 为 None）。thread_id 缺省空串——独立调用
    不隶属任何写作任务线程，订阅方按 session_id 过滤。
    引擎事件可能从其他线程分派，配对状态在锁内维护。
    """
    lock = threading.Lock()
    start_ids: dict[tuple[str, str | None], str] = {}

    def emit(
        type: str, unit: str, payload: dict[str, Any], parent_id: str | None
    ) -> EventEnvelope:
        envelope = new_envelope(
            type=type,
            unit=unit,
            payload=payload,
            trace_id=trace_id,
            session_id=session_id,
            thread_id=thread_id,
            parent_id=parent_id,
        )
        publish(envelope)
        return envelope

    def hook(event_type: str, payload: dict[str, Any]) -> None:
        unit = str(payload.get("unit", "subagent"))
        chapter_id = payload.get("chapter_id")
        key = (unit, chapter_id if isinstance(chapter_id, str) else None)
        with lock:
            if event_type == SUBAGENT_START:
                envelope = emit("subagent_start", unit, dict(payload), None)
                start_ids[key] = envelope.event_id
            elif event_type == SUBAGENT_PROGRESS:
                emit("progress", unit, dict(payload), start_ids.get(key))
            elif event_type == SUBAGENT_END:
                emit("subagent_end", unit, dict(payload), start_ids.pop(key, None))
            # CONTENT_DELTA 不出现在独立检索（写作子智能体才会发逐字流）；
            # 显式忽略以防误触可视化信封发布（逐字流只走业务通道，不入 graph_events）。
            elif event_type == CONTENT_DELTA:
                return

    return hook


class GraphRunEmitter:
    """一次图运行（首跑或恢复续跑）的事件翻译器：把 LangGraph 流块翻译为事件信封并发布。"""

    def __init__(
        self,
        *,
        publish: Callable[[EventEnvelope], None],
        trace_id: str,
        session_id: str,
        thread_id: str,
        publish_business: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """构造一次图运行的翻译器。

        ``publish`` 把可视化信封发布到全局 ``graph_hub``；
        ``publish_business`` 把逐字流 ``CONTENT_DELTA`` 翻译为业务通道事件
        发布到该任务的 ``entry.hub``（载荷剥离 ``unit`` 字段，避免与业务通道
        事件契约重复——业务事件 ``data`` 块按 ``content_delta`` 票据契约
        只带 chapter_id / mode / kind / delta / attempt / sequence）。
        ``publish_business`` 缺省 None 时 ``CONTENT_DELTA`` 被静默丢弃
        （独立检索等无业务通道的场景；图运行路径总会注入）。
        """
        self._publish = publish
        self._trace_id = trace_id
        self._session_id = session_id
        self._thread_id = thread_id
        self._publish_business = publish_business
        self._root_id: str | None = None
        self._lock = threading.RLock()
        """并行首写时子智能体钩子在执行器线程发事件、流块在驱动线程处理，
        两侧共享配对状态：全部公共入口在此锁内执行，保证配对与去重原子。"""
        self._node_start_ids: dict[str, str] = {}
        """节点名 → 该节点最近一次 node_start 的 event_id。"""
        self._task_start_ids: dict[str, str] = {}
        """LangGraph task id → 该任务 node_start 的 event_id：并行扇出时
        同名节点有多个任务在跑，node_end 按 task id 配对各自的 node_start。"""
        self._chapter_node_start_ids: dict[tuple[str, str], str] = {}
        """（节点名, 目标章 id）→ 该章并行分支 node_start 的 event_id：
        检索（reference_orchestrator）与首写（chapter_drafter）两段扇出的
        子智能体事件按章节确定性挂到所属分支；同一章在两段各有一个分支，
        故键须带节点名区分。子智能体事件先于 debug 任务块到达时由发射器
        代为发出该分支的 node_start（见 _ensure_chapter_node_start），
        真实任务块随后到达时复用不重复发。"""
        self._subagent_start_ids: dict[tuple[str, str | None], str] = {}
        """（子智能体单元名, chapter_id）→ 最近一次 subagent_start 的 event_id：
        并行首写时同一单元的多个实例按章节区分，成对事件不互相覆盖。"""
        self._current_node: str | None = None
        """当前执行中的主节点（task 已到、task_result 未到）。
        并行扇出时是最近启动的任务所在节点，仅用于错误归属与父链兜底。"""
        self._completed_chapter_ids: set[str] = set()
        """已观察到草稿的章节 id 累积：并行分支的 update 只带单章列表，
        chapters_completed 由此累积集合计数而非单次 update 的长度。"""
        self._material_ids: set[str] = set()
        """已观察到素材的 id 累积：并行检索分支的 update 只带单章素材，
        material_count 由此累积集合计数而非单次 update 的长度（合并 reducer
        跨章按 URL 去重丢弃的条目不从计数回收，计数语义为「观察到的素材」）。"""
        self._interrupt_payload: dict[str, Any] | None = None
        self._last_status: WorkflowStatus | None = None
        # 快照元数据累积器：跨流块累积，保证每条 state_snapshot 字段完整。
        self._snapshot: dict[str, Any] = {
            "status": None,
            "iteration_round": 0,
            "chapter_total": 0,
            "chapters_completed": 0,
            "material_count": 0,
            "citation_retry_count": 0,
            "citation_warning_count": 0,
        }

    @property
    def interrupt_payload(self) -> dict[str, Any] | None:
        """本次运行若停在人工中断点，此处为中断载荷；否则 None。"""
        return self._interrupt_payload

    @property
    def last_status(self) -> WorkflowStatus | None:
        """最近一次 updates 里观察到的状态机枚举值。"""
        return self._last_status

    def seed(self, values: Mapping[str, Any]) -> None:
        """用检查点中已有状态初始化快照元数据累积器（恢复/回滚场景，保证快照字段完整）。"""
        self._accumulate(values)

    def emit_root(self, *, type: str, unit: str, payload: dict[str, Any]) -> str:
        """发布本次运行的根事件并记住其 event_id 作为后续 node_start 的父 id，返回 event_id。"""
        envelope = self._emit(type=type, unit=unit, payload=payload, parent_id=None)
        self._root_id = envelope.event_id
        return envelope.event_id

    def handle(self, mode: str, chunk: Any) -> None:
        """处理 stream_mode=["updates","debug"] 产出的一个流块。"""
        with self._lock:
            if mode == "debug":
                self._handle_debug(chunk)
            elif mode == "updates":
                self._handle_updates(chunk)

    def handle_error(self, exc: BaseException) -> None:
        """图运行抛异常时发布 node_error（unit 为当前执行中节点，未知则 "graph"；payload 含异常类型与消息）。"""
        with self._lock:
            unit = self._current_node or "graph"
            self._emit(
                type="node_error",
                unit=unit,
                payload={"error_type": type(exc).__name__, "message": str(exc)},
                parent_id=self._parent_for(unit),
            )

    def make_subagent_hook(self) -> EventHook:
        """返回可注入子智能体适配层的事件钩子：把 subagent_start/subagent_end 转成信封发布，
        子智能体内部进度（SUBAGENT_PROGRESS）转成信封既有的 progress 类型、父指向当前 subagent_start。

        并行首写分支（rewriter_loop 的 draft 模式）按章节确定性挂接：
        subagent_start 的父 id 取该章 chapter_drafter 分支的 node_start，
        分支 node_start 未到达时由发射器代为发出（消除跨线程到达竞态）。
        """

        def hook(event_type: str, payload: dict[str, Any]) -> None:
            with self._lock:
                self._handle_subagent_event(event_type, payload)

        return hook

    def _handle_subagent_event(
        self, event_type: str, payload: dict[str, Any]
    ) -> None:
        """子智能体事件 → 信封：成对键为（单元名, chapter_id），并行实例互不覆盖。

        ``CONTENT_DELTA`` 例外：逐字流只走业务通道（``publish_business``），
        永不进可视化信封（ADR-0001：可视化通道只放元数据，逐字流正文属
        可丢级业务载荷）；``publish_business`` 缺省时静默丢弃（独立检索等
        无业务通道场景，但写作子智能体不会走 standalone 钩子）。
        """
        if event_type == CONTENT_DELTA:
            if self._publish_business is not None:
                self._publish_business(
                    "content_delta",
                    {k: v for k, v in payload.items() if k != "unit"},
                )
            return
        unit = str(payload.get("unit", "subagent"))
        chapter_id = payload.get("chapter_id")
        key = (unit, chapter_id if isinstance(chapter_id, str) else None)
        if event_type == SUBAGENT_START:
            envelope = self._emit(
                type="subagent_start",
                unit=unit,
                payload=dict(payload),
                parent_id=self._subagent_parent(payload),
            )
            self._subagent_start_ids[key] = envelope.event_id
            if unit == "rewriter_loop" and payload.get("mode") == "draft":
                self._emit(
                    type="pipeline_timing",
                    unit="reference_orchestrator",
                    payload={"chapter_id": chapter_id, "phase": "first_draft_started"},
                    parent_id=envelope.parent_id,
                )
        elif event_type == SUBAGENT_PROGRESS:
            self._emit(
                type="progress",
                unit=unit,
                payload=dict(payload),
                parent_id=self._subagent_start_ids.get(key, self._root_id),
            )
        elif event_type == SUBAGENT_END:
            self._emit(
                type="subagent_end",
                unit=unit,
                payload=dict(payload),
                parent_id=self._subagent_start_ids.pop(key, self._root_id),
            )
            if unit == "search_agent":
                parent_id = self._ensure_chapter_node_start(
                    "reference_orchestrator", str(chapter_id)
                )
                for phase in ("retrieval_completed", "first_draft_queued"):
                    self._emit(
                        type="pipeline_timing",
                        unit="reference_orchestrator",
                        payload={"chapter_id": chapter_id, "phase": phase},
                        parent_id=parent_id,
                    )

    def _subagent_parent(self, payload: dict[str, Any]) -> str | None:
        """subagent_start 的父 id：并行分支按章节配对，其余挂当前节点。

        首写流水线的 rewriter_loop draft 模式在 reference_orchestrator 分支
        内执行，故按章节挂到真实的检索分支 node_start，而不虚构独立节点；
        writing_orchestrator 的 draft 分支仅作防御、正常不可达。
        search_agent 带 chapter_id 时同理挂检索分支——例外是修订轮的
        增量检索从串行的 writing_orchestrator 发起（串行超步内任务块
        先于子智能体事件处理，_current_node 判定无竞态），挂当前节点；
        其余场景沿用「当前执行中节点的 node_start，未知退根事件」。
        """
        chapter_id = payload.get("chapter_id")
        if payload.get("mode") == "draft" and isinstance(chapter_id, str):
            return self._ensure_chapter_node_start(
                "reference_orchestrator", chapter_id
            )
        if (
            payload.get("unit") == "search_agent"
            and isinstance(chapter_id, str)
            and self._current_node != "writing_orchestrator"
        ):
            return self._ensure_chapter_node_start(
                "reference_orchestrator", chapter_id
            )
        return self._parent_for(self._current_node or "")

    def _ensure_chapter_node_start(
        self,
        node: str,
        chapter_id: str,
        step: Any = None,
        task_id: str | None = None,
    ) -> str:
        """取（或代发）指定章节并行分支（检索或首写）的 node_start，返回其 event_id。

        子智能体事件在执行器线程可能先于驱动线程处理该分支的 debug 任务块，
        此时代为发出 node_start 保证父链确定；真实任务块随后到达时复用同一
        event_id，不重复发 node_start，仅补记 task id 配对（供 node_end 使用）。
        """
        existing = self._chapter_node_start_ids.get((node, chapter_id))
        if existing is not None:
            if task_id is not None:
                self._task_start_ids[task_id] = existing
            return existing
        start_payload: dict[str, Any] = {"chapter_id": chapter_id}
        if step is not None:
            start_payload["step"] = step
        envelope = self._emit(
            type="node_start",
            unit=node,
            payload=start_payload,
            parent_id=self._root_id,
        )
        self._chapter_node_start_ids[(node, chapter_id)] = envelope.event_id
        self._node_start_ids[node] = envelope.event_id
        if task_id is not None:
            self._task_start_ids[task_id] = envelope.event_id
        self._current_node = node
        return envelope.event_id

    # ---- 内部实现 ----

    def _emit(
        self,
        *,
        type: str,
        unit: str,
        payload: dict[str, Any],
        parent_id: str | None,
    ) -> EventEnvelope:
        """构造并发布一条信封。"""
        envelope = new_envelope(
            type=type,
            unit=unit,
            payload=payload,
            trace_id=self._trace_id,
            session_id=self._session_id,
            thread_id=self._thread_id,
            parent_id=parent_id,
        )
        self._publish(envelope)
        return envelope

    def _parent_for(self, node: str) -> str | None:
        """节点派生事件的父 id：该节点的 node_start，未到达则退根事件。"""
        return self._node_start_ids.get(node, self._root_id)

    def _handle_debug(self, chunk: Any) -> None:
        """debug 流块：task → node_start，task_result → node_end，checkpoint 忽略。"""
        if not isinstance(chunk, dict):
            return
        chunk_type = chunk.get("type")
        payload = chunk.get("payload")
        if not isinstance(payload, dict):
            return
        node = payload.get("name")
        if not isinstance(node, str) or node not in GRAPH_UPDATE_NODES:
            return
        unit = graph_node_unit(node)
        task_id = payload.get("id")
        if chunk_type == "task":
            chapter_id = self._task_chapter_id(payload)
            if chapter_id is not None:
                # 并行分支（检索或首写）：经（节点, 章节）键控入口取（或代发）
                # node_start，子智能体事件先到时已代发过，这里复用不重复发。
                self._ensure_chapter_node_start(
                    unit,
                    chapter_id,
                    step=chunk.get("step"),
                    task_id=task_id if isinstance(task_id, str) else None,
                )
                self._current_node = unit
                return
            envelope = self._emit(
                type="node_start",
                unit=unit,
                payload={"step": chunk.get("step")},
                parent_id=self._root_id,
            )
            self._node_start_ids[unit] = envelope.event_id
            if isinstance(task_id, str):
                self._task_start_ids[task_id] = envelope.event_id
            self._current_node = unit
        elif chunk_type == "task_result":
            end_payload: dict[str, Any] = {"step": chunk.get("step")}
            chapter_id = self._task_chapter_id(payload)
            if chapter_id is not None:
                end_payload["chapter_id"] = chapter_id
            if payload.get("error") is not None:
                end_payload["error"] = str(payload["error"])
            if payload.get("interrupts"):
                end_payload["interrupted"] = True
            # 并行扇出时同名节点有多个任务：按 task id 配对各自的 node_start，
            # 取不到时回落该节点最近一次 node_start。
            parent_id = None
            if isinstance(task_id, str):
                parent_id = self._task_start_ids.pop(task_id, None)
            self._emit(
                type="node_end",
                unit=unit,
                payload=end_payload,
                parent_id=parent_id or self._parent_for(unit),
            )
            if self._current_node == unit:
                self._current_node = None

    @staticmethod
    def _task_chapter_id(payload: dict[str, Any]) -> str | None:
        """从 debug 任务载荷的 input 提取目标章 id（并行扇出的 Send 任务态携带）。"""
        task_input = payload.get("input")
        if isinstance(task_input, dict):
            for key in ("draft_chapter_id", "reference_chapter_id"):
                chapter_id = task_input.get(key)
                if isinstance(chapter_id, str):
                    return chapter_id
        return None

    def _handle_updates(self, chunk: Any) -> None:
        """updates 流块：中断 → gate_blocked；节点部分状态 → 派生元数据事件序列。"""
        if not isinstance(chunk, dict):
            return
        for key, value in chunk.items():
            if key == "__interrupt__":
                self._handle_interrupt(value)
            elif key in GRAPH_UPDATE_NODES and isinstance(value, dict):
                self._handle_node_update(graph_node_unit(key), value)

    def _handle_interrupt(self, value: Any) -> None:
        """中断块 → gate_blocked：载荷本身即元数据，原样入信封。"""
        payload: dict[str, Any] = {}
        interrupts = value if isinstance(value, (tuple, list)) else (value,)
        for interrupt in interrupts:
            raw = getattr(interrupt, "value", None)
            if isinstance(raw, dict):
                payload = dict(raw)
                break
        self._interrupt_payload = payload
        self._emit(
            type="gate_blocked",
            unit="human_review_gate",
            payload=payload,
            parent_id=self._parent_for("human_review_gate"),
        )

    def _accumulate(self, update: Mapping[str, Any]) -> None:
        """用部分状态更新快照元数据累积器（只存计数与枚举，绝不存正文）。"""
        if "status" in update:
            self._snapshot["status"] = status_text(update["status"])
        if "iteration_round" in update:
            self._snapshot["iteration_round"] = update["iteration_round"]
        if "citation_retry_count" in update:
            self._snapshot["citation_retry_count"] = update["citation_retry_count"]
        if "outline" in update:
            self._snapshot["chapter_total"] = len(update["outline"])
        if "chapter_drafts" in update:
            # 并行首写各分支的 update 只带单章列表：按章节 id 集合累积计数，
            # 串行节点回写完整列表时并集不变，两种形态计数都正确。
            self._completed_chapter_ids.update(
                draft["chapter_id"] if isinstance(draft, dict) else draft.chapter_id
                for draft in update["chapter_drafts"]
            )
            self._snapshot["chapters_completed"] = len(self._completed_chapter_ids)
        if "citation_library" in update:
            # 并行检索各分支的 update 只带单章素材：按素材 id 集合累积计数，
            # 串行节点回写完整列表时并集不变，两种形态计数都正确。
            self._material_ids.update(
                material
                if isinstance(material, str)
                else material["id"]
                if isinstance(material, dict)
                else material.id
                for material in update["citation_library"]
            )
            self._snapshot["material_count"] = len(self._material_ids)
        if "citation_warnings" in update:
            self._snapshot["citation_warning_count"] = len(update["citation_warnings"])

    def _handle_node_update(self, node: str, update: dict[str, Any]) -> None:
        """节点部分状态 → llm_config_used / state_snapshot / progress / branch_taken / loop_iteration。"""
        self._accumulate(update)
        status = update.get("status")
        if isinstance(status, WorkflowStatus):
            self._last_status = status
        parent_id = self._parent_for(node)

        llm_config = update.get("current_node_llm_config")
        if isinstance(llm_config, dict):
            self._emit(
                type="llm_config_used",
                unit=node,
                payload=dict(llm_config),
                parent_id=parent_id,
            )

        self._emit(
            type="state_snapshot",
            unit=node,
            payload=dict(self._snapshot),
            parent_id=parent_id,
        )

        self._emit(
            type="progress",
            unit=node,
            payload={
                "chapters_completed": self._snapshot["chapters_completed"],
                "chapter_total": self._snapshot["chapter_total"],
                "iteration_round": self._snapshot["iteration_round"],
                "status": self._snapshot["status"],
            },
            parent_id=parent_id,
        )

        if node in _BRANCH_RULES and isinstance(status, WorkflowStatus):
            self._emit_branch_and_loop(node, status, parent_id)

    def _emit_branch_and_loop(
        self, node: str, status: WorkflowStatus, parent_id: str | None
    ) -> None:
        """条件路由节点：按状态机值发布 branch_taken 与 loop_iteration。"""
        rule = _BRANCH_RULES[node].get(status)
        if rule is None and node == "document_reviewer":
            # 终审通过或重试超限：主路径进入人工中断点。
            rule = ("human_review_gate", "篇级终审通过或重试超限，进入人工中断点")
        if rule is not None:
            to, reason = rule
            self._emit(
                type="branch_taken",
                unit=node,
                payload={"from": node, "to": to, "reason": reason},
                parent_id=parent_id,
            )

        if node == "document_reviewer" and status is WorkflowStatus.CITATION_CHECKING:
            self._emit(
                type="loop_iteration",
                unit=node,
                payload={
                    "loop": "citation_retry",
                    "round": self._snapshot["citation_retry_count"],
                },
                parent_id=parent_id,
            )
        elif (
            node == "human_review_gate"
            and status is WorkflowStatus.AWAIT_USER_REVIEW
        ):
            self._emit(
                type="loop_iteration",
                unit=node,
                payload={
                    "loop": "revision",
                    "round": self._snapshot["iteration_round"],
                },
                parent_id=parent_id,
            )
