"""任务服务层：写作任务的生命周期管理与双通道事件发布。

TaskManager 把编译好的 LangGraph 图包装成任务粒度的服务对象：
创建任务、提交审阅、崩溃恢复、历史版本回滚、检查点清单与状态查询。

图节点内部用 asyncio.run 调子智能体，因此图的 stream 绝不能跑在服务
事件循环上：每次运行经 asyncio.to_thread 进独占工作线程同步驱动，
事件经线程安全的 EventHub 发回事件循环。

双通道隔离：
- 业务通道：每任务一个 EventHub，只发轻量业务事件（状态、审阅请求、
  定稿全文、错误）；任务定稿或失败后关闭，SSE 随之正常收尾。
- graph_event 可视化通道：全服务一个 EventHub，只发事件信封，
  永不关闭，过滤在订阅端完成。

子智能体事件钩子按运行动态路由：build_graph 只在建服务时调用一次，
而每次运行有自己的翻译器（emitter）。SubagentHookDispatcher 基于
threading.local 分发——每次运行独占一个工作线程，节点内 asyncio.run
仍在同一 OS 线程执行，线程本地变量可靠指向当前运行的钩子。
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

import observability
from bibliography import render_article
from event_broker import EventHub
from event_envelope import new_envelope
from graph import MAIN_NODES
from graph_event_stream import GraphRunEmitter
from state import WorkflowStatus, initial_state, status_text
from subagents import EventHook


class TaskNotFound(Exception):
    """任务、检查点或线程不存在。"""


class TaskConflict(Exception):
    """任务当前状态不允许该操作（运行中或未停在中断点）。"""


class InvalidReview(Exception):
    """审阅提交内容不符合恢复值契约。"""


class SubagentHookDispatcher:
    """基于 threading.local 的子智能体事件钩子分发器。

    构图时作为 EventHook 一次性注入打桩适配层；每次图运行在自己的
    工作线程开头登记当前运行的真实钩子，结束时清除。
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def set_hook(self, hook: EventHook) -> None:
        """登记当前线程（即当前运行）的真实钩子。"""
        self._local.hook = hook

    def clear_hook(self) -> None:
        """清除当前线程的钩子登记。"""
        self._local.hook = None

    def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        """把子智能体事件转发给当前线程登记的钩子；未登记则丢弃。"""
        hook = getattr(self._local, "hook", None)
        if hook is not None:
            hook(event_type, payload)


@dataclass
class _TaskEntry:
    """任务在内存中的登记信息。"""

    thread_id: str
    trace_id: str
    session_id: str
    hub: EventHub
    run_task: asyncio.Task[None] | None = field(default=None)

    @property
    def running(self) -> bool:
        """是否有本任务的图运行正在进行。"""
        return self.run_task is not None and not self.run_task.done()


class TaskManager:
    """写作任务生命周期管理器：单实例服务全部任务。"""

    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        graph_hub: EventHub,
        loop: asyncio.AbstractEventLoop,
        hook_dispatcher: SubagentHookDispatcher,
    ) -> None:
        self._graph = graph
        self._graph_hub = graph_hub
        self._loop = loop
        self._hook_dispatcher = hook_dispatcher
        self._tasks: dict[str, _TaskEntry] = {}

    # ---- 对外操作 ----

    def create_task(
        self, user_intent: str, user_identity: str, session_id: str
    ) -> tuple[str, str]:
        """创建写作任务并启动首跑，返回（thread_id, trace_id）。"""
        thread_id = uuid.uuid4().hex
        trace_id = uuid.uuid4().hex
        entry = _TaskEntry(
            thread_id=thread_id,
            trace_id=trace_id,
            session_id=session_id,
            hub=EventHub(self._loop),
        )
        self._tasks[thread_id] = entry

        emitter = self._new_emitter(entry)
        emitter.emit_root(
            type="progress",
            unit="graph",
            payload={"phase": "run_start", "user_identity": user_identity},
        )
        self._start_run(
            entry,
            emitter,
            initial_state(user_intent, user_identity, trace_id),
            self._thread_config(thread_id),
        )
        return thread_id, trace_id

    def submit_review(
        self, thread_id: str, action: str, feedback: str | None
    ) -> None:
        """提交人工审阅决定，从中断点恢复运行。"""
        entry = self._require_entry(thread_id)
        if entry.running:
            raise TaskConflict(f"任务 {thread_id} 已有运行在进行中")
        if action == "finalize":
            resume_value: dict[str, Any] = {"action": "finalize"}
        elif action == "revise":
            if not (isinstance(feedback, str) and feedback.strip()):
                raise InvalidReview("action=revise 时必须携带非空的 feedback 意见文本")
            resume_value = {"action": "revise", "feedback": feedback.strip()}
        else:
            raise InvalidReview(f"非法的审阅动作：{action!r}")

        config = self._thread_config(thread_id)
        snapshot = self._graph.get_state(config)
        if not self._at_interrupt(snapshot):
            raise TaskConflict(f"任务 {thread_id} 未停在人工中断点，不能提交审阅")

        emitter = self._new_emitter(entry)
        emitter.seed(snapshot.values)
        emitter.emit_root(
            type="gate_resumed",
            unit="human_review_gate",
            payload={"action": action},
        )
        self._start_run(entry, emitter, Command(resume=resume_value), config)

    def resume_task(self, thread_id: str, session_id: str) -> str:
        """崩溃后恢复：按检查点重建任务登记并按现场续跑，返回当前状态字符串。

        - 停在中断点：不重跑图，只在两条通道补发中断事件；
        - 中途被杀（有待执行节点且无中断）：以输入 None 继续驱动；
        - 已终态：仅重建登记（终态任务的业务通道随即关闭，SSE 正常收尾）。
        """
        entry = self._ensure_entry(thread_id, session_id)
        if entry.running:
            raise TaskConflict(f"任务 {thread_id} 已有运行在进行中")
        if session_id:
            # 调用方本次传入的非空 session_id 覆盖登记值（只透传不鉴权）。
            entry.session_id = session_id

        config = self._thread_config(thread_id)
        snapshot = self._graph.get_state(config)
        if not snapshot.values and not snapshot.next:
            raise TaskNotFound(f"任务不存在：{thread_id}（无检查点）")

        status = status_text(
            snapshot.values.get("status"), WorkflowStatus.IDLE.value
        )
        if self._at_interrupt(snapshot):
            # 停在中断点：补发 gate_blocked 信封与业务 review_required，不重跑图。
            payload = self._interrupt_payload_of(snapshot)
            self._graph_hub.publish(
                new_envelope(
                    type="gate_blocked",
                    unit="human_review_gate",
                    payload=payload,
                    trace_id=entry.trace_id,
                    session_id=entry.session_id,
                    thread_id=thread_id,
                )
            )
            self._publish_business(entry, "review_required", payload)
            return status

        if snapshot.next:
            # 中途被杀：以输入 None 从最近检查点继续驱动。
            emitter = self._new_emitter(entry)
            emitter.seed(snapshot.values)
            emitter.emit_root(
                type="progress", unit="graph", payload={"phase": "resume"}
            )
            self._start_run(entry, emitter, None, config)
            return status

        # 已到终态：业务通道关闭，订阅方正常收尾。
        entry.hub.close()
        return status

    def rollback(self, thread_id: str, checkpoint_id: str) -> None:
        """回滚到指定历史检查点：LangGraph 从该检查点分叉重放。

        重放到 human_review_gate 会重新中断，用户从该历史版本继续迭代。
        """
        entry = self._ensure_entry(thread_id)
        if entry.running:
            raise TaskConflict(f"任务 {thread_id} 已有运行在进行中")

        target = None
        for snapshot in self._graph.get_state_history(
            self._thread_config(thread_id)
        ):
            if (
                snapshot.config["configurable"].get("checkpoint_id")
                == checkpoint_id
            ):
                target = snapshot
                break
        if target is None:
            raise TaskNotFound(
                f"任务 {thread_id} 不存在检查点：{checkpoint_id}"
            )

        emitter = self._new_emitter(entry)
        emitter.seed(target.values)
        emitter.emit_root(
            type="progress",
            unit="graph",
            payload={"phase": "rollback", "checkpoint_id": checkpoint_id},
        )
        rollback_config = cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            },
        )
        self._start_run(entry, emitter, None, rollback_config)

    def list_checkpoints(self, thread_id: str) -> list[dict[str, Any]]:
        """检查点元数据清单（新到旧）：绝不含正文。"""
        self._ensure_entry(thread_id)
        checkpoints: list[dict[str, Any]] = []
        for snapshot in self._graph.get_state_history(
            self._thread_config(thread_id)
        ):
            values = snapshot.values
            checkpoints.append(
                {
                    "checkpoint_id": snapshot.config["configurable"].get(
                        "checkpoint_id"
                    ),
                    "ts": snapshot.created_at,
                    "status": status_text(
                        values.get("status"), WorkflowStatus.IDLE.value
                    ),
                    "iteration_round": values.get("iteration_round", 0),
                    "next": list(snapshot.next),
                }
            )
        if not checkpoints:
            raise TaskNotFound(f"任务不存在：{thread_id}（无检查点）")
        return checkpoints

    def get_status(self, thread_id: str) -> dict[str, Any]:
        """任务当前状态摘要。"""
        entry = self._ensure_entry(thread_id)
        snapshot = self._graph.get_state(self._thread_config(thread_id))
        return {
            "thread_id": thread_id,
            "status": status_text(
                snapshot.values.get("status"), WorkflowStatus.IDLE.value
            ),
            "iteration_round": snapshot.values.get("iteration_round", 0),
            "awaiting_review": self._at_interrupt(snapshot),
            "running": entry.running,
        }

    def render_bibliography(self, thread_id: str, format: str) -> dict[str, Any]:
        """按书目格式渲染最终交付：重编号正文 + 书目列表。

        引文内容存于 State 引文库，格式在交付时指定，两者完全解耦；
        尚无章节正文（框架或检索阶段）时不可渲染。
        """
        self._ensure_entry(thread_id)
        values = self._graph.get_state(self._thread_config(thread_id)).values
        drafts = values.get("chapter_drafts", [])
        if not drafts:
            raise TaskConflict(f"任务 {thread_id} 尚无章节正文，不能渲染书目")
        rendered = render_article(
            drafts, values.get("citation_library", []), format
        )
        return {
            "thread_id": thread_id,
            "format": rendered.format,
            "chapters": [
                {"chapter_id": chapter.chapter_id, "text": chapter.text}
                for chapter in rendered.chapters
            ],
            "bibliography": [
                {
                    "index": entry.index,
                    "material_id": entry.material_id,
                    "text": entry.text,
                }
                for entry in rendered.entries
            ],
        }

    def business_hub(self, thread_id: str) -> EventHub:
        """取任务的业务事件枢纽（SSE 订阅入口）。"""
        return self._require_entry(thread_id).hub

    # ---- 内部实现 ----

    def _require_entry(self, thread_id: str) -> _TaskEntry:
        entry = self._tasks.get(thread_id)
        if entry is None:
            raise TaskNotFound(f"任务不存在：{thread_id}")
        return entry

    def _ensure_entry(self, thread_id: str, session_id: str = "") -> _TaskEntry:
        """取任务的内存登记；丢失（进程重启）时按检查点重建，无检查点抛 TaskNotFound。"""
        entry = self._tasks.get(thread_id)
        if entry is not None:
            return entry
        snapshot = self._graph.get_state(self._thread_config(thread_id))
        if not snapshot.values and not snapshot.next:
            raise TaskNotFound(f"任务不存在：{thread_id}（无检查点）")
        entry = _TaskEntry(
            thread_id=thread_id,
            trace_id=str(snapshot.values.get("execution_trace_id", "")),
            session_id=session_id,
            hub=EventHub(self._loop),
        )
        self._tasks[thread_id] = entry
        return entry

    def _thread_config(self, thread_id: str) -> RunnableConfig:
        return cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})

    def _new_emitter(self, entry: _TaskEntry) -> GraphRunEmitter:
        """为一次运行构造事件翻译器，发布到全局 graph_event 枢纽。"""
        return GraphRunEmitter(
            publish=self._graph_hub.publish,
            trace_id=entry.trace_id,
            session_id=entry.session_id,
            thread_id=entry.thread_id,
        )

    @staticmethod
    def _at_interrupt(snapshot: Any) -> bool:
        """检查点是否停在人工中断点：有待执行节点且任务携带中断。"""
        return bool(snapshot.next) and any(
            task.interrupts for task in snapshot.tasks
        )

    @staticmethod
    def _interrupt_payload_of(snapshot: Any) -> dict[str, Any]:
        """从检查点任务中提取中断载荷（元数据）。"""
        for task in snapshot.tasks:
            for intr in task.interrupts:
                raw = getattr(intr, "value", None)
                if isinstance(raw, dict):
                    return dict(raw)
        return {}

    def _publish_business(
        self, entry: _TaskEntry, event_type: str, data: dict[str, Any]
    ) -> None:
        """向业务通道发布一条轻量业务事件（线程安全）。"""
        entry.hub.publish(
            {
                "event_id": uuid.uuid4().hex,
                "type": event_type,
                "thread_id": entry.thread_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
        )

    def _start_run(
        self,
        entry: _TaskEntry,
        emitter: GraphRunEmitter,
        graph_input: Any,
        config: RunnableConfig,
    ) -> None:
        """在事件循环上登记一次图运行（工作线程驱动 + 收尾）。"""
        entry.run_task = self._loop.create_task(
            self._run(entry, emitter, graph_input, config)
        )

    async def _run(
        self,
        entry: _TaskEntry,
        emitter: GraphRunEmitter,
        graph_input: Any,
        config: RunnableConfig,
    ) -> None:
        """一次图运行的完整生命周期：工作线程驱动、收尾发布与错误兜底。"""
        try:
            await asyncio.to_thread(
                self._drive, entry, emitter, graph_input, config
            )
        except asyncio.CancelledError:
            # 服务关停取消任务不是业务失败：原样上抛，不发 error 事件也不关业务枢纽。
            raise
        except Exception as exc:  # noqa: BLE001 —— 运行失败必须落业务错误事件。
            emitter.handle_error(exc)
            self._publish_business(entry, "error", {"message": str(exc)})
            entry.hub.close()
            return
        self._finish_run(entry, emitter)

    def _drive(
        self,
        entry: _TaskEntry,
        emitter: GraphRunEmitter,
        graph_input: Any,
        config: RunnableConfig,
    ) -> None:
        """工作线程内同步驱动图运行：翻译事件信封并发布业务状态事件。

        整次驱动包在 Langfuse 根 span 内（未启用时直通）：节点与子智能体
        span、LLM generation 都在本线程内产生，天然挂到这条 trace 之下。
        """
        self._hook_dispatcher.set_hook(emitter.make_subagent_hook())
        try:
            with observability.run_span(
                thread_id=entry.thread_id,
                session_id=entry.session_id,
                trace_id=entry.trace_id,
            ):
                for mode, chunk in self._graph.stream(
                    graph_input, config, stream_mode=["updates", "debug"]
                ):
                    emitter.handle(mode, chunk)
                    if mode == "updates":
                        self._publish_status_updates(entry, chunk)
        finally:
            self._hook_dispatcher.clear_hook()

    def _publish_status_updates(self, entry: _TaskEntry, chunk: Any) -> None:
        """updates 块含状态机值时向业务通道发布轻量状态事件。"""
        if not isinstance(chunk, dict):
            return
        for node, update in chunk.items():
            if node not in MAIN_NODES or not isinstance(update, dict):
                continue
            if "status" not in update:
                continue
            self._publish_business(
                entry,
                "status",
                {
                    "status": status_text(
                        update["status"], WorkflowStatus.IDLE.value
                    ),
                    "iteration_round": update.get("iteration_round", 0),
                    "node": node,
                },
            )

    def _finish_run(self, entry: _TaskEntry, emitter: GraphRunEmitter) -> None:
        """运行正常结束后的收尾：中断转审阅请求，终态发布定稿全文。"""
        if emitter.interrupt_payload is not None:
            self._publish_business(
                entry, "review_required", emitter.interrupt_payload
            )
            return
        if emitter.last_status is WorkflowStatus.FINISHED:
            values = self._graph.get_state(
                self._thread_config(entry.thread_id)
            ).values
            chapters = [
                {
                    "chapter_id": draft.chapter_id,
                    "text": draft.text,
                    "summary": draft.summary,
                }
                for draft in values.get("chapter_drafts", [])
            ]
            self._publish_business(
                entry,
                "finalized",
                {
                    "chapters": chapters,
                    "citation_warnings": list(
                        values.get("citation_warnings", [])
                    ),
                },
            )
            entry.hub.close()
