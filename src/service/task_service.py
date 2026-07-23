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
contextvars 分发——每次运行独占一个工作线程，LangGraph 把节点任务连同
copy_context() 派发进执行器线程池，并行首写分支内的钩子仍可靠指向
当前运行（详见 SubagentHookDispatcher docstring）。
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from llm import observability
from domain.bibliography import render_article
from domain.env_config import read_positive_int
from service.event_broker import EventHub
from service.event_envelope import new_envelope
from domain.units import MAIN_NODES
from service.graph_event_stream import GraphRunEmitter
from domain.state import WorkflowStatus, initial_state, status_text
from domain.events import EventHook


class TaskNotFound(Exception):
    """任务、检查点或线程不存在。"""


class TaskConflict(Exception):
    """任务当前状态不允许该操作（运行中或未停在中断点）。"""


class InvalidReview(Exception):
    """审阅提交内容不符合恢复值契约。"""


def build_resume_value(action: str, feedback: str | None) -> dict[str, Any]:
    """把审阅动作构造成人工中断点恢复值（恢复值契约的服务侧唯一产地）。

    与 human_review_gate 的恢复值校验同形：finalize / confirm 只携 action，
    revise 必须携非空 feedback；confirm 是否可用由节点按是否存在待确认清单
    裁决，服务层只透传。非法动作或缺意见抛 InvalidReview。
    """
    if action == "finalize":
        return {"action": "finalize"}
    if action == "confirm":
        return {"action": "confirm"}
    if action == "revise":
        if not (isinstance(feedback, str) and feedback.strip()):
            raise InvalidReview("action=revise 时必须携带非空的 feedback 意见文本")
        return {"action": "revise", "feedback": feedback.strip()}
    raise InvalidReview(f"非法的审阅动作：{action!r}")


def _review_pack_content(values: Mapping[str, Any]) -> dict[str, Any]:
    """从检查点 state 取审阅包六类内容（model_dump，章级粒度）。

    六类：当前轮大纲、各章正文（chapter_id/text/summary）、引文警告、
    篇级评审 warn、修订台账、引文库（素材全文）。outline / chapter_drafts /
    citation_library 在检查点里经 serializer 还原为 pydantic 模型实例，
    取 model_dump 与 GET /products 章级粒度形状对齐；空字段兜底为空列表。
    """
    return {
        "outline": [spec.model_dump() for spec in values.get("outline", [])],
        "chapters": [
            {
                "chapter_id": draft.chapter_id,
                "text": draft.text,
                "summary": draft.summary,
            }
            for draft in values.get("chapter_drafts", [])
        ],
        "citation_warnings": list(values.get("citation_warnings", [])),
        "review_warnings": list(values.get("review_warnings", [])),
        "revision_ledger": [
            round_.model_dump() for round_ in values.get("revision_ledger", [])
        ],
        "citation_library": [
            material.model_dump() for material in values.get("citation_library", [])
        ],
    }


def _review_pack_version(content: Mapping[str, Any], iteration_round: int) -> str:
    """审阅包轮次指纹：六类内容 + 迭代轮次的规范 JSON sha256 前 16 位。

    入参为已构建的六类内容 dict（``_review_pack_content`` 产物），避免 REST
    路径重复 model_dump。同状态同指纹（GET /review 重复调用幂等）；
    修订再停门后内容变化即指纹变化。sort_keys=True 保证字段顺序不影响指纹，
    只对内容本身敏感。
    """
    canonical = json.dumps(
        {"iteration_round": iteration_round, **content},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _review_required_routing(payload: Mapping[str, Any]) -> dict[str, Any]:
    """把人工中断载荷投影为 review_required 的纯路由元数据（全文走 GET /review）。

    保留四类分支信号：iteration_round / chapter_ids（正常审阅）、
    pending_confirmation（大扇出确认分支）、clarification_questions（含混回问分支）、
    error（上次提交契约不符或解析失败，须重新提交）；剥离 citation_warnings /
    review_warnings——两者属审阅包内容，全文只走 GET /tasks/{id}/review。
    中断载荷本身不变（human_review_gate 仍读全量做契约校验与回显）。
    """
    routing: dict[str, Any] = {
        "iteration_round": payload.get("iteration_round", 0),
        "chapter_ids": list(payload.get("chapter_ids", [])),
    }
    error = payload.get("error")
    if error is not None:
        routing["error"] = error
    questions = payload.get("clarification_questions")
    if questions:
        routing["clarification_questions"] = list(questions)
    pending = payload.get("pending_confirmation")
    if pending:
        routing["pending_confirmation"] = pending
    return routing


class SubagentHookDispatcher:
    """基于 contextvars 的子智能体事件钩子分发器。

    构图时作为 EventHook 一次性注入打桩适配层；每次图运行在自己的
    工作线程开头登记当前运行的真实钩子，结束时清除。
    用 ContextVar 而非 threading.local：LangGraph 把每个节点任务连同
    copy_context() 派发进执行器线程池（并行首写的 chapter_drafter 分支
    不在驱动线程上执行），contextvars 随任务复制、钩子在并行分支内依然
    指向当前运行；不同运行各占独立驱动线程，上下文互不串扰。
    """

    def __init__(self) -> None:
        self._hook_var: contextvars.ContextVar[EventHook | None] = (
            contextvars.ContextVar("subagent_hook", default=None)
        )

    def set_hook(self, hook: EventHook) -> None:
        """登记当前运行（当前上下文）的真实钩子。"""
        self._hook_var.set(hook)

    def clear_hook(self) -> None:
        """清除当前上下文的钩子登记。"""
        self._hook_var.set(None)

    def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        """把子智能体事件转发给当前上下文登记的钩子；未登记则丢弃。"""
        hook = self._hook_var.get()
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


@dataclass
class _ProductTracker:
    """一次图运行内结构化产物的已发登记：去重，避免重发同内容产物帧。

    产物事件只在产物新产出时发：outline 首次非空、某章素材集合增长、
    某章草稿文本变化。检查点恢复续跑不重发已完成产物——图不重跑已完成
    节点，updates 不会再现已落检查点的产物，故本登记 per-run 局部即可，
    无需跨运行持久化。
    """

    outline_done: bool = False
    material_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    """章 id → 已发素材 id 集合：集合增长时重发该章 materials_ready。"""
    draft_text: dict[str, str] = field(default_factory=dict)
    """章 id → 已发草稿文本：文本变化时重发该章 chapter_ready。"""


class TaskManager:
    """写作任务生命周期管理器：单实例服务全部任务。"""

    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        graph_hub: EventHub,
        loop: asyncio.AbstractEventLoop,
        hook_dispatcher: SubagentHookDispatcher,
        epoch: str,
        max_queue: int,
    ) -> None:
        self._graph = graph
        self._graph_hub = graph_hub
        self._loop = loop
        self._hook_dispatcher = hook_dispatcher
        self._epoch = epoch
        self._max_queue = max_queue
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
            hub=self._new_hub(thread_id),
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

    def submit_review(self, thread_id: str, action: str, feedback: str | None) -> None:
        """提交人工审阅决定，从中断点恢复运行。"""
        entry = self._require_entry(thread_id)
        if entry.running:
            raise TaskConflict(f"任务 {thread_id} 已有运行在进行中")
        resume_value = build_resume_value(action, feedback)

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

        status = status_text(snapshot.values.get("status"), WorkflowStatus.IDLE.value)
        if self._at_interrupt(snapshot):
            # 停在中断点：补发 gate_blocked 信封与审阅门双发（摘要产物 +
            # review_required 路由信号），不重跑图。
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
            self._publish_review_gate(entry, payload, snapshot.values)
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
        for snapshot in self._graph.get_state_history(self._thread_config(thread_id)):
            if snapshot.config["configurable"].get("checkpoint_id") == checkpoint_id:
                target = snapshot
                break
        if target is None:
            raise TaskNotFound(f"任务 {thread_id} 不存在检查点：{checkpoint_id}")

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
        for snapshot in self._graph.get_state_history(self._thread_config(thread_id)):
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

    def get_products(self, thread_id: str) -> dict[str, Any]:
        """运行中产物只读快照：目录/假说/各章素材与已完成章正文，章级粒度。

        纯只读检查点 state，不引入新状态、不加写路径；SSE 丢帧靠此 REST
        对账。任务不存在抛 TaskNotFound（与 get_status 一致）；刚创建尚无
        产物时返回空快照（outline 为空 → chapters 为空列表）。

        形状与章级 checkpoint 对齐：points / materials / draft 直接取对应
        pydantic 模型 model_dump，不做字段改名；章条目的 chapter_id 由
        ChapterSpec.id 映射，materials 按章分组、draft 未完成时为 null。
        """
        self._ensure_entry(thread_id)
        values = self._graph.get_state(self._thread_config(thread_id)).values
        outline = values.get("outline", [])
        drafts_by_id = {
            draft.chapter_id: draft for draft in values.get("chapter_drafts", [])
        }
        materials_by_chapter: dict[str, list[Any]] = {}
        for material in values.get("citation_library", []):
            materials_by_chapter.setdefault(material.chapter_id, []).append(material)
        chapters: list[dict[str, Any]] = []
        for spec in outline:
            chapters.append(
                {
                    "chapter_id": spec.id,
                    "title": spec.title,
                    "subsections": list(spec.subsections),
                    "chapter_type": spec.chapter_type,
                    "planned_summary": spec.planned_summary,
                    "points": [point.model_dump() for point in spec.points],
                    "materials": [
                        material.model_dump()
                        for material in materials_by_chapter.get(spec.id, [])
                    ],
                    "draft": (
                        drafts_by_id[spec.id].model_dump()
                        if spec.id in drafts_by_id
                        else None
                    ),
                }
            )
        return {
            "thread_id": thread_id,
            "status": status_text(values.get("status"), WorkflowStatus.IDLE.value),
            "iteration_round": values.get("iteration_round", 0),
            "chapters": chapters,
        }

    def get_review_pack(self, thread_id: str) -> dict[str, Any]:
        """人工审阅包全文：仅停在人工中断点时可取，否则抛 TaskConflict（409）。

        一次给齐六类内容（当前轮大纲、各章正文 chapter_id/text/summary、
        引文警告、篇级评审 warn、修订台账、引文库素材全文）+ pack_version
        轮次指纹；重复调用幂等（同检查点同指纹）。任务不存在抛 TaskNotFound；
        未停在中断点抛 TaskConflict（不返回半成品）。
        """
        self._ensure_entry(thread_id)
        config = self._thread_config(thread_id)
        snapshot = self._graph.get_state(config)
        if not self._at_interrupt(snapshot):
            raise TaskConflict(f"任务 {thread_id} 未停在人工中断点，审阅包不可取")
        values = snapshot.values
        iteration_round = values.get("iteration_round", 0)
        content = _review_pack_content(values)
        return {
            "thread_id": thread_id,
            "status": status_text(values.get("status"), WorkflowStatus.IDLE.value),
            "iteration_round": iteration_round,
            "pack_version": _review_pack_version(content, iteration_round),
            **content,
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
        rendered = render_article(drafts, values.get("citation_library", []), format)
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

    def shutdown(self) -> None:
        """服务关停时优雅关流：取消在跑图运行并关闭全部业务枢纽。

        枢纽关闭后订阅队列收到结束哨兵，SSE 生成器正常收尾，客户端连接
        随之收束；REST 真相源（检查点）不受影响。
        """
        for entry in self._tasks.values():
            run_task = entry.run_task
            if run_task is not None and not run_task.done():
                run_task.cancel()
            entry.hub.close()

    def _new_hub(self, thread_id: str) -> EventHub:
        """为任务建一条业务通道枢纽：带本进程世代 id 与带 thread_id 的 reconcile 载荷。"""
        return EventHub(
            self._loop,
            epoch=self._epoch,
            thread_id=thread_id,
            max_queue=self._max_queue,
        )

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
            hub=self._new_hub(thread_id),
        )
        self._tasks[thread_id] = entry
        return entry

    def _thread_config(self, thread_id: str) -> RunnableConfig:
        """构造运行配置：线程 id + 并行任务并发上限。

        max_concurrency 是 langgraph 读取的 RunnableConfig 顶层键，
        约束同一超步内并行任务（首写扇出的 chapter_drafter 分支）的并发数，
        按环境变量 GRAPH_MAX_CONCURRENCY 配置（缺省 4，兼顾服务商限流）。
        """
        return cast(
            RunnableConfig,
            {
                "configurable": {"thread_id": thread_id},
                "max_concurrency": read_positive_int(
                    os.environ, "GRAPH_MAX_CONCURRENCY", 4
                ),
            },
        )

    def _new_emitter(self, entry: _TaskEntry) -> GraphRunEmitter:
        """为一次运行构造事件翻译器，发布到全局 graph_event 枢纽。

        ``publish_business`` 把逐字流 ``CONTENT_DELTA`` 翻译为业务通道事件
        发布到该任务 ``entry.hub``：子智能体在工作线程发 CONTENT_DELTA →
        ``_publish_business`` → ``entry.hub.publish`` 经 call_soon_threadsafe
        调度到 loop 线程；合并已在 ``_stream_once`` 工作线程侧由 DeltaMerger
        完成，跨线程只传合并后的帧，避免逐 token 调度。
        """
        return GraphRunEmitter(
            publish=self._graph_hub.publish,
            trace_id=entry.trace_id,
            session_id=entry.session_id,
            thread_id=entry.thread_id,
            publish_business=lambda event_type, data: self._publish_business(
                entry, event_type, data
            ),
        )

    @staticmethod
    def _at_interrupt(snapshot: Any) -> bool:
        """检查点是否停在人工中断点：有任务携带中断即是。

        不能以 snapshot.next 非空作前置条件：同一节点内重新中断
        （安全汇点循环——契约错误、含混回问、大扇出确认）后，
        LangGraph 的 snapshot.next 为空而中断仍挂在任务上，
        此时任务依然可以且必须接受下一次审阅提交。
        """
        return any(task.interrupts for task in snapshot.tasks)

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
        """向业务通道发布一条轻量业务事件（线程安全）。

        事件 id（``{epoch}-{seq}``）与 ts 由枢纽在入站时分配盖戳，调用方
        只负责语义载荷；payload 形如 ``{event_id, type, thread_id, ts, data}``。
        """
        entry.hub.publish(
            {
                "type": event_type,
                "thread_id": entry.thread_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
        )

    def _publish_review_gate(
        self,
        entry: _TaskEntry,
        interrupt_payload: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> None:
        """停审阅门时双发：review_pack_ready 摘要产物事件 + review_required 路由信号。

        review_pack_ready 属可丢级（type=product），只推摘要 + pack_version，
        全文（章正文/素材全文/警告/台账）绝不入 SSE——丢了靠 GET /review 对账；
        review_required 降级为纯路由元数据（信号必达，不参与丢最旧）。
        先发产物后发信号，保证「全部产物事件先于 review_required」的到达序。
        """
        self._publish_business(
            entry,
            "product",
            {"kind": "review_pack_ready", **self._review_pack_summary(values)},
        )
        self._publish_business(
            entry, "review_required", _review_required_routing(interrupt_payload)
        )

    @staticmethod
    def _review_pack_summary(values: Mapping[str, Any]) -> dict[str, Any]:
        """审阅包摘要：轮次、章节 id 与各项计数 + pack_version，绝不含正文/素材全文。

        六类内容只构建一次（``_review_pack_content``），计数与指纹同源——
        指纹与 ``GET /review`` 全文同检查点同值。
        """
        iteration_round = values.get("iteration_round", 0)
        content = _review_pack_content(values)
        outline = content["outline"]
        return {
            "iteration_round": iteration_round,
            "chapter_ids": [spec["id"] for spec in outline],
            "chapter_total": len(outline),
            "chapter_completed": len(content["chapters"]),
            "material_count": len(content["citation_library"]),
            "citation_warning_count": len(content["citation_warnings"]),
            "review_warning_count": len(content["review_warnings"]),
            "revision_round_count": len(content["revision_ledger"]),
            "pack_version": _review_pack_version(content, iteration_round),
        }

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
            await asyncio.to_thread(self._drive, entry, emitter, graph_input, config)
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
        product_tracker = _ProductTracker()
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
                        self._publish_product_events(entry, chunk, product_tracker)
        finally:
            self._hook_dispatcher.clear_hook()

    def _publish_status_updates(self, entry: _TaskEntry, chunk: Any) -> None:
        """updates 块含状态机值时向业务通道发布轻量状态事件。"""
        for node, update in self._main_updates(chunk):
            if "status" not in update:
                continue
            self._publish_business(
                entry,
                "status",
                {
                    "status": status_text(update["status"], WorkflowStatus.IDLE.value),
                    "iteration_round": update.get("iteration_round", 0),
                    "node": node,
                },
            )

    @staticmethod
    def _main_updates(
        chunk: Any,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """updates 块里的主节点部分状态更新序列（节点名, 更新 dict）。

        状态事件与产物事件都从同形 updates 块派生，共用此迭代避免重复的
        ``isinstance`` + ``MAIN_NODES`` 过滤前奏（Duplicated Code 收敛点）。
        """
        if not isinstance(chunk, dict):
            return
        for node, update in chunk.items():
            if node in MAIN_NODES and isinstance(update, dict):
                yield node, update

    def _publish_product_events(
        self,
        entry: _TaskEntry,
        chunk: Any,
        tracker: _ProductTracker,
    ) -> None:
        """updates 块含结构化产物时向业务通道发布整块产物事件（可丢级）。

        产物事件属可丢级（票 #55 契约 ``type=product``），丢帧靠
        ``GET /tasks/{id}/products``（票 #56）对账；只在产物新产出时发，
        经 per-run 登记去重（见 _ProductTracker）。

        三类整块产物：outline 首次非空 → ``outline_ready``（含假说）、
        某章素材集合增长 → ``materials_ready``、某章草稿文本变化 →
        ``chapter_ready``。载荷取该次 update 携带的整块对象 model_dump，
        与 ``GET /products`` 章级粒度形状对齐，丢帧后 REST 取回同等。

        逐字流（content_delta）与审阅包（review_pack_ready）不在本票范围。
        """
        for node, update in self._main_updates(chunk):
            outline = update.get("outline")
            if outline and not tracker.outline_done:
                tracker.outline_done = True
                self._publish_business(
                    entry,
                    "product",
                    {
                        "kind": "outline_ready",
                        "outline": [spec.model_dump() for spec in outline],
                    },
                )
            materials = update.get("citation_library")
            if materials:
                by_chapter: dict[str, list[Any]] = {}
                for material in materials:
                    by_chapter.setdefault(material.chapter_id, []).append(material)
                for chapter_id, mats in by_chapter.items():
                    ids = frozenset(m.id for m in mats)
                    if ids - tracker.material_ids.get(chapter_id, frozenset()):
                        tracker.material_ids[chapter_id] = (
                            tracker.material_ids.get(chapter_id, frozenset()) | ids
                        )
                        self._publish_business(
                            entry,
                            "product",
                            {
                                "kind": "materials_ready",
                                "chapter_id": chapter_id,
                                "materials": [m.model_dump() for m in mats],
                            },
                        )
            drafts = update.get("chapter_drafts")
            if drafts:
                for draft in drafts:
                    if tracker.draft_text.get(draft.chapter_id) != draft.text:
                        tracker.draft_text[draft.chapter_id] = draft.text
                        self._publish_business(
                            entry,
                            "product",
                            {
                                "kind": "chapter_ready",
                                "chapter_id": draft.chapter_id,
                                "draft": draft.model_dump(),
                            },
                        )

    def _finish_run(self, entry: _TaskEntry, emitter: GraphRunEmitter) -> None:
        """运行正常结束后的收尾：中断转审阅门双发，终态发布定稿全文。"""
        if emitter.interrupt_payload is not None:
            values = self._graph.get_state(self._thread_config(entry.thread_id)).values
            self._publish_review_gate(entry, emitter.interrupt_payload, values)
            return
        if emitter.last_status is WorkflowStatus.FINISHED:
            values = self._graph.get_state(self._thread_config(entry.thread_id)).values
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
                    "citation_warnings": list(values.get("citation_warnings", [])),
                },
            )
            entry.hub.close()
