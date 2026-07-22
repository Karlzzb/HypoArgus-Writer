"""对外 FastAPI 服务：REST 任务接口与双 SSE 通道。

LangGraph 以纯库形态嵌入：lifespan 里构建编译图、事件枢纽与 TaskManager。
两条 SSE 通道严格隔离：业务通道只发轻量业务事件（每任务一个枢纽，
定稿或失败后正常收尾）；graph_event 可视化通道只发事件信封（全局一个
枢纽，永不主动关闭，按 thread_id / session_id / 事件类型过滤订阅）。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, field_validator, model_validator

from assembly.assembler_config import AssemblerConfig
from service.event_broker import EventHub
from service.event_envelope import GRAPH_EVENT_TYPES, EventEnvelope
from graph import build_graph, postgres_checkpointer
from llm import observability
from llm.llm_client import LLMFactory, default_llm_factory
from agents.chapter_reviewer import make_chapter_reviewer
from agents.contracts import (
    HypothesisPayload,
    MaterialPayload,
    PointPayload,
    SearchTask,
    Subagent,
)
from agents.rewriter_loop import make_rewriter_loop
from agents.search_agent import make_search_agent
from domain.events import SUBAGENT_END, EventHook
from search_agent.api import (
    SearchAgentConfigurationError,
    SearchAgentContractError,
)
from service.graph_event_stream import make_standalone_subagent_hook
from service.task_service import (
    InvalidReview,
    SubagentHookDispatcher,
    TaskConflict,
    TaskManager,
    TaskNotFound,
)

_SSE_HEADERS = {"Cache-Control": "no-cache"}

SubagentFactory = Callable[[EventHook], Subagent]
"""子智能体工厂形态：以应用内部事件分发器实例化（工厂签名与打桩/真实现一致）。"""


def _resolve_subagent(
    injected: Subagent | SubagentFactory | None,
    default_factory: SubagentFactory,
    event_hook: EventHook,
) -> Subagent:
    """解析注入的子智能体：实例直用，工厂以内部事件分发器实例化，None 走缺省工厂。

    工厂形态供测试注入打桩/假运行时的同时保留事件旁路（进度事件仍经
    dispatcher → emitter → SSE 流出）；实例形态调用方自管事件钩子。
    """
    if injected is None:
        return default_factory(event_hook)
    if isinstance(injected, Subagent):
        return injected
    return injected(event_hook)


class CreateTaskRequest(BaseModel):
    """创建任务请求体。"""

    user_intent: str
    user_identity: str = ""
    session_id: str = ""

    @field_validator("user_intent")
    @classmethod
    def _intent_not_blank(cls, value: str) -> str:
        """写作意图不允许为空白。"""
        if not value.strip():
            raise ValueError("user_intent 不允许为空白")
        return value


class CreateTaskResponse(BaseModel):
    """创建任务响应体。"""

    thread_id: str
    execution_trace_id: str


class ReviewRequest(BaseModel):
    """提交审阅请求体：契约与人工中断点恢复值一致。"""

    action: Literal["finalize", "revise"]
    feedback: str | None = None

    @model_validator(mode="after")
    def _revise_requires_feedback(self) -> "ReviewRequest":
        """action=revise 必须携带非空意见文本。"""
        if self.action == "revise" and not (
            self.feedback and self.feedback.strip()
        ):
            raise ValueError("action=revise 时必须携带非空的 feedback 意见文本")
        return self


class ResumeRequest(BaseModel):
    """崩溃恢复请求体：session_id 由调用方透传，本系统不鉴权。"""

    session_id: str = ""


class ResumeResponse(BaseModel):
    """崩溃恢复响应体。"""

    thread_id: str
    status: str


class RollbackRequest(BaseModel):
    """回滚请求体。"""

    checkpoint_id: str


class RetrievalHypothesis(BaseModel):
    """独立检索请求中的假说条目：契约 HypothesisPayload 的校验形态。"""

    id: str
    text: str
    refute_condition: str

    @field_validator("id", "text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """假说 id 与本文不允许为空白（空白反驳条件合法：不产生反向检索项）。"""
        if not value.strip():
            raise ValueError("假说 id 与 text 不允许为空白")
        return value


class RetrievalPoint(BaseModel):
    """独立检索请求中的论点条目：契约 PointPayload 的校验形态。"""

    id: str
    text: str

    @field_validator("id", "text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """论点 id 与本文不允许为空白。"""
        if not value.strip():
            raise ValueError("论点 id 与 text 不允许为空白")
        return value


class RetrievalRequest(BaseModel):
    """独立检索请求体：字段即 SearchTask 契约，points/genre/既有素材摘要可选带默认值。

    session_id 由调用方透传（与创建任务一致，本系统不鉴权），
    进度事件按其入信封供 /graph_events 过滤订阅。
    points 缺省为空：与假说一并聚合进查询构造（杠杆①），不传亦可检索。
    """

    chapter_id: str
    points: list[RetrievalPoint] = []
    hypotheses: list[RetrievalHypothesis]
    genre: str = ""
    existing_materials_digest: str = ""
    session_id: str = ""

    @field_validator("chapter_id")
    @classmethod
    def _chapter_not_blank(cls, value: str) -> str:
        """章节 id 不允许为空白。"""
        if not value.strip():
            raise ValueError("chapter_id 不允许为空白")
        return value

    @field_validator("hypotheses")
    @classmethod
    def _hypotheses_not_empty(
        cls, value: list[RetrievalHypothesis]
    ) -> list[RetrievalHypothesis]:
        """假说列表不允许为空：空章无检索语义。"""
        if not value:
            raise ValueError("hypotheses 不允许为空")
        return value


class RetrievalResponse(BaseModel):
    """独立检索响应体：SearchResult 素材列表 + 本次调用的诊断摘要块。

    diagnostics 与 subagent_end 事件携带的诊断摘要同源（打桩等无诊断
    实现下为空对象），供无 Langfuse 权限的调用方观察本次检索运行细节。
    素材 verdict 为三值枚举 pass/fail/inconclusive（杠杆②）：pass 强支撑、
    inconclusive 弱佐证（近似命中/补充）、fail 反例或不可用；消费方须按三值处理。
    diagnostics 可能含 weak_evidence_count（本章弱佐证条数）与 pass_below_threshold
    （pass 落库低于下限的薄弱章警告，杠杆①）。
    """

    materials: list[MaterialPayload]
    diagnostics: dict[str, Any]


# 领域异常 → HTTP 状态码：应用启动时逐类注册为全局异常处理器。
_DOMAIN_ERROR_STATUS: tuple[tuple[type[Exception], int], ...] = (
    (TaskNotFound, 404),
    (TaskConflict, 409),
    (InvalidReview, 422),
    # 检索引擎域异常（独立检索接口同步抛出）：契约违约按不可处理实体、
    # 通道/LLM 配置缺失按服务不可用。
    (SearchAgentContractError, 422),
    (SearchAgentConfigurationError, 503),
)


def _make_domain_error_handler(
    status_code: int,
) -> Callable[[Request, Exception], JSONResponse]:
    """构造把某类领域异常转为 JSON 错误响应的处理器（错误体与 HTTPException 同构）。"""

    def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handler


def _sse_frame(event_id: str, event_type: str, data: dict[str, Any]) -> str:
    """按 SSE 规范拼一帧：id / event / data 三行加空行分隔。"""
    return (
        f"id: {event_id}\n"
        f"event: {event_type}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


def create_app(
    *,
    llm_factory: LLMFactory = default_llm_factory,
    checkpointer: BaseCheckpointSaver | None = None,
    search_agent: Subagent | SubagentFactory | None = None,
    rewriter_loop: Subagent | SubagentFactory | None = None,
    chapter_reviewer: Subagent | SubagentFactory | None = None,
    document_review_max_retries: int | None = None,
    assembler_config: AssemblerConfig | None = None,
) -> FastAPI:
    """构建 FastAPI 应用：全部依赖在 lifespan 里装配。

    checkpointer 为 None 时走生产路径（Postgres 存档器，按环境变量连接）；
    测试注入 InMemorySaver。search_agent 与 rewriter_loop 未注入时均使用
    真实现工厂（make_search_agent / make_rewriter_loop），事件钩子
    经线程本地分发器按运行动态路由；注入工厂形态时同样以内部分发器实例化
    （保留事件旁路），注入实例形态时调用方自管事件钩子。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        with ExitStack() as stack:
            saver = (
                stack.enter_context(postgres_checkpointer())
                if checkpointer is None
                else checkpointer
            )
            hook_dispatcher = SubagentHookDispatcher()
            # 独立检索接口与主流程复用同一 search_agent 实例（同一信号量、
            # 同一运行时），故在传入 build_graph 前先解析并挂到应用状态。
            resolved_search_agent = _resolve_subagent(
                search_agent, make_search_agent, hook_dispatcher
            )
            graph = build_graph(
                llm_factory=llm_factory,
                checkpointer=saver,
                search_agent=resolved_search_agent,
                rewriter_loop=_resolve_subagent(
                    rewriter_loop,
                    lambda hook: make_rewriter_loop(llm_factory, hook),
                    hook_dispatcher,
                ),
                chapter_reviewer=_resolve_subagent(
                    chapter_reviewer,
                    lambda hook: make_chapter_reviewer(llm_factory, hook),
                    hook_dispatcher,
                ),
                document_review_max_retries=document_review_max_retries,
                assembler_config=assembler_config,
            )
            graph_hub = EventHub(loop)
            app.state.graph_hub = graph_hub
            app.state.search_agent = observability.wrap_subagent(
                resolved_search_agent
            )
            app.state.hook_dispatcher = hook_dispatcher
            app.state.manager = TaskManager(
                graph=graph,
                graph_hub=graph_hub,
                loop=loop,
                hook_dispatcher=hook_dispatcher,
            )
            yield

    app = FastAPI(title="HypoArgus-Writer", lifespan=lifespan)
    for exc_type, status_code in _DOMAIN_ERROR_STATUS:
        app.add_exception_handler(
            exc_type, _make_domain_error_handler(status_code)
        )

    def _manager() -> TaskManager:
        manager: TaskManager = app.state.manager
        return manager

    @app.post("/tasks", status_code=201, response_model=CreateTaskResponse)
    async def create_task(request: CreateTaskRequest) -> CreateTaskResponse:
        """创建写作任务并启动首跑。"""
        thread_id, trace_id = _manager().create_task(
            request.user_intent, request.user_identity, request.session_id
        )
        return CreateTaskResponse(thread_id=thread_id, execution_trace_id=trace_id)

    @app.post("/retrieval", response_model=RetrievalResponse)
    async def run_retrieval(request: RetrievalRequest) -> RetrievalResponse:
        """独立阻塞式检索：一章假说列表同步换素材与诊断，不启动写作任务。

        与主流程同一套任务包/结果契约、同一 search_agent 实例（lifespan
        构建）；进度事件带调用方 session_id 经全局 /graph_events 通道流出，
        subagent_start 即本次调用的根事件。诊断摘要从 subagent_end 事件
        载荷截获进响应 diagnostics 块；事件与诊断依赖内部分发器路由
        （工厂/缺省注入形态），实例形态注入时调用方自管事件钩子、
        diagnostics 为空对象。
        """
        agent: Subagent = app.state.search_agent
        dispatcher: SubagentHookDispatcher = app.state.hook_dispatcher
        envelope_hook = make_standalone_subagent_hook(
            publish=app.state.graph_hub.publish,
            trace_id=uuid.uuid4().hex,
            session_id=request.session_id,
        )
        diagnostics: dict[str, Any] = {}

        def hook(event_type: str, payload: dict[str, Any]) -> None:
            summary = payload.get("diagnostics")
            if event_type == SUBAGENT_END and isinstance(summary, dict):
                diagnostics.update(summary)
            envelope_hook(event_type, payload)

        task = SearchTask(
            chapter_id=request.chapter_id,
            points=[
                PointPayload(id=point.id, text=point.text)
                for point in request.points
            ],
            hypotheses=[
                HypothesisPayload(
                    id=hypothesis.id,
                    text=hypothesis.text,
                    refute_condition=hypothesis.refute_condition,
                )
                for hypothesis in request.hypotheses
            ],
            genre=request.genre,
            existing_materials_digest=request.existing_materials_digest,
        )
        # contextvars 按请求隔离：钩子登记只在本请求上下文内生效，
        # 与并发请求及图运行的工作线程互不串扰。
        dispatcher.set_hook(hook)
        try:
            result = await agent.run(dict(task))
        finally:
            dispatcher.clear_hook()
        return RetrievalResponse(
            materials=result["materials"], diagnostics=diagnostics
        )

    @app.get("/tasks/{thread_id}")
    async def get_task_status(thread_id: str) -> dict[str, Any]:
        """查询任务当前状态摘要。"""
        return _manager().get_status(thread_id)

    @app.post("/tasks/{thread_id}/review", status_code=202)
    async def submit_review(thread_id: str, request: ReviewRequest) -> dict[str, str]:
        """提交人工审阅决定（定稿或修订），从中断点恢复运行。"""
        _manager().submit_review(thread_id, request.action, request.feedback)
        return {"thread_id": thread_id, "action": request.action}

    @app.post("/tasks/{thread_id}/resume", response_model=ResumeResponse)
    async def resume_task(thread_id: str, request: ResumeRequest) -> ResumeResponse:
        """崩溃后按检查点恢复任务。"""
        status = _manager().resume_task(thread_id, request.session_id)
        return ResumeResponse(thread_id=thread_id, status=status)

    @app.post("/tasks/{thread_id}/rollback", status_code=202)
    async def rollback_task(
        thread_id: str, request: RollbackRequest
    ) -> dict[str, str]:
        """回滚到指定历史检查点并从该版本继续迭代。"""
        _manager().rollback(thread_id, request.checkpoint_id)
        return {"thread_id": thread_id, "checkpoint_id": request.checkpoint_id}

    @app.get("/tasks/{thread_id}/bibliography")
    async def render_bibliography(
        thread_id: str, format: str = Query(default="gbt7714")
    ) -> dict[str, Any]:
        """按书目格式渲染最终交付（重编号正文 + 书目），格式与内容解耦。"""
        try:
            return _manager().render_bibliography(thread_id, format)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.get("/tasks/{thread_id}/checkpoints")
    async def list_checkpoints(thread_id: str) -> list[dict[str, Any]]:
        """检查点元数据清单（仅元数据，不含正文）。"""
        return _manager().list_checkpoints(thread_id)

    @app.get("/tasks/{thread_id}/stream")
    async def stream_task(thread_id: str) -> StreamingResponse:
        """业务数据 SSE 通道：任务定稿或失败后流正常结束。"""
        hub = _manager().business_hub(thread_id)

        async def generate() -> AsyncIterator[str]:
            async for item in hub.subscribe():
                yield _sse_frame(item["event_id"], item["type"], item)

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    @app.get("/graph_events")
    async def graph_events(
        thread_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        types: str | None = Query(default=None),
    ) -> StreamingResponse:
        """graph_event 可视化 SSE 通道：全局流，按参数过滤，由客户端断开。"""
        type_filter: frozenset[str] | None = None
        if types is not None:
            requested = frozenset(
                item.strip() for item in types.split(",") if item.strip()
            )
            unknown = requested - GRAPH_EVENT_TYPES
            if unknown or not requested:
                raise HTTPException(
                    status_code=400,
                    detail=f"非法的事件类型过滤值：{sorted(unknown) or types}",
                )
            type_filter = requested
        hub: EventHub = app.state.graph_hub

        async def generate() -> AsyncIterator[str]:
            async for envelope in hub.subscribe():
                if not isinstance(envelope, EventEnvelope):
                    continue
                if thread_id is not None and envelope.thread_id != thread_id:
                    continue
                if session_id is not None and envelope.session_id != session_id:
                    continue
                if type_filter is not None and envelope.type not in type_filter:
                    continue
                yield _sse_frame(
                    envelope.event_id, envelope.type, envelope.model_dump()
                )

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    return app
