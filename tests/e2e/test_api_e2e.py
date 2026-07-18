"""HTTP 层端到端测试：REST 接口、双 SSE 通道、断点续跑与版本回滚。

注入 FakeLLM 与 InMemorySaver，起真实 uvicorn 服务（随机端口、真套接字、
uvicorn 驱动 lifespan，与生产完全同构）后用 httpx 全链路驱动：
创建任务 → 并发消费两条 SSE → 审阅迭代 → 定稿；
以及事件信封完备性、过滤参数、崩溃恢复、历史回滚与错误路径。

不用 httpx ASGITransport：它把响应整体缓冲到应用返回后才交付
（httpx 0.28 实测），无法边跑边消费 SSE 流。

FakeLLM 响应计划复用 test_graph_e2e 的编排方式；
所有 SSE 读取都包 asyncio.wait_for 防挂死，读到目标事件即断开。
"""

import asyncio
import json
import threading
from contextlib import ExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Callable

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from service.app import create_app
from graph import build_graph, postgres_checkpointer
from llm.llm_client import FakeLLM
from domain.state import initial_state
from tests.llm_response_plans import (
    FIRST_PASS_RESPONSES,
    REVISE_ROUND_RESPONSES,
    SEMANTIC_PASS,
    TRUNK_RESPONSES,
)
from tests.e2e.test_graph_e2e import TEST_PG_DSN, _pg_reachable

TIMEOUT = 30.0


def _make_app(responses: list[str], checkpointer: Any = None) -> FastAPI:
    """带 FakeLLM 与 InMemorySaver 构建应用。"""
    fake = FakeLLM(list(responses))
    return create_app(
        llm_factory=lambda unit: fake,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


@asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """在后台线程起真实 uvicorn 服务（随机端口），挂 httpx 客户端。"""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        timeout_graceful_shutdown=3,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        async with asyncio.timeout(TIMEOUT):
            while not server.started:
                await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=httpx.Timeout(10.0, read=TIMEOUT),
        ) as client:
            yield client
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def _read_sse(
    client: httpx.AsyncClient,
    url: str,
    stop: Callable[[list[dict]], bool],
    timeout: float = TIMEOUT,
) -> tuple[list[dict], bool]:
    """消费 SSE 流：按空行分帧解析 id/event/data，stop 命中即断开。

    返回（帧列表, 流是否自然结束）。data 行解析为 JSON 对象。
    """
    frames: list[dict] = []
    ended = False

    async def _consume() -> None:
        nonlocal ended
        async with client.stream("GET", url) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            frame: dict = {}
            async for line in response.aiter_lines():
                if line == "":
                    if frame:
                        frames.append(frame)
                        if stop(frames):
                            return
                        frame = {}
                    continue
                key, _, value = line.partition(": ")
                frame[key] = json.loads(value) if key == "data" else value
        ended = True

    await asyncio.wait_for(_consume(), timeout)
    return frames, ended


def _stop_on_review_count(n: int) -> Callable[[list[dict]], bool]:
    """业务流的停止条件：收到第 n 条 review_required。"""
    return lambda frames: (
        sum(1 for f in frames if f["event"] == "review_required") >= n
    )


def _stop_on_types(required: set[str]) -> Callable[[list[dict]], bool]:
    """graph_event 流的停止条件：已见事件类型覆盖 required 集合。"""
    return lambda frames: required <= {f["event"] for f in frames}


async def _create_task(
    client: httpx.AsyncClient, session_id: str = "sess-e2e"
) -> tuple[str, str]:
    """创建任务，返回（thread_id, trace_id）。"""
    response = await client.post(
        "/tasks",
        json={
            "user_intent": "写一篇人才培养方案",
            "user_identity": "专业撰稿人",
            "session_id": session_id,
        },
    )
    assert response.status_code == 201
    body = response.json()
    return body["thread_id"], body["execution_trace_id"]


def test_主干闭环_创建审阅迭代到定稿且双通道各司其职():
    async def main() -> None:
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            thread_id, _ = await _create_task(client)

            # 并发消费两条 SSE：业务流等 review_required，可视化流等 gate_blocked。
            (business, _), (graph_frames, _) = await asyncio.gather(
                _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                ),
                _read_sse(
                    client,
                    f"/graph_events?thread_id={thread_id}",
                    _stop_on_types({"gate_blocked"}),
                ),
            )

            # 双通道严格隔离：业务流无事件信封字段，可视化流无业务事件类型。
            assert all("payload" not in f["data"] for f in business)
            assert all("unit" in f["data"] for f in graph_frames)
            assert not [
                f
                for f in graph_frames
                if f["event"] in {"status", "review_required", "finalized"}
            ]

            # 审阅请求载荷仅元数据，不含正文字段。
            review = next(f for f in business if f["event"] == "review_required")
            payload = review["data"]["data"]
            assert payload["chapter_ids"] == ["ch1", "ch2"]
            assert payload["citation_warnings"] == []
            assert "text" not in json.dumps(payload, ensure_ascii=False)
            # 业务流沿途有轻量状态事件。
            statuses = [
                f["data"]["data"]["status"]
                for f in business
                if f["event"] == "status"
            ]
            assert "FRAMEWORK_BUILDING" in statuses

            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "AWAIT_USER_REVIEW"
            assert status["awaiting_review"] is True
            assert status["running"] is False

            # 提交修订：迭代一轮后再次到达中断点。
            response = await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章口吻克制些"},
            )
            assert response.status_code == 202
            business, _ = await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(2)
            )
            second_review = [
                f for f in business if f["event"] == "review_required"
            ][1]
            assert second_review["data"]["data"]["iteration_round"] == 1

            # 定稿：业务流收到含全文章节的 finalized，且流正常结束。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            business, ended = await _read_sse(
                client, f"/tasks/{thread_id}/stream", lambda frames: False
            )
            assert ended, "定稿后业务流必须正常收尾"
            finalized = next(f for f in business if f["event"] == "finalized")
            chapters = finalized["data"]["data"]["chapters"]
            assert [c["chapter_id"] for c in chapters] == ["ch1", "ch2"]
            assert all(c["text"] and c["summary"] for c in chapters)
            assert "口吻更克制" in chapters[1]["text"]

            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "FINISHED"

    asyncio.run(main())


def test_事件信封完备_字段齐全父子链路可拼接且快照无正文():
    required_types = {
        "node_start",
        "node_end",
        "state_snapshot",
        "llm_config_used",
        "progress",
        "branch_taken",
        "gate_blocked",
        "gate_resumed",
        "subagent_start",
        "subagent_end",
    }

    async def main() -> None:
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            thread_id, trace_id = await _create_task(client, "sess-envelope")
            await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(1)
            )
            await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章口吻克制些"},
            )
            await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(2)
            )
            await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            business, _ = await _read_sse(
                client, f"/tasks/{thread_id}/stream", lambda frames: False
            )
            assert any(f["event"] == "finalized" for f in business)

            frames, _ = await _read_sse(
                client,
                f"/graph_events?thread_id={thread_id}",
                _stop_on_types(required_types),
            )
            envelopes = [f["data"] for f in frames]

            # 字段齐全：event_id 全局唯一，trace/session/thread 正确，ts 非空。
            assert len({e["event_id"] for e in envelopes}) == len(envelopes)
            for envelope in envelopes:
                assert envelope["trace_id"] == trace_id
                assert envelope["session_id"] == "sess-envelope"
                assert envelope["thread_id"] == thread_id
                assert envelope["ts"]
                assert envelope["type"] in required_types | {
                    "node_error",
                    "loop_iteration",
                }

            # 父子链路可拼接：node_end.parent_id 指向同单元的某条 node_start。
            start_ids_by_unit: dict[str, set[str]] = {}
            for envelope in envelopes:
                if envelope["type"] == "node_start":
                    start_ids_by_unit.setdefault(envelope["unit"], set()).add(
                        envelope["event_id"]
                    )
            node_ends = [e for e in envelopes if e["type"] == "node_end"]
            assert node_ends
            for envelope in node_ends:
                assert envelope["parent_id"] in start_ids_by_unit[envelope["unit"]]

            # 快照事件只含元数据：绝不出现正文类字段。
            snapshots = [e for e in envelopes if e["type"] == "state_snapshot"]
            assert snapshots
            for snapshot in snapshots:
                assert {"text", "summary", "excerpt"}.isdisjoint(snapshot["payload"])

    asyncio.run(main())


def test_过滤参数_按类型与session隔离且非法类型400():
    async def main() -> None:
        # 两个任务先后跑（共享 FakeLLM 应答须串行消费保证确定性）。
        app = _make_app([*FIRST_PASS_RESPONSES, *FIRST_PASS_RESPONSES])
        async with _client(app) as client:
            thread_a, _ = await _create_task(client, "sess-A")
            await _read_sse(
                client, f"/tasks/{thread_a}/stream", _stop_on_review_count(1)
            )
            thread_b, _ = await _create_task(client, "sess-B")
            await _read_sse(
                client, f"/tasks/{thread_b}/stream", _stop_on_review_count(1)
            )

            # types=progress：只收到 progress。
            frames, _ = await _read_sse(
                client,
                f"/graph_events?thread_id={thread_a}&types=progress",
                lambda fs: len(fs) >= 3,
            )
            assert frames and all(f["event"] == "progress" for f in frames)

            # 按 session_id 过滤：两个任务并存时只收到对应任务事件。
            frames, _ = await _read_sse(
                client,
                "/graph_events?session_id=sess-B",
                _stop_on_types({"gate_blocked"}),
            )
            assert frames
            assert all(f["data"]["session_id"] == "sess-B" for f in frames)
            assert all(f["data"]["thread_id"] == thread_b for f in frames)

            # 组合过滤同样生效。
            frames, _ = await _read_sse(
                client,
                "/graph_events?session_id=sess-A&types=gate_blocked",
                lambda fs: len(fs) >= 1,
            )
            assert all(
                f["event"] == "gate_blocked"
                and f["data"]["thread_id"] == thread_a
                for f in frames
            )

            # 非法事件类型 → 400。
            response = await client.get("/graph_events?types=not_a_type")
            assert response.status_code == 400
            response = await client.get("/graph_events?types=progress,bogus")
            assert response.status_code == 400

    asyncio.run(main())


def test_断点续跑_模拟进程死亡后resume重发审阅请求并可定稿():
    async def main() -> None:
        saver = InMemorySaver()
        app1 = _make_app(FIRST_PASS_RESPONSES, checkpointer=saver)
        async with _client(app1) as client:
            thread_id, _ = await _create_task(client, "sess-crash")
            await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(1)
            )
        # app1 连同 TaskManager 内存登记一并丢弃，模拟进程死亡。

        app2 = _make_app([], checkpointer=saver)
        async with _client(app2) as client:
            # 恢复前内存无登记：状态查询按检查点自动重建登记，直接可用。
            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "AWAIT_USER_REVIEW"

            response = await client.post(
                f"/tasks/{thread_id}/resume", json={"session_id": "sess-crash-2"}
            )
            assert response.status_code == 200
            assert response.json() == {
                "thread_id": thread_id,
                "status": "AWAIT_USER_REVIEW",
            }

            # 业务流重发 review_required（不重跑图）。
            business, _ = await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(1)
            )
            review = next(f for f in business if f["event"] == "review_required")
            assert review["data"]["data"]["chapter_ids"] == ["ch1", "ch2"]

            # 可视化通道补发 gate_blocked，session 取本次恢复传入值。
            frames, _ = await _read_sse(
                client,
                f"/graph_events?thread_id={thread_id}",
                _stop_on_types({"gate_blocked"}),
            )
            gate = next(f for f in frames if f["event"] == "gate_blocked")
            assert gate["data"]["session_id"] == "sess-crash-2"

            # 定稿收束：FakeLLM 无应答也能定稿（定稿分支不调 LLM）。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            business, ended = await _read_sse(
                client, f"/tasks/{thread_id}/stream", lambda frames: False
            )
            assert ended
            assert any(f["event"] == "finalized" for f in business)
            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "FINISHED"

    asyncio.run(main())


def test_回滚_回到首轮中断点检查点后可继续迭代并定稿():
    async def main() -> None:
        # 回滚后新一轮 revise 的意见解析应答：ch2 落实与首轮不同的独特指令。
        post_rollback_directive = json.dumps(
            [
                {
                    "target_chapter_id": "ch2",
                    "type": "rewrite_only",
                    "instruction": "结尾更有力",
                }
            ],
            ensure_ascii=False,
        )
        app = _make_app(
            [
                *FIRST_PASS_RESPONSES,
                *REVISE_ROUND_RESPONSES,
                post_rollback_directive,
                SEMANTIC_PASS,
            ]
        )
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, "sess-rollback")
            await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(1)
            )
            await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章口吻克制些"},
            )
            await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(2)
            )

            # 检查点清单仅元数据；找到首轮 AWAIT_USER_REVIEW 的中断点检查点。
            response = await client.get(f"/tasks/{thread_id}/checkpoints")
            assert response.status_code == 200
            checkpoints = response.json()
            for checkpoint in checkpoints:
                assert set(checkpoint) == {
                    "checkpoint_id",
                    "ts",
                    "status",
                    "iteration_round",
                    "next",
                }
            target = next(
                c
                for c in checkpoints
                if "human_review_gate" in c["next"]
                and c["status"] == "AWAIT_USER_REVIEW"
                and c["iteration_round"] == 0
            )

            # 未知检查点 → 404。
            response = await client.post(
                f"/tasks/{thread_id}/rollback", json={"checkpoint_id": "nope"}
            )
            assert response.status_code == 404

            # 回滚：重放到 human_review_gate 重新中断，回到历史版本。
            response = await client.post(
                f"/tasks/{thread_id}/rollback",
                json={"checkpoint_id": target["checkpoint_id"]},
            )
            assert response.status_code == 202
            business, _ = await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(3)
            )
            rollback_review = [
                f for f in business if f["event"] == "review_required"
            ][2]
            assert rollback_review["data"]["data"]["iteration_round"] == 0

            # 回滚后可继续迭代：提交新一轮修订，收到新的审阅请求。
            response = await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章结尾再有力些"},
            )
            assert response.status_code == 202
            business, _ = await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(4)
            )
            post_rollback_review = [
                f for f in business if f["event"] == "review_required"
            ][3]
            assert post_rollback_review["data"]["data"]["iteration_round"] == 1

            # 新一轮迭代后定稿：从回滚版本出发只落实了回滚后的修订指令。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            business, ended = await _read_sse(
                client, f"/tasks/{thread_id}/stream", lambda frames: False
            )
            assert ended
            finalized = next(f for f in business if f["event"] == "finalized")
            chapters = finalized["data"]["data"]["chapters"]
            # 回滚丢弃了回滚前那轮修订，回滚后的新指令已落实。
            assert "口吻更克制" not in chapters[1]["text"]
            assert "结尾更有力" in chapters[1]["text"]

    asyncio.run(main())


def test_断点续跑_图运行中途死亡后resume续跑至中断点():
    saver = InMemorySaver()
    thread_id = "crash-mid-run"
    # 手工驱动同款图（相同响应计划 + 共享存档器）：framework 完成的
    # 检查点落库后立即停止迭代，模拟图运行中途进程死亡。
    # 节点内部用 asyncio.run 调子智能体，手工驱动必须在事件循环之外。
    fake = FakeLLM(list(FIRST_PASS_RESPONSES))
    graph = build_graph(llm_factory=lambda unit: fake, checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    for mode, chunk in graph.stream(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-mid"),
        config,
        stream_mode=["updates", "debug"],
    ):
        if (
            mode == "debug"
            and isinstance(chunk, dict)
            and chunk.get("type") == "checkpoint"
            and "reference_orchestrator" in (chunk["payload"].get("next") or [])
        ):
            break
    # 死亡现场：最近检查点的待执行节点是 reference_orchestrator。
    assert graph.get_state(config).next == ("reference_orchestrator",)

    async def main() -> None:
        # 新进程只需剩余阶段的应答：2 章语义核查各一条。
        app = _make_app([SEMANTIC_PASS, SEMANTIC_PASS], checkpointer=saver)
        async with _client(app) as client:
            response = await client.post(
                f"/tasks/{thread_id}/resume", json={"session_id": "sess-mid"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "FRAMEWORK_BUILDING"

            # 续跑走完剩余节点到人工中断点：业务流收到审阅请求。
            business, _ = await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(1)
            )
            review = next(f for f in business if f["event"] == "review_required")
            assert review["data"]["data"]["chapter_ids"] == ["ch1", "ch2"]

            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "AWAIT_USER_REVIEW"
            assert status["awaiting_review"] is True

    asyncio.run(main())


def test_崩溃后免resume_直接查状态检查点并回滚到中断点():
    saver = InMemorySaver()
    thread_id = "crash-then-rollback"
    # 手工驱动同款图到人工中断点后丢弃，模拟进程死亡（须在事件循环之外）。
    fake = FakeLLM(list(FIRST_PASS_RESPONSES))
    graph = build_graph(llm_factory=lambda unit: fake, checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    graph.invoke(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-roll"), config
    )
    assert graph.get_state(config).next == ("human_review_gate",)

    async def main() -> None:
        # 新进程不先 resume：状态与检查点清单直接可用（登记按检查点自动重建）。
        app = _make_app([], checkpointer=saver)
        async with _client(app) as client:
            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "AWAIT_USER_REVIEW"

            response = await client.get(f"/tasks/{thread_id}/checkpoints")
            assert response.status_code == 200
            target = next(
                c
                for c in response.json()
                if "human_review_gate" in c["next"]
                and c["status"] == "AWAIT_USER_REVIEW"
            )

            # 直接回滚成功：重放到 human_review_gate 重新中断。
            response = await client.post(
                f"/tasks/{thread_id}/rollback",
                json={"checkpoint_id": target["checkpoint_id"]},
            )
            assert response.status_code == 202
            business, _ = await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(1)
            )
            review = next(f for f in business if f["event"] == "review_required")
            assert review["data"]["data"]["chapter_ids"] == ["ch1", "ch2"]

    asyncio.run(main())


def test_错误路径_未知任务404与非中断点提交409与非法入参422():
    async def main() -> None:
        app = _make_app(FIRST_PASS_RESPONSES)
        async with _client(app) as client:
            # 未知 thread：status / review / rollback / resume / stream 都 404。
            assert (await client.get("/tasks/nope")).status_code == 404
            response = await client.post(
                "/tasks/nope/review", json={"action": "finalize"}
            )
            assert response.status_code == 404
            response = await client.post(
                "/tasks/nope/rollback", json={"checkpoint_id": "x"}
            )
            assert response.status_code == 404
            response = await client.post(
                "/tasks/nope/resume", json={"session_id": ""}
            )
            assert response.status_code == 404
            response = await client.get("/tasks/nope/stream")
            assert response.status_code == 404

            # 非法入参：空白意图 422；revise 无 feedback 422。
            response = await client.post("/tasks", json={"user_intent": "   "})
            assert response.status_code == 422
            response = await client.post(
                "/tasks/nope/review", json={"action": "revise"}
            )
            assert response.status_code == 422

            # 中断点之外提交审阅 → 409：先正常定稿再重复提交。
            thread_id, _ = await _create_task(client, "sess-err")
            await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(1)
            )
            await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            _, ended = await _read_sse(
                client, f"/tasks/{thread_id}/stream", lambda frames: False
            )
            assert ended
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 409

    asyncio.run(main())


# ---- 端到端主干验收（issue #7）：混合两类修订分支 + 引文门禁 + 书目渲染 ----


@pytest.mark.parametrize(
    "backend",
    [
        "memory",
        pytest.param(
            "postgres",
            marks=pytest.mark.skipif(
                not _pg_reachable(TEST_PG_DSN), reason="测试 Postgres 不可达"
            ),
        ),
    ],
)
def test_端到端主干_混合修订两类分支经引文门禁定稿并渲染书目(backend: str):
    async def main(saver: Any) -> None:
        app = _make_app(TRUNK_RESPONSES, checkpointer=saver)
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, f"sess-trunk-{backend}")

            # 并发消费两条 SSE：业务流到审阅请求，可视化流到 gate_blocked。
            (business, _), (graph_frames, _) = await asyncio.gather(
                _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                ),
                _read_sse(
                    client,
                    f"/graph_events?thread_id={thread_id}",
                    _stop_on_types({"gate_blocked"}),
                ),
            )
            # 主干链路完整：可视化流可见五个主节点与两个子智能体的活动。
            units = {f["data"]["unit"] for f in graph_frames}
            assert {
                "framework_orchestrator",
                "reference_orchestrator",
                "writing_orchestrator",
                "citation_validator",
                "human_review_gate",
                "search_agent",
                "rewriter_loop",
            } <= units
            # 引文门禁首轮通过：审阅请求不带未决引文警告。
            review = next(f for f in business if f["event"] == "review_required")
            assert review["data"]["data"]["citation_warnings"] == []

            # 提交混合两类分支的修订意见，迭代一轮后回到中断点。
            response = await client.post(
                f"/tasks/{thread_id}/review",
                json={
                    "action": "revise",
                    "feedback": "引言口吻克制些；第二章补充行业数据",
                },
            )
            assert response.status_code == 202
            business, _ = await _read_sse(
                client, f"/tasks/{thread_id}/stream", _stop_on_review_count(2)
            )
            second = [f for f in business if f["event"] == "review_required"][1]
            assert second["data"]["data"]["iteration_round"] == 1
            assert second["data"]["data"]["citation_warnings"] == []

            # 定稿：两类修订都已落实到对应章节。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            business, ended = await _read_sse(
                client, f"/tasks/{thread_id}/stream", lambda frames: False
            )
            assert ended
            finalized = next(f for f in business if f["event"] == "finalized")
            chapters = {
                c["chapter_id"]: c["text"]
                for c in finalized["data"]["data"]["chapters"]
            }
            assert "引言口吻更克制" in chapters["ch1"]
            assert "补充行业数据佐证" in chapters["ch2"]

            # 书目渲染：正文角标重编号为数字序号，条目按格式产出。
            response = await client.get(f"/tasks/{thread_id}/bibliography")
            assert response.status_code == 200
            rendered = response.json()
            assert rendered["format"] == "gbt7714"
            texts = " ".join(c["text"] for c in rendered["chapters"])
            assert "[1]" in texts
            assert "[m-" not in texts, "正文仍残留素材 ID 角标，未完成重编号"
            assert rendered["bibliography"]
            assert rendered["bibliography"][0]["text"].startswith("[1] ")

            # 格式与内容解耦：同一引文库按另一格式渲染出不同条目文本。
            apa = (
                await client.get(
                    f"/tasks/{thread_id}/bibliography?format=apa"
                )
            ).json()
            assert [e["material_id"] for e in apa["bibliography"]] == [
                e["material_id"] for e in rendered["bibliography"]
            ]
            assert apa["bibliography"][0]["text"] != rendered["bibliography"][0]["text"]

            # 非法格式 400；未知任务 404。
            response = await client.get(
                f"/tasks/{thread_id}/bibliography?format=chicago"
            )
            assert response.status_code == 400
            response = await client.get("/tasks/nope/bibliography")
            assert response.status_code == 404

            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "FINISHED"

    with ExitStack() as stack:
        saver = (
            stack.enter_context(postgres_checkpointer(TEST_PG_DSN))
            if backend == "postgres"
            else InMemorySaver()
        )
        asyncio.run(main(saver))
