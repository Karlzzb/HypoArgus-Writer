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
import os
import re
import threading
from contextlib import ExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Callable

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver


from agents.chapter_reviewer import make_stub_chapter_reviewer
from agents.rewriter_loop import make_stub_rewriter_loop
from agents.search_agent import (
    FakeSearchAgentRuntime,
    make_search_agent,
    make_stub_search_agent,
)
from domain.events import SUBAGENT_END, SUBAGENT_PROGRESS, SUBAGENT_START
from service.app import create_app
from service.event_broker import EventHub
from service.event_envelope import GRAPH_EVENT_TYPES
from service.mock_scenarios import DEFAULT_SCENARIO, DEGRADATION_SCENARIO, MockScenario
from graph import build_graph, checkpoint_serializer, postgres_checkpointer
from llm.llm_client import FakeLLM
from domain.state import ChapterDraft, SelfCheck, initial_state
from service.llm_response_plans import (
    DOCUMENT_REVIEW_PASS,
    FIRST_PASS_RESPONSES,
    FRAMEWORK_KEYED_RESPONSES,
    REVISE_ROUND_RESPONSES,
    SEMANTIC_PASS,
    TRUNK_RESPONSES,
    WRITER_KEYED_RESPONSES,
)
from tests.e2e.test_graph_e2e import TEST_PG_DSN, _pg_reachable

TIMEOUT = 30.0

# 测试用固定世代标识：便于以 ``{epoch}-0`` 续传从流首回放，确定性捕获整段历史。
TEST_EPOCH = "ep-test"
# 从流首续传的 Last-Event-ID：声明"已收至 seq 0"，服务端回放 seq>0 的全部保留事件。
FROM_START = f"{TEST_EPOCH}-0"
MATERIAL_ID_PATTERN = re.compile(r"m_[0-9A-Z]{26}")


def _assert_opaque_material_id(material_id: str) -> None:
    assert MATERIAL_ID_PATTERN.fullmatch(material_id)


def _assert_text_has_material_marker(text: str) -> None:
    assert re.search(r"\[m_[0-9A-Z]{26}\]", text)


def _make_app(
    responses: list[str],
    checkpointer: Any = None,
    keyed: dict[str, list[str]] | None = None,
    *,
    rewriter_stub: bool = True,
    search_agent: Any = make_stub_search_agent,
    epoch: str = TEST_EPOCH,
    ping_interval: int = 3600,
    max_queue: int | None = None,
    mock_scenario: MockScenario | None = None,
) -> FastAPI:
    """带 FakeLLM 与 InMemorySaver 构建应用。

    rewriter_stub=True（缺省）显式注入打桩改写器：本文件多数用例验收 HTTP 层
    与事件通道，不依赖写作真实现（其契约在 tests/agents/rewriter_loop/ 覆盖）。
    置 False 走 create_app 缺省的真实现链路（事件钩子接内部分发器），
    调用方须在 keyed 里给足写作与自审应答。
    search_agent 缺省注入打桩工厂（以应用内部事件分发器实例化，保留事件
    旁路）；中断续跑用例注入真适配层工厂 + 假引擎运行时。
    epoch 固定为 TEST_EPOCH，便于用 ``{epoch}-0`` 续传从流首回放整段历史
    （替代旧"订阅先全量回放"语义）；ping_interval 默认拉高以避免 keepalive
    ping 干扰断言，ping 专项用例单独调小。
    mock_scenario 透传 create_app 的 mock 栈场景；缺省 None 时 mock 图用
    DEFAULT_SCENARIO（自带 FakeLLM + 打桩子智能体，与真栈 FakeLLM 互不
    干涉）。真栈仍由注入的 responses/keyed FakeLLM 驱动；mock 任务按
    thread_id 前缀路由到 mock 图，与真任务共享同一 checkpointer。
    """
    fake = FakeLLM(
        list(responses),
        keyed_responses=keyed if keyed is not None else FRAMEWORK_KEYED_RESPONSES,
    )
    subagent_kwargs: dict[str, Any] = (
        {"rewriter_loop": make_stub_rewriter_loop()} if rewriter_stub else {}
    )
    # 章级评审真实现会顺次消费 FakeLLM 应答，抢占其他节点的应答；本文件的
    # 应答计划未给评审预留应答，故各路径一律注入打桩评审（不调 LLM，恒通过）。
    subagent_kwargs["chapter_reviewer"] = make_stub_chapter_reviewer()
    subagent_kwargs["search_agent"] = search_agent
    return create_app(
        llm_factory=lambda unit: fake,
        checkpointer=checkpointer
        if checkpointer is not None
        else InMemorySaver(serde=checkpoint_serializer()),
        epoch=epoch,
        ping_interval=ping_interval,
        max_queue=max_queue,
        mock_scenario=mock_scenario,
        **subagent_kwargs,
    )


@asynccontextmanager
async def _client(
    app: FastAPI, read_timeout: float = TIMEOUT
) -> AsyncIterator[httpx.AsyncClient]:
    """在后台线程起真实 uvicorn 服务（随机端口），挂 httpx 客户端。

    read_timeout 供长耗时场景（门控真实链路 E2E）放宽读超时，缺省不变。
    """
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
            timeout=httpx.Timeout(10.0, read=read_timeout),
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
    last_event_id: str | None = None,
) -> tuple[list[dict], bool]:
    """消费 SSE 流：按空行分帧解析 id/event/data，stop 命中即断开。

    返回（帧列表, 流是否自然结束）。data 行解析为 JSON 对象；keepalive ping
    注释行（以 ``:`` 开头）跳过，不污染帧。last_event_id 非空时作为
    ``Last-Event-ID`` 请求头携带，触发服务端续传（``{epoch}-0`` 即从流首回放）。
    """
    frames: list[dict] = []
    ended = False
    headers = {"Last-Event-ID": last_event_id} if last_event_id is not None else None

    async def _consume() -> None:
        nonlocal ended
        async with client.stream("GET", url, headers=headers) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            frame: dict = {}
            async for line in response.aiter_lines():
                if line.startswith(":"):
                    # keepalive ping 注释行，跳过。
                    continue
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
    return lambda frames: sum(1 for f in frames if f["event"] == "review_required") >= n


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

            # 并发消费两条 SSE（均以 {epoch}-0 续传从流首回放，确定性捕获整段
            # 历史）：业务流等 review_required，可视化流等 gate_blocked。
            (business, _), (graph_frames, _) = await asyncio.gather(
                _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                    last_event_id=FROM_START,
                ),
                _read_sse(
                    client,
                    f"/graph_events?thread_id={thread_id}",
                    _stop_on_types({"gate_blocked"}),
                    last_event_id=FROM_START,
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

            # 事件 id 形如 {epoch}-{seq}，单流内单调递增。
            ids = [f["id"] for f in business]
            assert ids == sorted(ids, key=lambda i: int(i.split("-")[-1]))
            assert all(i.startswith(f"{TEST_EPOCH}-") for i in ids)

            # 审阅请求载荷仅路由元数据，不含正文字段，也不含引文警告/篇级 warn 全文。
            review = next(f for f in business if f["event"] == "review_required")
            payload = review["data"]["data"]
            assert payload["chapter_ids"] == ["ch1", "ch2"]
            assert "citation_warnings" not in payload
            assert "review_warnings" not in payload
            assert "text" not in json.dumps(payload, ensure_ascii=False)
            # 审阅包全文走 REST：六类内容齐备 + pack_version，重复调用幂等。
            pack = (await client.get(f"/tasks/{thread_id}/review")).json()
            assert pack["pack_version"]
            assert [c["chapter_id"] for c in pack["chapters"]] == ["ch1", "ch2"]
            assert all(c["text"] and c["summary"] for c in pack["chapters"])
            assert pack["citation_warnings"] == []
            assert pack["review_warnings"] == []
            assert pack["revision_ledger"] == []
            assert pack["iteration_round"] == 0
            first_version = pack["pack_version"]
            pack_again = (await client.get(f"/tasks/{thread_id}/review")).json()
            assert pack_again["pack_version"] == first_version
            # 业务流沿途有轻量状态事件。
            statuses = [
                f["data"]["data"]["status"] for f in business if f["event"] == "status"
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
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(2),
                last_event_id=FROM_START,
            )
            second_review = [f for f in business if f["event"] == "review_required"][1]
            assert second_review["data"]["data"]["iteration_round"] == 1
            # 修订再停门后审阅包内容变化 → pack_version 变化，REST 重取得新一轮内容。
            pack2 = (await client.get(f"/tasks/{thread_id}/review")).json()
            assert pack2["iteration_round"] == 1
            assert pack2["pack_version"] != first_version
            assert [c["chapter_id"] for c in pack2["chapters"]] == ["ch1", "ch2"]

            # 定稿：业务流收到含全文章节的 finalized，且流正常结束。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            business, ended = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                last_event_id=FROM_START,
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
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章口吻克制些"},
            )
            await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(2),
                last_event_id=FROM_START,
            )
            await client.post(f"/tasks/{thread_id}/review", json={"action": "finalize"})
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                last_event_id=FROM_START,
            )
            assert any(f["event"] == "finalized" for f in business)

            frames, _ = await _read_sse(
                client,
                f"/graph_events?thread_id={thread_id}",
                _stop_on_types(required_types),
                last_event_id=FROM_START,
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
        # 两个任务先后跑（共享 FakeLLM 应答须串行消费保证确定性），
        # 键控假说应答按任务数翻倍。
        app = _make_app(
            [*FIRST_PASS_RESPONSES, *FIRST_PASS_RESPONSES],
            keyed={
                key: values * 2 for key, values in FRAMEWORK_KEYED_RESPONSES.items()
            },
        )
        async with _client(app) as client:
            thread_a, _ = await _create_task(client, "sess-A")
            await _read_sse(
                client,
                f"/tasks/{thread_a}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            thread_b, _ = await _create_task(client, "sess-B")
            await _read_sse(
                client,
                f"/tasks/{thread_b}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )

            # types=progress：只收到 progress。
            frames, _ = await _read_sse(
                client,
                f"/graph_events?thread_id={thread_a}&types=progress",
                lambda fs: len(fs) >= 3,
                last_event_id=FROM_START,
            )
            assert frames and all(f["event"] == "progress" for f in frames)

            # 按 session_id 过滤：两个任务并存时只收到对应任务事件。
            frames, _ = await _read_sse(
                client,
                "/graph_events?session_id=sess-B",
                _stop_on_types({"gate_blocked"}),
                last_event_id=FROM_START,
            )
            assert frames
            assert all(f["data"]["session_id"] == "sess-B" for f in frames)
            assert all(f["data"]["thread_id"] == thread_b for f in frames)

            # 组合过滤同样生效。
            frames, _ = await _read_sse(
                client,
                "/graph_events?session_id=sess-A&types=gate_blocked",
                lambda fs: len(fs) >= 1,
                last_event_id=FROM_START,
            )
            assert all(
                f["event"] == "gate_blocked" and f["data"]["thread_id"] == thread_a
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
        saver = InMemorySaver(serde=checkpoint_serializer())
        app1 = _make_app(FIRST_PASS_RESPONSES, checkpointer=saver)
        async with _client(app1) as client:
            thread_id, _ = await _create_task(client, "sess-crash")
            await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
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

            # 业务流重发 review_required（不重跑图）：续传从流首回放捕获。
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            review = next(f for f in business if f["event"] == "review_required")
            assert review["data"]["data"]["chapter_ids"] == ["ch1", "ch2"]

            # 可视化通道补发 gate_blocked，session 取本次恢复传入值。
            frames, _ = await _read_sse(
                client,
                f"/graph_events?thread_id={thread_id}",
                _stop_on_types({"gate_blocked"}),
                last_event_id=FROM_START,
            )
            gate = next(f for f in frames if f["event"] == "gate_blocked")
            assert gate["data"]["session_id"] == "sess-crash-2"

            # 定稿收束：FakeLLM 无应答也能定稿（定稿分支不调 LLM）。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            business, ended = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                last_event_id=FROM_START,
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
                DOCUMENT_REVIEW_PASS,
            ]
        )
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, "sess-rollback")
            await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章口吻克制些"},
            )
            await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(2),
                last_event_id=FROM_START,
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
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(3),
                last_event_id=FROM_START,
            )
            rollback_review = [f for f in business if f["event"] == "review_required"][
                2
            ]
            assert rollback_review["data"]["data"]["iteration_round"] == 0

            # 回滚后可继续迭代：提交新一轮修订，收到新的审阅请求。
            response = await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章结尾再有力些"},
            )
            assert response.status_code == 202
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(4),
                last_event_id=FROM_START,
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
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                last_event_id=FROM_START,
            )
            assert ended
            finalized = next(f for f in business if f["event"] == "finalized")
            chapters = finalized["data"]["data"]["chapters"]
            # 回滚丢弃了回滚前那轮修订，回滚后的新指令已落实。
            assert "口吻更克制" not in chapters[1]["text"]
            assert "结尾更有力" in chapters[1]["text"]

    asyncio.run(main())


def test_断点续跑_图运行中途死亡后resume续跑至中断点():
    saver = InMemorySaver(serde=checkpoint_serializer())
    thread_id = "crash-mid-run"
    # 手工驱动同款图（相同响应计划 + 共享存档器）：framework 完成的
    # 检查点落库后立即停止迭代，模拟图运行中途进程死亡。
    # 节点内部用 asyncio.run 调子智能体，手工驱动必须在事件循环之外。
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    graph = build_graph(
        llm_factory=lambda unit: fake,
        checkpointer=saver,
        search_agent=make_stub_search_agent(),
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
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
    # 死亡现场：最近检查点的待执行任务是检索并行扇出的 2 个单章分支。
    assert graph.get_state(config).next == (
        "reference_orchestrator",
        "reference_orchestrator",
    )

    async def main() -> None:
        # 新进程只需剩余阶段的应答：2 章语义核查各一条 + 篇级评审一条。
        app = _make_app(
            [SEMANTIC_PASS, SEMANTIC_PASS, DOCUMENT_REVIEW_PASS], checkpointer=saver
        )
        async with _client(app) as client:
            response = await client.post(
                f"/tasks/{thread_id}/resume", json={"session_id": "sess-mid"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "FRAMEWORK_BUILDING"

            # 续跑走完剩余节点到人工中断点：业务流收到审阅请求。
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            review = next(f for f in business if f["event"] == "review_required")
            assert review["data"]["data"]["chapter_ids"] == ["ch1", "ch2"]

            # 断点续跑不重发已完成产物（issue #58）：framework 已在崩溃前
            # 完成、续跑不重跑，故无 outline_ready；续跑产出的素材/草稿照常发。
            kinds = [
                f["data"]["data"]["kind"] for f in business if f["event"] == "product"
            ]
            assert "outline_ready" not in kinds
            assert sorted(k for k in kinds if k == "materials_ready") == [
                "materials_ready",
                "materials_ready",
            ]
            assert sorted(k for k in kinds if k == "chapter_ready") == [
                "chapter_ready",
                "chapter_ready",
            ]

            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "AWAIT_USER_REVIEW"
            assert status["awaiting_review"] is True

    asyncio.run(main())


def test_检索中断续跑_已完成章节零重复检索且事件配对产物与不中断路径等价():
    """检索中断续跑 E2E（issue #37，ADR-0001 约束 4 的检索真适配层版）。

    链路口径：真适配层（make_search_agent：契约映射、信号量限流、进度桥、
    诊断摘要）+ 引擎运行时边界的假实现（FakeSearchAgentRuntime，模拟时延与
    副作用）——仅在桩上通过的验收不算通过（约束 4）。
    故障注入：ch2 检索的副作用回调先让位并行的 ch1 分支完成，再抛致命异常，
    图调用在检索超步内崩溃并向外抛出，等价于「一章完成、下一章进行中」时
    进程被 kill；已完成 ch1 分支的写入作为 pending write 被 checkpoint 保留。
    新「进程」（HTTP 应用 + 同一存档器 + 健康假运行时）resume 续跑到人工
    中断点。断言：已完成章节零重复检索（两个假运行时的载荷记录为证）、
    钩子与信封两层事件成对且父子链正确（含中断分支的残链）、
    最终产物与不中断路径完全等价。
    """
    saver = InMemorySaver(serde=checkpoint_serializer())
    thread_id = "retrieval-crash-resume"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    async def _crash_on_ch2(payload: dict[str, Any]) -> None:
        if payload["paragraph"]["paragraph_id"] == "ch2":
            # 让位并行的 ch1 分支先完成，使「一章完成、另一章进行中」
            # 的死亡现场确定性成立（同 test_graph_e2e 的检索中断用例）。
            await asyncio.sleep(0.3)
            raise RuntimeError("故障注入：进程死于 ch2 检索")

    # 第一个「进程」：手工驱动同款图（真适配层 + 崩溃注入假运行时 +
    # 共享存档器）。节点内部用 asyncio.run 调子智能体，须在事件循环之外。
    crash_runtime = FakeSearchAgentRuntime(
        latency_seconds=0.05, side_effect=_crash_on_ch2
    )
    events_before: list[tuple[str, dict]] = []
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    graph = build_graph(
        llm_factory=lambda unit: fake,
        checkpointer=saver,
        search_agent=make_search_agent(
            lambda etype, payload: events_before.append((etype, payload)),
            runtime=crash_runtime,
        ),
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
    with pytest.raises(RuntimeError, match="故障注入：进程死于 ch2 检索"):
        graph.invoke(
            initial_state("写一篇人才培养方案", "专业撰稿人", "trace-ref-resume"),
            config,
        )
    # 死亡现场：两章检索均已发起，已完成的 ch1 分支素材作为 pending write
    # 被 checkpoint 保留，待执行任务只剩失败的 ch2 检索分支。
    assert sorted(p["paragraph"]["paragraph_id"] for p in crash_runtime.payloads) == [
        "ch1",
        "ch2",
    ]
    snapshot = graph.get_state(config)
    assert snapshot.next == ("reference_orchestrator",)
    assert {
        material.chapter_id for material in snapshot.values["citation_library"]
    } == {"ch1"}

    # 钩子层事件链（崩溃进程）：ch1 成对完整、progress 全落在区间内且
    # 步骤序与假引擎回放一致（1 假说 = 正反 2 检索项）；结束事件带诊断摘要。
    ch1_events = [(t, p) for t, p in events_before if p["chapter_id"] == "ch1"]
    ch2_events = [(t, p) for t, p in events_before if p["chapter_id"] == "ch2"]
    ch1_types = [t for t, _ in ch1_events]
    assert ch1_types[0] == SUBAGENT_START and ch1_types[-1] == SUBAGENT_END
    assert all(t == SUBAGENT_PROGRESS for t in ch1_types[1:-1])
    assert all(p["unit"] == "search_agent" for _, p in ch1_events)
    # 假引擎每章仅 1 pass（< 缺省下限 3），engine_call_end 前必发薄弱章警告（杠杆①）。
    assert [p["step"] for _, p in ch1_events[1:-1]] == [
        "engine_call_start",
        "task.start",
        "task.retrieved",
        "verdict.done",
        "task.start",
        "task.retrieved",
        "verdict.done",
        "judge.batches_done",
        "weak_chapter_warning",
        "engine_call_end",
    ]
    warning = next(p for _, p in ch1_events if p.get("step") == "weak_chapter_warning")
    assert warning["pass_count"] == 1 and warning["threshold"] == 3
    verdict_events = [p for _, p in ch1_events if p.get("step") == "verdict.done"]
    assert [(p["done_count"], p["item_total"]) for p in verdict_events] == [
        (1, 2),
        (2, 2),
    ]
    assert ch1_events[-1][1]["diagnostics"]["call_counts"] == {
        "web_search": 2,
        "web_fetch": 2,
    }
    # 中断分支残链：有 start 无 end，死于引擎调用中途，事件流如实反映现场。
    assert [t for t, _ in ch2_events] == [SUBAGENT_START, SUBAGENT_PROGRESS]
    assert ch2_events[1][1]["step"] == "engine_call_start"

    # 第二个「进程」：HTTP 应用 + 同一存档器 + 健康假运行时 resume 续跑，
    # 只备剩余阶段应答（2 章语义核查 + 篇级评审；首写走打桩改写器不调 LLM）。
    healthy_runtime = FakeSearchAgentRuntime(latency_seconds=0.05)

    async def main() -> None:
        app = _make_app(
            [SEMANTIC_PASS, SEMANTIC_PASS, DOCUMENT_REVIEW_PASS],
            checkpointer=saver,
            search_agent=lambda hook: make_search_agent(hook, runtime=healthy_runtime),
        )
        async with _client(app) as client:
            response = await client.post(
                f"/tasks/{thread_id}/resume", json={"session_id": "sess-ref-resume"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "REFERENCE_FETCHING"

            # 续跑走完剩余节点到人工中断点：两章齐全。
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            review = next(f for f in business if f["event"] == "review_required")
            assert review["data"]["data"]["chapter_ids"] == ["ch1", "ch2"]

            # 信封层事件链（恢复进程）：只有 ch2 分支的检索活动，
            # progress 与 subagent_end 全部挂在本次 subagent_start 之下，
            # subagent_start 挂在检索分支的 node_start 之下。
            frames, _ = await _read_sse(
                client,
                f"/graph_events?thread_id={thread_id}",
                _stop_on_types({"gate_blocked"}),
                last_event_id=FROM_START,
            )
            envelopes = [f["data"] for f in frames]
            search_events = [e for e in envelopes if e["unit"] == "search_agent"]
            starts = [e for e in search_events if e["type"] == "subagent_start"]
            ends = [e for e in search_events if e["type"] == "subagent_end"]
            progresses = [e for e in search_events if e["type"] == "progress"]
            assert [e["payload"]["chapter_id"] for e in starts] == ["ch2"]
            assert [e["payload"]["chapter_id"] for e in ends] == ["ch2"]
            assert progresses
            assert all(e["payload"]["chapter_id"] == "ch2" for e in progresses)
            start_id = starts[0]["event_id"]
            assert all(e["parent_id"] == start_id for e in progresses)
            assert ends[0]["parent_id"] == start_id
            assert progresses[0]["payload"]["step"] == "engine_call_start"
            assert progresses[-1]["payload"]["step"] == "engine_call_end"
            reference_start_ids = {
                e["event_id"]
                for e in envelopes
                if e["type"] == "node_start" and e["unit"] == "reference_orchestrator"
            }
            assert starts[0]["parent_id"] in reference_start_ids

    asyncio.run(main())

    # 已完成章节零重复检索：恢复进程的运行时只见过失败的 ch2 分支，
    # ch1 的外部检索与 LLM 成本零重复支付。
    assert [p["paragraph"]["paragraph_id"] for p in healthy_runtime.payloads] == ["ch2"]

    # 最终产物与不中断路径完全等价：同一确定性假运行时与应答计划保证
    # 引文库、章节草稿与引文核查报告可逐字段比对。
    resumed = graph.get_state(config).values
    baseline_fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    baseline_graph = build_graph(
        llm_factory=lambda unit: baseline_fake,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        search_agent=make_search_agent(runtime=FakeSearchAgentRuntime()),
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
    baseline = baseline_graph.invoke(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-ref-base"),
        {"configurable": {"thread_id": "retrieval-crash-baseline"}},
    )
    assert resumed["citation_library"] == baseline["citation_library"]
    assert resumed["chapter_drafts"] == baseline["chapter_drafts"]
    assert resumed["citation_report"] == baseline["citation_report"]


def test_崩溃后免resume_直接查状态检查点并回滚到中断点():
    saver = InMemorySaver(serde=checkpoint_serializer())
    thread_id = "crash-then-rollback"
    # 手工驱动同款图到人工中断点后丢弃，模拟进程死亡（须在事件循环之外）。
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    graph = build_graph(
        llm_factory=lambda unit: fake,
        checkpointer=saver,
        search_agent=make_stub_search_agent(),
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
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
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
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
            assert (await client.get("/tasks/nope/review")).status_code == 404
            response = await client.post(
                "/tasks/nope/rollback", json={"checkpoint_id": "x"}
            )
            assert response.status_code == 404
            response = await client.post("/tasks/nope/resume", json={"session_id": ""})
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
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            await client.post(f"/tasks/{thread_id}/review", json={"action": "finalize"})
            _, ended = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                last_event_id=FROM_START,
            )
            assert ended
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 409
            # 未停在中断点取审阅包 → 409（不返回半成品）。
            assert (
                await client.get(f"/tasks/{thread_id}/review")
            ).status_code == 409

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
        # 端到端主干走 rewriter_loop 真实现链路：写作与自审经键控应答分派。
        app = _make_app(
            TRUNK_RESPONSES,
            checkpointer=saver,
            keyed={**FRAMEWORK_KEYED_RESPONSES, **WRITER_KEYED_RESPONSES},
            rewriter_stub=False,
        )
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, f"sess-trunk-{backend}")

            # 并发消费两条 SSE（{epoch}-0 续传从流首回放）：业务流到审阅请求，
            # 可视化流到 gate_blocked。
            (business, _), (graph_frames, _) = await asyncio.gather(
                _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                    last_event_id=FROM_START,
                ),
                _read_sse(
                    client,
                    f"/graph_events?thread_id={thread_id}",
                    _stop_on_types({"gate_blocked"}),
                    last_event_id=FROM_START,
                ),
            )
            # 主干链路完整：可视化流可见首跑主路径的主节点与两个子智能体的活动
            # （writing_orchestrator 只在修订轮运行，首跑走 chapter_drafter 并行首写）。
            units = {f["data"]["unit"] for f in graph_frames}
            assert {
                "framework_orchestrator",
                "reference_orchestrator",
                "chapter_drafter",
                "document_reviewer",
                "human_review_gate",
                "search_agent",
                "rewriter_loop",
            } <= units
            # 引文门禁首轮通过：审阅请求只携路由元数据，警告全文走 GET /review。
            review = next(f for f in business if f["event"] == "review_required")
            assert "citation_warnings" not in review["data"]["data"]
            pack = (await client.get(f"/tasks/{thread_id}/review")).json()
            assert pack["citation_warnings"] == []
            assert pack["pack_version"]

            # 提交混合两类分支的修订意见：2/2 章受影响超过大纲一半，
            # 先携解析清单重新中断待确认（issue #49 大扇出确认）。
            response = await client.post(
                f"/tasks/{thread_id}/review",
                json={
                    "action": "revise",
                    "feedback": "引言口吻克制些；第二章补充行业数据",
                },
            )
            assert response.status_code == 202
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(2),
                last_event_id=FROM_START,
            )
            second = [f for f in business if f["event"] == "review_required"][1]
            confirmation = second["data"]["data"]["pending_confirmation"]
            assert confirmation["affected_chapter_ids"] == ["ch1", "ch2"]
            assert confirmation["total_chapters"] == 2

            # 确认清单后执行，迭代一轮后回到中断点。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "confirm"}
            )
            assert response.status_code == 202
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(3),
                last_event_id=FROM_START,
            )
            third = [f for f in business if f["event"] == "review_required"][2]
            assert third["data"]["data"]["iteration_round"] == 1
            assert "citation_warnings" not in third["data"]["data"]
            assert "review_warnings" not in third["data"]["data"]
            assert "pending_confirmation" not in third["data"]["data"]

            # 定稿：两类修订都已落实到对应章节。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            business, ended = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                last_event_id=FROM_START,
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
            # 类型标识按来源通道产出（打桩按假说 ID 确定性分派：
            # ch1 落结构化数据、ch2 落联网），联网来源带真实链接位。
            entry_texts = [e["text"] for e in rendered["bibliography"]]
            assert any("[DS]" in text for text in entry_texts)
            assert any(
                "[EB/OL]" in text and "https://stub.example/" in text
                for text in entry_texts
            )

            # 格式与内容解耦：同一引文库按另一格式渲染出不同条目文本。
            apa = (
                await client.get(f"/tasks/{thread_id}/bibliography?format=apa")
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
            else InMemorySaver(serde=checkpoint_serializer())
        )
        asyncio.run(main(saver))


def test_独立检索接口_同步返回素材与诊断且事件按session_id可订阅():
    """独立阻塞式检索接口 E2E（issue #38）。

    真适配层 + 假引擎运行时：POST /retrieval 同步返回素材（回链假说 ID、
    verdict、url/source_kind）与 diagnostics 块；进度事件带调用方 session_id
    经全局 /graph_events 通道可过滤订阅，start/progress/end 父子链正确。
    与主流程同一实例（工厂只实例化一次）、同一契约映射
    （引擎载荷与 engine_payload_from_task 逐字段相等）。
    """
    from agents.search_agent.mapping import engine_payload_from_task

    runtime = FakeSearchAgentRuntime(latency_seconds=0.01)
    factory_hooks: list[Any] = []

    def factory(hook: Any) -> Any:
        factory_hooks.append(hook)
        return make_search_agent(hook, runtime=runtime)

    async def main() -> None:
        app = _make_app([], search_agent=factory)
        async with _client(app) as client:
            request_body = {
                "chapter_id": "ch-solo",
                "hypotheses": [
                    {"id": "h1", "text": "假说一", "refute_condition": "反驳条件一"},
                    {"id": "h2", "text": "假说二", "refute_condition": ""},
                ],
                "session_id": "sess-retrieval",
            }
            response = await client.post("/retrieval", json=request_body)
            assert response.status_code == 200
            body = response.json()

            # 素材逐条回链假说 ID，verdict / url / source_kind 齐备：
            # 正向支撑素材 pass，反向（反驳条件）素材一律 fail，
            # 仅联网来源带链接。
            materials = body["materials"]
            assert materials
            assert {m["hypothesis_id"] for m in materials} <= {"h1", "h2"}
            for material in materials:
                _assert_opaque_material_id(material["id"])
                assert material["source_ref"]
                assert material["verdict"] in ("pass", "fail")
                assert material["source_kind"] in (
                    "web",
                    "knowledge_base",
                    "structured_data",
                )
                assert (material["source_kind"] == "web") == (
                    material["url"] is not None
                )
            assert any(m["verdict"] == "pass" for m in materials)
            assert any(m["verdict"] == "fail" for m in materials)

            # 诊断块：与 subagent_end 事件同源的摘要子集
            # （2 正向项 + 1 反向项 = 3 个检索项）。
            diagnostics = body["diagnostics"]
            assert diagnostics["call_counts"] == {"web_search": 3, "web_fetch": 3}
            assert "total_elapsed_ms" in diagnostics
            assert diagnostics["judge_integrity"]["judge_missing_candidate_count"] == 0

            # 事件按 session_id 过滤订阅到，成对且父子链正确（{epoch}-0 续传
            # 从流首回放检索期间已发布的全部事件，确定性捕获完整事件链）。
            frames, _ = await _read_sse(
                client,
                "/graph_events?session_id=sess-retrieval",
                _stop_on_types({"subagent_end"}),
                last_event_id=FROM_START,
            )
            envelopes = [f["data"] for f in frames]
            assert envelopes
            assert all(e["session_id"] == "sess-retrieval" for e in envelopes)
            starts = [e for e in envelopes if e["type"] == "subagent_start"]
            ends = [e for e in envelopes if e["type"] == "subagent_end"]
            progresses = [e for e in envelopes if e["type"] == "progress"]
            assert [e["payload"]["chapter_id"] for e in starts] == ["ch-solo"]
            assert [e["payload"]["chapter_id"] for e in ends] == ["ch-solo"]
            assert starts[0]["parent_id"] is None
            start_id = starts[0]["event_id"]
            assert progresses
            assert all(e["parent_id"] == start_id for e in progresses)
            assert ends[0]["parent_id"] == start_id
            assert progresses[0]["payload"]["step"] == "engine_call_start"
            assert progresses[-1]["payload"]["step"] == "engine_call_end"
            assert ends[0]["payload"]["diagnostics"] == diagnostics

    asyncio.run(main())

    # 同一实例：lifespan 只经工厂实例化一次，主流程与独立接口共用。
    assert len(factory_hooks) == 1
    # 同一契约：独立接口的引擎载荷与主流程适配层的映射函数逐字段相等
    # （genre 与既有素材摘要走缺省值）。
    expected_task = {
        "chapter_id": "ch-solo",
        "hypotheses": [
            {"id": "h1", "text": "假说一", "refute_condition": "反驳条件一"},
            {"id": "h2", "text": "假说二", "refute_condition": ""},
        ],
        "genre": "",
        "existing_materials_digest": "",
    }
    assert runtime.payloads == [engine_payload_from_task(expected_task)]


def test_独立检索接口_请求体校验失败返回422():
    async def main() -> None:
        app = _make_app([])
        async with _client(app) as client:
            valid_hypotheses = [{"id": "h1", "text": "假说一", "refute_condition": ""}]
            # 缺 hypotheses 字段。
            response = await client.post("/retrieval", json={"chapter_id": "ch1"})
            assert response.status_code == 422
            # 空假说列表。
            response = await client.post(
                "/retrieval", json={"chapter_id": "ch1", "hypotheses": []}
            )
            assert response.status_code == 422
            # 空白 chapter_id。
            response = await client.post(
                "/retrieval",
                json={"chapter_id": "  ", "hypotheses": valid_hypotheses},
            )
            assert response.status_code == 422
            # 空白假说本文。
            response = await client.post(
                "/retrieval",
                json={
                    "chapter_id": "ch1",
                    "hypotheses": [{"id": "h1", "text": " ", "refute_condition": ""}],
                },
            )
            assert response.status_code == 422

    asyncio.run(main())


def test_独立检索接口_引擎域异常沿既有映射模式转状态码():
    """引擎域异常经既有「异常→状态码」注册模式映射：配置缺失 503、契约违约 422。"""
    from search_agent.api import (
        SearchAgentConfigurationError,
        SearchAgentContractError,
    )

    def raise_configuration_error(payload: dict[str, Any]) -> None:
        raise SearchAgentConfigurationError("检索通道配置缺失")

    def raise_contract_error(payload: dict[str, Any]) -> None:
        raise SearchAgentContractError("检索任务不符合引擎入参契约")

    async def main() -> None:
        request_body = {
            "chapter_id": "ch1",
            "hypotheses": [{"id": "h1", "text": "假说一", "refute_condition": ""}],
        }
        app = _make_app(
            [],
            search_agent=lambda hook: make_search_agent(
                hook,
                runtime=FakeSearchAgentRuntime(side_effect=raise_configuration_error),
            ),
        )
        async with _client(app) as client:
            response = await client.post("/retrieval", json=request_body)
            assert response.status_code == 503
            assert "检索通道配置缺失" in response.json()["detail"]

        app = _make_app(
            [],
            search_agent=lambda hook: make_search_agent(
                hook,
                runtime=FakeSearchAgentRuntime(side_effect=raise_contract_error),
            ),
        )
        async with _client(app) as client:
            response = await client.post("/retrieval", json=request_body)
            assert response.status_code == 422
            assert "引擎入参契约" in response.json()["detail"]

    asyncio.run(main())


# ---- 传输底座（issue #54）：keepalive / 断线检测 / 优雅关流 / Last-Event-ID 续传 ----


def test_新订阅不带LastEventID只收实时不回放历史():
    """AC：不带 Last-Event-ID 的新订阅只收实时事件，不回放历史。"""

    async def main() -> None:
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, "sess-fresh")
            # 跑到首轮审阅中断点：枢纽已累积 iteration_round=0 的历史事件。
            first, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            historical_last_seq = int(first[-1]["id"].rsplit("-", 1)[-1])
            # 新订阅（不带 Last-Event-ID）：先开流注册，再提交修订触发新事件。
            reader = asyncio.create_task(
                _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                )
            )
            await asyncio.sleep(0.1)  # 让订阅完成注册（实时订阅）
            await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章口吻克制些"},
            )
            business, _ = await reader
            # 不回放历史：新订阅首帧 seq 严格大于历史末帧 seq（只收订阅之后的事件）。
            assert int(business[0]["id"].rsplit("-", 1)[-1]) > historical_last_seq
            # 不回放首轮历史 review_required：只收到修订后 iteration_round=1 的。
            rounds = [
                f["data"]["data"]["iteration_round"]
                for f in business
                if f["event"] == "review_required"
            ]
            assert rounds == [1]

    asyncio.run(main())


def test_断线重连带LastEventID只续推该id之后事件不重复不回放():
    """AC：带 Last-Event-ID 重连只收到该 id 之后的事件，不重复不回放。"""

    async def main() -> None:
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, "sess-resume")
            # 首轮：续传从流首读至审阅中断点，记下最后一条事件 id。
            first, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            last_id = first[-1]["id"]
            last_seq = int(last_id.rsplit("-", 1)[-1])
            # 提交修订触发新事件后断线重连：带 Last-Event-ID 续传。
            await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章口吻克制些"},
            )
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=last_id,
            )
            # 续传不重复：首帧 seq 严格大于 last_id 的 seq。
            first_seq = int(business[0]["id"].rsplit("-", 1)[-1])
            assert first_seq > last_seq
            # 不回放历史：只含修订后 iteration_round=1 的 review_required。
            rounds = [
                f["data"]["data"]["iteration_round"]
                for f in business
                if f["event"] == "review_required"
            ]
            assert rounds == [1]

    asyncio.run(main())


def test_世代失配重连立即收到reconcile_required后转实时():
    """AC：携带上一世代 Last-Event-ID 重连，立即收到 reconcile_required。"""
    saver = InMemorySaver(serde=checkpoint_serializer())

    async def main() -> None:
        # 第一个"进程"：epoch=ep-old，跑到审阅中断点并记录旧世代 last id。
        app1 = _make_app(FIRST_PASS_RESPONSES, checkpointer=saver, epoch="ep-old")
        async with _client(app1) as client:
            thread_id, _ = await _create_task(client, "sess-epoch")
            first, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id="ep-old-0",
            )
            old_last_id = first[-1]["id"]
            assert old_last_id.startswith("ep-old-")
        # 第二个"进程"：epoch=ep-new（进程重启世代切换），同存档器。
        app2 = _make_app([], checkpointer=saver, epoch="ep-new")
        async with _client(app2) as client:
            # 状态查询按检查点重建登记与业务枢纽（世代 ep-new）。
            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "AWAIT_USER_REVIEW"
            hub = app2.state.manager.business_hub(thread_id)
            # 带旧世代 Last-Event-ID 重连开流：立即收 reconcile_required。
            reader = asyncio.create_task(
                _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    lambda fs: any(f["event"] == "review_required" for f in fs),
                    last_event_id=old_last_id,
                )
            )
            # 等到流已开并注册（reconcile 已投递），再触发实时事件。
            for _ in range(50):
                if hub.subscriber_count == 1:
                    break
                await asyncio.sleep(0.02)
            assert hub.subscriber_count == 1
            await client.post(
                f"/tasks/{thread_id}/resume",
                json={"session_id": "sess-epoch-2"},
            )
            business, _ = await reader
            # reconcile_required 先于转实时的 review_required（后转实时推送）。
            rec_idx = next(
                i for i, f in enumerate(business) if f["event"] == "reconcile_required"
            )
            rev_idx = next(
                i for i, f in enumerate(business) if f["event"] == "review_required"
            )
            assert rec_idx < rev_idx
            rec = business[rec_idx]
            assert rec["data"]["reason"] == "epoch_mismatch"
            assert rec["data"]["last_event_id"] == old_last_id
            assert "GET /tasks/{id}" in rec["data"]["reconcile_via"]
            # reconcile 帧与转实时帧的传输 id 均属新世代 ep-new，形如 {epoch}-{seq}。
            assert rec["id"].startswith("ep-new-")
            assert business[rev_idx]["id"].startswith("ep-new-")
            # 转实时的 review_required seq 严格大于 reconcile 的 seq（单调递增）。
            assert int(business[rev_idx]["id"].rsplit("-", 1)[-1]) > int(
                rec["id"].rsplit("-", 1)[-1]
            )

    asyncio.run(main())


def test_keepalive_ping周期性收到心跳():
    """AC：SSE 长连接周期性收到 keepalive ping（缺省 15s）。"""
    from service.app import _DEFAULT_PING_INTERVAL

    # 缺省 ping 间隔为 15s（create_app 不传 ping_interval 时生效）。
    assert _DEFAULT_PING_INTERVAL == 15

    async def main() -> None:
        # ping 间隔调小到 1s 以便快速观测；缺省 15s 已由上面的常量断言保证。
        app = _make_app([], ping_interval=1)
        async with _client(app) as client:
            # 全局 graph_events 永不主动关闭、无历史事件：唯一会到达的就是 ping。
            got_ping = False

            async def consume() -> None:
                nonlocal got_ping
                async with client.stream("GET", "/graph_events") as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if line.startswith(":"):
                            got_ping = True
                            return

            await asyncio.wait_for(consume(), timeout=5.0)
            assert got_ping, "未在 5s 内收到 keepalive ping 注释行"

    asyncio.run(main())


def test_全局通道世代失配也下发reconcile_required():
    """全局可视化通道带旧世代 Last-Event-ID 重连同样下发 reconcile_required。"""

    async def main() -> None:
        app = _make_app([], epoch="ep-graph-new")
        async with _client(app) as client:
            frames, _ = await _read_sse(
                client,
                "/graph_events",
                lambda fs: any(f["event"] == "reconcile_required" for f in fs),
                last_event_id="ep-graph-old-1",
            )
            rec = next(f for f in frames if f["event"] == "reconcile_required")
            assert rec["data"]["reason"] == "epoch_mismatch"
            assert rec["data"]["last_event_id"] == "ep-graph-old-1"
            # 全局通道无可重取的 REST：reconcile_via 为空并附说明。
            assert rec["data"]["reconcile_via"] == []
            assert "无可重取" in rec["data"]["note"]
            assert rec["id"].startswith("ep-graph-new-")

    asyncio.run(main())


def test_客户端断开被服务端检测且停止推流():
    """AC：客户端断开被服务端检测并停止推流（订阅队列注销）。"""

    async def main() -> None:
        app = _make_app([])
        async with _client(app) as client:
            graph_hub: EventHub = app.state.graph_hub
            assert graph_hub.subscriber_count == 0
            # 打开一条全局可视化流并保持连接。
            async with client.stream("GET", "/graph_events") as response:
                assert response.status_code == 200
                # 等待订阅注册完成。
                for _ in range(50):
                    if graph_hub.subscriber_count == 1:
                        break
                    await asyncio.sleep(0.02)
                assert graph_hub.subscriber_count == 1
            # 客户端断开：生成器被取消、队列在 finally 中注销。
            for _ in range(50):
                if graph_hub.subscriber_count == 0:
                    break
                await asyncio.sleep(0.02)
            assert graph_hub.subscriber_count == 0, "断开后订阅队列未注销"

    asyncio.run(main())


def test_服务关停优雅关流():
    """AC：服务关停优雅关流（lifespan 关闭枢纽，生成器收到结束哨兵）。"""

    async def main() -> None:
        app = _make_app([])
        async with _client(app) as client:
            graph_hub: EventHub = app.state.graph_hub
            assert not graph_hub.closed
            # 开一条全局可视化流（永不主动关闭），确认运行中枢纽未关。
            async with client.stream("GET", "/graph_events") as response:
                assert response.status_code == 200
                assert not graph_hub.closed
        # _client 退出 → server.should_exit → lifespan 关停 → manager.shutdown()
        # 与 graph_hub.close() 落地，SSE 生成器收到结束哨兵后优雅关流。
        assert app.state.graph_hub.closed

    asyncio.run(main())


def test_stats端点可观测背压():
    """AC：dropped 与 subscriber_count 可经 stats 端点观测（issue #55）。"""

    async def main() -> None:
        app = _make_app([])
        async with _client(app) as client:
            # 未知任务的业务 stats → 404。
            miss = await client.get("/tasks/nope/stream/stats")
            assert miss.status_code == 404

            # 全局可视化 stats：无订阅者、无丢弃、世代即测试固定值。
            stats = (await client.get("/graph_events/stats")).json()
            assert stats["subscriber_count"] == 0
            assert stats["dropped"] == 0
            assert stats["epoch"] == TEST_EPOCH

            # 业务 stats：建一个任务，无订阅者时计数为 0。
            thread_id, _ = await _create_task(client)
            biz = (await client.get(f"/tasks/{thread_id}/stream/stats")).json()
            assert biz["thread_id"] == thread_id
            assert biz["subscriber_count"] == 0
            assert biz["dropped"] == 0
            assert biz["epoch"] == TEST_EPOCH

            # 开一条全局流并保持连接，期间 subscriber_count 经端点观测为 1。
            async with client.stream("GET", "/graph_events") as response:
                assert response.status_code == 200
                for _ in range(50):
                    live = (await client.get("/graph_events/stats")).json()
                    if live["subscriber_count"] >= 1:
                        break
                    await asyncio.sleep(0.02)
                assert live["subscriber_count"] >= 1
                assert live["epoch"] == TEST_EPOCH
            # 客户端断开后，订阅者数回落为 0。
            for _ in range(50):
                after = (await client.get("/graph_events/stats")).json()
                if after["subscriber_count"] == 0:
                    break
                await asyncio.sleep(0.02)
            assert after["subscriber_count"] == 0

    asyncio.run(main())


# ---- 运行中产物快照（issue #56）：GET /tasks/{id}/products REST 真相源 ----


def _drive_to_framework_complete(saver: Any, thread_id: str) -> Any:
    """手工驱动同款图到框架完成检查点后停止，返回图对象供断言对照。

    与 test_断点续跑_图运行中途死亡后resume续跑至中断点 同款范式：共享
    存档器 + FakeLLM + 打桩子智能体，断点取 reference_orchestrator 待执行
    的检查点（框架已完成、检索与写作未开始）。
    """
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    graph = build_graph(
        llm_factory=lambda unit: fake,
        checkpointer=saver,
        search_agent=make_stub_search_agent(),
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    for mode, chunk in graph.stream(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-framework"),
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
    return graph


def _drive_to_review_gate(saver: Any, thread_id: str) -> Any:
    """手工驱动同款图到人工审阅中断点后停止，返回图对象供断言对照。

    检查点停在 human_review_gate：目录/假说/各章素材/已完成章正文齐全。
    """
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    graph = build_graph(
        llm_factory=lambda unit: fake,
        checkpointer=saver,
        search_agent=make_stub_search_agent(),
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    graph.invoke(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-gate"), config
    )
    assert graph.get_state(config).next == ("human_review_gate",)
    return graph


def test_产物快照_空快照与未知任务404():
    """(a) 刚创建 IDLE 无产物 → 空快照；(e) 不存在 thread_id → 404。"""
    saver = InMemorySaver(serde=checkpoint_serializer())
    thread_id = "products-idle"
    # 手工驱动到首个检查点（初始状态：IDLE、outline 为空）即停，模拟刚创建。
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    graph = build_graph(
        llm_factory=lambda unit: fake,
        checkpointer=saver,
        search_agent=make_stub_search_agent(),
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    for mode, chunk in graph.stream(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-idle"),
        config,
        stream_mode=["updates", "debug"],
    ):
        if (
            mode == "debug"
            and isinstance(chunk, dict)
            and chunk.get("type") == "checkpoint"
            and "framework_orchestrator" in (chunk["payload"].get("next") or [])
        ):
            break
    # 初始检查点：IDLE、outline 为空。
    snapshot = graph.get_state(config)
    assert snapshot.values.get("status") == "IDLE"

    async def main() -> None:
        app = _make_app([], checkpointer=saver)
        async with _client(app) as client:
            # (a) 空快照：chapters=[]，status=IDLE，iteration_round=0。
            response = await client.get(f"/tasks/{thread_id}/products")
            assert response.status_code == 200
            body = response.json()
            assert body["thread_id"] == thread_id
            assert body["status"] == "IDLE"
            assert body["iteration_round"] == 0
            assert body["chapters"] == []

            # (e) 不存在 thread_id → 404。
            response = await client.get("/tasks/nope/products")
            assert response.status_code == 404

    asyncio.run(main())


def test_产物快照_框架完成无草稿素材标注未产出():
    """(b) 框架完成、尚无草稿/素材 → 目录/假说齐全，materials=[]、draft=null。"""
    saver = InMemorySaver(serde=checkpoint_serializer())
    thread_id = "products-framework"
    graph = _drive_to_framework_complete(saver, thread_id)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    assert snapshot.next == ("reference_orchestrator", "reference_orchestrator")

    async def main() -> None:
        app = _make_app(
            [SEMANTIC_PASS, SEMANTIC_PASS, DOCUMENT_REVIEW_PASS],
            checkpointer=saver,
        )
        async with _client(app) as client:
            response = await client.get(f"/tasks/{thread_id}/products")
            assert response.status_code == 200
            body = response.json()
            assert body["thread_id"] == thread_id
            assert body["status"] == "FRAMEWORK_BUILDING"
            assert body["iteration_round"] == 0
            chapters = body["chapters"]
            assert len(chapters) == 2
            # 每章目录与假说齐全，且无素材、无草稿（draft=null 标注未产出）。
            for chapter in chapters:
                assert chapter["chapter_id"] in {"ch1", "ch2"}
                assert chapter["title"]
                assert isinstance(chapter["subsections"], list)
                assert isinstance(chapter["chapter_type"], (str, type(None)))
                assert isinstance(chapter["planned_summary"], str)
                assert chapter["materials"] == []
                assert chapter["draft"] is None
                assert len(chapter["points"]) >= 1
                for point in chapter["points"]:
                    assert point["id"]
                    assert point["text"]
                    assert isinstance(point["hypotheses"], list)
                    for hyp in point["hypotheses"]:
                        assert {"id", "text", "refute_condition", "angle"} <= set(hyp)

            # 与检查点 state 逐字段一致：outline 的 ChapterSpec 与 products 的章条目。
            outline = snapshot.values.get("outline", [])
            assert [c["chapter_id"] for c in chapters] == [spec.id for spec in outline]
            for chapter, spec in zip(chapters, outline):
                assert chapter["title"] == spec.title
                assert chapter["subsections"] == list(spec.subsections)
                assert chapter["chapter_type"] == spec.chapter_type
                assert chapter["points"] == [p.model_dump() for p in spec.points]

    asyncio.run(main())


def test_产物快照_部分章完成未完成章标注未产出():
    """验收标准1的"部分章完成"态：已写完的章正文出现，未完成章 draft=null。

    并行首写在单一超步内完成全部章草稿，committed 检查点观察不到部分草稿
    态；故以 graph.update_state 在框架完成态（无草稿）上确定性注入单章草稿，
    构造"ch1 已完成、ch2 未完成"检查点，直接验证章级标记逻辑。
    """
    saver = InMemorySaver(serde=checkpoint_serializer())
    thread_id = "products-partial"
    graph = _drive_to_framework_complete(saver, thread_id)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    # 框架完成态：outline 含 ch1/ch2，无草稿。
    pre_outline = graph.get_state(config).values.get("outline", [])
    assert {spec.id for spec in pre_outline} == {"ch1", "ch2"}
    assert graph.get_state(config).values.get("chapter_drafts", []) == []

    # 注入单章草稿：merge_chapter_drafts 在空库上接受 [ch1_draft] → 仅 ch1 有草稿。
    graph.update_state(
        config,
        {
            "chapter_drafts": [
                ChapterDraft(
                    chapter_id="ch1",
                    text="第一章已写完的正文。",
                    summary="第一章摘要。",
                    self_check=SelfCheck(),
                )
            ]
        },
    )
    drafts = graph.get_state(config).values.get("chapter_drafts", [])
    assert [d.chapter_id for d in drafts] == ["ch1"]

    async def main() -> None:
        app = _make_app([], checkpointer=saver)
        async with _client(app) as client:
            response = await client.get(f"/tasks/{thread_id}/products")
            assert response.status_code == 200
            chapters = {c["chapter_id"]: c for c in response.json()["chapters"]}
            # ch1：已完成，正文/摘要出现。
            assert chapters["ch1"]["draft"] is not None
            assert chapters["ch1"]["draft"]["text"] == "第一章已写完的正文。"
            assert chapters["ch1"]["draft"]["summary"] == "第一章摘要。"
            # ch2：未完成，明确标注未产出（draft=null），目录/假说仍在。
            assert chapters["ch2"]["draft"] is None
            assert chapters["ch2"]["title"]
            assert chapters["ch2"]["materials"] == []

    asyncio.run(main())


def test_产物快照_审阅门产物与检查点及finalized事件一致并定稿后正文齐全():
    """(c) 停在审阅门 → 产物与检查点一致且与 finalized 事件正文/摘要一致；
    (d) 定稿后 status=FINISHED、各章正文齐全。"""
    saver = InMemorySaver(serde=checkpoint_serializer())
    thread_id = "products-gate"
    graph = _drive_to_review_gate(saver, thread_id)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    checkpoint_drafts = snapshot.values.get("chapter_drafts", [])
    checkpoint_library = snapshot.values.get("citation_library", [])
    checkpoint_outline = snapshot.values.get("outline", [])

    async def main() -> None:
        # 续跑只备定稿后所需：finalize 不调 LLM，故空应答即可。
        app = _make_app([], checkpointer=saver)
        async with _client(app) as client:
            # 先 resume 让 HTTP 进程登记任务（停在审阅门，不重跑图）。
            response = await client.post(
                f"/tasks/{thread_id}/resume", json={"session_id": "sess-gate"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "AWAIT_USER_REVIEW"

            # (c) 审阅门产物快照：目录/假说/各章素材/已完成章正文齐全。
            response = await client.get(f"/tasks/{thread_id}/products")
            assert response.status_code == 200
            body = response.json()
            assert body["thread_id"] == thread_id
            assert body["status"] == "AWAIT_USER_REVIEW"
            assert body["iteration_round"] == 0
            chapters = body["chapters"]
            assert len(chapters) == len(checkpoint_outline)

            # 素材按章分组、与检查点引文库逐字段一致。
            for chapter, spec in zip(chapters, checkpoint_outline):
                assert chapter["chapter_id"] == spec.id
                assert chapter["title"] == spec.title
                assert chapter["subsections"] == list(spec.subsections)
                assert chapter["points"] == [p.model_dump() for p in spec.points]
                expected_materials = [
                    m.model_dump()
                    for m in checkpoint_library
                    if m.chapter_id == spec.id
                ]
                assert chapter["materials"] == expected_materials
                # 该章已完成正文：draft 非空且与检查点草稿逐字段一致。
                expected_draft = next(
                    d for d in checkpoint_drafts if d.chapter_id == spec.id
                )
                assert chapter["draft"] == expected_draft.model_dump()

            # 定稿：finalized 事件正文/摘要与快照 draft 一致。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            business, ended = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                last_event_id=FROM_START,
            )
            assert ended
            finalized = next(f for f in business if f["event"] == "finalized")
            finalized_chapters = {
                c["chapter_id"]: c for c in finalized["data"]["data"]["chapters"]
            }
            for chapter in chapters:
                fc = finalized_chapters[chapter["chapter_id"]]
                assert chapter["draft"]["text"] == fc["text"]
                assert chapter["draft"]["summary"] == fc["summary"]

            # (d) 定稿后 status=FINISHED，各章正文齐全。
            response = await client.get(f"/tasks/{thread_id}/products")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "FINISHED"
            for chapter in body["chapters"]:
                assert chapter["draft"] is not None
                assert chapter["draft"]["text"]

    asyncio.run(main())


def test_产物快照_并发读不影响图运行():
    """(f) 运行中并发 GET /products 不影响图运行：全部 200、形状一致、图照常到审阅门。"""

    async def main() -> None:
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, "sess-concurrent")

            # 边消费 SSE（图在跑）边并发发多个 GET /products。
            sse_task = asyncio.create_task(
                _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                    last_event_id=FROM_START,
                )
            )
            # 等图运行起来再并发读。
            await asyncio.sleep(0.1)

            responses = await asyncio.gather(
                *(client.get(f"/tasks/{thread_id}/products") for _ in range(5))
            )
            assert all(r.status_code == 200 for r in responses)
            bodies = [r.json() for r in responses]
            # 形状一致：thread_id 正确、字段齐全、chapters 为列表。
            for body in bodies:
                assert body["thread_id"] == thread_id
                assert "status" in body
                assert "iteration_round" in body
                assert isinstance(body["chapters"], list)
            # 图运行照常到达审阅门（review_required 仍收到）。
            business, _ = await sse_task
            assert any(f["event"] == "review_required" for f in business)

    asyncio.run(main())


# ---- 产物流（issue #58）：结构化产物整块事件上网线 ----


def _product_kinds(frames: list[dict]) -> list[str]:
    """业务流里的产物事件 kind 序列（按到达顺序）。"""
    return [f["data"]["data"]["kind"] for f in frames if f["event"] == "product"]


def test_产物流_结构化产物按序整块上网线且可视化通道无产物全文():
    """mock 替身起服务跑一单任务：业务 SSE 上按产出顺序收到
    outline_ready → 各章 materials_ready → 各章 chapter_ready，载荷为整块产物；
    产物事件与 status/review_required 在同一条流上复用、type 可区分；
    全局可视化通道不出产物全文、12 元数据事件类型不变。
    """

    async def main() -> None:
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, "sess-product")

            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )

            # 产物事件与既有 status/review_required 在同一条流复用、type 可区分。
            types_seen = {f["event"] for f in business}
            assert "product" in types_seen
            assert "status" in types_seen
            assert "review_required" in types_seen

            kinds = _product_kinds(business)
            assert kinds.count("outline_ready") == 1
            assert sorted(k for k in kinds if k == "materials_ready") == [
                "materials_ready",
                "materials_ready",
            ]
            assert sorted(k for k in kinds if k == "chapter_ready") == [
                "chapter_ready",
                "chapter_ready",
            ]

            # 按产出顺序：outline_ready 先于各 materials_ready 先于各 chapter_ready，
            # 全部产物事件先于 review_required（停审阅门即收齐产物）。
            outline_idx = kinds.index("outline_ready")
            mat_idxs = [i for i, k in enumerate(kinds) if k == "materials_ready"]
            chap_idxs = [i for i, k in enumerate(kinds) if k == "chapter_ready"]
            assert outline_idx < min(mat_idxs) < min(chap_idxs)
            review_idx = next(
                i for i, f in enumerate(business) if f["event"] == "review_required"
            )
            product_positions = [
                i for i, f in enumerate(business) if f["event"] == "product"
            ]
            assert max(product_positions) < review_idx

            # 载荷为整块产物：outline 含各章骨架与论点/假说。
            outline_event = next(
                f
                for f in business
                if f["event"] == "product"
                and f["data"]["data"]["kind"] == "outline_ready"
            )
            outline = outline_event["data"]["data"]["outline"]
            assert {c["id"] for c in outline} == {"ch1", "ch2"}
            ch1_spec = next(c for c in outline if c["id"] == "ch1")
            assert ch1_spec["title"] == "第一章"
            assert ch1_spec["points"] and ch1_spec["points"][0]["hypotheses"]

            # 各章素材整块：ch1/ch2 各一条素材，回链假说 id。
            mats_by_chapter = {
                f["data"]["data"]["chapter_id"]: f["data"]["data"]["materials"]
                for f in business
                if f["event"] == "product"
                and f["data"]["data"]["kind"] == "materials_ready"
            }
            assert set(mats_by_chapter) == {"ch1", "ch2"}
            assert [m["hypothesis_id"] for m in mats_by_chapter["ch1"]] == [
                "ch1-p1-h1"
            ]
            assert [m["hypothesis_id"] for m in mats_by_chapter["ch2"]] == [
                "ch2-p1-h1"
            ]
            for mats in mats_by_chapter.values():
                for material in mats:
                    _assert_opaque_material_id(material["id"])
                    assert material["source_ref"] is not None
            assert all(
                m["verdict"] == "pass"
                for mats in mats_by_chapter.values()
                for m in mats
            )

            # 各章正文整块锚：draft 文本非空、含原位角标。
            drafts_by_chapter = {
                f["data"]["data"]["chapter_id"]: f["data"]["data"]["draft"]
                for f in business
                if f["event"] == "product"
                and f["data"]["data"]["kind"] == "chapter_ready"
            }
            assert set(drafts_by_chapter) == {"ch1", "ch2"}
            for chapter_id, draft in drafts_by_chapter.items():
                assert draft["chapter_id"] == chapter_id
                assert draft["text"]
                assert draft["summary"]

            # 全局可视化通道：不出产物全文、12 元数据事件类型不变。
            graph_frames, _ = await _read_sse(
                client,
                f"/graph_events?thread_id={thread_id}",
                _stop_on_types({"gate_blocked"}),
                last_event_id=FROM_START,
            )
            graph_types = {f["event"] for f in graph_frames}
            assert graph_types <= GRAPH_EVENT_TYPES
            assert "product" not in graph_types
            blob = json.dumps([f["data"] for f in graph_frames], ensure_ascii=False)
            assert "打桩摘录" not in blob  # 素材 excerpt 不进可视化通道
            assert "打桩正文" not in blob  # 草稿 text 不进可视化通道
            assert "假说一" not in blob  # 假说正文不进可视化通道

    asyncio.run(main())


def test_审阅包_摘要事件与REST全文同源且review_required仅路由元数据():
    """票 #60：停审阅门时 SSE 双发——review_pack_ready 摘要产物事件
    （可丢级，含 pack_version，不含章正文/素材全文）+ review_required 纯路由
    信号（必达）。GET /tasks/{id}/review 取六类内容全文，与摘要 pack_version 同源；
    重复调用幂等，修订再停门 pack_version 变化。
    """

    async def main() -> None:
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, "sess-review-pack")

            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )

            pack_event = next(
                f
                for f in business
                if f["event"] == "product"
                and f["data"]["data"]["kind"] == "review_pack_ready"
            )
            summary = pack_event["data"]["data"]
            # 摘要只推计数 + pack_version，绝不含章正文或素材全文。
            assert summary["chapter_ids"] == ["ch1", "ch2"]
            assert summary["chapter_total"] == 2
            assert summary["chapter_completed"] == 2
            assert summary["material_count"] >= 1
            assert summary["iteration_round"] == 0
            assert summary["pack_version"]
            summary_blob = json.dumps(summary, ensure_ascii=False)
            assert "打桩正文" not in summary_blob
            assert "打桩摘录" not in summary_blob

            # 摘要与 REST 全文同源：pack_version 一致、计数与全文长度对齐。
            pack = (await client.get(f"/tasks/{thread_id}/review")).json()
            assert pack["pack_version"] == summary["pack_version"]
            assert pack["chapters"][0]["text"]  # 全文只在 REST
            assert summary["chapter_total"] == len(pack["outline"])
            assert summary["material_count"] == len(pack["citation_library"])

            # review_required 紧随其后（先产物后信号），且仅路由元数据。
            pack_idx = business.index(pack_event)
            review_idx = next(
                i
                for i, f in enumerate(business)
                if f["event"] == "review_required"
            )
            assert pack_idx < review_idx
            routing = next(
                f for f in business if f["event"] == "review_required"
            )["data"]["data"]
            assert set(routing) <= {
                "iteration_round",
                "chapter_ids",
                "error",
                "clarification_questions",
                "pending_confirmation",
            }
            routing_blob = json.dumps(routing, ensure_ascii=False)
            assert "打桩正文" not in routing_blob
            assert "citation_warnings" not in routing
            assert "review_warnings" not in routing

            # 重复调用幂等（同检查点同 pack_version）。
            assert (
                await client.get(f"/tasks/{thread_id}/review")
            ).json()["pack_version"] == pack["pack_version"]

            # 修订再停门后内容变化 → pack_version 变化，REST 重取得新一轮内容。
            await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": "第二章口吻克制些"},
            )
            await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(2),
                last_event_id=FROM_START,
            )
            pack2 = (await client.get(f"/tasks/{thread_id}/review")).json()
            assert pack2["pack_version"] != pack["pack_version"]
            assert pack2["iteration_round"] == 1

    asyncio.run(main())


def test_产物流_产物事件载荷与products快照逐字段对账():
    """产物事件（可丢级，票 #55 已证 ``type=product`` 参与两级丢弃）丢帧后，
    经 ``GET /tasks/{id}/products``（票 #56）取回同等内容——逐字段对账。

    正常速率消费者不丢不重（dropped=0）；产物事件载荷与 REST 快照逐字段一致，
    即丢帧不丢内容、REST 是真相源。慢消费者丢产物帧的丢弃机制本身在
    ``tests/service/test_event_broker.py`` 已确定性单测覆盖，本用例聚焦
    SSE 产物事件载荷与 REST 真相源的逐字段对账（票 #55/#56/#58 联动验证）。
    """

    async def main() -> None:
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, "sess-product-parity")

            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )

            # 正常速率消费者不丢不重。
            stats = (await client.get(f"/tasks/{thread_id}/stream/stats")).json()
            assert stats["dropped"] == 0

            body = (await client.get(f"/tasks/{thread_id}/products")).json()
            assert body["status"] == "AWAIT_USER_REVIEW"
            chapters = {c["chapter_id"]: c for c in body["chapters"]}

            # outline_ready 载荷与 REST 大纲逐字段一致（含论点/假说）。
            outline_event = next(
                f
                for f in business
                if f["event"] == "product"
                and f["data"]["data"]["kind"] == "outline_ready"
            )
            outline = {c["id"]: c for c in outline_event["data"]["data"]["outline"]}
            assert set(outline) == set(chapters)
            for chapter_id, spec in outline.items():
                rest = chapters[chapter_id]
                assert spec["title"] == rest["title"]
                assert spec["subsections"] == rest["subsections"]
                assert spec["chapter_type"] == rest["chapter_type"]
                assert spec["planned_summary"] == rest["planned_summary"]
                assert spec["points"] == rest["points"]

            # 各章 materials_ready 载荷与 REST 该章素材逐字段一致。
            mats_events = {
                f["data"]["data"]["chapter_id"]: f["data"]["data"]["materials"]
                for f in business
                if f["event"] == "product"
                and f["data"]["data"]["kind"] == "materials_ready"
            }
            assert set(mats_events) == set(chapters)
            for chapter_id, mats in mats_events.items():
                assert mats == chapters[chapter_id]["materials"]

            # 各章 chapter_ready 载荷与 REST 该章草稿逐字段一致。
            draft_events = {
                f["data"]["data"]["chapter_id"]: f["data"]["data"]["draft"]
                for f in business
                if f["event"] == "product"
                and f["data"]["data"]["kind"] == "chapter_ready"
            }
            assert set(draft_events) == set(chapters)
            for chapter_id, draft in draft_events.items():
                assert draft == chapters[chapter_id]["draft"]

            # AC #2：产物事件与 finalized 在同一条流上复用、type 可区分。
            # 提交定稿后重读整段历史，product 与 finalized 同流共存。
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            full, ended = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                last_event_id=FROM_START,
            )
            assert ended
            types_full = {f["event"] for f in full}
            assert "product" in types_full
            assert "finalized" in types_full
            assert any(
                f["data"]["data"]["kind"] == "chapter_ready"
                for f in full
                if f["event"] == "product"
            )

    asyncio.run(main())


# ---- 逐字流 content_delta（issue #59）：HTTP 级别验收 ----


class _EnvOverride:
    """临时覆盖 os.environ 键值对，退出时恢复（测试用、非线程安全）。"""

    def __init__(self, **overrides: str) -> None:
        self._overrides = overrides
        self._saved: dict[str, str | None] = {}

    def __enter__(self) -> "_EnvOverride":
        for key, value in self._overrides.items():
            self._saved[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc: Any) -> None:
        for key, original in self._saved.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _content_delta_payloads(frames: list[dict]) -> list[dict[str, Any]]:
    """从业务帧列表提取 content_delta 事件的 data 块。"""
    return [
        f["data"]["data"]
        for f in frames
        if f["event"] == "content_delta"
    ]


def test_逐字流_content_delta拼接一致与退化重试attempt递增():
    """mock 真链路（rewriter_stub=False）：业务 SSE 收 content_delta 帧，
    拼接 == chapter_ready 整块正文；退化重试 attempt 递增、sequence 复位；
    content_delta 不出现在可视化通道。"""

    async def main() -> None:
        # ch1 draft 前插一段未闭合 JSON 强制退化重试：attempt 1 失败、attempt 2 成功。
        bad_ch1_draft = '{"chapter_text": "部分未闭合'
        keyed = {
            **FRAMEWORK_KEYED_RESPONSES,
            **WRITER_KEYED_RESPONSES,
            # 覆盖 ch1 的 draft 键控序列：malformed 在前、valid draft 在后。
            "- 标题：第一章": [
                bad_ch1_draft,
                WRITER_KEYED_RESPONSES["- 标题：第一章"][0],
                WRITER_KEYED_RESPONSES["- 标题：第一章"][1],
            ],
        }
        # 小阈值 + 关时间窗口：确定性多帧、便于拼接断言。
        # env 须覆盖整个 async with _client 区间：make_rewriter_loop 在 lifespan
        # 启动期（uvicorn 起服务时）才读 env，_make_app 只构建 FastAPI 对象。
        with _EnvOverride(WRITER_DELTA_FLUSH_CHARS="4", WRITER_DELTA_FLUSH_MS="0"):
            app = _make_app(
                [*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES],
                keyed=keyed,
                rewriter_stub=False,
            )
            async with _client(app) as client:
                thread_id, _ = await _create_task(client, "sess-delta")

                business, _ = await _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                    last_event_id=FROM_START,
                )

                deltas = _content_delta_payloads(business)
                assert deltas, "draft/revise 期间须发 content_delta 事件"
                # 每帧字段齐全：chapter_id / mode / kind / delta / attempt / sequence。
                for payload in deltas:
                    assert payload["chapter_id"] in {"ch1", "ch2"}
                    assert payload["mode"] in {"draft", "revise"}
                    assert payload["kind"] in {"content", "thinking"}
                    assert isinstance(payload["delta"], str)
                    assert isinstance(payload["attempt"], int)
                    assert isinstance(payload["sequence"], int)

                # 退化重试：ch1 draft 的 attempt 1（malformed）有帧、attempt 2（valid）有帧；
                # attempt 2 的 sequence 从 0 复位（丢弃重建语义）。
                ch1_draft_deltas = [
                    p for p in deltas
                    if p["chapter_id"] == "ch1" and p["mode"] == "draft"
                ]
                attempt1 = [p for p in ch1_draft_deltas if p["attempt"] == 1]
                attempt2 = [p for p in ch1_draft_deltas if p["attempt"] == 2]
                assert attempt1, "ch1 draft 退化重试 attempt 1 须有 content_delta 帧"
                assert attempt2, "ch1 draft 退化重试 attempt 2 须有 content_delta 帧"
                assert attempt2[0]["sequence"] == 0, "新 attempt 的 sequence 须从 0 复位"
                # attempt 内 sequence 单调递增。
                assert [p["sequence"] for p in attempt2] == sorted(
                    p["sequence"] for p in attempt2
                )

                # 拼接 ch1 draft 最终 attempt 的 content 帧 == chapter_ready 整块正文。
                ch1_ready = next(
                    f["data"]["data"]
                    for f in business
                    if f["event"] == "product"
                    and f["data"]["data"]["kind"] == "chapter_ready"
                    and f["data"]["data"]["chapter_id"] == "ch1"
                )
                ch1_final_attempt = max(p["attempt"] for p in ch1_draft_deltas)
                ch1_final_content = "".join(
                    p["delta"]
                    for p in ch1_draft_deltas
                    if p["attempt"] == ch1_final_attempt and p["kind"] == "content"
                )
                assert ch1_final_content == ch1_ready["draft"]["text"], (
                    "逐字流最终 attempt 拼接须与 chapter_ready 整块正文一致"
                )

                # ch2 draft（无退化）：拼接 == chapter_ready 正文。
                ch2_draft_deltas = [
                    p for p in deltas
                    if p["chapter_id"] == "ch2" and p["mode"] == "draft"
                ]
                ch2_ready = next(
                    f["data"]["data"]
                    for f in business
                    if f["event"] == "product"
                    and f["data"]["data"]["kind"] == "chapter_ready"
                    and f["data"]["data"]["chapter_id"] == "ch2"
                )
                ch2_final_attempt = max(p["attempt"] for p in ch2_draft_deltas)
                ch2_final_content = "".join(
                    p["delta"]
                    for p in ch2_draft_deltas
                    if p["attempt"] == ch2_final_attempt and p["kind"] == "content"
                )
                assert ch2_final_content == ch2_ready["draft"]["text"]

                # 可视化通道不出 content_delta（逐字流只走业务通道）。
                graph_frames, _ = await _read_sse(
                    client,
                    f"/graph_events?thread_id={thread_id}",
                    _stop_on_types({"gate_blocked"}),
                    last_event_id=FROM_START,
                )
                graph_types = {f["event"] for f in graph_frames}
                assert "content_delta" not in graph_types
                assert graph_types <= GRAPH_EVENT_TYPES

    asyncio.run(main())


def test_逐字流_合并粒度可配且stub链路不逐字流():
    """合并粒度（字符数阈值）经 WRITER_DELTA_FLUSH_CHARS 配置：
    小阈值→多帧、大阈值→少帧；rewriter_stub=True 的桩链路不逐字流。"""

    async def run_once(flush_chars: str) -> list[dict]:
        keyed = {**FRAMEWORK_KEYED_RESPONSES, **WRITER_KEYED_RESPONSES}
        with _EnvOverride(
            WRITER_DELTA_FLUSH_CHARS=flush_chars, WRITER_DELTA_FLUSH_MS="0"
        ):
            app = _make_app(
                [*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES],
                keyed=keyed,
                rewriter_stub=False,
            )
            async with _client(app) as client:
                thread_id, _ = await _create_task(client, f"sess-granule-{flush_chars}")
                business, _ = await _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                    last_event_id=FROM_START,
                )
                return _content_delta_payloads(business)

    async def main() -> None:
        small = await run_once("4")
        large = await run_once("10000")
        assert len(small) > len(large), (
            f"小阈值帧数 {len(small)} 须多于大阈值帧数 {len(large)}"
        )
        # 大阈值下 content_delta 帧数仍 ≥ 2（ch1 + ch2 draft 至少各一帧，由
        # flush_remaining 兜底）。
        assert len(large) >= 2

        # stub 链路不逐字流：rewriter_stub=True 时桩改写器不调 LLM、不发 content_delta。
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            thread_id, _ = await _create_task(client, "sess-stub-no-delta")
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            assert _content_delta_payloads(business) == []

    asyncio.run(main())


# ---- mock 档（issue #61）：双栈装配 + 形如真场景库 + 清理 ----

import re

_SNAKE_CASE_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def _assert_snake_case(obj: Any) -> None:
    """递归断言 dict/list 的所有键名皆为 snake_case（小写+下划线，无 camelCase）。"""
    if isinstance(obj, dict):
        for key in obj:
            assert isinstance(key, str), f"键非字符串类型：{key!r}"
            assert _SNAKE_CASE_KEY.match(key), f"键非 snake_case：{key!r}"
            _assert_snake_case(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            _assert_snake_case(item)


async def _create_mock_task(
    client: httpx.AsyncClient, session_id: str = "sess-mock"
) -> tuple[str, str]:
    """创建 mock 任务（``mock:true``），返回（thread_id, trace_id）。"""
    response = await client.post(
        "/tasks",
        json={
            "user_intent": "写一篇人才培养方案",
            "user_identity": "专业撰稿人",
            "session_id": session_id,
            "mock": True,
        },
    )
    assert response.status_code == 201
    body = response.json()
    return body["thread_id"], body["execution_trace_id"]


def test_mock档_秒回审阅门且事件序列与真实档同形字段snake_case():
    """mock 任务秒级走完到审阅门：业务/可视化双通道事件类型与真栈同形，
    所有业务帧 data 递归键名皆为 snake_case；状态/审阅包 REST 带 mock 标记。"""

    async def main() -> None:
        # 真栈 FakeLLM 给足应答（本用例只跑 mock 栈）；mock 栈用 DEFAULT_SCENARIO。
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            tid, _ = await _create_mock_task(client, "sess-mock-shape")
            assert tid.startswith("mock-")

            # 并发消费两条 SSE：业务流等 review_required、可视化流等 gate_blocked。
            (business, _), (graph_frames, _) = await asyncio.gather(
                _read_sse(
                    client,
                    f"/tasks/{tid}/stream",
                    _stop_on_review_count(1),
                    last_event_id=FROM_START,
                ),
                _read_sse(
                    client,
                    f"/graph_events?thread_id={tid}",
                    _stop_on_types({"gate_blocked"}),
                    last_event_id=FROM_START,
                ),
            )

            # 业务流事件类型集 ⊆ 真栈同形集合（逐字流在真 rewriter 下可能出现
            # content_delta；DEFAULT_SCENARIO 的 flush_chars 缺省 64、草稿短，
            # 出现与否均合规，断言其若出现则字段 snake_case 即可）。
            business_types = {f["event"] for f in business}
            allowed = {"status", "product", "content_delta",
                       "review_pack_ready", "review_required"}
            assert business_types <= allowed, (
                f"业务流事件类型超出允许集：{business_types - allowed}"
            )

            # 所有业务帧 data 递归键名皆 snake_case（与真栈 HTTP 契约同形）。
            for frame in business:
                _assert_snake_case(frame["data"])

            # 可视化帧 data 同样要求 snake_case（事件信封字段已定型）。
            for frame in graph_frames:
                _assert_snake_case(frame["data"])

            # 收到 review_required：路由元数据含 chapter_ids == ["ch1","ch2"]。
            review = next(f for f in business if f["event"] == "review_required")
            assert review["data"]["data"]["chapter_ids"] == ["ch1", "ch2"]

            # 收到 product 事件：kind 落在四类产物分支之一。
            product_kinds = {
                f["data"]["data"]["kind"]
                for f in business
                if f["event"] == "product"
            }
            assert product_kinds <= {
                "outline_ready", "materials_ready",
                "chapter_ready", "review_pack_ready",
            }
            assert product_kinds, "mock 任务到门须至少发一类 product 事件"

            # 状态 REST：停审阅门、带 mock 标记。
            status = (await client.get(f"/tasks/{tid}")).json()
            assert status["status"] == "AWAIT_USER_REVIEW"
            assert status["awaiting_review"] is True
            assert status["mock"] is True

            # 审阅包 REST：2 章、正文含角标、篇级 warn 非空、带 mock 标记。
            pack = (await client.get(f"/tasks/{tid}/review")).json()
            assert [c["chapter_id"] for c in pack["chapters"]] == ["ch1", "ch2"]
            for chapter in pack["chapters"]:
                _assert_text_has_material_marker(chapter["text"])
            assert pack["review_warnings"], "篇级终审 warn 须非空"
            assert pack["mock"] is True

    asyncio.run(main())


def test_mock档_thread_id带前缀且状态响应带mock标记():
    """mock 与真任务 thread_id 前缀区分；状态/产物/审阅包响应的 mock 字段
    按前缀正确回填 True/False。"""

    async def main() -> None:
        app = _make_app([*FIRST_PASS_RESPONSES, *REVISE_ROUND_RESPONSES])
        async with _client(app) as client:
            mock_tid, _ = await _create_mock_task(client, "sess-prefix-mock")
            real_tid, _ = await _create_task(client, "sess-prefix-real")

            assert mock_tid.startswith("mock-")
            assert not real_tid.startswith("mock-")

            # mock 任务到门（秒回）。
            await _read_sse(
                client,
                f"/tasks/{mock_tid}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )

            mock_status = (await client.get(f"/tasks/{mock_tid}")).json()
            assert mock_status["mock"] is True

            real_status = (await client.get(f"/tasks/{real_tid}")).json()
            assert real_status["mock"] is False

            # 产物快照与审阅包的 mock 字段按前缀路由回填。
            mock_products = (await client.get(f"/tasks/{mock_tid}/products")).json()
            assert mock_products["mock"] is True

            mock_pack = (await client.get(f"/tasks/{mock_tid}/review")).json()
            assert mock_pack["mock"] is True

    asyncio.run(main())


def test_mock档与真实档共享checkpointer_重启后可查可回滚可续跑():
    """mock 任务与真任务共享同一 checkpointer：进程重启（TaskManager 内存
    登记丢失）后按检查点自动重建；rollback 返回 202 并重新中断在审阅门；
    resume 补发 review_required（停在原中断点不重跑图）。"""

    async def main() -> None:
        saver = InMemorySaver(serde=checkpoint_serializer())
        # app1：跑一个 mock 任务到门后丢弃，模拟进程死亡。
        app1 = _make_app([], checkpointer=saver)
        async with _client(app1) as client:
            tid, _ = await _create_mock_task(client, "sess-restart-1")
            await _read_sse(
                client,
                f"/tasks/{tid}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )

        # app2：同 saver 重启；登记按检查点自动重建。
        app2 = _make_app([], checkpointer=saver)
        async with _client(app2) as client:
            # 重启后状态可查、mock 标记保留。
            status = (await client.get(f"/tasks/{tid}")).json()
            assert status["status"] == "AWAIT_USER_REVIEW"
            assert status["awaiting_review"] is True
            assert status["mock"] is True

            # 检查点清单可列：找到停门的历史检查点。
            checkpoints = (await client.get(f"/tasks/{tid}/checkpoints")).json()
            target = next(
                c for c in checkpoints
                if "human_review_gate" in c["next"]
                and c["status"] == "AWAIT_USER_REVIEW"
            )

            # rollback 到该检查点 → 202；重放后重新中断在审阅门。
            rollback = await client.post(
                f"/tasks/{tid}/rollback",
                json={"checkpoint_id": target["checkpoint_id"]},
            )
            assert rollback.status_code == 202
            # 等待 rollback 重放产生的 review_required（兜底应答仍走完状态机到门）；
            # 超时即视作回滚后直接停在门，校验状态即可。
            try:
                await _read_sse(
                    client,
                    f"/tasks/{tid}/stream",
                    _stop_on_review_count(1),
                    last_event_id=FROM_START,
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                pass
            post_rollback_status = (await client.get(f"/tasks/{tid}")).json()
            assert post_rollback_status["status"] == "AWAIT_USER_REVIEW"
            assert post_rollback_status["mock"] is True

            # resume 补发：停在原中断点不重跑图，状态仍 AWAIT_USER_REVIEW。
            resume = await client.post(
                f"/tasks/{tid}/resume", json={"session_id": "sess-restart-2"}
            )
            assert resume.status_code == 200
            # resume 后业务流再收到 review_required（停门补发）。
            await _read_sse(
                client,
                f"/tasks/{tid}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            final_status = (await client.get(f"/tasks/{tid}")).json()
            assert final_status["status"] == "AWAIT_USER_REVIEW"
            assert final_status["mock"] is True

    asyncio.run(main())


def test_mock档场景库_多章大纲角标正文篇级warn与退化重试attempt递增():
    """场景库双分支覆盖：(a) DEFAULT_SCENARIO 多章大纲/角标正文/篇级 warn；
    (b) DEGRADATION_SCENARIO + 小 flush 阈值触发 ch1 退化重试 attempt 1→2、
    sequence 复位。"""

    async def main() -> None:
        # (a) 默认场景：不开 env 覆盖。
        app = _make_app([], mock_scenario=DEFAULT_SCENARIO)
        async with _client(app) as client:
            tid, _ = await _create_mock_task(client, "sess-scen-default")
            await _read_sse(
                client,
                f"/tasks/{tid}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            pack = (await client.get(f"/tasks/{tid}/review")).json()
            assert [c["chapter_id"] for c in pack["chapters"]] == ["ch1", "ch2"]
            for chapter in pack["chapters"]:
                _assert_text_has_material_marker(chapter["text"])
            assert pack["review_warnings"], "篇级 transition warn 须非空"
            assert any(
                "章间衔接" in w for w in pack["review_warnings"]
            ), "篇级 warn 须含章间衔接"

        # (b) 退化场景：env 小阈值 + DEGRADATION_SCENARIO（ch1 键控序列已覆盖
        # [malformed, valid draft, valid revise]）。
        with _EnvOverride(WRITER_DELTA_FLUSH_CHARS="4", WRITER_DELTA_FLUSH_MS="0"):
            app = _make_app([], mock_scenario=DEGRADATION_SCENARIO)
            async with _client(app) as client:
                tid, _ = await _create_mock_task(client, "sess-scen-degrade")
                business, _ = await _read_sse(
                    client,
                    f"/tasks/{tid}/stream",
                    _stop_on_review_count(1),
                    last_event_id=FROM_START,
                )
                deltas = _content_delta_payloads(business)
                ch1_draft = [
                    p for p in deltas
                    if p["chapter_id"] == "ch1" and p["mode"] == "draft"
                ]
                attempts = {p["attempt"] for p in ch1_draft}
                assert {1, 2} <= attempts, (
                    f"ch1 draft 退化重试 attempt 须见 1 与 2，实际 {attempts}"
                )
                attempt2_seq0 = [
                    p for p in ch1_draft if p["attempt"] == 2
                ][0]["sequence"] == 0
                assert attempt2_seq0, "新 attempt 的 sequence 须从 0 复位"

    asyncio.run(main())


def test_mock档清理策略_按前缀清mock线程():
    """``purge_mock_threads`` 按 ``mock-`` 前缀清理检查点线程：清后该任务
    GET 返回 404（检查点已删、_ensure_entry 重建时无检查点 → TaskNotFound）。"""

    async def main() -> None:
        saver = InMemorySaver(serde=checkpoint_serializer())
        app = _make_app([], checkpointer=saver, mock_scenario=DEFAULT_SCENARIO)
        async with _client(app) as client:
            tid, _ = await _create_mock_task(client, "sess-purge")
            await _read_sse(
                client,
                f"/tasks/{tid}/stream",
                _stop_on_review_count(1),
                last_event_id=FROM_START,
            )
            # lifespan 启动后 manager 落在 app.state.manager。
            manager = app.state.manager
            n = manager.purge_mock_threads()
            assert n >= 1, "至少清理掉本用例创建的 mock 任务"
            # 清理后 GET 状态返回 404（检查点已删）。
            resp = await client.get(f"/tasks/{tid}")
            assert resp.status_code == 404

    asyncio.run(main())
