#!/usr/bin/env python
"""空转全流程演示脚本：起真实服务，驱动一遍完整写作闭环并渲染书目。

流程：创建任务 → 并发消费业务与 graph_event 双 SSE 流 → 提交混合两类
分支（纯改写 + 补充佐证）的修订意见 → 引文门禁 → 定稿 → 按两种书目
格式渲染最终交付。

缺省为空转模式：确定性假 LLM + 内存存档器 + 打桩子智能体，
不依赖任何外部设施，可离线复现。
加 --real 切换生产同构模式：真实 LLM 配置（.env 各单元变量）+
Postgres 存档器（HYPOARGUS_PG_DSN）+ Langfuse 上报（LANGFUSE_* 已配置时）。

用法：
    python scripts/demo.py           # 空转演示
    python scripts/demo.py --real    # 生产同构演示（需 .env 就绪）
"""

import argparse
import asyncio
import json
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

TIMEOUT = 300.0

MIXED_FEEDBACK = "引言口吻克制些；第二章补充行业数据佐证"


def _build_app(real: bool):
    """按模式构建应用：空转注入假 LLM 与内存存档器，--real 走生产路径。"""
    from app import create_app

    if real:
        return create_app()

    from langgraph.checkpoint.memory import InMemorySaver

    from llm_client import FakeLLM
    from tests.llm_response_plans import TRUNK_RESPONSES

    fake = FakeLLM(list(TRUNK_RESPONSES))
    return create_app(
        llm_factory=lambda unit: fake, checkpointer=InMemorySaver()
    )


async def _watch_graph_events(client: httpx.AsyncClient, thread_id: str) -> None:
    """持续打印 graph_event 可视化通道的事件信封摘要（由主流程取消收尾）。"""
    async with client.stream(
        "GET", f"/graph_events?thread_id={thread_id}"
    ) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            envelope = json.loads(line[len("data: ") :])
            print(f"  [graph_event] {envelope['type']:<16} unit={envelope['unit']}")


async def _consume_business(
    client: httpx.AsyncClient, thread_id: str, on_review, timeout: float = TIMEOUT
) -> dict | None:
    """消费业务流：打印事件，遇 review_required 交给回调，返回 finalized 载荷。"""

    async def _consume() -> dict | None:
        async with client.stream(
            "GET", f"/tasks/{thread_id}/stream"
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: ") :])
                data = event["data"]
                if event["type"] == "status":
                    print(
                        f"[业务] 状态 {data['status']}"
                        f"（节点 {data['node']}，第 {data['iteration_round']} 轮）"
                    )
                elif event["type"] == "review_required":
                    print(f"[业务] 到达人工中断点：章节 {data['chapter_ids']}")
                    await on_review(data)
                elif event["type"] == "finalized":
                    print("[业务] 已定稿")
                    return data
                elif event["type"] == "error":
                    raise RuntimeError(f"运行失败：{data['message']}")
        return None

    return await asyncio.wait_for(_consume(), timeout)


async def _drive(client: httpx.AsyncClient) -> None:
    """驱动一遍完整闭环并渲染书目。"""
    response = await client.post(
        "/tasks",
        json={
            "user_intent": "写一篇论证「结构化写作智能体的工程价值」的行业评论",
            "user_identity": "专业撰稿人",
            "session_id": "demo-session",
        },
    )
    response.raise_for_status()
    thread_id = response.json()["thread_id"]
    print(f"任务已创建：thread_id={thread_id}")

    watcher = asyncio.create_task(_watch_graph_events(client, thread_id))
    reviewed = False

    async def on_review(data: dict) -> None:
        nonlocal reviewed
        if data["citation_warnings"]:
            print(f"[业务] 未决引文警告：{data['citation_warnings']}")
        if not reviewed:
            reviewed = True
            print(f"[演示] 提交混合修订意见：{MIXED_FEEDBACK}")
            await client.post(
                f"/tasks/{thread_id}/review",
                json={"action": "revise", "feedback": MIXED_FEEDBACK},
            )
        else:
            print("[演示] 提交定稿")
            await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )

    try:
        finalized = await _consume_business(client, thread_id, on_review)
    finally:
        watcher.cancel()

    assert finalized is not None
    for chapter in finalized["chapters"]:
        print(f"\n===== 章节 {chapter['chapter_id']} =====\n{chapter['text']}")

    for fmt in ("gbt7714", "markdown"):
        response = await client.get(
            f"/tasks/{thread_id}/bibliography?format={fmt}"
        )
        response.raise_for_status()
        rendered = response.json()
        print(f"\n===== 书目（{fmt}）=====")
        for entry in rendered["bibliography"]:
            print(entry["text"])


async def _main(real: bool) -> None:
    app = _build_app(real)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="warning",
        timeout_graceful_shutdown=3,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        async with asyncio.timeout(30):
            while not server.started:
                await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        print(f"服务已就绪：http://127.0.0.1:{port}（{'生产同构' if real else '空转'}模式）")
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=httpx.Timeout(10.0, read=TIMEOUT),
        ) as client:
            await _drive(client)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    print("\n演示完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="生产同构模式：真实 LLM 配置 + Postgres 存档器 + Langfuse 上报",
    )
    asyncio.run(_main(parser.parse_args().real))
