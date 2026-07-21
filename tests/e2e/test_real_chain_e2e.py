"""issue #39 门控真实链路 E2E：真实 LLM + 真实三通道检索的收口验收。

按环境变量门控（参考源项目 gated real-LLM E2E 模式）：
显式设 HYPOARGUS_REAL_E2E=1 且 LLM 与检索通道凭据齐备才运行，
否则整个模块跳过——离线全量测试不触网、不花费真实调用成本。

两发验收（均起真实 uvicorn + httpx 驱动，存档器注入 InMemorySaver，
其余全部走生产装配路径）：
1. 独立接口 POST /retrieval 一发：素材逐条回链假说、联网来源带真实链接、
   响应携 diagnostics、进度事件按 session_id 走 /graph_events 且密度不低于假说数。
2. 主流程一发：真实检索素材支撑正文角标、参考文献按 source_kind 输出正确的
   GB/T 7714 类型标识且联网条目带真实链接；混合修订意见触发"补充佐证"型
   增量检索（携既有引文库摘要），最终书目按 URL 判重无重复素材；
   每次检索调用的进度事件数量不少于该章假说数（防事件悄悄变稀）。

真实调用天然非确定性，断言只锚定稳定的外部行为
（回链、链接、类型标识、事件配对与密度），不锚定素材内容。
"""

import asyncio
import json
import os
import re
from typing import Any

import httpx
import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

# 只在显式开启门控时读 .env：离线全量测试不因收集本模块把真实凭据注入环境。
if os.environ.get("HYPOARGUS_REAL_E2E") == "1":
    load_dotenv()

# 凭据齐备的判定只取各通道的启用必需项（Bisheng token 在部分部署下为空，
# 不作为门槛）；结构化通道按环境自身配置决定是否启用，不进门槛。
_REQUIRED_ENV = (
    "LLM_API_KEY",
    "VOLCANO_SEARCH_API_KEY",
    "BISHENG_BASE_URL",
)
_MISSING_ENV = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
_OPTED_IN = os.environ.get("HYPOARGUS_REAL_E2E") == "1"

pytestmark = [
    pytest.mark.real_e2e,
    pytest.mark.skipif(
        not _OPTED_IN or bool(_MISSING_ENV),
        reason=(
            "门控真实链路 E2E：需 HYPOARGUS_REAL_E2E=1 显式开启"
            + (f"，且补齐凭据 {_MISSING_ENV}" if _MISSING_ENV else "")
        ),
    ),
]

from service.app import create_app  # noqa: E402
from graph import checkpoint_serializer  # noqa: E402
from scripts.demo import load_baseline_task  # noqa: E402
from tests.e2e.test_api_e2e import _client, _read_sse, _stop_on_review_count  # noqa: E402

# 独立检索一发的上限：多通道检索 + 逐项裁决，给足真实调用余量。
RETRIEVAL_TIMEOUT = 1200.0
# 主流程一发的上限：真实基准输入全流程（首写 + 修订 + 终审）约 20-40 分钟。
TRUNK_TIMEOUT = 3600.0

SOURCE_KINDS = {"web", "knowledge_base", "structured_data"}
# 与 scripts/demo.py 一致的混合修订意见：前半纯改写、后半触发"补充佐证"。
MIXED_FEEDBACK = "引言口吻克制些；第二章补充行业数据佐证"


def _watch_graph_events(
    client: httpx.AsyncClient, url: str, envelopes: list[dict[str, Any]]
) -> "asyncio.Task[None]":
    """起后台任务持续收集 /graph_events 事件信封，配 _stop_watcher 收尾。"""

    async def watch() -> None:
        async with client.stream("GET", url) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    envelopes.append(json.loads(line[len("data: "):]))

    return asyncio.create_task(watch())


async def _stop_watcher(watcher: "asyncio.Task[None]") -> None:
    """取消并等待收集任务：解析异常不被静默吞掉，取消本身不视为失败。"""
    watcher.cancel()
    try:
        await watcher
    except (asyncio.CancelledError, httpx.HTTPError):
        pass


def _search_progress_by_call(envelopes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 subagent_start 事件分组 search_agent 的进度事件，附各调用的假说数。

    返回 {start_event_id: {"start": 信封, "steps": [进度信封...]}}；
    假说数取适配层 engine_call_start 步骤携带的 hypothesis_count。
    """
    calls: dict[str, dict[str, Any]] = {}
    for envelope in envelopes:
        if envelope["unit"] != "search_agent":
            continue
        if envelope["type"] == "subagent_start":
            calls[envelope["event_id"]] = {"start": envelope, "steps": []}
        elif envelope["type"] == "progress":
            call = calls.get(envelope.get("parent_id") or "")
            if call is not None:
                call["steps"].append(envelope)
    return calls


def _assert_progress_density(calls: dict[str, dict[str, Any]]) -> None:
    """进度事件密度冒烟：每次检索调用的进度事件数不少于该次假说数。

    engine_call_start / engine_call_end 是适配层的最低保证；密度下限锚定
    假说数量级，防引擎内部结构调整导致事件悄悄变稀。
    """
    assert calls, "未观察到任何 search_agent 检索调用"
    for call in calls.values():
        steps = [step["payload"]["step"] for step in call["steps"]]
        assert "engine_call_start" in steps and "engine_call_end" in steps
        start_step = next(
            step for step in call["steps"]
            if step["payload"]["step"] == "engine_call_start"
        )
        hypothesis_count = start_step["payload"]["hypothesis_count"]
        assert len(call["steps"]) >= hypothesis_count, (
            f"章节 {call['start']['payload'].get('chapter_id')} 的进度事件仅 "
            f"{len(call['steps'])} 条，低于假说数 {hypothesis_count}"
        )


def _bibliography_urls(markdown_entries: list[dict[str, Any]]) -> list[str]:
    """从 markdown 格式书目条目提取全部链接（`i. [来源](链接)`）。"""
    urls: list[str] = []
    for entry in markdown_entries:
        match = re.search(r"\((https?://[^)]+)\)", entry["text"])
        if match:
            urls.append(match.group(1))
    return urls


def test_独立检索接口_真实调用素材可溯源且携诊断与达标进度事件():
    async def main() -> None:
        app = create_app(
            checkpointer=InMemorySaver(serde=checkpoint_serializer())
        )
        async with _client(app, read_timeout=RETRIEVAL_TIMEOUT) as client:
            session_id = "sess-real-retrieval"
            # 用权威统计口径的强事实假说：真实检索的域名过滤与证据裁决严格，
            # 弱观点类假说可能一条有效引用都留不下（引擎按设计裁 INCONCLUSIVE）。
            hypotheses = [
                {
                    "id": "h1",
                    "text": "2024 年我国国内生产总值比上年增长 5% 左右。",
                    "refute_condition": "存在权威统计数据表明 2024 年我国国内生产总值增速显著低于 5%。",
                },
                {
                    "id": "h2",
                    "text": "截至 2024 年底我国累计建成 5G 基站超过 400 万个。",
                    "refute_condition": "",
                },
                {
                    "id": "h3",
                    "text": "2024 年我国数据资源生产总量保持增长。",
                    "refute_condition": "",
                },
            ]

            envelopes: list[dict[str, Any]] = []
            watcher = _watch_graph_events(
                client, f"/graph_events?session_id={session_id}", envelopes
            )
            await asyncio.sleep(0.5)  # 确保订阅先于检索启动，事件不漏头。
            try:
                response = await client.post(
                    "/retrieval",
                    json={
                        "chapter_id": "real-e2e-ch1",
                        "genre": "科技产业分析",
                        "hypotheses": hypotheses,
                        "session_id": session_id,
                    },
                    timeout=httpx.Timeout(10.0, read=RETRIEVAL_TIMEOUT),
                )
                await asyncio.sleep(0.5)  # 给事件通道留排空时间。
            finally:
                await _stop_watcher(watcher)

            assert response.status_code == 200, response.text
            body = response.json()

            # 素材逐条回链假说，字段全部落在契约值域内。
            materials = body["materials"]
            assert materials, "真实检索一发未返回任何素材"
            hypothesis_ids = {h["id"] for h in hypotheses}
            for material in materials:
                assert material["hypothesis_id"] in hypothesis_ids
                assert material["verdict"] in {"pass", "fail"}
                assert material["source_kind"] in SOURCE_KINDS
                assert material["source"].strip()
                if material["source_kind"] == "web":
                    assert material["url"] and material["url"].startswith("http")
            # 联网来源带真实链接（AC：素材带真实来源链接）。
            assert any(
                m["source_kind"] == "web" and m["url"] for m in materials
            ), "真实检索未产出任何带链接的联网来源素材"

            # 响应携诊断块，且含引擎诊断摘要的已知键。
            diagnostics = body["diagnostics"]
            assert diagnostics, "独立接口响应 diagnostics 为空"
            assert set(diagnostics) & {
                "total_elapsed_ms",
                "call_counts",
                "deadline_reached",
                "gap_retrieval",
                "judge_integrity",
            }, f"diagnostics 缺少已知诊断摘要键：{sorted(diagnostics)}"

            # 事件走全局 /graph_events：按 session 过滤可见、成对且密度达标。
            starts = [e for e in envelopes if e["type"] == "subagent_start"]
            ends = [e for e in envelopes if e["type"] == "subagent_end"]
            assert len(starts) == 1 and len(ends) == 1
            assert all(e["session_id"] == session_id for e in envelopes)
            calls = _search_progress_by_call(envelopes)
            _assert_progress_density(calls)
            assert len(next(iter(calls.values()))["steps"]) >= len(hypotheses)

    asyncio.run(main())


def test_主流程真实一发_角标可溯源修订增量检索不重且书目类型标识正确():
    async def main() -> None:
        app = create_app(
            checkpointer=InMemorySaver(serde=checkpoint_serializer())
        )
        async with _client(app, read_timeout=TRUNK_TIMEOUT) as client:
            task_payload = load_baseline_task()
            response = await client.post("/tasks", json=task_payload)
            assert response.status_code == 201
            thread_id = response.json()["thread_id"]

            envelopes: list[dict[str, Any]] = []
            watcher = _watch_graph_events(
                client, f"/graph_events?thread_id={thread_id}", envelopes
            )
            try:
                # 首跑到第一次人工中断点。
                business, _ = await _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                    timeout=TRUNK_TIMEOUT,
                )
                # 中断点上图已静止，短暂排空后可视化通道不再有首跑事件在途；
                # 检索启动/结束成对是排空充分的守卫（首跑检索先于成章完成）。
                await asyncio.sleep(1.0)
                first_pass_mark = len(envelopes)
                first_pass_starts = [
                    e for e in envelopes[:first_pass_mark]
                    if e["type"] == "subagent_start" and e["unit"] == "search_agent"
                ]
                first_pass_ends = [
                    e for e in envelopes[:first_pass_mark]
                    if e["type"] == "subagent_end" and e["unit"] == "search_agent"
                ]
                assert len(first_pass_starts) == len(first_pass_ends), (
                    "首跑检索事件未成对，排空不充分或调用未完成"
                )

                # 提交混合修订意见：后半句触发"补充佐证"型增量检索。
                response = await client.post(
                    f"/tasks/{thread_id}/review",
                    json={"action": "revise", "feedback": MIXED_FEEDBACK},
                )
                assert response.status_code == 202
                business, _ = await _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(2),
                    timeout=TRUNK_TIMEOUT,
                )
                second = [
                    f for f in business if f["event"] == "review_required"
                ][1]
                assert second["data"]["data"]["iteration_round"] == 1

                # 定稿。
                response = await client.post(
                    f"/tasks/{thread_id}/review", json={"action": "finalize"}
                )
                assert response.status_code == 202
                business, ended = await _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    lambda frames: False,
                    timeout=TRUNK_TIMEOUT,
                )
                assert ended
                finalized = next(
                    f for f in business if f["event"] == "finalized"
                )
                chapters = finalized["data"]["data"]["chapters"]
                assert chapters
                await asyncio.sleep(0.5)  # 给可视化通道留排空时间。
            finally:
                await _stop_watcher(watcher)

            # ── 检索调用形态：首跑逐章调用 + 修订轮增量检索，事件成对。──
            calls = _search_progress_by_call(envelopes)
            starts = [e for e in envelopes if (
                e["type"] == "subagent_start" and e["unit"] == "search_agent"
            )]
            ends = [e for e in envelopes if (
                e["type"] == "subagent_end" and e["unit"] == "search_agent"
            )]
            assert len(starts) == len(ends)
            # 首跑逐章检索的稳定契约：每章至多一次首跑调用（不重复检索）；
            # 真实 LLM 下某章可零假说而按设计跳过检索，故不锚定调用数等于
            # 章节数，改锚定"带角标的章节必有对应检索调用"（角标只能来自
            # 落库素材，素材只能来自检索）。
            first_pass_chapter_ids = [
                e["payload"]["chapter_id"] for e in first_pass_starts
            ]
            assert len(first_pass_chapter_ids) == len(set(first_pass_chapter_ids)), (
                f"首跑存在同章重复检索调用：{first_pass_chapter_ids}"
            )
            retrieved_chapter_ids = {e["payload"]["chapter_id"] for e in starts}
            cited_chapter_ids = {
                chapter["chapter_id"]
                for chapter in chapters
                if re.search(r"\[m-[A-Za-z0-9_\-]+-cit-[A-Za-z0-9]+\]", chapter["text"])
            }
            assert cited_chapter_ids, "定稿正文无任何素材角标"
            assert cited_chapter_ids <= retrieved_chapter_ids, (
                f"带角标章节 {sorted(cited_chapter_ids - retrieved_chapter_ids)} "
                "未观察到对应检索调用"
            )
            # 修订轮出现增量检索（"补充佐证"分支真实触发）。
            revision_starts = [
                e for e in envelopes[first_pass_mark:]
                if e["type"] == "subagent_start" and e["unit"] == "search_agent"
            ]
            assert revision_starts, "修订轮未观察到增量检索调用"
            # 进度事件密度冒烟：每次调用不少于该章假说数。
            _assert_progress_density(calls)

            # ── 书目：角标重编号可溯源，类型标识按通道正确，链接真实。──
            rendered = (
                await client.get(f"/tasks/{thread_id}/bibliography?format=gbt7714")
            ).json()
            entries = rendered["bibliography"]
            assert entries, "定稿书目为空：正文角标背后无落库素材"
            joined = " ".join(c["text"] for c in rendered["chapters"])
            assert "[1]" in joined, "正文角标未重编号，书目不可溯源"
            for entry in entries:
                assert re.match(
                    r"^\[\d+\] .+\[(EB/OL|DB/OL|DS)\]\.", entry["text"]
                ), f"书目条目类型标识不合 GB/T 7714 约定：{entry['text']}"
            web_entries = [e["text"] for e in entries if "[EB/OL]" in e["text"]]
            assert web_entries, "书目无联网来源条目"
            assert all("http" in text for text in web_entries), (
                "联网来源条目缺真实链接"
            )

            # ── 修订增量检索不检回重复素材：最终书目按 URL 判重无重复。──
            markdown = (
                await client.get(f"/tasks/{thread_id}/bibliography?format=markdown")
            ).json()
            urls = _bibliography_urls(markdown["bibliography"])
            assert len(urls) == len(set(urls)), (
                f"书目存在重复 URL：{[u for u in urls if urls.count(u) > 1]}"
            )

    asyncio.run(main())
