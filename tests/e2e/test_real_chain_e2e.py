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
from collections import Counter
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

from domain.env_config import read_positive_int  # noqa: E402
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


def _business_product_kinds(frames: list[dict[str, Any]]) -> list[str]:
    """业务流里的产物事件 kind 序列（按到达顺序）。"""
    return [
        f["data"]["data"]["kind"]
        for f in frames
        if f.get("event") == "product"
    ]


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

            # 响应携诊断块，且含引擎诊断摘要的已知键。
            materials = body["materials"]
            diagnostics = body["diagnostics"]
            assert diagnostics, "独立接口响应 diagnostics 为空"
            assert set(diagnostics) & {
                "total_elapsed_ms",
                "call_counts",
                "deadline_reached",
                "gap_retrieval",
                "judge_integrity",
            }, f"diagnostics 缺少已知诊断摘要键：{sorted(diagnostics)}"

            # ── T1 检索漏斗放宽收口（issue #45/#50）──
            # AC：每章 pass 素材落库数达下限或有显式警告事件；下限计数排除 inconclusive。
            # 真实 LLM 下裁决器偶发限流可致本章零 pass——此时薄弱章警告须显式触发
            # （杠杆①不阻断不补检、单轮），"达标或警告"二者必居其一。
            pass_count = sum(1 for m in materials if m["verdict"] == "pass")
            weak_count = sum(1 for m in materials if m["verdict"] == "inconclusive")
            floor_warning = [
                e for e in envelopes
                if e["type"] == "progress"
                and e.get("unit") == "search_agent"
                and e["payload"].get("step") == "weak_chapter_warning"
            ]
            if "pass_below_threshold" in diagnostics:
                below = diagnostics["pass_below_threshold"]
                # 下限计数与素材实际 pass 数一致（排除 inconclusive 的硬约束）。
                assert below["pass_count"] == pass_count, (
                    f"下限计数未排除 inconclusive：pass_below_threshold="
                    f"{below['pass_count']} 实际 pass={pass_count}"
                )
                assert below["threshold"] >= 1
                assert below["pass_count"] < below["threshold"]
                assert floor_warning, "pass 低于下限但未发薄弱章警告事件"
            else:
                # 达下限：pass 必有产出（命中率无回归）。
                assert pass_count >= 1, "真实检索未产出任何 pass 素材（命中率回归）"
            # 弱佐证计数一致性（杠杆②）：摘要计数与素材 inconclusive 数一致。
            if "weak_evidence_count" in diagnostics:
                assert diagnostics["weak_evidence_count"] == weak_count
                assert weak_count >= 1
            # 裁决通过率无回归：裁决器不丢候选（judge_missing_candidate_count==0）。
            judge_integrity = diagnostics.get("judge_integrity", {})
            assert judge_integrity.get("judge_missing_candidate_count", 0) == 0, (
                f"裁决完整性回归：{judge_integrity}"
            )

            # 素材契约（逐条回链假说、字段值域、联网来源带真实链接）：
            # 裁决器未限流产素材时验；限流致零素材时上方薄弱章警告已收口。
            if materials:
                hypothesis_ids = {h["id"] for h in hypotheses}
                for material in materials:
                    assert material["hypothesis_id"] in hypothesis_ids
                    # T1 杠杆②放行后 inconclusive 弱佐证进可引池（CITABLE_VERDICTS），
                    # 真实检索的 SUPPLEMENT 关系映射为 inconclusive，故值域含三分支。
                    assert material["verdict"] in {"pass", "fail", "inconclusive"}
                    assert material["source_kind"] in SOURCE_KINDS
                    assert material["source"].strip()
                    if material["source_kind"] == "web":
                        assert material["url"] and material["url"].startswith("http")
                # 联网来源带真实链接（AC：素材带真实来源链接）。
                assert any(
                    m["source_kind"] == "web" and m["url"] for m in materials
                ), "真实检索未产出任何带链接的联网来源素材"

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
                second_data = second["data"]["data"]
                if "pending_confirmation" in second_data:
                    # 真实解析非确定：意见若被判定触及超过大纲一半章节，
                    # 会先携解析清单重新中断（issue #49 大扇出确认），确认后继续。
                    response = await client.post(
                        f"/tasks/{thread_id}/review", json={"action": "confirm"}
                    )
                    assert response.status_code == 202
                    business, _ = await _read_sse(
                        client,
                        f"/tasks/{thread_id}/stream",
                        _stop_on_review_count(3),
                        timeout=TRUNK_TIMEOUT,
                    )
                    second_data = [
                        f for f in business if f["event"] == "review_required"
                    ][2]["data"]["data"]
                assert second_data["iteration_round"] == 1

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

            # ── 写作侧每章调用次数上限（issue #50）：write + review + 至多一次
            #    rewrite、章级无终态复审；修订轮无二次重写叠加。──
            # subagent 启动事件按 (unit, chapter_id, mode) 切分首跑 / 修订轮，
            # 以首跑检索扇出完成位点 first_pass_mark 为界。
            def _starts_by_chapter(envelopes_slice, unit, mode):
                ids = [
                    e["payload"]["chapter_id"]
                    for e in envelopes_slice
                    if e["type"] == "subagent_start"
                    and e.get("unit") == unit
                    and e["payload"].get("mode") == mode
                ]
                return Counter(ids)

            first_pass = envelopes[:first_pass_mark]
            revision = envelopes[first_pass_mark:]
            # 首跑每章：draft ≤1、review ≤1（max_rewrites=1 + 单次章级评审）。
            for unit, mode in (
                ("rewriter_loop", "draft"),
                ("chapter_reviewer", "review"),
            ):
                counts = _starts_by_chapter(first_pass, unit, mode)
                for ch, n in counts.items():
                    assert n <= 1, (
                        f"首跑章 {ch} 的 {unit}/{mode} 启动 {n} 次，超过上限 1"
                    )
            # revise 全章合计上限放宽至 1 + DOCUMENT_REVIEW_MAX_RETRIES（缺省 2 → 3）：
            # rewriter_loop/revise 同模式可发自两条合法路径——chapter_drafter 章级
            # 重写（max_rewrites=1，至多 1 次）与 writing_orchestrator 篇级终审回退
            # 重写（ADR-0008，受 DOCUMENT_REVIEW_MAX_RETRIES 约束）。回退在首跑段
            # （终审未过→回退）与修订轮（修订后终审未过→回退）均可合法出现。
            # 并行首写下 subagent_start 的 parent_id 经 _current_node 兜底、按章
            # 归属不可靠（执行器线程发事件、驱动线程置 _current_node，见
            # graph_event_stream 文档），故不按来源切分、仅锚定全章合计上限。
            max_review_retries = read_positive_int(
                os.environ, "DOCUMENT_REVIEW_MAX_RETRIES", 2
            )
            revise_cap = 1 + max_review_retries
            for ch, n in _starts_by_chapter(
                first_pass, "rewriter_loop", "revise"
            ).items():
                assert n <= revise_cap, (
                    f"首跑章 {ch} 的 rewriter_loop/revise 启动 {n} 次，"
                    f"超过 1+DOCUMENT_REVIEW_MAX_RETRIES={revise_cap} 上限"
                )
            # 章级无终态复审：rewrite 之后再无同章 chapter_reviewer 启动——
            # 每章至多一次 review，且若有 revise，review 必先于 revise。
            # 篇级终审回退 revise 可命中空稿短路章（无素材→空稿→章级评审短路、
            # 无 chapter_reviewer/review 事件，writing_orchestrator 仍据终审报告
            # 回退改写），此时同章无 review，不强制 review 存在；有 review 者
            # 其 review 必先于该章任意 revise。
            review_starts_fp = [
                (e["payload"]["chapter_id"], idx)
                for idx, e in enumerate(first_pass)
                if e["type"] == "subagent_start"
                and e.get("unit") == "chapter_reviewer"
                and e["payload"].get("mode") == "review"
            ]
            revise_starts_fp = [
                (e["payload"]["chapter_id"], idx)
                for idx, e in enumerate(first_pass)
                if e["type"] == "subagent_start"
                and e.get("unit") == "rewriter_loop"
                and e["payload"].get("mode") == "revise"
            ]
            review_idx_fp = {ch: i for ch, i in review_starts_fp}
            for ch, ri in revise_starts_fp:
                if ch not in review_idx_fp:
                    continue
                assert review_idx_fp[ch] < ri, (
                    f"首跑章 {ch} 的 review 未先于 revise（终态复审顺序回归）"
                )
            # 修订轮：writing_orchestrator revise ≤1（修订指令单次改写，ADR-0007）
            # + 篇级终审回退 revise ≤ max_review_retries，合计 ≤ revise_cap。
            for ch, n in _starts_by_chapter(
                revision, "rewriter_loop", "revise"
            ).items():
                assert n <= revise_cap, (
                    f"修订轮章 {ch} 的 rewriter_loop/revise 启动 {n} 次，"
                    f"超过 1+DOCUMENT_REVIEW_MAX_RETRIES={revise_cap} 上限"
                )

            # ── 篇级终审（issue #48/#50）：error 打回 / 语义 warn 呈人工，端到端成立。──
            # 终审主节点必运行；其路由出口（error→回退重写 / 通过→人工）必有 branch_taken。
            assert any(
                e["type"] == "node_start" and e.get("unit") == "document_reviewer"
                for e in envelopes
            ), "篇级终审 document_reviewer 未运行"
            assert any(
                e["type"] == "branch_taken"
                and e["payload"].get("from") == "document_reviewer"
                for e in envelopes
            ), "篇级终审无路由流出（error 打回 / 通过收束未接通）"
            # warn 通道接通（ADR-0008）：每次人工中断点载荷必携 review_warnings 字段，
            # 语义 warn 呈人工不打回；真实 LLM 下可为空但字段必在。
            # 业务流 /stream 订阅即回放历史，finalize 流含全链路所有中断点。
            review_requireds = [
                f for f in business if f["event"] == "review_required"
            ]
            assert review_requireds, "全链路未出现人工中断点"
            for rr in review_requireds:
                assert "review_warnings" in rr["data"]["data"], (
                    "人工中断点载荷缺 review_warnings 字段（warn 通道未接通）"
                )

            # ── 检索下限警告一致性（issue #45/#50 全链路收口）──
            # 每章检索调用的 subagent_end 诊断：若 pass_below_threshold 触发，
            # 其 pass_count < threshold；弱佐证计数 ≥1。
            search_ends = [
                e for e in envelopes
                if e["type"] == "subagent_end" and e.get("unit") == "search_agent"
            ]
            for end in search_ends:
                diag = end["payload"].get("diagnostics") or {}
                if "pass_below_threshold" in diag:
                    below = diag["pass_below_threshold"]
                    assert below["pass_count"] < below["threshold"], (
                        f"薄弱章下限计数失真：{below}"
                    )
                if "weak_evidence_count" in diag:
                    assert diag["weak_evidence_count"] >= 1

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


def test_v2交付通道_真实链路产物流逐字流审阅门定稿与续传对账():
    """issue #62 收口（AC#2/#3）：真实链路上一单任务的 v2 交付通道事件序列。

    与主流程一发（五维机检、含修订轮增量检索）解耦：本发只跑首写到审阅门
    再定稿，不触修订轮——把 v2 交付通道（产物流→逐字流→审阅门→定稿）与
    续传/对账在真实链路上独立验收，避免修订轮增量检索的外部检索服务波动
    干扰 v2 契约的稳定结构断言。

    真实调用天然非确定性，断言只锚定稳定的外部结构：
    - AC#2 产物流按序整块上网线：outline_ready → materials_ready →
      chapter_ready → review_pack_ready，全部先于 review_required；
      逐字流 content_delta 帧（attempt/sequence 语义）在 draft 期间上网线；
      定稿 finalized 收尾。
    - AC#2 审阅门摘要与 REST 全文同源：review_pack_ready 摘要的 pack_version
      与 GET /tasks/{id}/review 全文一致。
    - AC#3 运行中产物快照对账：GET /tasks/{id}/products 只读检查点与审阅包同源。
    - AC#3 断线重连续传：携审阅门摘要帧 id 重连只补其后事件、不重复；
      携异世代 id 重连立即收 reconcile_required(epoch_mismatch)。
    """

    async def main() -> None:
        # 固定 epoch：便于以 ``{epoch}-0`` 从流首回放整段历史、以异世代 id 触发
        # reconcile_required。InMemorySaver 与生产 Postgres 存档器同构（resume/
        # rollback 照常），真实链路验收只验 v2 交付通道、不验存档器介质。
        app = create_app(
            checkpointer=InMemorySaver(serde=checkpoint_serializer()),
            epoch="ep-real-v2",
        )
        async with _client(app, read_timeout=TRUNK_TIMEOUT) as client:
            task_payload = load_baseline_task()
            response = await client.post("/tasks", json=task_payload)
            assert response.status_code == 201
            thread_id = response.json()["thread_id"]

            # 首跑到第一次人工中断点（首写 + 篇级终审首跑过后）。
            business, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                _stop_on_review_count(1),
                timeout=TRUNK_TIMEOUT,
            )

            # ── AC#3 运行中产物快照对账：GET /products 只读检查点，与审阅包同源。──
            products = (await client.get(f"/tasks/{thread_id}/products")).json()
            assert products["status"] == "AWAIT_USER_REVIEW"
            assert products["chapters"], "运行中产物快照无章正文"

            # ── AC#2 审阅门摘要 + REST 全文同源：review_pack_ready 摘要的
            #    pack_version 与 GET /review 全文 pack_version 一致。──
            pack_event = next(
                f for f in business
                if f.get("event") == "product"
                and f["data"]["data"]["kind"] == "review_pack_ready"
            )
            pack_summary = pack_event["data"]["data"]
            review_pack = (await client.get(f"/tasks/{thread_id}/review")).json()
            assert review_pack["pack_version"] == pack_summary["pack_version"]
            assert review_pack["chapters"][0]["text"]  # 全文只在 REST
            assert pack_summary["chapter_total"] == len(review_pack["outline"])

            # ── AC#3 断线重连续传：携审阅门摘要帧 id 重连，只补该 id 之后事件，
            #    不重复 review_pack_ready、只收其后 review_required。──
            resume_frames, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: any(
                    f["event"] == "review_required" for f in frames
                ),
                timeout=TRUNK_TIMEOUT,
                last_event_id=pack_event["id"],
            )
            resume_events = {f["event"] for f in resume_frames}
            assert "review_required" in resume_events, (
                "Last-Event-ID 续传未补到其后的 review_required"
            )
            assert not any(
                f.get("event") == "product"
                and f["data"]["data"]["kind"] == "review_pack_ready"
                for f in resume_frames
            ), "Last-Event-ID 续传重复回放了已收的 review_pack_ready"

            # ── AC#3 reconcile_required：携异世代 Last-Event-ID 重连，立即收失配
            #    控制事件后转实时（停审阅门，枢纽未关、转实时无新事件故按 stop 收。）。──
            rec_frames, _ = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: any(
                    f["event"] == "reconcile_required" for f in frames
                ),
                timeout=TRUNK_TIMEOUT,
                last_event_id="ep-foreign-0",
            )
            rec = next(
                f for f in rec_frames if f["event"] == "reconcile_required"
            )
            assert rec["data"]["reason"] == "epoch_mismatch"
            assert rec["data"]["last_event_id"] == "ep-foreign-0"

            # ── 定稿（不调 LLM，直接收束到 FINISHED）。──
            response = await client.post(
                f"/tasks/{thread_id}/review", json={"action": "finalize"}
            )
            assert response.status_code == 202
            await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                timeout=TRUNK_TIMEOUT,
            )

            # ── AC#2 真相源回放：定稿后业务枢纽关闭，携 ``{epoch}-0`` 重连从流首
            #    回放整段历史后自然收尾。历史缓冲 max_history=10000 保留全段、
            #    不经每订阅者队列两级丢弃——稳拿完整 v2 事件序列。──
            full_frames, full_ended = await _read_sse(
                client,
                f"/tasks/{thread_id}/stream",
                lambda frames: False,
                timeout=TRUNK_TIMEOUT,
                last_event_id="ep-real-v2-0",
            )
            assert full_ended, "定稿后回放流未自然收尾（业务枢纽未关）"

            # 产物流按序整块上网线：outline → materials → chapter → review_pack。
            kinds = _business_product_kinds(full_frames)
            assert kinds.count("outline_ready") == 1, "缺 outline_ready 产物事件"
            assert "materials_ready" in kinds, "缺 materials_ready 产物事件"
            assert "chapter_ready" in kinds, "缺 chapter_ready 产物事件"
            assert "review_pack_ready" in kinds, "缺 review_pack_ready 产物事件"
            assert kinds.index("outline_ready") < kinds.index("materials_ready")
            assert kinds.index("materials_ready") < kinds.index("chapter_ready")
            assert kinds.index("chapter_ready") < kinds.index("review_pack_ready")

            # 逐字流 content_delta：真实 LLM stream 路径在业务通道上网线，
            # 帧载荷字段齐全、attempt/sequence 类型正确（attempt≥1、sequence≥0）。
            # mode 含 draft 与 revise：篇级终审首轮不合格会触发回退重写
            # （ADR-0008），rewriter_loop 以 mode=revise 重写本章，亦上网线。
            deltas = [
                f["data"]["data"]
                for f in full_frames
                if f.get("event") == "content_delta"
            ]
            assert deltas, "真实链路 draft 期间未发 content_delta 事件"
            for payload in deltas:
                assert payload["chapter_id"]
                assert payload["mode"] in {"draft", "revise"}
                assert payload["kind"] in {"content", "thinking"}
                assert isinstance(payload["delta"], str) and payload["delta"]
                assert isinstance(payload["attempt"], int) and payload["attempt"] >= 1
                assert isinstance(payload["sequence"], int) and payload["sequence"] >= 0

            # 全部产物事件先于首个人工中断点 review_required；finalized 收尾。
            full_events = [f["event"] for f in full_frames]
            first_review = full_events.index("review_required")
            product_positions = [
                i for i, e in enumerate(full_events) if e == "product"
            ]
            assert product_positions, "回放历史无任何产物事件"
            assert max(product_positions) < first_review, (
                "产物事件未全部先于首个人工中断点"
            )
            assert full_events[-1] == "finalized", "回放历史未以 finalized 收尾"

    asyncio.run(main())


def test_断点续跑_进程死亡后resume不重跑已完成章且可定稿():
    """issue #50 收口：真·LLM 下中断续跑语义不变。

    真实 LLM 天然非确定性，是 stub 测不出的回归暴露面：若中断续跑路径
    误把已完成章重新扇出，真模型会重算并产出漂移草稿。本发锚定稳定不变量——
    停在人工中断点（首写已完成、篇级终审首跑过后），进程死亡后 resume：

    - 不重跑图：可视化通道只补发 gate_blocked，无任何 subagent_start
      （已完成章零重复调用，真·LLM 下亦不因非确定性重算）；
    - 业务通道重发 review_required（携 review_warnings warn 通道）；
    - 续跑可定稿收束（FINISHED）。

    ADR-0001 约束 1（崩溃只损失进行中分支）的 stub 覆盖见 test_graph_e2e
    与 test_api_e2e；本发补真·LLM 下"已完成章零重复调用"的收口断言。
    """

    async def main() -> None:
        # 同一 InMemorySaver 跨两个 app 实例：app1 丢弃即模拟进程死亡，
        # app2 按检查点重建登记续跑（与生产 Postgres 存档器同构）。
        saver = InMemorySaver(serde=checkpoint_serializer())
        app1 = create_app(checkpointer=saver)
        async with _client(app1, read_timeout=TRUNK_TIMEOUT) as client:
            task_payload = load_baseline_task()
            response = await client.post("/tasks", json=task_payload)
            assert response.status_code == 201
            thread_id = response.json()["thread_id"]

            envelopes: list[dict[str, Any]] = []
            watcher = _watch_graph_events(
                client, f"/graph_events?thread_id={thread_id}", envelopes
            )
            try:
                # 首跑到第一个人工中断点（首写 + 篇级终审首跑过后）。
                await _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                    timeout=TRUNK_TIMEOUT,
                )
                await asyncio.sleep(1.0)  # 给可视化通道排空首跑在途事件。
            finally:
                await _stop_watcher(watcher)
            # 首跑确有真实 LLM 调用（subagent 启动过），否则续跑断言无意义。
            first_pass_starts = [
                e for e in envelopes
                if e["type"] == "subagent_start"
                and e.get("unit") in (
                    "rewriter_loop", "chapter_reviewer", "search_agent"
                )
            ]
            assert first_pass_starts, "首跑未观察到任何 subagent 调用"
        # app1 连同 TaskManager 内存登记一并丢弃，模拟进程死亡。

        app2 = create_app(checkpointer=saver)
        async with _client(app2, read_timeout=TRUNK_TIMEOUT) as client:
            resume_envelopes: list[dict[str, Any]] = []
            watcher = _watch_graph_events(
                client, f"/graph_events?thread_id={thread_id}", resume_envelopes
            )
            try:
                await asyncio.sleep(0.5)  # 确保订阅先于 resume，事件不漏头。
                response = await client.post(
                    f"/tasks/{thread_id}/resume",
                    json={"session_id": "sess-real-resume-2"},
                )
                assert response.status_code == 200
                assert response.json()["status"] == "AWAIT_USER_REVIEW"

                # 业务流重发 review_required（不重跑图）。
                business, _ = await _read_sse(
                    client,
                    f"/tasks/{thread_id}/stream",
                    _stop_on_review_count(1),
                    timeout=TRUNK_TIMEOUT,
                )
                await asyncio.sleep(1.0)  # 给可视化通道排空补发事件。
            finally:
                await _stop_watcher(watcher)

            # 中断续跑语义不变（核心断言）：resume 停在中断点不重跑图——
            # 可视化通道只补发 gate_blocked，无任何 subagent 启动；
            # 已完成章零重复调用，真·LLM 下亦不因非确定性重算。
            resume_starts = [
                e for e in resume_envelopes if e["type"] == "subagent_start"
            ]
            assert not resume_starts, (
                "resume 重跑了已完成章（subagent 启动）："
                f"{[(e.get('unit'), e['payload'].get('chapter_id')) for e in resume_starts]}"
            )
            assert any(
                e["type"] == "gate_blocked" for e in resume_envelopes
            ), "resume 未补发 gate_blocked 信封"
            # warn 通道随中断点重发（ADR-0008）。
            review = next(f for f in business if f["event"] == "review_required")
            assert "review_warnings" in review["data"]["data"]

            # 续跑可定稿收束：finalize 不调 LLM，直接收束到 FINISHED。
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
            assert any(f["event"] == "finalized" for f in business)
            status = (await client.get(f"/tasks/{thread_id}")).json()
            assert status["status"] == "FINISHED"

    asyncio.run(main())
