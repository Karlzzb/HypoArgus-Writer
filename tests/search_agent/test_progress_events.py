"""引擎进度事件密度测试：离线全流程跑 ParallelSourcesFlow，收集 progress. 事件。

用空结果的假联网通道 + 无 LLM 依赖离线跑真实流程（candidate_passthrough），
断言逐任务进度事件密度与元数据边界；回调走 SafeTraceEmitter 的可调用分派，
与宿主适配层桥同一条通路。
"""

import asyncio
from typing import Any, cast

from search_agent.evidence_retrieval.claim_logic import AtomicClaim, ClaimLogicOperator
from search_agent.evidence_retrieval.config import EvidenceRetrievalConfig
from search_agent.evidence_retrieval.dependencies import EvidenceRetrievalDependencies
from search_agent.evidence_retrieval.evidence_judge import EvidenceJudge
from search_agent.evidence_retrieval.flows.parallel_sources_flow import (
    ParallelSourcesFlow,
)
from search_agent.evidence_retrieval.schemas import (
    LineType,
    RetrievalGoal,
    RetrievalTask,
)
from search_agent.evidence_retrieval.tracing import SafeTraceEmitter


class _EmptyWebSearch:
    """空结果联网检索假实现：证明进度事件不依赖任何候选产出。"""

    async def search(self, name: str, query: Any) -> list[Any]:
        return []


def _task(task_id: str, line_type: LineType) -> RetrievalTask:
    return RetrievalTask(
        task_id=task_id,
        request_id="chapter-ch-1",
        document_id="document-1",
        user_id="user-1",
        paragraph_id="ch-1",
        line_type=line_type,
        node_id=f"node-{task_id}",
        item_id=f"item-{task_id}",
        target_text="远程办公是否提高研发团队交付效率",
        paragraph_text="远程办公显著提高了研发团队交付效率。",
        retrieval_goal=RetrievalGoal.VERIFY_ORIGINAL,
        atomic_claims=[
            AtomicClaim(
                claim_id=f"claim-{task_id}",
                subject="远程办公",
                source_text_span="远程办公提高研发团队交付效率",
            )
        ],
        claim_logic_operator=ClaimLogicOperator.SINGLE,
    )


def _run_flow_and_collect() -> list[tuple[str, dict[str, Any]]]:
    """离线跑一次真实流程（一正一反两任务），返回收集到的全部引擎事件。"""
    events: list[tuple[str, dict[str, Any]]] = []

    def collect(event: str, payload: dict[str, Any]) -> None:
        events.append((event, payload))

    config = EvidenceRetrievalConfig(
        evidence_output_mode="candidate_passthrough",
        shadow_mode=False,
    )
    dependencies = EvidenceRetrievalDependencies(
        web_search=_EmptyWebSearch(),
        web_fetcher=object(),
        kb_client=object(),
        structured_client=object(),
        # candidate_passthrough 不触碰裁决器：占位对象一旦被调用立即暴露。
        judge=cast(EvidenceJudge, object()),
        batch_judge=object(),
    )
    trace = SafeTraceEmitter(config, [collect])
    flow = ParallelSourcesFlow(config, dependencies, trace)
    tasks = [_task("task-1", LineType.FORWARD), _task("task-2", LineType.REVERSE)]

    async def main() -> None:
        await flow.run(tasks, scenarios={})
        # emit_nowait 走队列异步分派，冲净后事件清单才完整。
        await trace.flush(3000)

    asyncio.run(main())
    return events


def test_逐任务进度事件_密度不少于任务数() -> None:
    events = _run_flow_and_collect()

    progress = [row for row in events if row[0].startswith("progress.")]
    # 事件密度防退化：一次检索的进度事件数量不少于检索项（任务）数量级。
    assert len(progress) >= 2

    by_event: dict[str, list[dict[str, Any]]] = {}
    for event, payload in progress:
        by_event.setdefault(event, []).append(payload)
    # 每个任务一条开始、一条检索完成、一条裁决完成。
    for name in ("progress.task.start", "progress.task.retrieved", "progress.verdict.done"):
        assert {payload["task_id"] for payload in by_event[name]} == {
            "task-1",
            "task-2",
        }
    for payload in by_event["progress.task.start"]:
        assert payload["line_type"] in ("forward", "reverse")
    for payload in by_event["progress.task.retrieved"]:
        assert payload["candidate_count"] == 0
    # 每任务恰好一条裁决完成事件（passthrough 与 judge 两分支互斥，不重复发射）。
    assert len(by_event["progress.verdict.done"]) == 2
    for payload in by_event["progress.verdict.done"]:
        assert payload["verdict"] == "INCONCLUSIVE"
    # 通道调用级事件：每任务四条（web / 公共知识库 / 指定知识库 / 结构化）。
    channels = {
        (payload["task_id"], payload["channel"])
        for payload in by_event["progress.channel.done"]
    }
    assert channels == {
        (task_id, channel)
        for task_id in ("task-1", "task-2")
        for channel in ("web", "public_kb", "selected_kb", "structured")
    }


def test_进度事件载荷经消毒_不含正文全文() -> None:
    events = _run_flow_and_collect()

    progress_payloads = [
        payload for event, payload in events if event.startswith("progress.")
    ]
    assert progress_payloads

    def walk(value: Any) -> None:
        """递归检查载荷全部层级的字符串值，不只顶层。"""
        if isinstance(value, str):
            assert "远程办公" not in value
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    for payload in progress_payloads:
        walk(payload)
