"""search_agent 真实现的接口契约测试：与打桩契约测试互为镜像。

真编排（工厂 + 契约映射 + 信号量限流）+ 引擎运行时假实现（唯一新增
测试接缝），只断言外部行为：素材回链、verdict、url/source_kind、
事件成对、限流与失败语义。
"""

import asyncio
import threading
from typing import Any

import pytest

from agents.search_agent import (
    FakeSearchAgentRuntime,
    make_search_agent,
    make_stub_search_agent,
)
from tests.agents.test_search_agent import SEARCH_TASK


def test_真实现_素材字段合规且逐条回链假说() -> None:
    runtime = FakeSearchAgentRuntime()
    adapter = make_search_agent(runtime=runtime)
    result = asyncio.run(adapter.run(dict(SEARCH_TASK)))

    assert set(result.keys()) == {"materials"}
    materials = result["materials"]
    hypothesis_ids = {hypothesis["id"] for hypothesis in SEARCH_TASK["hypotheses"]}
    assert {material["hypothesis_id"] for material in materials} == hypothesis_ids
    for material in materials:
        assert set(material.keys()) == {
            "id",
            "hypothesis_id",
            "source",
            "url",
            "source_kind",
            "excerpt",
            "relevance_score",
            "verdict",
        }
        assert material["verdict"] in ("pass", "fail")
        assert material["source_kind"] in ("web", "knowledge_base", "structured_data")

    # 每条假说至少一条 pass 素材（正向线），非空反驳条件另有反向 fail 素材。
    pass_linked = {
        material["hypothesis_id"]
        for material in materials
        if material["verdict"] == "pass"
    }
    assert pass_linked == hypothesis_ids


def test_真实现与打桩同工厂签名且结果键一致() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def hook(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    adapter = make_search_agent(hook, runtime=FakeSearchAgentRuntime())
    stub = make_stub_search_agent(hook)
    assert adapter.unit == stub.unit == "search_agent"

    result = asyncio.run(adapter.run(dict(SEARCH_TASK)))
    assert set(result.keys()) == {"materials"}

    # 适配层事件首尾成对、进度事件夹在其间，全部携带章节上下文
    # （ADR-0001 约束 2 的载荷规矩）。
    event_types = [event_type for event_type, _ in events]
    assert event_types[0] == "subagent_start"
    assert event_types[-1] == "subagent_end"
    assert set(event_types[1:-1]) == {"subagent_progress"}
    assert len(event_types) > 2
    for _, payload in events:
        assert payload["unit"] == "search_agent"
        assert payload["chapter_id"] == SEARCH_TASK["chapter_id"]


def _run_and_collect_events(
    task: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """跑一次真实现适配器（假引擎运行时），返回（结果, 事件清单）。"""
    events: list[tuple[str, dict[str, Any]]] = []

    def hook(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    adapter = make_search_agent(hook, runtime=FakeSearchAgentRuntime())
    result = asyncio.run(adapter.run(task))
    return result, events


def test_进度事件密度_不少于假说数量() -> None:
    task = dict(SEARCH_TASK)
    _, events = _run_and_collect_events(task)

    progress = [payload for event_type, payload in events if event_type == "subagent_progress"]
    # 事件密度防退化：一次检索的进度事件数量不少于假说数量级
    # （引擎逐项事件 + 适配层 engine_call 首尾事件）。
    assert len(progress) >= len(task["hypotheses"])
    steps = [payload["step"] for payload in progress]
    # 适配层自身的最低进度保证：即使引擎内部事件全丢也有首尾两步。
    assert steps[0] == "engine_call_start"
    assert steps[-1] == "engine_call_end"
    # 引擎逐项进度事件经桥翻译进入同一钩子（step 去掉 progress. 前缀）。
    assert "task.start" in steps
    assert "verdict.done" in steps

    # 裁决完成事件附带 x/y 累计计数：分母为检索项总数（正向+非空反驳条件反向）。
    verdict_payloads = [payload for payload in progress if payload["step"] == "verdict.done"]
    item_total = verdict_payloads[0]["item_total"]
    assert item_total >= len(task["hypotheses"])
    assert [payload["done_count"] for payload in verdict_payloads] == list(
        range(1, len(verdict_payloads) + 1)
    )
    assert all(payload["item_total"] == item_total for payload in verdict_payloads)


def test_进度事件载荷只含元数据_不含正文全文() -> None:
    task = dict(SEARCH_TASK)
    _, events = _run_and_collect_events(task)

    hypothesis_texts = [hypothesis["text"] for hypothesis in task["hypotheses"]]
    for _event_type, payload in events:
        # 白名单键之外的正文类字段一律不透传。
        for banned in ("content", "text", "excerpt", "target_text", "paragraph_text"):
            assert banned not in payload
        # 载荷值里不得出现假说正文（进度事件只带 id / 计数 / 步骤）。
        for value in payload.values():
            if isinstance(value, str):
                for text in hypothesis_texts:
                    assert text not in value


def test_引擎非进度诊断事件_不翻译为宿主进度() -> None:
    _, events = _run_and_collect_events(dict(SEARCH_TASK))

    steps = [
        payload["step"]
        for event_type, payload in events
        if event_type == "subagent_progress"
    ]
    # 假运行时回放里混入的 parallel.finalize 等非 progress. 前缀事件被桥过滤。
    assert not any("finalize" in step for step in steps)


def test_结束事件携带诊断摘要_结果契约不含诊断() -> None:
    task = dict(SEARCH_TASK)
    result, events = _run_and_collect_events(task)

    assert set(result.keys()) == {"materials"}
    end_payloads = [
        payload for event_type, payload in events if event_type == "subagent_end"
    ]
    assert len(end_payloads) == 1
    diagnostics = end_payloads[0]["diagnostics"]
    # 摘要子集只放元数据：总耗时、外部调用计数、补漏与裁决完成度、截止命中。
    assert diagnostics["total_elapsed_ms"] >= 0
    assert diagnostics["deadline_reached"] is False
    assert isinstance(diagnostics["call_counts"], dict)
    assert diagnostics["gap_retrieval"]["triggered_count"] == 0
    assert diagnostics["judge_integrity"]["judge_missing_candidate_count"] == 0


def test_运行时假实现_记录载荷并模拟时延() -> None:
    runtime = FakeSearchAgentRuntime(latency_seconds=0.01)
    adapter = make_search_agent(runtime=runtime)

    asyncio.run(adapter.run(dict(SEARCH_TASK)))
    assert len(runtime.payloads) == 1
    assert runtime.payloads[0]["paragraph"]["paragraph_id"] == SEARCH_TASK["chapter_id"]


def test_运行时副作用异常向上抛_失败即整体失败() -> None:
    def explode(payload: dict[str, Any]) -> None:
        raise RuntimeError("故障注入：检索通道超时")

    adapter = make_search_agent(
        runtime=FakeSearchAgentRuntime(side_effect=explode)
    )
    with pytest.raises(RuntimeError, match="故障注入：检索通道超时"):
        asyncio.run(adapter.run(dict(SEARCH_TASK)))


class _并发探针运行时(FakeSearchAgentRuntime):
    """带并发水位记录的假运行时：检验信号量限流的真实并发上限。"""

    def __init__(self, latency_seconds: float) -> None:
        super().__init__(latency_seconds=latency_seconds)
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    async def retrieve(
        self, payload: dict[str, Any], **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            return await super().retrieve(payload, **kwargs)
        finally:
            with self._lock:
                self._active -= 1


def _run_chapters_concurrently(adapter: Any, count: int) -> None:
    """并发跑多章检索：与检索扇出同形（每分支独立任务包）。"""

    async def main() -> None:
        await asyncio.gather(
            *(
                adapter.run(dict(SEARCH_TASK, chapter_id=f"ch-{index}"))
                for index in range(count)
            )
        )

    asyncio.run(main())


def test_信号量限流_并发上限为注入阈值() -> None:
    runtime = _并发探针运行时(latency_seconds=0.05)
    adapter = make_search_agent(runtime=runtime, max_concurrent_calls=1)
    _run_chapters_concurrently(adapter, 3)
    assert runtime.max_active == 1


def test_信号量阈值经环境变量配置(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_AGENT_MAX_CONCURRENT_CALLS", "3")
    runtime = _并发探针运行时(latency_seconds=0.05)
    adapter = make_search_agent(runtime=runtime)
    _run_chapters_concurrently(adapter, 5)
    assert runtime.max_active > 1
    assert runtime.max_active <= 3


def test_信号量阈值非法值报错并指明变量名(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEARCH_AGENT_MAX_CONCURRENT_CALLS", "零")
    with pytest.raises(ValueError, match="SEARCH_AGENT_MAX_CONCURRENT_CALLS"):
        make_search_agent(runtime=FakeSearchAgentRuntime())
