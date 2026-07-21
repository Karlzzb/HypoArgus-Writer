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

    # 适配层事件成对且携带章节上下文（ADR-0001 约束 2 的载荷规矩）。
    assert [event_type for event_type, _ in events] == [
        "subagent_start",
        "subagent_end",
    ]
    for _, payload in events:
        assert payload["unit"] == "search_agent"
        assert payload["chapter_id"] == SEARCH_TASK["chapter_id"]


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

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            return await super().retrieve(payload)
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
