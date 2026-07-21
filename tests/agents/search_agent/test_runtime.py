"""引擎运行时封装的配置过滤测试：检索图只能看到回调，不见父图存档配置。

父图 checkpointer 配置经 contextvar 泄漏进检索子图会引发 loop 亲和性问题
（源项目适配层的已知坑），EngineRuntime 在调用引擎前必须把 LangChain
运行配置收窄到只剩回调，调用后恢复原配置。
"""

import asyncio
from typing import Any

from langchain_core.runnables.config import var_child_runnable_config

from agents.search_agent import EngineRuntime

PARENT_CONFIG: dict[str, Any] = {
    "callbacks": ["宿主回调"],
    "configurable": {
        "thread_id": "父图线程",
        "checkpoint_ns": "父图命名空间",
        "__pregel_checkpointer": object(),
    },
    "metadata": {"langgraph_node": "reference_orchestrator"},
}


def _run_with_parent_config(runtime: EngineRuntime) -> dict[str, Any] | None:
    """在带父图配置的 contextvar 环境里调用运行时，返回调用后残留的环境配置。"""

    async def main() -> dict[str, Any] | None:
        token = var_child_runnable_config.set(dict(PARENT_CONFIG))
        try:
            await runtime.retrieve({"request_id": "chapter-ch1"})
            return var_child_runnable_config.get()
        finally:
            var_child_runnable_config.reset(token)

    return asyncio.run(main())


def test_调用检索图时环境配置只保留回调() -> None:
    seen: dict[str, Any] = {}

    async def probing_invoke(
        payload: dict[str, Any], callbacks: list[Any]
    ) -> dict[str, Any]:
        seen["ambient"] = var_child_runnable_config.get()
        seen["callbacks"] = callbacks
        seen["payload"] = payload
        return {"results": [], "citations": []}

    runtime = EngineRuntime(invoke=probing_invoke)
    after = _run_with_parent_config(runtime)

    # 引擎调用期间的环境配置只剩回调：thread_id / checkpoint_ns /
    # checkpointer 等父图 configurable 一律被过滤。
    assert seen["ambient"] == {"callbacks": ["宿主回调"]}
    assert seen["callbacks"] == ["宿主回调"]
    assert seen["payload"] == {"request_id": "chapter-ch1"}

    # 调用结束后恢复父图配置，不污染同一运行内的后续节点逻辑。
    assert after == PARENT_CONFIG


def test_引擎抛错时环境配置同样恢复() -> None:
    async def failing_invoke(
        payload: dict[str, Any], callbacks: list[Any]
    ) -> dict[str, Any]:
        raise RuntimeError("故障注入")

    runtime = EngineRuntime(invoke=failing_invoke)

    async def main() -> dict[str, Any] | None:
        token = var_child_runnable_config.set(dict(PARENT_CONFIG))
        try:
            try:
                await runtime.retrieve({"request_id": "chapter-ch1"})
            except RuntimeError:
                pass
            return var_child_runnable_config.get()
        finally:
            var_child_runnable_config.reset(token)

    assert asyncio.run(main()) == PARENT_CONFIG


def test_无环境配置时回调为空列表() -> None:
    seen: dict[str, Any] = {}

    async def probing_invoke(
        payload: dict[str, Any], callbacks: list[Any]
    ) -> dict[str, Any]:
        seen["ambient"] = var_child_runnable_config.get()
        seen["callbacks"] = callbacks
        return {"results": [], "citations": []}

    asyncio.run(EngineRuntime(invoke=probing_invoke).retrieve({}))
    # 显式空回调列表即宿主管理模式：引擎不自动挂载自己的 Langfuse handler。
    assert seen["callbacks"] == []
    assert seen["ambient"] == {"callbacks": []}
