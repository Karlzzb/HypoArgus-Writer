"""search_agent 子智能体包：真实检索适配层（契约映射 + 引擎运行时边界）与打桩。

真实现见 ``agent``（工厂与信号量限流）、``mapping``（契约映射纯函数）、
``runtime``（引擎运行时边界：真实一次性调用封装与假实现测试接缝）；
打桩实现见 ``stub``（同包共存，供空转与测试显式注入）。
本包对外 re-export 工厂与接缝，导入路径保持 ``agents.search_agent`` 不变。
"""

from agents.search_agent.agent import (
    DEFAULT_MAX_CONCURRENT_CALLS,
    MAX_CONCURRENT_CALLS_ENV,
    make_search_agent,
)
from agents.search_agent.mapping import (
    ENGINE_DOCUMENT_ID,
    engine_payload_from_task,
    forward_item_id,
    reverse_item_id,
    search_result_from_engine_output,
    split_item_id,
)
from agents.search_agent.runtime import (
    EngineRuntime,
    FakeSearchAgentRuntime,
    SearchAgentRuntimeSeam,
    ambient_callbacks,
    fake_engine_output,
)
from agents.search_agent.stub import (
    UNIT,
    make_stub_search_agent,
    stub_search_agent_run,
)

__all__ = [
    "DEFAULT_MAX_CONCURRENT_CALLS",
    "ENGINE_DOCUMENT_ID",
    "EngineRuntime",
    "FakeSearchAgentRuntime",
    "MAX_CONCURRENT_CALLS_ENV",
    "SearchAgentRuntimeSeam",
    "UNIT",
    "ambient_callbacks",
    "engine_payload_from_task",
    "fake_engine_output",
    "forward_item_id",
    "make_search_agent",
    "make_stub_search_agent",
    "reverse_item_id",
    "search_result_from_engine_output",
    "split_item_id",
    "stub_search_agent_run",
]
