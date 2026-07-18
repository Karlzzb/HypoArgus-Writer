"""Langfuse 可观测接入层：全链路 LLM 调用与运行单元 trace 上报。

启用条件：环境变量 LANGFUSE_PUBLIC_KEY 与 LANGFUSE_SECRET_KEY 均非空
（接口地址取 LANGFUSE_BASE_URL，自建实例必填）。
未启用时本模块所有入口都是直通实现：不建客户端、不发网络请求、
不改变任何行为，测试与本地开发无需 Langfuse 设施。

覆盖面对齐 PRD「全部 7 个运行单元」：
- 每次图运行一条 trace：task_service 在工作线程内用 run_span 包住整次
  stream 驱动，trace 关联 thread_id / session_id / execution_trace_id；
- 5 个主节点：graph 构图时用 traced_node 包装节点函数，成为 trace 下的 span；
- 2 个子智能体：适配层调用被 wrap_subagent 包住，发 subagent 级 span；
- LLM 调用本身：llm_client 在启用时改用 langfuse.openai 官方插桩客户端，
  每次调用自动上报 generation（输入输出、token 用量、耗时、成本），
  并嵌套在所属节点 span 之下，由此定位到运行单元。

span 上下文经线程内 contextvars 传递：一次图运行独占一个工作线程，
节点内 asyncio.run 复制当前上下文，父子关系天然成立。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langfuse import Langfuse
    from openai import OpenAI

    from agents.contracts import Subagent

RUN_SPAN_NAME = "writing_task_run"

_client: "Langfuse | None" = None
_client_override: "Langfuse | None" = None


def langfuse_enabled() -> bool:
    """是否启用 Langfuse 上报：公私钥环境变量齐备即启用。"""
    if _client_override is not None:
        return True
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    )


def use_client(client: "Langfuse | None") -> None:
    """注入 Langfuse 客户端（测试接缝）；传 None 恢复按环境变量决定。"""
    global _client_override
    _client_override = client


def _get_client() -> "Langfuse | None":
    """惰性单例：未启用返回 None，启用后按环境变量创建一次。"""
    if _client_override is not None:
        return _client_override
    if not langfuse_enabled():
        return None
    global _client
    if _client is None:
        from langfuse import Langfuse

        timeout_raw = os.environ.get("LANGFUSE_TIMEOUT", "").strip()
        _client = Langfuse(timeout=int(timeout_raw) if timeout_raw else None)
    return _client


def openai_client_class() -> "type[OpenAI]":
    """LLM 封装层使用的 OpenAI 客户端类：启用时换成 Langfuse 官方插桩版。

    插桩版是 openai.OpenAI 的子类，每次 chat.completions 调用自动上报
    generation（输入输出、token 用量、耗时、成本）。
    本项目 LLM 封装直连 OpenAI 兼容接口而非 langchain Runnable，
    官方 LangChain 回调处理器捕捉不到这些调用，故采用官方插桩客户端。
    """
    if langfuse_enabled():
        from langfuse.openai import OpenAI as LangfuseOpenAI

        return LangfuseOpenAI
    from openai import OpenAI

    return OpenAI


@contextmanager
def run_span(
    *, thread_id: str, session_id: str, trace_id: str
) -> Iterator[None]:
    """包住一次图运行的根 span，并把关联标识传播到 trace 与全部子观测。"""
    client = _get_client()
    if client is None:
        yield
        return
    from langfuse import propagate_attributes

    try:
        with client.start_as_current_observation(
            name=RUN_SPAN_NAME, as_type="span"
        ):
            with propagate_attributes(
                trace_name=RUN_SPAN_NAME,
                session_id=session_id or None,
                metadata={
                    "thread_id": thread_id,
                    "execution_trace_id": trace_id,
                },
            ):
                yield
    finally:
        # 每次运行结束强制送出，服务常驻也能及时在 Langfuse 查到。
        client.flush()


@contextmanager
def _unit_span(name: str) -> Iterator[None]:
    """运行单元级 span；人工中断不算失败，先干净收 span 再重抛。"""
    client = _get_client()
    if client is None:
        yield
        return
    from langgraph.errors import GraphInterrupt

    interrupt: BaseException | None = None
    with client.start_as_current_observation(name=name, as_type="span") as span:
        try:
            yield
        except GraphInterrupt as exc:
            span.update(metadata={"gate_blocked": True})
            interrupt = exc
    if interrupt is not None:
        raise interrupt


def traced_node(unit: str, fn: Any) -> Any:
    """把主节点函数包进单元 span；未启用时原样返回，零包装开销。"""
    if not langfuse_enabled():
        return fn

    def wrapper(state: Any) -> Any:
        with _unit_span(f"node:{unit}"):
            return fn(state)

    return wrapper


def wrap_subagent(subagent: "Subagent") -> "Subagent":
    """把子智能体适配层包进单元 span；未启用时原样返回。"""
    if not langfuse_enabled():
        return subagent
    return _TracedSubagent(subagent)


class _TracedSubagent:
    """子智能体黑盒调用的 span 包装：对内实现零侵入。"""

    def __init__(self, inner: "Subagent") -> None:
        self._inner = inner

    @property
    def unit(self) -> str:
        """运行单元名，透传内层。"""
        return self._inner.unit

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        with _unit_span(f"subagent:{self._inner.unit}"):
            return await self._inner.run(task)
