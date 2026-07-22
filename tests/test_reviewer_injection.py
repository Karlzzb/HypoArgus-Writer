"""chapter_reviewer 构图注入的装配测试：可经 build_graph 参数注入、stub 可替换。

验收（issue #44）：chapter_reviewer 可经构图注入参数注入，stub 可替换。
注入 stub 时不触碰真实现工厂（不请求 chapter_reviewer 单元 LLM）；
未注入时走真实现工厂（构造期按单元名请求 LLM）。
"""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from agents.chapter_reviewer import make_stub_chapter_reviewer
from graph import build_graph
from llm.llm_client import FakeLLM
from nodes.writing_orchestrator import make_writing_orchestrator_node


def _recording_factory(seen: list[str]) -> Any:
    def factory(unit: str) -> FakeLLM:
        seen.append(unit)
        return FakeLLM([])

    return factory


def test_构图注入_stub评审可替换_不请求真实现单元() -> None:
    seen: list[str] = []
    build_graph(
        llm_factory=_recording_factory(seen),
        checkpointer=InMemorySaver(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
    # 注入 stub：构造期不按 chapter_reviewer 单元请求 LLM（真实现工厂未被触发）。
    assert "chapter_reviewer" not in seen


def test_构图注入_未注入_走真实现工厂请求单元() -> None:
    seen: list[str] = []
    build_graph(
        llm_factory=_recording_factory(seen),
        checkpointer=InMemorySaver(),
    )
    # 未注入：真实现工厂构造期按单元名请求一次 LLM。
    assert "chapter_reviewer" in seen


def test_节点工厂_接受评审注入参数() -> None:
    # 节点工厂接受 chapter_reviewer 注入接缝（T3 起消费），构造不报错。
    node = make_writing_orchestrator_node(
        make_stub_chapter_reviewer(),
        make_stub_chapter_reviewer(),
        None,
        make_stub_chapter_reviewer(),
    )
    assert callable(node)
