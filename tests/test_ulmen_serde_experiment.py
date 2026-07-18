"""ulmen-langgraph 压缩 serde 实验：验证「关闭开关后历史存档必须仍可读取」约束。

PRD 约定：ulmen 压缩仅作为实验性可选 serde 接入存档器，默认关闭；
关闭开关后历史存档必须仍可读取，做不到则不启用。

实验结论（本文件固化为回归证据）：
- 正向兼容成立：开启压缩的存档器可以读取此前未压缩写入的存档
  （UlmenSerde 按 ULMZ 魔数前缀判别，无前缀则透传内层 serde）。
- 反向兼容不成立：压缩写入的存档在关闭开关后，纯 PostgresSaver
  读取时不报错而是**静默解出错误数据**（msgpack 把 ULMZ 魔数首字节
  当作整数解码并丢弃其余内容），属于最危险的静默数据损坏。
- 因此本项目不启用 ulmen serde，graph.py 不提供接入开关；
  本实验测试保留 ulmen-langgraph 于 experiment 依赖组，供复现结论。
"""

import uuid
from typing import Any, TypedDict

import pytest

from tests.test_graph_e2e import TEST_PG_DSN, _pg_reachable

ulmen_langgraph = pytest.importorskip(
    "ulmen.ext.langgraph", reason="未安装 experiment 依赖组（ulmen-langgraph）"
)

from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

UlmenCheckpointer = ulmen_langgraph.UlmenCheckpointer

pytestmark = pytest.mark.skipif(
    not _pg_reachable(TEST_PG_DSN), reason="实验 Postgres 不可达"
)


class _State(TypedDict):
    materials: list[Any]
    meta: dict[str, Any]


def _node(state: _State) -> _State:
    return {
        "materials": state["materials"] + [{"id": "m2", "excerpt": "第二条"}],
        "meta": {**state["meta"], "round": 2},
    }


def _builder() -> StateGraph:
    builder = StateGraph(_State)
    builder.add_node("n", _node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    return builder


INITIAL = {"materials": [{"id": "m1", "excerpt": "首条"}], "meta": {"round": 1}}
EXPECTED = {
    "materials": [{"id": "m1", "excerpt": "首条"}, {"id": "m2", "excerpt": "第二条"}],
    "meta": {"round": 2},
}


def _run_with(checkpointer_wrapper, thread_id: str) -> None:
    with PostgresSaver.from_conn_string(TEST_PG_DSN) as saver:
        saver.setup()
        graph = _builder().compile(checkpointer=checkpointer_wrapper(saver))
        graph.invoke(INITIAL, {"configurable": {"thread_id": thread_id}})


def test_正向兼容_开启压缩可读取未压缩历史存档():
    thread_id = f"ulmen-exp-plain-{uuid.uuid4().hex[:8]}"
    _run_with(lambda saver: saver, thread_id)

    with PostgresSaver.from_conn_string(TEST_PG_DSN) as saver:
        tup = UlmenCheckpointer(saver).get_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
    assert tup is not None
    assert tup.checkpoint["channel_values"]["materials"] == EXPECTED["materials"]
    assert tup.checkpoint["channel_values"]["meta"] == EXPECTED["meta"]


def test_反向不兼容_关闭开关后压缩存档静默损坏_故不启用():
    """固化不启用结论：压缩写入的存档关闭开关后读出的是损坏数据。

    若某天 ulmen-langgraph 修复了该问题，本测试会失败，
    届时可重新评估接入；在此之前 graph.py 不得提供 ulmen 开关。
    """
    thread_id = f"ulmen-exp-ulmz-{uuid.uuid4().hex[:8]}"
    _run_with(UlmenCheckpointer, thread_id)

    with PostgresSaver.from_conn_string(TEST_PG_DSN) as saver:
        try:
            tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
        except Exception:
            # 报错读不出同样意味着反向不兼容，结论一致。
            return
    assert tup is not None
    values = tup.checkpoint["channel_values"]
    assert values["materials"] != EXPECTED["materials"], (
        "纯 PostgresSaver 竟能正确读取压缩存档：反向兼容已成立，"
        "请重新评估启用 ulmen serde 并更新文档结论"
    )
