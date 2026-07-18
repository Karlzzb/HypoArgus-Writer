"""端到端空跑测试：注入假 LLM 跑整张图，验证状态机流转与 Postgres 持久化。

Postgres 连接串取环境变量 HYPOARGUS_TEST_PG_DSN，缺省指向本地测试库；
库不可达时跳过持久化用例（其余用例仍必须全绿）。
"""

import os
import socket
import uuid
from urllib.parse import urlparse

import pytest

from graph import MAIN_NODES, NODE_STATUS, build_graph, postgres_checkpointer
from llm_client import FakeLLM
from llm_config import RUNTIME_UNITS
from state import WorkflowStatus, initial_state

TEST_PG_DSN = os.environ.get(
    "HYPOARGUS_TEST_PG_DSN",
    "postgresql://postgres:postgres@127.0.0.1:15432/postgres",
)

EXPECTED_STATUS_ORDER = [
    WorkflowStatus.FRAMEWORK_BUILDING,
    WorkflowStatus.REFERENCE_FETCHING,
    WorkflowStatus.ARTICLE_WRITING,
    WorkflowStatus.CITATION_CHECKING,
    WorkflowStatus.AWAIT_USER_REVIEW,
]


def _pg_reachable(dsn: str) -> bool:
    parsed = urlparse(dsn)
    try:
        with socket.create_connection(
            (parsed.hostname or "127.0.0.1", parsed.port or 5432), timeout=2
        ):
            return True
    except OSError:
        return False


def test_主节点清单与运行单元清单一致():
    # 5 个主节点必须都是合法运行单元，防止两处常量清单漂移。
    assert set(MAIN_NODES) <= set(RUNTIME_UNITS)
    assert len(MAIN_NODES) == 5


def test_假LLM端到端空跑_状态机按序流转():
    fake = FakeLLM()
    graph = build_graph(llm_factory=lambda unit: fake)

    observed: list[WorkflowStatus] = []
    for update in graph.stream(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-e2e"),
        stream_mode="updates",
    ):
        for node_name, node_update in update.items():
            assert node_update["status"] == NODE_STATUS[node_name]
            observed.append(node_update["status"])

    assert observed == EXPECTED_STATUS_ORDER


def test_每个主节点各经统一封装层调用一次LLM():
    fake = FakeLLM()
    units_seen: list[str] = []

    def factory(unit: str) -> FakeLLM:
        units_seen.append(unit)
        return fake

    graph = build_graph(llm_factory=factory)
    result = graph.invoke(initial_state("意图", "身份", "trace-llm"))

    assert units_seen == list(MAIN_NODES)
    assert len(fake.calls) == len(MAIN_NODES)
    # 终态记录的是最后一个节点（human_review_gate）的配置元数据，且不含密钥。
    assert result["current_node_llm_config"]["unit"] == "human_review_gate"
    assert "api_key" not in result["current_node_llm_config"]


@pytest.mark.skipif(
    not _pg_reachable(TEST_PG_DSN), reason="测试 Postgres 不可达"
)
def test_状态经Postgres存档器持久化():
    import psycopg

    thread_id = f"e2e-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    with postgres_checkpointer(TEST_PG_DSN) as saver:
        graph = build_graph(
            llm_factory=lambda unit: FakeLLM(), checkpointer=saver
        )
        result = graph.invoke(
            initial_state("持久化测试", "专业撰稿人", "trace-pg"), config
        )
        assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW

        # 存档器可读回最新 checkpoint，且状态与终态一致（断点续跑的前提）。
        snapshot = graph.get_state(config)
        assert snapshot.values["status"] == WorkflowStatus.AWAIT_USER_REVIEW

    # 直接查库：该 thread_id 下确有 checkpoint 记录（每个节点一步 + 起始步）。
    with psycopg.connect(TEST_PG_DSN) as conn:
        row = conn.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = %s", (thread_id,)
        ).fetchone()
        assert row is not None and row[0] >= len(MAIN_NODES)
