"""Langfuse 可观测接入层测试：未启用零侵入 + 启用后运行单元 span 全覆盖。"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.types import Command

from llm import observability
from agents.rewriter_loop import make_stub_rewriter_loop
from agents.search_agent import make_stub_search_agent
from domain.units import MAIN_NODES
from graph import build_graph, checkpoint_serializer
from llm.llm_client import FakeLLM, OpenAICompatibleLLM
from llm.llm_config import LLMConfig
from domain.state import initial_state
from tests.llm_response_plans import FIRST_PASS_RESPONSES, FRAMEWORK_KEYED_RESPONSES

STUB_MODEL = "stub-observability-model"
STUB_COMPLETION = "插桩应答"


@pytest.fixture(autouse=True)
def _clean_langfuse_env(monkeypatch):
    """默认清空 Langfuse 环境变量与客户端注入，保证测试相互隔离。"""
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_TRACING_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    observability.use_client(None)


def test_未配置时全部入口零侵入():
    from openai import OpenAI

    assert observability.langfuse_enabled() is False
    assert observability.openai_client_class() is OpenAI

    def node_fn(state):
        return state

    assert observability.traced_node("framework_orchestrator", node_fn) is node_fn

    class _Agent:
        unit = "search_agent"

        async def run(self, task):
            return {}

    agent = _Agent()
    assert observability.wrap_subagent(agent) is agent
    # 更新当前 span 元数据同样零侵入：不建客户端、不抛错。
    observability.update_current_span_metadata({"total_elapsed_ms": 1})


def test_启用时把诊断元数据转交当前span():
    class _RecordingClient:
        def __init__(self) -> None:
            self.metadata_updates: list[dict] = []

        def update_current_span(self, *, metadata: dict) -> None:
            self.metadata_updates.append(metadata)

    client = _RecordingClient()
    observability.use_client(client)  # type: ignore[arg-type]

    observability.update_current_span_metadata(
        {"search_agent_flow_metrics": {"total_elapsed_ms": 12}}
    )
    assert client.metadata_updates == [
        {"search_agent_flow_metrics": {"total_elapsed_ms": 12}}
    ]


def test_配置公私钥后选用Langfuse插桩客户端(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    from langfuse.openai import OpenAI as LangfuseOpenAI

    assert observability.langfuse_enabled() is True
    assert observability.openai_client_class() is LangfuseOpenAI


def test_官方总开关设false时全部入口回到直通实现(monkeypatch):
    from openai import OpenAI

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "False")

    assert observability.langfuse_enabled() is False
    assert observability.openai_client_class() is OpenAI

    def node_fn(state):
        return state

    assert observability.traced_node("framework_orchestrator", node_fn) is node_fn


def test_官方总开关非false取值不改变启用行为(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    # 与 SDK 同口径：只有字面 "false"（忽略大小写）才关，其他取值一律开。
    for value in ("true", "1", "0", "no", "anything"):
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", value)
        assert observability.langfuse_enabled() is True, value


def test_启用后一次完整运行span覆盖全部运行单元(captured_spans):
    names = [span.name for span in captured_spans]

    assert observability.RUN_SPAN_NAME in names
    # writing_orchestrator 只在修订与终审回退时运行，首跑主路径不经过它。
    for node in MAIN_NODES:
        if node == "writing_orchestrator":
            continue
        assert f"node:{node}" in names, f"缺少主节点 span：{node}"
    for unit in ("search_agent", "rewriter_loop"):
        assert f"subagent:{unit}" in names, f"缺少子智能体 span：{unit}"


def test_人工中断不把门禁span标记为错误(captured_spans):
    from opentelemetry.trace import StatusCode

    gate_spans = [
        span for span in captured_spans if span.name == "node:human_review_gate"
    ]
    assert gate_spans
    for span in gate_spans:
        assert span.status.status_code is not StatusCode.ERROR


def test_启用后LLM调用经官方插桩上报generation(captured_spans):
    """插桩客户端把真实（此处为桩服务）LLM 调用上报为 generation 观测。"""
    generation_spans = [
        span
        for span in captured_spans
        if STUB_MODEL in (span.attributes or {}).values()
    ]
    assert generation_spans, "未捕获到插桩客户端上报的 generation span"
    values = [
        str(value)
        for span in generation_spans
        for value in (span.attributes or {}).values()
    ]
    # 输出以 JSON 转义（\uXXXX）形式记录，两种形态命中其一即可。
    escaped = json.dumps(STUB_COMPLETION)[1:-1]
    assert any(STUB_COMPLETION in v or escaped in v for v in values), (
        "generation 观测未记录模型输出"
    )
    assert any('"total_tokens": 5' in v for v in values), (
        "generation 观测未记录 token 用量"
    )


class _StubOpenAIHandler(BaseHTTPRequestHandler):
    """最小 OpenAI 兼容桩服务：任何 POST 都返回一条固定的 chat completion。"""

    def do_POST(self) -> None:  # noqa: N802 —— BaseHTTPRequestHandler 约定命名。
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": STUB_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": STUB_COMPLETION,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """静默桩服务访问日志。"""


@pytest.fixture(scope="module")
def captured_spans():
    """注入内存导出器的真实 Langfuse 客户端，跑一遍闭环并返回全部 span。

    OTel 全局 TracerProvider 每进程只能安装一次，因此本 fixture 按模块
    只跑一次闭环，需要断言 span 的测试共用同一份捕获结果。
    图运行之外再经官方插桩客户端调一次 OpenAI 兼容桩服务，
    覆盖 generation 上报路径（FakeLLM 不走 openai 客户端）。
    """
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:1",
        span_exporter=exporter,
        tracing_enabled=True,
    )
    observability.use_client(client)
    server = HTTPServer(("127.0.0.1", 0), _StubOpenAIHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        fake = FakeLLM(
            list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
        )
        graph = build_graph(
            llm_factory=lambda unit: fake,
            checkpointer=InMemorySaver(serde=checkpoint_serializer()),
            # 本测试验收 span 覆盖而非子智能体真实现：显式注入两个打桩。
            search_agent=make_stub_search_agent(),
            rewriter_loop=make_stub_rewriter_loop(),
        )
        config: RunnableConfig = {"configurable": {"thread_id": "obs-test"}}
        with observability.run_span(
            thread_id="obs-test", session_id="sess", trace_id="trace"
        ):
            graph.invoke(initial_state("意图", "身份", "trace"), config)
            graph.invoke(Command(resume={"action": "finalize"}), config)
            # 图运行同一 trace 内经插桩客户端调桩服务，上报一条 generation。
            llm = OpenAICompatibleLLM(
                LLMConfig(
                    model=STUB_MODEL,
                    base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                    api_key="sk-stub",
                )
            )
            assert llm.invoke([{"role": "user", "content": "你好"}]) == (
                STUB_COMPLETION
            )
        client.flush()
        return exporter.get_finished_spans()
    finally:
        server.shutdown()
        server.server_close()
        observability.use_client(None)
