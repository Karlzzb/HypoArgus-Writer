"""统一 LLM 调用封装层（测试注入点）的单元测试。"""

from types import SimpleNamespace

import pytest

from llm import observability
from llm.llm_client import FakeLLM, OpenAICompatibleLLM, StreamChunk
from llm.llm_config import LLMConfig


def test_假LLM按序返回预置应答并记录调用():
    fake = FakeLLM(responses=["第一", "第二"])
    assert fake.invoke([{"role": "user", "content": "a"}]) == "第一"
    assert fake.invoke([{"role": "user", "content": "b"}]) == "第二"
    assert fake.invoke([{"role": "user", "content": "c"}]) == "假LLM应答#3"
    assert len(fake.calls) == 3
    assert fake.calls[0][0]["content"] == "a"


def test_假LLM配置元数据不含密钥():
    assert "api_key" not in FakeLLM().metadata


def test_假LLM_stream按定长分块吐流且与invoke一致():
    """stream 复用应答分派，再按 chunk_size 定长分块；拼接结果与 invoke 一致。"""
    response = '{"chapter_text": "正文正文正文"}'
    fake = FakeLLM(responses=[response], chunk_size=4)

    streamed = list(fake.stream([{"role": "user", "content": "q"}]))

    # 全部为 content kind，拼接还原完整应答。
    assert all(chunk.kind == "content" for chunk in streamed)
    assert "".join(chunk.text for chunk in streamed) == response
    # 与 invoke 路径返回同一文本（应答分派一致）。
    assert (
        FakeLLM(responses=[response]).invoke([{"role": "user", "content": "q"}])
        == response
    )
    # stream 同样记录调用，供断言。
    assert len(fake.calls) == 1


def test_假LLM_stream多次运行输出一致():
    """确定性分块：同配置多次独立实例输出逐字一致（供 mock 档与场景库复用）。"""
    response = "假LLM应答#1" * 5
    first = [
        chunk.text
        for chunk in FakeLLM(responses=[response], chunk_size=3).stream(
            [{"role": "user", "content": "q"}]
        )
    ]
    second = [
        chunk.text
        for chunk in FakeLLM(responses=[response], chunk_size=3).stream(
            [{"role": "user", "content": "q"}]
        )
    ]
    assert first == second
    assert "".join(first) == response


def test_假LLM_stream键控应答同样可分块():
    """键控应答也经 stream 分派分块，应答与提示词内容绑定。"""
    fake = FakeLLM(
        keyed_responses={"锚词": ["键控应答一"]},
        chunk_size=2,
    )
    streamed = list(fake.stream([{"role": "user", "content": "请围绕锚词展开"}]))
    assert "".join(chunk.text for chunk in streamed) == "键控应答一"


def test_假LLM_stream与invoke共用同一调用序号():
    """stream 与 invoke 共享应答池：stream 消费一条后 invoke 取下一条。"""
    fake = FakeLLM(responses=["第一", "第二"], chunk_size=4)
    streamed = "".join(c.text for c in fake.stream([{"role": "user", "content": "a"}]))
    assert streamed == "第一"
    # stream 已弹出「第一」，invoke 取「第二」。
    assert fake.invoke([{"role": "user", "content": "b"}]) == "第二"


def test_StreamChunk为不可变值对象():
    chunk = StreamChunk("content", "片段")
    assert chunk.kind == "content"
    assert chunk.text == "片段"
    # frozen dataclass：不可赋值。
    with pytest.raises(Exception):
        chunk.kind = "thinking"  # type: ignore[misc]


class _RecordingCompletions:
    calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content="{\"ok\": true}"))
                    ]
                )
            ])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="{\"ok\": true}"))
            ]
        )


class _RecordingOpenAI:
    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.chat = SimpleNamespace(completions=_RecordingCompletions())


def _recording_llm(*, enable_thinking: bool = False) -> OpenAICompatibleLLM:
    _RecordingCompletions.calls = []
    return OpenAICompatibleLLM(
        LLMConfig(
            model="stub-model",
            base_url="https://example.invalid/v1",
            api_key="sk-test",
            enable_thinking=enable_thinking,
        )
    )


def test_OpenAI兼容LLM_JSON提示启用响应格式(monkeypatch):
    monkeypatch.setattr(observability, "openai_client_class", lambda: _RecordingOpenAI)
    llm = _recording_llm()

    assert llm.invoke([{"role": "system", "content": "只输出 JSON"}]) == "{\"ok\": true}"

    call = _RecordingCompletions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"enable_thinking": False}


def test_OpenAI兼容LLM_stream_JSON提示启用响应格式(monkeypatch):
    monkeypatch.setattr(observability, "openai_client_class", lambda: _RecordingOpenAI)
    llm = _recording_llm()

    chunks = list(llm.stream([{"role": "system", "content": "Return JSON."}]))

    assert "".join(chunk.text for chunk in chunks) == "{\"ok\": true}"
    call = _RecordingCompletions.calls[0]
    assert call["stream"] is True
    assert call["response_format"] == {"type": "json_object"}


def test_OpenAI兼容LLM_thinking模式不启用响应格式(monkeypatch):
    monkeypatch.setattr(observability, "openai_client_class", lambda: _RecordingOpenAI)
    llm = _recording_llm(enable_thinking=True)

    llm.invoke([{"role": "system", "content": "只输出 JSON"}])

    call = _RecordingCompletions.calls[0]
    assert "response_format" not in call
    assert call["extra_body"] == {"enable_thinking": True}
