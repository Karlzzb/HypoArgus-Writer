"""统一 LLM 调用封装层（测试接缝）的单元测试。"""

from llm.llm_client import FakeLLM


def test_假LLM按序返回预置应答并记录调用():
    fake = FakeLLM(responses=["第一", "第二"])
    assert fake.invoke([{"role": "user", "content": "a"}]) == "第一"
    assert fake.invoke([{"role": "user", "content": "b"}]) == "第二"
    assert fake.invoke([{"role": "user", "content": "c"}]) == "假LLM应答#3"
    assert len(fake.calls) == 3
    assert fake.calls[0][0]["content"] == "a"


def test_假LLM配置元数据不含密钥():
    assert "api_key" not in FakeLLM().metadata
