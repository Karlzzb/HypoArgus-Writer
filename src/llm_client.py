"""统一 LLM 调用封装层。

所有模型统一按 OpenAI 兼容接口封装；本层是测试注入确定性假 LLM 的接缝：
节点代码只依赖 LLM 协议与 LLMFactory，不直接触碰 openai SDK。
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from openai import OpenAI

from llm_config import LLMConfig, load_llm_config

Message = dict[str, str]


class LLM(Protocol):
    """最小 LLM 调用协议：输入 OpenAI 格式消息列表，返回文本。"""

    def invoke(self, messages: list[Message]) -> str: ...

    @property
    def metadata(self) -> dict[str, str]:
        """供状态与事件上报的配置元数据，不得包含密钥。"""
        ...


class OpenAICompatibleLLM:
    """按 OpenAI 兼容接口调用真实模型的封装。"""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: OpenAI | None = None

    @property
    def metadata(self) -> dict[str, str]:
        """供状态与事件上报的配置元数据，不含密钥。"""
        return {"model": self._config.model, "base_url": self._config.base_url}

    def invoke(self, messages: list[Message]) -> str:
        if self._client is None:
            # 惰性创建，避免仅构造对象（如配置校验场景）就建立网络客户端。
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._config.base_url, api_key=self._config.api_key
            )
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
        )
        content = response.choices[0].message.content
        return content or ""


class FakeLLM:
    """确定性测试替身：依次返回预置应答，耗尽后返回带调用序号的固定文本。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[Message]] = []

    @property
    def metadata(self) -> dict[str, str]:
        """供状态与事件上报的配置元数据。"""
        return {"model": "fake-llm", "base_url": "fake://"}

    def invoke(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return f"假LLM应答#{len(self.calls)}"


LLMFactory = Callable[[str], LLM]
"""按运行单元名构造 LLM 实例的工厂类型。"""


def default_llm_factory(unit: str) -> LLM:
    """缺省工厂：读取该单元的环境变量配置并构造真实 OpenAI 兼容客户端。"""
    return OpenAICompatibleLLM(load_llm_config(unit))
