"""统一 LLM 调用封装层。

所有模型统一按 OpenAI 兼容接口封装；本层是测试注入确定性假 LLM 的接缝：
节点代码只依赖 LLM 协议与 LLMFactory，不直接触碰 openai SDK。
Langfuse 启用时客户端换成官方插桩版，每次调用自动上报 generation。
"""

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from openai import OpenAI

from llm import observability
from llm.llm_config import LLMConfig, load_llm_config

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
        # 节点内并发调用同一实例时，锁保证惰性客户端只创建一次。
        self._client_lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, str]:
        """供状态与事件上报的配置元数据，不含密钥。"""
        return {
            "model": self._config.model,
            "base_url": self._config.base_url,
            "enable_thinking": "1" if self._config.enable_thinking else "0",
        }

    def invoke(self, messages: list[Message]) -> str:
        with self._client_lock:
            if self._client is None:
                # 惰性创建，避免仅构造对象（如配置校验场景）就建立网络客户端；
                # 客户端类按 Langfuse 启用与否选择（插桩版是原版的子类，接口一致）。
                client_class = observability.openai_client_class()

                self._client = client_class(
                    base_url=self._config.base_url, api_key=self._config.api_key
                )
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            # 思考开关显式下发（OpenAI 兼容扩展参数，qwen 系等服务商支持；
            # 不认识该参数的服务商会忽略）。缺省关闭：结构化 JSON 任务上
            # 思考 token 可达可见输出的数倍，延迟放大 5 倍以上。
            extra_body={"enable_thinking": self._config.enable_thinking},
        )
        content = response.choices[0].message.content
        return content or ""


class FakeLLM:
    """确定性测试替身：线程安全，支持顺序应答与按提示词内容键控的应答。

    应答分派规则（每次调用依次尝试）：
    1. 键控应答：keyed_responses 里首个「键出现在消息文本中且尚有剩余应答」的键，
       弹出该键的下一条应答——用于并发调用场景，应答与提示词内容绑定而非依赖调用顺序；
    2. 顺序应答：从预置列表头部弹出；
    3. 全部耗尽后返回带调用序号的固定文本。
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        keyed_responses: dict[str, list[str]] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._keyed_responses = {
            key: list(values) for key, values in (keyed_responses or {}).items()
        }
        self._lock = threading.Lock()
        self.calls: list[list[Message]] = []

    @property
    def metadata(self) -> dict[str, str]:
        """供状态与事件上报的配置元数据。"""
        return {"model": "fake-llm", "base_url": "fake://"}

    def invoke(self, messages: list[Message]) -> str:
        with self._lock:
            self.calls.append(messages)
            text = "\n".join(message.get("content", "") for message in messages)
            for key, values in self._keyed_responses.items():
                if values and key in text:
                    return values.pop(0)
            if self._responses:
                return self._responses.pop(0)
            return f"假LLM应答#{len(self.calls)}"


LLMFactory = Callable[[str], LLM]
"""按运行单元名构造 LLM 实例的工厂类型。"""


def default_llm_factory(unit: str) -> LLM:
    """缺省工厂：读取该单元的环境变量配置并构造真实 OpenAI 兼容客户端。"""
    return OpenAICompatibleLLM(load_llm_config(unit))
