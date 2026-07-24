"""统一 LLM 调用封装层。

所有模型统一按 OpenAI 兼容接口封装；本层是测试替换确定性假 LLM 的注入点：
节点代码只依赖 LLM 协议与 LLMFactory，不直接触碰 openai SDK。
Langfuse 启用时客户端换成官方插桩版，每次调用自动上报 generation。
"""

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from openai import OpenAI, Stream
    from openai.types.chat import ChatCompletionChunk

from llm import observability
from llm.llm_config import LLMConfig, load_llm_config

Message = dict[str, str]

# 流式片段种类：正文（content）/ 推理（thinking，思考开启时）。
# 取值对齐 CONTEXT.md 逐字流词汇表「按 kind 区分（content / thinking）」，
# 逐字流 content_delta.kind 即由本类型映射；provider SDK 字段名
# （reasoning_content / reasoning）是另一层概念，不混入此处。
StreamKind = Literal["content", "thinking"]


@dataclass(frozen=True)
class StreamChunk:
    """LLM 流式单个片段：正文或推理，逐字流据此区分 ``kind`` 外发。

    PRD #53 原草 ``stream -> Iterator[str]``，但思考开启时正文与推理 CoT
    需逐片段区分 kind（逐字流 ``content_delta`` 的 ``kind`` 字段即由本类型
    映射），故协议片段承载类型化的 ``kind`` 而非裸串——这是「最低注入点」
    之上、为 kind 区分所做的必要承载。
    """

    kind: StreamKind
    text: str


class LLM(Protocol):
    """最小 LLM 调用协议：输入 OpenAI 格式消息列表，返回文本。"""

    def invoke(self, messages: list[Message]) -> str: ...

    def stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """流式调用：逐片段产出正文，思考开启时一并产出推理片段。"""
        ...

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
        client = self._ensure_client()
        completions = cast(Any, client.chat.completions)
        response = completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            **self._structured_output_kwargs(messages),
            # 思考开关显式下发（OpenAI 兼容扩展参数，qwen 系等服务商支持；
            # 不认识该参数的服务商会忽略）。缺省关闭：结构化 JSON 任务上
            # 思考 token 可达可见输出的数倍，延迟放大 5 倍以上。
            extra_body={"enable_thinking": self._config.enable_thinking},
        )
        content = response.choices[0].message.content
        return content or ""

    def stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """流式调用：``stream=True`` 迭代 ``delta.content``。

        思考开启时一并按 provider 字段名抽推理片段：``delta.reasoning_content``
        （qwen 系等）优先，回退 ``delta.reasoning``（部分 provider）。正文片段
        ``kind=content``、推理片段 ``kind=thinking``，逐字流据此映射
        ``content_delta.kind``（取值对齐 CONTEXT.md 词汇表 content / thinking）。
        """
        client = self._ensure_client()
        completions = cast(Any, client.chat.completions)
        response = completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            **self._structured_output_kwargs(messages),
            extra_body={"enable_thinking": self._config.enable_thinking},
        )
        # SDK 在 stream=True 下的返回为 Stream[ChatCompletionChunk]；mypy 对带
        # extra_body 的重载推断不稳，显式定型到流式迭代器，避免 union-attr 噪声。
        stream = cast("Stream[ChatCompletionChunk]", response)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield StreamChunk("content", content)
            if self._config.enable_thinking:
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning:
                    yield StreamChunk("thinking", reasoning)

    def _ensure_client(self) -> "OpenAI":
        """惰性创建并返回网络客户端，锁保证并发下只创建一次。

        invoke 与 stream 共用：客户端类按 Langfuse 启用与否选择（插桩版是
        原版的子类，接口一致）；仅构造对象（如配置校验场景）不触发创建。
        """
        with self._client_lock:
            if self._client is None:
                client_class = observability.openai_client_class()
                self._client = client_class(
                    base_url=self._config.base_url, api_key=self._config.api_key
                )
            return self._client

    def _structured_output_kwargs(
        self, messages: list[Message]
    ) -> dict[str, Any]:
        """Use provider-enforced JSON mode when the prompt asks for JSON output.

        Qwen/DashScope supports OpenAI-compatible ``response_format`` for
        non-thinking models. Thinking mode cannot combine with JSON object mode,
        so in that case we leave the request prompt-only.
        """
        if self._config.enable_thinking:
            return {}
        if any("JSON" in message.get("content", "") for message in messages):
            return {"response_format": {"type": "json_object"}}
        return {}


class FakeLLM:
    """确定性测试替身：线程安全，支持顺序应答与按提示词内容键控的应答。

    应答分派规则（每次调用依次尝试）：
    1. 键控应答：keyed_responses 里首个「键出现在消息文本中且尚有剩余应答」的键，
       弹出该键的下一条应答——用于并发调用场景，应答与提示词内容绑定而非依赖调用顺序；
    2. 顺序应答：从预置列表头部弹出；
    3. 全部耗尽后返回带调用序号的固定文本。

    ``stream`` 复用同一应答分派拿到完整文本，再按 ``chunk_size`` 确定性定长分块
    吐流（全部为 ``content`` kind，FakeLLM 不产生推理）；多次运行同一配置输出一致，
    供 mock 档与场景库复用。
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        keyed_responses: dict[str, list[str]] | None = None,
        *,
        chunk_size: int = 8,
    ) -> None:
        self._responses = list(responses or [])
        self._keyed_responses = {
            key: list(values) for key, values in (keyed_responses or {}).items()
        }
        self._chunk_size = chunk_size
        self._lock = threading.Lock()
        self.calls: list[list[Message]] = []

    @property
    def metadata(self) -> dict[str, str]:
        """供状态与事件上报的配置元数据。"""
        return {"model": "fake-llm", "base_url": "fake://"}

    def _dispatch(self, messages: list[Message]) -> str:
        """应答分派：键控优先、顺序次之、耗尽兜底（线程安全由调用方持锁）。"""
        self.calls.append(messages)
        text = "\n".join(message.get("content", "") for message in messages)
        for key, values in self._keyed_responses.items():
            if values and key in text:
                return values.pop(0)
        if self._responses:
            return self._responses.pop(0)
        return f"假LLM应答#{len(self.calls)}"

    def invoke(self, messages: list[Message]) -> str:
        with self._lock:
            return self._dispatch(messages)

    def stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """确定性分块吐流：按 ``chunk_size`` 切完整应答为定长片段。"""
        with self._lock:
            text = self._dispatch(messages)
        # 跨 ``range`` 切片：定长、可复现，多次运行同一应答输出逐字一致。
        for start in range(0, len(text), self._chunk_size):
            yield StreamChunk("content", text[start : start + self._chunk_size])


LLMFactory = Callable[[str], LLM]
"""按运行单元名构造 LLM 实例的工厂类型。"""


def default_llm_factory(unit: str) -> LLM:
    """缺省工厂：读取该单元的环境变量配置并构造真实 OpenAI 兼容客户端。"""
    return OpenAICompatibleLLM(load_llm_config(unit))
