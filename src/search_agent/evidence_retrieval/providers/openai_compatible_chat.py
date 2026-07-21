"""Small async OpenAI-compatible chat client used by the real Judge CLI.

The evidence flow already owns the Judge timeout, batching, validation and
Langfuse GENERATION lifecycle.  Pulling in the general LangChain/OpenAI SDK
adds several seconds of process start-up without adding a capability here, so
this adapter implements only the single ``ainvoke`` operation the Judge needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from ...env import env_str


def _chat_completions_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    suffix = "/chat/completions"
    if value.endswith(suffix):
        return value
    return value + suffix


@dataclass(slots=True)
class OpenAICompatibleMessage:
    content: str
    response_metadata: dict[str, Any] = field(default_factory=dict)
    usage_metadata: dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleChatClient:
    """Reusable HTTP/1.1 pool with the LangChain-like ``ainvoke`` seam.

    The configured model gateway has been observed returning prematurely
    closed JSON for concurrent HTTP/2 streams. A bounded HTTP/1.1 pool retains
    keep-alive/concurrency without that transport issue; its size is aligned
    with the Judge semaphore by the runtime factory.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 35.0,
        max_output_tokens: int = 4096,
        max_connections: int = 4,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.endpoint = _chat_completions_url(endpoint)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            http2=False,
            timeout=httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds)),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            headers={"Authorization": f"Bearer {api_key}"},
            trust_env=False,
        )

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None,
        timeout_seconds: float = 35.0,
        max_connections: int = 4,
    ):
        api_key = env_str("LLM_KEY")
        base_url = env_str("LLM_BASE_URL")
        if not api_key or not base_url:
            return None
        return cls(
            endpoint=base_url,
            api_key=api_key,
            model=model or env_str("LLM_MODEL") or "qwen-turbo",
            timeout_seconds=timeout_seconds,
            max_connections=max_connections,
        )

    async def ainvoke(self, prompt: str) -> OpenAICompatibleMessage:
        response = await self.client.post(
            self.endpoint,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self.max_output_tokens,
            },
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        content = message.get("content") or ""
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        metadata = {
            "id": payload.get("id"),
            "model": payload.get("model") or self.model,
            "finish_reason": choice.get("finish_reason"),
            "model_provider": "openai_compatible",
            "usage": usage,
        }
        return OpenAICompatibleMessage(
            content=str(content),
            response_metadata={key: value for key, value in metadata.items() if value is not None},
            usage_metadata=usage,
        )

    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
        tool_choice: str | dict[str, Any] | None = None,
    ):
        """LangChain-compatible Tool Calling binding for the configured gateway."""
        definitions = []
        for tool in tools:
            schema_model = getattr(tool, "args_schema", None)
            parameters = schema_model.model_json_schema() if schema_model is not None else {
                "type": "object", "properties": {},
            }
            definitions.append({
                "type": "function",
                "function": {
                    "name": str(tool.name),
                    "description": str(tool.description or ""),
                    "parameters": parameters,
                },
            })
        return BoundOpenAICompatibleTools(
            parent=self,
            tools=definitions,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()


@dataclass(slots=True)
class BoundOpenAICompatibleTools:
    parent: OpenAICompatibleChatClient
    tools: list[dict[str, Any]]
    parallel_tool_calls: bool = True
    tool_choice: str | dict[str, Any] | None = None

    async def ainvoke(self, prompt: Any):
        from langchain_core.messages import AIMessage

        if isinstance(prompt, list):
            messages = []
            for message in prompt:
                role = getattr(message, "type", None) or getattr(message, "role", None) or "user"
                role = {"human": "user", "ai": "assistant", "tool": "tool"}.get(str(role), str(role))
                messages.append({"role": role, "content": str(getattr(message, "content", message))})
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        body: dict[str, Any] = {
            "model": self.parent.model,
            "messages": messages,
            "tools": self.tools,
            "parallel_tool_calls": self.parallel_tool_calls,
            "temperature": 0,
            "max_tokens": self.parent.max_output_tokens,
        }
        if self.tool_choice is not None:
            body["tool_choice"] = self.tool_choice
        response = await self.parent.client.post(self.parent.endpoint, json=body)
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        tool_calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            raw_args = function.get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {"_invalid_json": raw_args}
            tool_calls.append({
                "name": str(function.get("name") or ""),
                "args": raw_args if isinstance(raw_args, dict) else {},
                "id": str(call.get("id") or f"tool-call-{index + 1}"),
                "type": "tool_call",
            })
        metadata = {
            "id": payload.get("id"),
            "model": payload.get("model") or self.parent.model,
            "finish_reason": choice.get("finish_reason"),
            "model_provider": "openai_compatible",
            "usage": payload.get("usage") or {},
        }
        return AIMessage(
            content=str(message.get("content") or ""),
            tool_calls=tool_calls,
            response_metadata={key: value for key, value in metadata.items() if value is not None},
        )
