"""issue #59 门控真实流式逐字流 E2E：真实 LLM stream 路径的接缝验收。

按环境变量门控：显式设 HYPOARGUS_REAL_E2E=1 且
LLM_API_KEY 齐备才运行，否则整个模块跳过——离线全量测试不触网、不花费真实调用成本。

本模块只验收 writer LLM 的 stream 路径（draft 一发），不触检索通道——故门槛只取
LLM_API_KEY，不要求 VOLCANO_SEARCH_API_KEY / BISHENG_BASE_URL。

真实调用天然非确定性，断言只锚定稳定的外部结构：
- content_delta 帧载荷字段齐全（chapter_id / mode / kind / delta / attempt / sequence）；
- 最终 attempt 的 content 帧拼接 == 终态信封 chapter_text（JsonFieldExtractor
  在真实分块 JSON 上的纯正文抽取正确）；
- 若模型 enable_thinking=1，至少有一帧 kind=thinking；
- attempt / sequence 字段存在且类型正确。
"""

import asyncio
import os
from typing import Any

import pytest
from dotenv import load_dotenv

# 只在显式开启门控时读 .env：离线全量测试不因收集本模块把真实凭据注入环境。
if os.environ.get("HYPOARGUS_REAL_E2E") == "1":
    load_dotenv()

_REQUIRED_ENV = ("LLM_API_KEY",)
_MISSING_ENV = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
_OPTED_IN = os.environ.get("HYPOARGUS_REAL_E2E") == "1"

pytestmark = [
    pytest.mark.real_e2e,
    pytest.mark.skipif(
        not _OPTED_IN or bool(_MISSING_ENV),
        reason=(
            "门控真实流式逐字流 E2E：需 HYPOARGUS_REAL_E2E=1 显式开启"
            + (f"，且补齐凭据 {_MISSING_ENV}" if _MISSING_ENV else "")
        ),
    ),
]

from agents.rewriter_loop.llm_adapter import (  # noqa: E402
    LlmWriterClient,
)
from agents.rewriter_loop.stub import UNIT  # noqa: E402
from domain.events import CONTENT_DELTA  # noqa: E402
from llm.llm_client import default_llm_factory  # noqa: E402

# 真实流式一发上限：真实 LLM 单章 draft 约 30-120 秒，给足余量。
_STREAM_TIMEOUT = 300.0


def _capture_hook() -> tuple[list[tuple[str, dict]], Any]:
    """构造捕获钩子与事件列表：(event_type, payload) 入列。"""
    events: list[tuple[str, dict]] = []
    return events, (lambda etype, payload: events.append((etype, dict(payload))))


def _draft_task() -> dict[str, Any]:
    """最小 draft 任务包：单章单论点、无素材（逐字流只验收 stream 抽取，不验素材角标）。"""
    return {
        "mode": "draft",
        "doc_type": "通用公文",
        "doc_variant": None,
        "chapter_spec": {
            "id": "real-stream-ch1",
            "title": "一、总则",
            "chapter_type": None,
            "points": [{"id": "p-1", "text": "本专业面向智能制造领域培养高素质人才。"}],
            "hypotheses": [
                {
                    "id": "h-1",
                    "text": "智能制造领域对高素质人才有持续需求。",
                    "refute_condition": "",
                }
            ],
        },
        "materials": [],
        "prev_chapter_summary": "",
    }


def test_真实流式_draft逐字流拼接一致且字段齐全():
    """真实 LLM stream 路径一发：content_delta 帧载荷结构正确、拼接 == chapter_text。

    非确定性内容只锚定稳定结构：字段齐全、类型正确、最终 attempt 的 content
    帧拼接与终态信封 chapter_text 逐字一致（JsonFieldExtractor 在真实分块
    JSON 上纯正文抽取正确）。enable_thinking=1 时断言至少一帧 thinking。
    """

    async def main() -> None:
        llm = default_llm_factory(UNIT)
        events, hook = _capture_hook()
        # flush_chars=8 + flush_ms=0：确定性按字符数驱动、小阈值多帧便于结构断言。
        client = LlmWriterClient(
            llm, flush_chars=8, flush_ms=0, event_hook=hook
        )
        task = _draft_task()
        style_prose = "通用公文风格指南片段：公文范式、口语黑名单。"
        # LlmWriterClient._stream_once 在调用方线程同步消费 stream，hook 在同线程
        # 同步触发，无跨线程调度；经 to_thread 避免阻塞事件循环。
        envelope = await asyncio.to_thread(
            client.draft, task, style_prose
        )

        deltas = [p for et, p in events if et == CONTENT_DELTA]
        assert deltas, "真实流式 draft 须产出 content_delta 帧"

        # 字段齐全 + 类型正确。
        for payload in deltas:
            assert payload["unit"] == UNIT
            assert payload["chapter_id"] == "real-stream-ch1"
            assert payload["mode"] == "draft"
            assert payload["kind"] in {"content", "thinking"}
            assert isinstance(payload["delta"], str) and payload["delta"]
            assert isinstance(payload["attempt"], int) and payload["attempt"] >= 1
            assert isinstance(payload["sequence"], int) and payload["sequence"] >= 0

        # 最终 attempt 的 content 帧拼接 == 终态信封 chapter_text。
        final_attempt = max(p["attempt"] for p in deltas)
        final_content = "".join(
            p["delta"]
            for p in deltas
            if p["attempt"] == final_attempt and p["kind"] == "content"
        )
        assert final_content == envelope.chapter_text, (
            "最终 attempt 的 content 帧拼接须与终态信封 chapter_text 逐字一致"
        )
        assert envelope.chapter_text.strip(), "终态信封 chapter_text 非空"

        # sequence 在最终 attempt 内单调递增、从 0 起。
        final_seqs = [p["sequence"] for p in deltas if p["attempt"] == final_attempt]
        assert final_seqs == sorted(final_seqs)
        assert final_seqs[0] == 0

        # 若模型 enable_thinking=1，至少一帧 thinking。
        enable_thinking = llm.metadata.get("enable_thinking") == "1"
        if enable_thinking:
            thinking_deltas = [
                p for p in deltas if p["kind"] == "thinking"
            ]
            assert thinking_deltas, "enable_thinking=1 须产出 thinking 帧"
        # enable_thinking=0 时不断言 thinking（FakeLLM 不产 thinking，真实无思考模型同）。

    asyncio.run(asyncio.wait_for(main(), timeout=_STREAM_TIMEOUT))
