"""LLM JSON 应答的共享解析工具。

多个主节点的 LLM 调用都要求「只输出 JSON」，解析与顶层类型校验收敛到本模块，
应答不合法时抛含步骤名的 ValueError。
环境变量 LLM_DEBUG_TIMING=1 开启逐次调用的计时日志（步骤名、耗时、出入字符数），
供性能调测定位慢调用。
"""

import json
import os
import time
from typing import Any

from llm.llm_client import LLM, Message

JSON_ONLY_RULE = "只输出 JSON，不要输出任何多余文字、解释或代码围栏。"

# 应答不合法时的有界重试上限（含首次尝试的总次数）。
_MAX_ATTEMPTS = 3

# 计时日志开关环境变量名：取值 1 开启，其余关闭。
_DEBUG_TIMING_ENV = "LLM_DEBUG_TIMING"


def timing_enabled() -> bool:
    """LLM 计时日志是否开启（每次调用现读环境变量，便于运行中切换）。"""
    return os.environ.get(_DEBUG_TIMING_ENV) == "1"


def parse_json(raw: str, step: str) -> Any:
    """剥掉围栏等噪音，从首个 JSON 起始符解析；失败抛含步骤名的 ValueError。

    LLM 常在长文本字段里输出未转义的换行、制表等控制字符，
    严格模式会整体拒收，故用 strict=False 容忍字符串值内的控制字符。
    """
    for index, char in enumerate(raw):
        if char in "[{":
            try:
                value, _ = json.JSONDecoder(strict=False).raw_decode(raw[index:])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"步骤「{step}」的 LLM 应答不是合法 JSON：{exc}"
                ) from None
            return value
    raise ValueError(f"步骤「{step}」的 LLM 应答中找不到 JSON：{raw[:200]!r}")


def invoke_json(
    llm: LLM, step: str, system: str, user: str, expect: type | tuple[type, ...]
) -> Any:
    """执行一次 LLM 调用并解析 JSON，同时校验顶层类型。

    真实 LLM 偶发输出残缺或顶层类型不符的应答；应答不合法时把原始应答
    与解析错误作为纠错反馈追加进对话并有界重试（共 ``_MAX_ATTEMPTS`` 次尝试），
    仍失败才抛错——避免长链路运行因单次应答瑕疵整体失败。
    """
    debug = timing_enabled()
    messages: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error: ValueError | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if debug:
            t0 = time.perf_counter()
            in_chars = sum(len(message["content"]) for message in messages)
            print(
                f"[llm-timing] step={step} start attempt={attempt} in_chars={in_chars}",
                flush=True,
            )
        raw = llm.invoke(messages)
        if debug:
            print(
                f"[llm-timing] step={step} end attempt={attempt}"
                f" dur={time.perf_counter() - t0:.1f}s out_chars={len(raw)}",
                flush=True,
            )
        try:
            payload = parse_json(raw, step)
            if not isinstance(payload, expect):
                expected = "对象" if expect is dict else "数组"
                raise ValueError(
                    f"步骤「{step}」的 LLM 应答顶层必须是 JSON {expected}，"
                    f"实际应答：{raw[:200]!r}"
                )
            return payload
        except ValueError as exc:
            last_error = exc
            messages = messages[:2] + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"你上一次的应答不合法：{exc}\n"
                        f"请修正后重新输出完整应答。{JSON_ONLY_RULE}"
                    ),
                },
            ]
    assert last_error is not None
    raise last_error
