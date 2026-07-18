"""LLM JSON 应答的共享解析工具。

多个主节点的 LLM 调用都要求「只输出 JSON」，解析与顶层类型校验收敛到本模块，
应答不合法时抛含步骤名的 ValueError。
"""

import json
from typing import Any

from llm.llm_client import LLM

JSON_ONLY_RULE = "只输出 JSON，不要输出任何多余文字、解释或代码围栏。"


def parse_json(raw: str, step: str) -> Any:
    """剥掉围栏等噪音，从首个 JSON 起始符解析；失败抛含步骤名的 ValueError。"""
    for index, char in enumerate(raw):
        if char in "[{":
            try:
                value, _ = json.JSONDecoder().raw_decode(raw[index:])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"步骤「{step}」的 LLM 应答不是合法 JSON：{exc}"
                ) from None
            return value
    raise ValueError(f"步骤「{step}」的 LLM 应答中找不到 JSON：{raw[:200]!r}")


def invoke_json(llm: LLM, step: str, system: str, user: str, expect: type) -> Any:
    """执行一次 LLM 调用并解析 JSON，同时校验顶层类型。"""
    payload = parse_json(
        llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        ),
        step,
    )
    if not isinstance(payload, expect):
        expected = "对象" if expect is dict else "数组"
        raise ValueError(f"步骤「{step}」的 LLM 应答顶层必须是 JSON {expected}")
    return payload
