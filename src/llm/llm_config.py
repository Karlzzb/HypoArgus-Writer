"""运行单元 LLM 配置读取与全局回落。

7 个运行单元各自支持在环境变量中以「单元名前缀 + LLM_MODEL / LLM_BASE_URL /
LLM_API_KEY / LLM_ENABLE_THINKING」独立配置；某字段未配置（或为空字符串）时
逐字段回落到无前缀的全局缺省变量。

思考模式（enable_thinking）缺省关闭：结构化 JSON 任务上思考 token 可达可见
输出的数倍，延迟放大 5 倍以上；需要深度推理的单元显式置 1 开启。
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

from domain.units import RUNTIME_UNITS

# 进程启动即把 .env 载入环境变量（不覆盖已有值），使「.env 中独立配置」真实生效。
load_dotenv()

_FIELD_SUFFIXES = ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY")

_THINKING_SUFFIX = "LLM_ENABLE_THINKING"

_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


@dataclass(frozen=True)
class LLMConfig:
    """单个运行单元最终生效的 LLM 配置。"""

    model: str
    base_url: str
    api_key: str
    enable_thinking: bool = False
    """思考模式开关：关闭时调用层显式请求服务商不产生思考 token。"""


def _normalize_base_url(base_url: str) -> str:
    """base_url 止于 OpenAI 兼容根路径：剥掉多余的 /chat/completions 后缀与尾部斜杠。"""
    url = base_url.rstrip("/")
    if url.endswith(_CHAT_COMPLETIONS_SUFFIX):
        url = url[: -len(_CHAT_COMPLETIONS_SUFFIX)]
    return url.rstrip("/")


def _read_bool(env: Mapping[str, str], prefixed_name: str, global_name: str) -> bool:
    """读取布尔开关：前缀变量优先、回落全局变量、都未配置回落 False。

    合法取值 "1"/"0"；其他非空值抛 ValueError 指明变量名。
    """
    for name in (prefixed_name, global_name):
        raw = env.get(name, "").strip()
        if not raw:
            continue
        if raw == "1":
            return True
        if raw == "0":
            return False
        raise ValueError(f"环境变量 {name} 只接受 1 或 0，当前值：{raw!r}")
    return False


def load_llm_config(unit: str, env: Mapping[str, str] | None = None) -> LLMConfig:
    """读取指定运行单元的 LLM 配置，逐字段回落到全局缺省变量。

    任一字段在前缀变量与全局变量中都取不到非空值时抛出 ValueError，指明缺失的变量名。
    """
    if unit not in RUNTIME_UNITS:
        raise ValueError(f"未知运行单元：{unit}，合法取值：{RUNTIME_UNITS}")
    if env is None:
        env = os.environ

    prefix = unit.upper()
    values: dict[str, str] = {}
    for suffix in _FIELD_SUFFIXES:
        prefixed_name = f"{prefix}_{suffix}"
        value = env.get(prefixed_name, "").strip() or env.get(suffix, "").strip()
        if not value:
            raise ValueError(
                f"运行单元 {unit} 缺少 LLM 配置：请设置 {prefixed_name} 或全局缺省 {suffix}"
            )
        values[suffix] = value

    return LLMConfig(
        model=values["LLM_MODEL"],
        base_url=_normalize_base_url(values["LLM_BASE_URL"]),
        api_key=values["LLM_API_KEY"],
        enable_thinking=_read_bool(
            env, f"{prefix}_{_THINKING_SUFFIX}", _THINKING_SUFFIX
        ),
    )
