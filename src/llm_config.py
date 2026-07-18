"""运行单元 LLM 配置读取与全局回落。

7 个运行单元各自支持在环境变量中以「单元名前缀 + LLM_MODEL / LLM_BASE_URL / LLM_API_KEY」
独立配置；某字段未配置（或为空字符串）时逐字段回落到无前缀的全局缺省变量。
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

# 进程启动即把 .env 载入环境变量（不覆盖已有值），使「.env 中独立配置」真实生效。
load_dotenv()

RUNTIME_UNITS: tuple[str, ...] = (
    "framework_orchestrator",
    "reference_orchestrator",
    "writing_orchestrator",
    "citation_validator",
    "human_review_gate",
    "search_agent",
    "rewriter_loop",
)

_FIELD_SUFFIXES = ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY")

_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


@dataclass(frozen=True)
class LLMConfig:
    """单个运行单元最终生效的 LLM 配置。"""

    model: str
    base_url: str
    api_key: str


def _normalize_base_url(base_url: str) -> str:
    """base_url 止于 OpenAI 兼容根路径：剥掉多余的 /chat/completions 后缀与尾部斜杠。"""
    url = base_url.rstrip("/")
    if url.endswith(_CHAT_COMPLETIONS_SUFFIX):
        url = url[: -len(_CHAT_COMPLETIONS_SUFFIX)]
    return url.rstrip("/")


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
    )
