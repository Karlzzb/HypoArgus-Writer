"""检索引擎包的 ``.env`` 加载工具。

包导入时自动加载包级 ``.env``（见 ``__init__``），支持的配置项：

    LLM_KEY / LLM_BASE_URL / LLM_MODEL   -> 引擎内 OpenAI 兼容 LLM 客户端
    LANGFUSE_PUBLIC_KEY / _SECRET_KEY / _BASE_URL -> Langfuse 追踪

加载只能通过删除 ``.env`` 关闭，绝不覆盖进程环境中已存在的变量；
读取的是与本包同目录的 ``.env``，与调用方工作目录无关。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).with_name(".env")


def load_env() -> None:
    """Load the package-local ``.env`` if present (never overriding real env)."""
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=False)


def env_str(name: str) -> str | None:
    """Return a stripped, non-empty environment value, or ``None``."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


__all__ = ["load_env", "env_str"]
