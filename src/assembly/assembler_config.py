"""上下文装配的压缩阈值配置读取。

四个阈值各自支持在环境变量中独立配置；未设置（或为空字符串）时回落缺省值。

token 预算以字符数近似：本项目面向中文写作场景，中文文本的字符数与
token 数量级接近（一个汉字约一个 token），用字符数做阈值判断足够稳健，
且避免引入分词器依赖。这是有意的近似决策。
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

from domain.env_config import read_positive_int

# 进程启动即把 .env 载入环境变量（不覆盖已有值），使「.env 中独立配置」真实生效。
load_dotenv()


@dataclass(frozen=True)
class AssemblerConfig:
    """上下文装配最终生效的压缩阈值。"""

    summary_chain_max_chars: int
    """摘要链总字符阈值：超过才触发「摘要的摘要」压缩。"""
    summary_digest_max_chars: int
    """压缩后每章摘要保留的字符数。"""
    ledger_keep_rounds: int
    """修订台账保留原文的最近轮数 K。"""
    ledger_digest_max_chars: int
    """更早轮次一句话摘要的字符数。"""


@dataclass(frozen=True)
class BudgetOverride:
    """每份装配配方的专属 token 预算覆盖。

    四个字段与 AssemblerConfig 同名同义；值为 None 的字段沿用全局配置。
    """

    summary_chain_max_chars: int | None = None
    summary_digest_max_chars: int | None = None
    ledger_keep_rounds: int | None = None
    ledger_digest_max_chars: int | None = None


# 各环境变量名与缺省值。
_CONFIG_DEFAULTS: tuple[tuple[str, int], ...] = (
    ("ASSEMBLER_SUMMARY_CHAIN_MAX_CHARS", 800),
    ("ASSEMBLER_SUMMARY_DIGEST_MAX_CHARS", 60),
    ("ASSEMBLER_LEDGER_KEEP_ROUNDS", 2),
    ("ASSEMBLER_LEDGER_DIGEST_MAX_CHARS", 60),
)


def load_assembler_config(env: Mapping[str, str] | None = None) -> AssemblerConfig:
    """读取上下文装配阈值配置，未设置或为空的变量回落缺省值。

    某变量设置了但不是正整数时抛出 ValueError，指明该变量名。
    """
    if env is None:
        env = os.environ

    values = {
        name: read_positive_int(env, name, default) for name, default in _CONFIG_DEFAULTS
    }
    return AssemblerConfig(
        summary_chain_max_chars=values["ASSEMBLER_SUMMARY_CHAIN_MAX_CHARS"],
        summary_digest_max_chars=values["ASSEMBLER_SUMMARY_DIGEST_MAX_CHARS"],
        ledger_keep_rounds=values["ASSEMBLER_LEDGER_KEEP_ROUNDS"],
        ledger_digest_max_chars=values["ASSEMBLER_LEDGER_DIGEST_MAX_CHARS"],
    )
