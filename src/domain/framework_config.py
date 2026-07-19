"""论证体系数量上限与并发度配置读取。

各配置项支持在环境变量中独立配置；未设置（或为空字符串）时回落缺省值。
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

from domain.env_config import read_positive_int

# 进程启动即把 .env 载入环境变量（不覆盖已有值），使「.env 中独立配置」真实生效。
load_dotenv()


@dataclass(frozen=True)
class FrameworkLimits:
    """论证体系最终生效的数量上限与假说生成并发度。"""

    max_points_per_chapter: int
    max_hypotheses_per_point: int
    max_hypotheses_total: int
    max_concurrent_chapters: int = 4


# 各环境变量名与缺省值。
_LIMIT_DEFAULTS: tuple[tuple[str, int], ...] = (
    ("FRAMEWORK_MAX_POINTS_PER_CHAPTER", 4),
    ("FRAMEWORK_MAX_HYPOTHESES_PER_POINT", 3),
    ("FRAMEWORK_MAX_HYPOTHESES_TOTAL", 60),
    ("FRAMEWORK_MAX_CONCURRENT_CHAPTERS", 4),
)


def load_framework_limits(env: Mapping[str, str] | None = None) -> FrameworkLimits:
    """读取论证体系数量上限配置，未设置或为空的变量回落缺省值。

    某变量设置了但不是正整数时抛出 ValueError，指明该变量名。
    """
    if env is None:
        env = os.environ

    values = {
        name: read_positive_int(env, name, default) for name, default in _LIMIT_DEFAULTS
    }
    return FrameworkLimits(
        max_points_per_chapter=values["FRAMEWORK_MAX_POINTS_PER_CHAPTER"],
        max_hypotheses_per_point=values["FRAMEWORK_MAX_HYPOTHESES_PER_POINT"],
        max_hypotheses_total=values["FRAMEWORK_MAX_HYPOTHESES_TOTAL"],
        max_concurrent_chapters=values["FRAMEWORK_MAX_CONCURRENT_CHAPTERS"],
    )
