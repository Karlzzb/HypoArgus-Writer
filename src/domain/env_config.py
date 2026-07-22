"""整数环境变量的共享读取逻辑。

多个配置模块（论证体系数量上限、终审重试上限、章级重写次数上限）都遵循
同一约定：未设置或为空字符串回落缺省值，设置了但越界（非正 / 负）抛 ValueError
并指明变量名。正整数与非负整数两种口径分列（后者允许 0，用于「可关闭」的开关型上限）。
"""

from collections.abc import Mapping


def read_positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    """读取单个正整数变量：空值回落缺省，非正整数抛 ValueError 并指明变量名。"""
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"环境变量 {name} 必须是正整数，当前值：{raw!r}") from None
    if value <= 0:
        raise ValueError(f"环境变量 {name} 必须是正整数，当前值：{raw!r}")
    return value


def read_nonnegative_int(env: Mapping[str, str], name: str, default: int) -> int:
    """读取单个非负整数变量：空值回落缺省，负数或非整数抛 ValueError 并指明变量名。

    与 ``read_positive_int`` 的唯一区别是允许 0——用于「可关闭」的上限
    （章级重写次数上限设 0 即关掉评审重写、只保留纯首写）。
    """
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"环境变量 {name} 必须是非负整数，当前值：{raw!r}") from None
    if value < 0:
        raise ValueError(f"环境变量 {name} 必须是非负整数，当前值：{raw!r}")
    return value
