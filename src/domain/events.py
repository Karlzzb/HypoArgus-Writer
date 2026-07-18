"""事件词汇：跨层共享的事件类型常量与挂钩类型。

子智能体适配层在调用前后发事件，服务层订阅渲染；
双方只共享本模块的中立契约，互不导入对方实现。
"""

from collections.abc import Callable
from typing import Any

SUBAGENT_START = "subagent_start"
SUBAGENT_END = "subagent_end"

EventHook = Callable[[str, dict[str, Any]], None]
"""事件挂钩：(事件类型, 载荷)；由调用方注入，缺省空实现。"""


def noop_hook(event_type: str, payload: dict[str, Any]) -> None:
    """缺省事件挂钩：不做任何事。"""
