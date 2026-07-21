"""事件信封：graph_event 可视化通道的统一事件封装。

信封字段与事件类型枚举为设计定稿（字段即决策），对外契约见 docs/api.md。
事件信封只携带元数据与轻量载荷，正文全文绝不入信封（state_snapshot 同受此约束）。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

# 七大类别展开后的 12 个事件类型完整枚举
# （节点启停与门禁、分支流转、循环迭代、子智能体调用、状态快照、模型配置、进度）。
GRAPH_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "node_start",
        "node_end",
        "node_error",
        "gate_blocked",
        "gate_resumed",
        "branch_taken",
        "loop_iteration",
        "subagent_start",
        "subagent_end",
        "state_snapshot",
        "llm_config_used",
        "progress",
    }
)


class EventEnvelope(BaseModel):
    """graph_event 通道单条事件的信封结构。

    parent_id 指向父事件的 event_id，前端据此拼接完整执行拓扑。
    session_id 由外部调用方传入，本系统只透传不鉴权。
    """

    model_config = {"frozen": True}

    event_id: str
    trace_id: str
    session_id: str
    thread_id: str
    parent_id: str | None
    ts: str
    type: str
    unit: str
    payload: dict[str, Any]


def new_envelope(
    *,
    type: str,
    unit: str,
    payload: dict[str, Any],
    trace_id: str,
    session_id: str,
    thread_id: str,
    parent_id: str | None = None,
) -> EventEnvelope:
    """构造一条新事件：自动生成全局唯一 event_id 与 UTC ISO8601 时间戳。"""
    if type not in GRAPH_EVENT_TYPES:
        raise ValueError(f"未知事件类型：{type}")
    return EventEnvelope(
        event_id=uuid.uuid4().hex,
        trace_id=trace_id,
        session_id=session_id,
        thread_id=thread_id,
        parent_id=parent_id,
        ts=datetime.now(timezone.utc).isoformat(),
        type=type,
        unit=unit,
        payload=payload,
    )
