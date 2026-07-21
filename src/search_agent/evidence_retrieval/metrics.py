"""Request/task metrics with event-loop-safe atomic updates."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class CallMetric:
    count: int = 0
    elapsed_ms: int = 0
    error_count: int = 0
    timeout_count: int = 0
    success_count: int = 0


class RequestMetricsCollector:
    """Synchronous updates are atomic between asyncio suspension points."""

    def __init__(self):
        self.started_at = time.monotonic()
        self.calls: dict[str, CallMetric] = defaultdict(CallMetric)
        self.tasks: dict[str, dict[str, Any]] = {}

    def start_task(self, task_id: str) -> float:
        started = time.monotonic()
        self.tasks[task_id] = {"started_at": started, "elapsed_ms": 0, "stage_timings_ms": {}, "timeouts": []}
        return started

    def finish_task(self, task_id: str) -> int:
        row = self.tasks.setdefault(task_id, {"started_at": time.monotonic(), "stage_timings_ms": {}, "timeouts": []})
        row["elapsed_ms"] = int((time.monotonic() - row["started_at"]) * 1000)
        return row["elapsed_ms"]

    def record_stage(self, task_id: str, stage: str, elapsed_ms: int, *, timeout: bool = False) -> None:
        row = self.tasks.setdefault(task_id, {"started_at": time.monotonic(), "elapsed_ms": 0, "stage_timings_ms": {}, "timeouts": []})
        timings = row["stage_timings_ms"]
        timings[stage] = timings.get(stage, 0) + max(0, int(elapsed_ms))
        if timeout:
            row["timeouts"].append(stage)

    def record_call(self, name: str, elapsed_ms: int, *, error: bool = False, timeout: bool = False) -> None:
        metric = self.calls[name]
        metric.count += 1
        metric.elapsed_ms += max(0, int(elapsed_ms))
        metric.error_count += int(error)
        metric.timeout_count += int(timeout)
        metric.success_count += int(not error)

    def begin_call(self, name: str) -> float:
        self.calls[name].count += 1
        return time.monotonic()

    def end_call(self, name: str, started_at: float, *, error: bool = False, timeout: bool = False) -> None:
        metric = self.calls[name]
        metric.elapsed_ms += max(0, int((time.monotonic() - started_at) * 1000))
        metric.error_count += int(error)
        metric.timeout_count += int(timeout)
        metric.success_count += int(not error)

    def snapshot(self) -> dict[str, Any]:
        elapsed = int((time.monotonic() - self.started_at) * 1000)
        task_elapsed = [int(row.get("elapsed_ms", 0)) for row in self.tasks.values()]
        return {
            "total_elapsed_ms": elapsed,
            "critical_path_ms": max(task_elapsed, default=0),
            "calls": {name: vars(metric).copy() for name, metric in self.calls.items()},
            "call_counts": {name: metric.count for name, metric in self.calls.items()},
            "task_metrics": {key: {k: v for k, v in value.items() if k != "started_at"} for key, value in self.tasks.items()},
        }


class TaskMetricsCollector:
    def __init__(self, request: RequestMetricsCollector, task_id: str):
        self.request = request
        self.task_id = task_id
        self.request.start_task(task_id)

    def stage(self, name: str, started_at: float, *, timeout: bool = False) -> int:
        elapsed = int((time.monotonic() - started_at) * 1000)
        self.request.record_stage(self.task_id, name, elapsed, timeout=timeout)
        return elapsed

    def finish(self) -> int:
        return self.request.finish_task(self.task_id)
