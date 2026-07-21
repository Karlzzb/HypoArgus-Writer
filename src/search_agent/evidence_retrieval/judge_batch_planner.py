"""Token-aware, lossless Batch Judge planning."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .schemas import EvidenceCandidate, PreparedContext, RetrievalTask


JudgeGroup = tuple[RetrievalTask, list[EvidenceCandidate], PreparedContext]


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate for mixed Chinese/ASCII prompts."""
    value = text or ""
    cjk = len(re.findall(r"[\u3400-\u9fff]", value))
    non_cjk = max(0, len(value) - cjk)
    return cjk + math.ceil(non_cjk / 4)


def compact_group_shape(group: JudgeGroup, candidate_max_chars: int) -> dict[str, Any]:
    """Return the compact shape used for planning (content length is bounded)."""
    task, rows, context = group
    return {
        "task_id": task.task_id,
        "target": task.target_text,
        "context": context.paragraph_text,
        "boundary": task.boundary,
        "required_slots": task.required_slots,
        "candidates": [
            {
                "candidate_id": row.candidate_id,
                "source": row.source_type.value,
                "title": row.title,
                "content": (row.content or "")[:candidate_max_chars],
            }
            for row in rows
        ],
    }


def estimate_groups(groups: list[JudgeGroup], candidate_max_chars: int) -> tuple[int, int]:
    payload = {"tasks": [compact_group_shape(group, candidate_max_chars) for group in groups]}
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return len(serialized), estimate_tokens(serialized)


@dataclass(slots=True)
class JudgeBatch:
    batch_id: str
    groups: list[JudgeGroup]
    task_ids: list[str]
    candidate_ids: list[str]
    candidate_keys: list[tuple[str, str]]
    candidate_count: int
    input_chars: int
    estimated_input_tokens: int
    expected_output_tokens: int
    over_token_limit: bool = False


@dataclass(slots=True)
class JudgeBatchPlan:
    batches: list[JudgeBatch]
    input_candidate_count: int
    batched_candidate_count: int
    duplicate_candidate_count: int
    missing_candidate_count: int
    candidate_to_batch: dict[tuple[str, str], str] = field(default_factory=dict)


class JudgeBatchPlanner:
    """Keep task candidates together when possible and never silently drop one."""

    def __init__(
        self,
        *,
        max_tasks: int,
        max_candidates: int,
        max_input_tokens: int,
        candidate_max_chars: int,
        expected_output_tokens_per_candidate: int,
    ):
        self.max_tasks = max_tasks
        self.max_candidates = max_candidates
        self.max_input_tokens = max_input_tokens
        self.candidate_max_chars = candidate_max_chars
        self.expected_output_tokens_per_candidate = expected_output_tokens_per_candidate

    def _fits(self, groups: list[JudgeGroup]) -> bool:
        task_count = len({task.task_id for task, _, _ in groups})
        candidate_count = sum(len(rows) for _, rows, _ in groups)
        _, tokens = estimate_groups(groups, self.candidate_max_chars)
        return (
            task_count <= self.max_tasks
            and candidate_count <= self.max_candidates
            and tokens <= self.max_input_tokens
        )

    def _split_task(self, group: JudgeGroup) -> list[JudgeGroup]:
        task, rows, context = group
        if not rows:
            return []
        pieces: list[JudgeGroup] = []
        current: list[EvidenceCandidate] = []
        for candidate in rows:
            proposed = (task, [*current, candidate], context)
            if current and not self._fits([proposed]):
                pieces.append((task, current, context))
                current = [candidate]
            else:
                current.append(candidate)
        if current:
            pieces.append((task, current, context))
        return pieces

    def plan(self, groups: list[JudgeGroup], *, batch_id_prefix: str = "judge-batch") -> JudgeBatchPlan:
        input_keys = [
            (task.task_id, candidate.candidate_id)
            for task, rows, _ in groups for candidate in rows
        ]
        duplicate_count = len(input_keys) - len(set(input_keys))
        if duplicate_count:
            raise ValueError(f"duplicate Judge candidate keys: {duplicate_count}")

        pieces = [piece for group in groups for piece in self._split_task(group)]
        raw_batches: list[list[JudgeGroup]] = []
        current: list[JudgeGroup] = []
        for piece in pieces:
            proposed = [*current, piece]
            if current and not self._fits(proposed):
                raw_batches.append(current)
                current = [piece]
            else:
                current = proposed
        if current:
            raw_batches.append(current)

        batches: list[JudgeBatch] = []
        candidate_to_batch: dict[tuple[str, str], str] = {}
        for index, batch_groups in enumerate(raw_batches, start=1):
            batch_id = f"{batch_id_prefix}-{index:03d}"
            keys = [
                (task.task_id, candidate.candidate_id)
                for task, rows, _ in batch_groups for candidate in rows
            ]
            input_chars, input_tokens = estimate_groups(batch_groups, self.candidate_max_chars)
            for key in keys:
                candidate_to_batch[key] = batch_id
            batches.append(JudgeBatch(
                batch_id=batch_id,
                groups=batch_groups,
                task_ids=list(dict.fromkeys(task.task_id for task, _, _ in batch_groups)),
                candidate_ids=[candidate_id for _, candidate_id in keys],
                candidate_keys=keys,
                candidate_count=len(keys),
                input_chars=input_chars,
                estimated_input_tokens=input_tokens,
                expected_output_tokens=len(keys) * self.expected_output_tokens_per_candidate,
                over_token_limit=input_tokens > self.max_input_tokens,
            ))

        batched_keys = [key for batch in batches for key in batch.candidate_keys]
        missing = len(set(input_keys) - set(batched_keys))
        duplicates = len(batched_keys) - len(set(batched_keys))
        if missing or duplicates or set(input_keys) != set(batched_keys):
            raise RuntimeError(
                f"lossless Judge planning failed: missing={missing}, duplicates={duplicates}"
            )
        return JudgeBatchPlan(
            batches=batches,
            input_candidate_count=len(input_keys),
            batched_candidate_count=len(batched_keys),
            duplicate_candidate_count=duplicates,
            missing_candidate_count=missing,
            candidate_to_batch=candidate_to_batch,
        )


__all__ = [
    "JudgeBatch", "JudgeBatchPlan", "JudgeBatchPlanner", "JudgeGroup",
    "compact_group_shape", "estimate_groups", "estimate_tokens",
]
