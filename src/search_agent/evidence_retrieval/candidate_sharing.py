"""High-confidence paragraph/request candidate sharing with provenance."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .schemas import EvidenceCandidate, RetrievalTask
from .query_normalization import extract_numeric_expressions


def _terms(task: RetrievalTask) -> set[str]:
    text = " ".join([task.target_text, task.boundary or "", *task.required_slots])
    return {item.casefold() for item in re.findall(r"[\u3400-\u9fffA-Za-z0-9]{2,}", text) if item not in {"是否", "以及", "并且"}}


def match_candidate_to_task(candidate: EvidenceCandidate, task: RetrievalTask) -> tuple[bool, list[str]]:
    body = f"{candidate.title} {candidate.content}"
    reasons: list[str] = []
    terms = _terms(task)
    if terms and any(term in body for term in terms):
        reasons.append("same_subject_or_metric")
    if any(value.replace(" ", "") in body for value in extract_numeric_expressions(task.target_text)):
        reasons.append("numeric_match")
    years = re.findall(r"(?:19|20)\d{2}", task.target_text)
    if years and any(year in body for year in years):
        reasons.append("same_year")
    return bool(reasons), reasons


def share_candidates(candidates: Iterable[EvidenceCandidate], tasks: Iterable[RetrievalTask]) -> dict[str, list[EvidenceCandidate]]:
    output: dict[str, list[EvidenceCandidate]] = defaultdict(list)
    task_list = list(tasks)
    for candidate in candidates:
        matched: list[str] = []
        reasons_by_task: dict[str, list[str]] = {}
        for task in task_list:
            if task.task_id == candidate.task_id:
                matched.append(task.task_id)
                reasons_by_task[task.task_id] = ["original_task"]
                continue
            ok, reasons = match_candidate_to_task(candidate, task)
            if ok:
                matched.append(task.task_id)
                reasons_by_task[task.task_id] = reasons
        for task_id in matched:
            metadata = {
                **candidate.metadata,
                "source_candidate_id": candidate.candidate_id,
                "original_task_id": candidate.task_id,
                "matched_task_ids": matched,
                "match_reasons": reasons_by_task,
            }
            output[task_id].append(candidate.model_copy(update={"task_id": task_id, "metadata": metadata}))
    return dict(output)


__all__ = ["match_candidate_to_task", "share_candidates"]
