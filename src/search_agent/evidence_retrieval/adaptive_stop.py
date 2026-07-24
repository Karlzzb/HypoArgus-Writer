"""低边际收益反向检索的保守止损策略。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReverseStopDecision:
    skip: bool
    reason: str
    evidence: dict[str, int | bool]


def decide_reverse_stop(*, enabled: bool, prior_attempts: int, prior_admitted: int, minimum_attempts: int, unresolved: bool, high_priority: bool, coverage_protected: bool) -> ReverseStopDecision:
    """仅在连续零收益后止损，受保护任务始终继续。"""
    evidence = {"prior_attempts": prior_attempts, "prior_admitted": prior_admitted, "minimum_attempts": minimum_attempts, "unresolved": unresolved, "high_priority": high_priority, "coverage_protected": coverage_protected}
    if not enabled:
        return ReverseStopDecision(False, "DISABLED", evidence)
    if unresolved or high_priority or coverage_protected:
        return ReverseStopDecision(False, "PROTECTED_REQUIRED_PATH", evidence)
    if prior_attempts < minimum_attempts:
        return ReverseStopDecision(False, "INSUFFICIENT_HISTORY", evidence)
    if prior_admitted:
        return ReverseStopDecision(False, "MARGINAL_YIELD_NONZERO", evidence)
    return ReverseStopDecision(True, "CONSECUTIVE_ZERO_ADMISSION_REVERSE_ATTEMPTS", evidence)
