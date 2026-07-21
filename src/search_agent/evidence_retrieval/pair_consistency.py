"""Forward/reverse logical-opposite consistency checks."""
from __future__ import annotations


def check_pair_consistency(forward_verdict: str, reverse_verdict: str) -> dict:
    forward = str(forward_verdict).upper()
    reverse = str(reverse_verdict).upper()
    consistent = (forward == "SUPPORTED" and reverse == "REFUTED") or (forward == "REFUTED" and reverse == "SUPPORTED")
    incomplete = forward == "INCONCLUSIVE" or reverse == "INCONCLUSIVE"
    return {
        "relationship": "logical_opposite",
        "consistency_status": "CONSISTENT" if consistent else "INCOMPLETE" if incomplete else "CONFLICT",
        "conflicts": [] if consistent or incomplete else [f"forward={forward},reverse={reverse}"],
    }


__all__ = ["check_pair_consistency"]
