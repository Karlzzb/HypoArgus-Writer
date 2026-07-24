"""Shared citation material policy for writer-visible agent prompts."""

from typing import Any

from agents.contracts import MaterialPayload
from domain.state import CITABLE_VERDICTS

EMPTY_CITABLE_MATERIALS_INSTRUCTION = (
    "本章无可引素材（素材池为空）：正文不得出现任何 `[...]` 角标，"
    "不得生成 `[1]` 等最终展示编号，不得列参考文献列表，不得臆造素材 id 或来源。"
)
"""Prompt contract for chapters with no citable material IDs."""


def citable_materials(task: dict[str, Any]) -> list[MaterialPayload]:
    """Return current-chapter materials that may be shown to writer-visible agents.

    The visible citation pool is strictly the current chapter's hypothesis-bound
    pass/inconclusive materials. If the task lacks a valid chapter hypothesis list,
    return an empty pool rather than risk leaking unrelated material IDs.
    """
    hypotheses = task.get("chapter_spec", {}).get("hypotheses", [])
    if not isinstance(hypotheses, list) or not hypotheses:
        return []
    hypothesis_ids = {
        hypothesis["id"]
        for hypothesis in hypotheses
        if isinstance(hypothesis, dict) and isinstance(hypothesis.get("id"), str)
    }
    if not hypothesis_ids:
        return []
    return [
        material
        for material in task["materials"]
        if material["verdict"] in CITABLE_VERDICTS
        and material["hypothesis_id"] in hypothesis_ids
    ]
