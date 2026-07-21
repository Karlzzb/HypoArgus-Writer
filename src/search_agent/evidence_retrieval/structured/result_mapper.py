"""Map StructuredToolResult data rows into normal Judge candidates."""
from __future__ import annotations

from ..schemas import EvidenceCandidate, SourceRef, SourceType, stable_json_hash
from .contracts import StructuredToolResult


def map_structured_tool_result(result: StructuredToolResult, task_by_id: dict[str, object]) -> list[EvidenceCandidate]:
    if result.status != "SUCCESS" or not result.rows:
        return []
    candidates: list[EvidenceCandidate] = []
    # The Judge and downstream citation builder need a readable fact surface,
    # not a JSON transport envelope.  Preserve field labels so every quoted
    # value can still be audited against the authoritative row.
    lines: list[str] = []
    for index, row in enumerate(result.rows, 1):
        ordered = [name for name in result.columns if name in row]
        ordered.extend(name for name in row if name not in ordered)
        facts = "；".join(f"{name}={row[name]}" for name in ordered if row[name] is not None)
        if facts:
            lines.append(f"记录{index}：{facts}。")
    content = "\n".join(lines)
    if not content:
        return []
    for task_id in result.target_task_ids:
        if task_id not in task_by_id:
            continue
        candidates.append(EvidenceCandidate(
            candidate_id=f"structured-{stable_json_hash([task_id, result.tool_call_id, result.scenario_key])[:20]}",
            task_id=task_id,
            source_type=SourceType.STRUCTURED,
            source_name=result.tool_name,
            source_ref=SourceRef(
                scenario_name=result.scenario_key,
                record_id=result.tool_call_id,
                dataset_id=result.dataset_id,
                query_execution_id=result.query_execution_id,
                query_params_hash=stable_json_hash(result.arguments),
            ),
            title=result.query_summary,
            content=content,
            initial_score=1.0,
            metadata={
                "tool_call_id": result.tool_call_id,
                "scenario_key": result.scenario_key,
                "dataset_id": result.dataset_id,
                "query_execution_id": result.query_execution_id,
                "columns": result.columns,
                "row_count": result.row_count,
                "matched_task_ids": result.target_task_ids,
            },
        ))
    return candidates


__all__ = ["map_structured_tool_result"]
