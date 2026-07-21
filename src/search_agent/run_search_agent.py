"""Single production CLI for the SearchAgent public contract.

Uses SearchAgentRuntime — no duplicated config/dependency/graph assembly.

Usage:
    python -u run_search_agent.py --input manual_inputs/v12_embodied_intelligence_4task.json --structured-intent-llm --evidence-judge-llm
    python -u run_search_agent.py --input ... --structured-intent-llm --evidence-judge-llm --output manual_outputs/output.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from search_agent import SearchAgentRuntime
from search_agent.evidence_retrieval.cli_io import emit_public_json


ROOT = Path(__file__).resolve().parent


async def run(
    input_path: Path,
    *,
    output_path: Path | None,
    structured_intent_llm_enabled: bool,
    evidence_judge_llm_enabled: bool,
) -> dict:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    runtime = SearchAgentRuntime.from_env(
        structured_intent_llm_enabled=structured_intent_llm_enabled or None,
        evidence_judge_llm_enabled=evidence_judge_llm_enabled or None,
    )
    try:
        result = await runtime.ainvoke(raw)
    finally:
        await runtime.aclose()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Output written to: {output_path.resolve()}", file=sys.stderr)

    emit_public_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the production SearchAgent graph")
    parser.add_argument(
        "--input",
        default=str(ROOT / "manual_inputs" / "v12_embodied_intelligence_4task.json"),
        help="search-agent-input/v1 JSON file",
    )
    parser.add_argument("--output", help="optional local file containing only search-agent-output/v1")
    parser.add_argument("--structured-intent-llm", action="store_true",
                        help="enable Structured Intent LLM (select Doris Function Tools)")
    parser.add_argument("--evidence-judge-llm", action="store_true",
                        help="enable Evidence Judge LLM (StructuredLLMEvidenceJudge)")
    args = parser.parse_args()

    try:
        asyncio.run(run(
            Path(args.input).resolve(),
            output_path=Path(args.output).resolve() if args.output else None,
            structured_intent_llm_enabled=args.structured_intent_llm,
            evidence_judge_llm_enabled=args.evidence_judge_llm,
        ))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
