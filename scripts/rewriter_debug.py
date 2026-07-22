#!/usr/bin/env python
"""rewriter_loop 独立调测脚本：绕开主图直接驱动真实现，供提示词与风格规则调优。

缺省为空转模式（零成本、确定性）：真适配器 ``LlmWriterClient`` + ``FakeLLM``
固定信封应答，走真实 JSON-in-text 解析路径；空转自审固定报一条「派生未标」，
使修订环节在全流程与 --step revise-fix 下均确定可达。
加 --real 后按 rewriter_loop 单元的环境配置（REWRITER_LOOP_* 前缀回落全局）调真实模型。

用法：
  uv run python scripts/rewriter_debug.py                        # 空转全流程
  uv run python scripts/rewriter_debug.py --step lint            # 跑到校验环节停
  uv run python scripts/rewriter_debug.py --mode revise          # 定向改写模式
  uv run python scripts/rewriter_debug.py --task scripts/rewriter_task.sample.json
  uv run python scripts/rewriter_debug.py --real                 # 真实模型（有花费）

--step 取值与内部环节的对应（跑到该环节停并打印中间产物；缺省整跑）：
  write=起草（一次 draft/revise 写作调用）  lint=风格校验
  audit=引用自审  revise-fix=违规修订（恰好一次修一次调用）

整跑（无 --step）经 ``make_rewriter_loop`` 工厂走与主图完全一致的编排链路，
进度事件（subagent_start/end 与各 step 的 SUBAGENT_PROGRESS）直接打印到终端；
--step 模式绕开编排、逐环节直调 LLM 注入点客户端与校验器，便于观察单环节产物。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# audit_issues_to_violations 是编排层公开导出的自审折叠函数；调测脚本直接复用
# 以保证 --step revise-fix 的违规口径与真编排零漂移（只读消费，不改 src）。
from agents.rewriter_loop import (  # noqa: E402
    UNIT,
    LlmWriterClient,
    Violation,
    audit_issues_to_violations,
    audit_items_for,
    lint,
    load_prose,
    make_rewriter_loop,
)
from agents.rewriter_loop.writer_client import WriterEnvelope, citable_materials  # noqa: E402
from domain.doc_types import carried_doc_facts, tier_from_variant  # noqa: E402
from llm.llm_client import LLM, FakeLLM, default_llm_factory  # noqa: E402

# 缺省样例任务包路径：按脚本自身位置解析，任意 cwd 下均可直接运行。
# draft 上下文与 revise 字段（current_text / revision_note）齐备，两种模式皆可空转。
_DEFAULT_TASK_PATH = Path(__file__).resolve().parent / "rewriter_task.sample.json"

# 任务包必备键（编排链路真正依赖的字段；mode 可被 --mode 覆盖故单独处理，
# 文种字段缺失时与真编排同口径落通用公文兑底，不在此强制）。
_REQUIRED_TASK_KEYS = ("chapter_spec", "materials", "prev_chapter_summary")
_REQUIRED_SPEC_KEYS = ("id", "title", "points", "hypotheses")


def print_hook(event_type: str, payload: dict[str, Any]) -> None:
    """打印型事件挂钩：整跑时把进度事件流实时落到终端，无需 SSE 服务。"""
    print(f"[事件] {event_type}：{json.dumps(payload, ensure_ascii=False)}")


def load_task(path: str | None) -> dict[str, Any]:
    """读取任务包 JSON（缺省用随包样例文件），并做最小契约校验给出可读报错。"""
    if path is None:
        print(f"[任务包] 未指定 --task，使用样例任务包：{_DEFAULT_TASK_PATH}")
        resolved = _DEFAULT_TASK_PATH
    else:
        print(f"[任务包] 读取：{path}")
        resolved = Path(path)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"任务包须为 JSON 对象，当前为 {type(raw).__name__}")
    task = raw
    missing = [k for k in _REQUIRED_TASK_KEYS if k not in task]
    if missing:
        raise SystemExit(f"任务包缺少必备字段：{missing}（契约见 agents/contracts.py RewriteTask）")
    spec = task["chapter_spec"]
    spec_missing = [k for k in _REQUIRED_SPEC_KEYS if k not in spec]
    if spec_missing:
        raise SystemExit(f"chapter_spec 缺少必备字段：{spec_missing}")
    return task


def build_fake_llm(task: dict[str, Any]) -> FakeLLM:
    """按任务包构造确定性 FakeLLM：应答按提示词内容键控，与调用顺序解耦。

    写作应答正文取自真实现契约测试的已知干净文本（不触发 lint 规则），
    pass 素材角标逐条落位；自审固定报一条「派生未标」，确定性触发一次修订。
    """
    materials = citable_materials(task)
    markers = "".join(f"[{m['id']}]" for m in materials)
    clean_sentence = "本专业面向智能制造领域培养高素质人才。"
    if task["mode"] == "revise":
        # 附注口径对齐 rewriter_loop 打桩（stub.py）：用户指令与规则违规
        # 修改指导逐条落「（修订落实：…）」附注，避免两处口径分叉。
        note = task.get("revision_note") or {}
        parts: list[str] = []
        directives = (note.get("user_directives") or "").strip()
        if directives:
            parts.append(f"（修订落实：{directives}）")
        for entry in note.get("rule_violations", []):
            parts.append(f"（修订落实：{entry['guidance']}）")
        body = f"{task.get('current_text', '')}{''.join(parts)}"
    else:
        prev = task["prev_chapter_summary"]
        lead = f"承接上文：{prev}" if prev else ""
        body = f"{lead}{clean_sentence}{markers}"

    def envelope_json(text: str, summary: str) -> str:
        return json.dumps(
            {"chapter_text": text, "chapter_summary": summary}, ensure_ascii=False
        )

    audit_issues: list[dict[str, str]] = []
    if materials:
        audit_issues = [
            {
                "item": "unmarked_derived_content",
                "material_id": materials[0]["id"],
                "excerpt": clean_sentence.rstrip("。"),
            }
        ]
    return FakeLLM(
        responses=[envelope_json(body, "一行公文摘要（空转样例）")],
        keyed_responses={
            # 自审提示词固定携带【章节自审】标签；修一次提示词固定携带违规清单导语。
            # 第二条空裁决供修后复检的自审（ADR-0004）消费：复检出清、终态干净。
            "【章节自审】": [
                json.dumps({"issues": audit_issues}, ensure_ascii=False),
                json.dumps({"issues": []}, ensure_ascii=False),
            ],
            "检出以下违规": [
                envelope_json(f"{body}（已按违规清单修订）", "修订后一行公文摘要（空转样例）")
            ],
        },
    )


def _print_envelope(label: str, envelope: WriterEnvelope) -> None:
    print(f"\n=== {label} ===")
    print(
        f"attempts={envelope.attempts} degraded={envelope.degraded}"
        f" 正文 {len(envelope.chapter_text)} 字符"
    )
    print(envelope.chapter_text)
    print(f"[摘要] {envelope.chapter_summary}")


def _print_violations(violations: list[Violation]) -> None:
    if not violations:
        print("（无违规）")
    for v in violations:
        print(f"- [{v.rule}] {v.message}")


def run_stepwise(
    client: LlmWriterClient, task: dict[str, Any], doc_type: str, doc_variant: str | None, step: str
) -> None:
    """逐环节直调 LLM 注入点与校验器，跑到 ``step`` 指定环节停并打印中间产物。

    环节顺序与真编排（writer.make_writer_run）一致：起草 → lint → 自审 → 修一次；
    自审折叠违规复用编排层同一折叠函数，保证违规口径零漂移。
    与真编排的差异：不产出 self_check 折叠结论（那是整跑的产物）；revise 起草
    不做「现有正文预 lint 并入」（ADR-0004 的合并属整跑链路，--step 逐环节独立观察）；
    修一次后不做修后复检（同理，复检结论看整跑的 self_check）。
    """
    mode = task["mode"]
    style_prose = load_prose(doc_type)

    # 环节一：起草（draft/revise 共用「写作调用」语义）。
    if mode == "revise":
        envelope = client.revise(task, style_prose)
    else:
        envelope = client.draft(task, style_prose)
    _print_envelope("起草（write）" if mode == "draft" else "定向改写（write）", envelope)
    if step == "write":
        return
    if not envelope.chapter_text.strip():
        print("\n正文为空（写作退化），后续环节不执行——与真编排的短路口径一致。")
        return

    # 环节二：风格校验（纯函数，参数口径与真编排一致）。
    spec = task["chapter_spec"]
    materials = citable_materials(task)
    violations = lint(
        envelope.chapter_text,
        doc_type,
        doc_variant,
        materials=materials,
        hypotheses=spec["hypotheses"],
    )
    print(f"\n=== 校验（lint）：{len(violations)} 条违规 ===")
    _print_violations(violations)
    if step == "lint":
        return

    # 环节三：LLM 自审（适用裁决项为空则跳过，与真编排口径一致，ADR-0005 按文种分派）。
    if audit_items_for(doc_type, has_materials=bool(materials)):
        audit = client.audit(envelope.chapter_text, task)
        print(
            f"\n=== 自审（audit）：{len(audit.issues)} 条违规"
            f"（attempts={audit.attempts} degraded={audit.degraded}）==="
        )
        for issue in audit.issues:
            source = f"素材 {issue.material_id}" if issue.material_id else (issue.label or issue.item)
            print(f"- {source}：{issue.excerpt}")
        violations.extend(audit_issues_to_violations(audit.issues))
    else:
        print("\n=== 自审（audit）：无适用裁决项，跳过（与真编排口径一致）===")
    if step == "audit":
        return

    # 环节四：违规修订（恰好一次修一次；无违规则不触发）。
    if not violations:
        print("\n=== 修订（revise-fix）：无违规，未触发 ===")
        return
    print(f"\n=== 修订（revise-fix）：携带 {len(violations)} 条违规重写 ===")
    if mode == "revise":
        fixed = client.revise(task, style_prose, fix_violations=violations)
    else:
        fixed = client.draft(task, style_prose, fix_violations=violations)
    _print_envelope("修订产物", fixed)


def run_full(llm_for_fake: LLM | None, task: dict[str, Any]) -> None:
    """整跑：经 ``make_rewriter_loop`` 工厂走与主图完全一致的编排链路。

    ``llm_for_fake`` 为 None 时用真实工厂（按单元环境配置调模型，有花费）。
    """
    if llm_for_fake is None:
        adapter = make_rewriter_loop(default_llm_factory, print_hook)
    else:
        fake = llm_for_fake
        adapter = make_rewriter_loop(lambda unit: fake, print_hook)
    result = asyncio.run(adapter.run(task))
    print("\n=== 最终产物（RewriteResult）===")
    print(result["chapter_text"])
    print(f"[摘要] {result['chapter_summary']}")
    print(f"[自检] {json.dumps(result['self_check'], ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["draft", "revise"],
        default=None,
        help="调用模式：覆盖任务包内 mode；缺省沿用任务包（任务包也没有时为 draft）",
    )
    parser.add_argument(
        "--step",
        choices=["write", "lint", "audit", "revise-fix"],
        default=None,
        help="跑到指定环节停并打印中间产物；缺省经真编排整跑",
    )
    parser.add_argument(
        "--task",
        default=None,
        metavar="PATH",
        help="任务包 JSON 路径；缺省用随包样例 scripts/rewriter_task.sample.json",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="调真实模型（rewriter_loop 单元环境配置，有花费）；缺省为 FakeLLM 空转",
    )
    args = parser.parse_args()

    task = load_task(args.task)
    task["mode"] = args.mode or task.get("mode") or "draft"
    # 文种与变体来自任务包（ADR-0005）；兑底与层次推导与真编排同一口径。
    doc_type, doc_variant = carried_doc_facts(task)
    tier = tier_from_variant(doc_variant)
    print(
        f"[配置] mode={task['mode']} doc_type={doc_type}"
        f" doc_variant={doc_variant} tier={tier} real={args.real}"
    )

    if args.step is None:
        run_full(None if args.real else build_fake_llm(task), task)
        return

    # --step：绕开编排逐环节直调，客户端构造口径与 make_rewriter_loop 工厂一致。
    llm: LLM = default_llm_factory(UNIT) if args.real else build_fake_llm(task)
    client = LlmWriterClient(llm)
    run_stepwise(client, task, doc_type, doc_variant, args.step)


if __name__ == "__main__":
    main()
