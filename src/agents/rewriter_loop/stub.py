"""rewriter_loop 子智能体：统包首写（draft）与纯改写（revise）双模式。

本期为打桩实现；真实现落地时按 contracts.RewriteTask/RewriteResult
同一接口规范替换，工厂签名不变。
"""

from typing import Any

from agents.contracts import SelfCheckPayload, SubagentAdapter
from domain.doc_types import carried_doc_facts
from domain.events import EventHook, noop_hook
from domain.state import CITABLE_VERDICTS

UNIT = "rewriter_loop"


async def stub_rewriter_loop_run(task: dict[str, Any]) -> dict[str, Any]:
    """rewriter_loop 打桩：产出含原位角标的确定性正文、章节摘要与自检结果。

    draft 模式承接上一章摘要生成正文；revise 模式在 current_text 基础上
    逐条附注修订指令，保证两种模式的接口都可空转。

    正文含 ## 标题行（spec['title'] 已含模板编号「一、」等，原样输出），
    满足章节编号连续唯一校验（issue #18）。
    """
    spec = task["chapter_spec"]
    # 写作池 = pass 强支撑 + inconclusive 弱佐证（杠杆②放宽），与真实适配层同口径。
    citable = [
        material
        for material in task["materials"]
        if material["verdict"] in CITABLE_VERDICTS
    ]

    if task["mode"] == "revise":
        directives = task.get("revision_directives", [])
        notes = "".join(
            f"（修订落实：{directive['instruction']}）" for directive in directives
        )
        chapter_text = f"{task.get('current_text', '')}{notes}"
    else:
        paragraphs: list[str] = [f"## {spec['title']}"]
        prev_summary = task["prev_chapter_summary"]
        if prev_summary:
            paragraphs.append(f"承接上一章：{prev_summary}")
        paragraphs.append("本章围绕以下论点展开（打桩正文）。")
        for point in spec["points"]:
            paragraphs.append(f"论点：{point['text']}（打桩论证）")
        for material in citable:
            paragraphs.append(
                f"素材佐证假说 {material['hypothesis_id']}（打桩）[{material['id']}]"
            )
        chapter_text = "\n\n".join(paragraphs)

    point_digest = "；".join(point["text"] for point in spec["points"])
    chapter_summary = f"《{spec['title']}》要点：{point_digest or '（无论点）'}（打桩摘要）"
    doc_type, doc_variant = carried_doc_facts(task)
    return {
        "chapter_text": chapter_text,
        "chapter_summary": chapter_summary,
        "self_check": SelfCheckPayload(citations_ok=True, issues=[]),
        "doc_type": doc_type,
        "doc_variant": doc_variant,
    }


def make_stub_rewriter_loop(event_hook: EventHook = noop_hook) -> SubagentAdapter:
    """构造 rewriter_loop 打桩适配器。"""
    return SubagentAdapter(UNIT, stub_rewriter_loop_run, event_hook)
