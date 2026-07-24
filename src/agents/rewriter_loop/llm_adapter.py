"""llm_adapter：写作 LLM 注入点的真实适配器（本项目 LLM 协议纯文本调用）。

只依赖注入的 ``llm.invoke(messages) -> str``（``llm.llm_client.LLM`` 协议），
不触碰任何 SDK、API key 或环境变量。结构化输出为 JSON-in-text：
system 尾部声明 JSON 字段并附 ``JSON_ONLY_RULE``，应答经 ``parse_json`` 解析。

自源仓库 HypoArgus-RewriteLoop ``infra/llm_adapters.py`` 移植的要点：

- 引用角标语义全部改为本项目单方括号 ``[素材id]``（可并列叠加、同素材复用同 id、
  仅可用素材池内 id、禁止杜撰）；不产出 reference_list——书目由下游渲染层统一处理。
- draft / revise 退化重试：异常、解析失败、结构非法或空正文均视为退化 → 重试至
  ``max_attempts``；拿到过合法信封（哪怕正文为空）则返回最后一次诚实结果
  （正文为空时 ``degraded=True``）；从未拿到信封但抛过异常则重抛最后一个异常。
- audit 自审永不阻断：``issues: []`` 是合法非退化结果（不重试）；重试耗尽降级为
  空裁决（``degraded=True``）；非法条目防御性丢弃。
  audit 的指令置于 system（修复源仓库漏挂 system 的缺陷）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from agents.citation_policy import (
    EMPTY_CITABLE_MATERIALS_INSTRUCTION,
    citable_materials,
)
from agents.contracts import MaterialPayload
from agents.rewriter_loop.delta_merger import DeltaMerger
from agents.rewriter_loop.style_linter import (
    AuditItem,
    Violation,
    audit_items_for,
    word_count_prompt_block,
)
from agents.rewriter_loop.stub import UNIT
from agents.rewriter_loop.writer_client import (
    AuditEnvelope,
    AuditIssue,
    WriterEnvelope,
)
from domain.doc_types import carried_doc_facts, tier_from_variant
from domain.events import CONTENT_DELTA, EventHook, noop_hook
from llm.llm_client import LLM
from llm.llm_json import JSON_ONLY_RULE, parse_json
from llm.stream_json import JsonFieldExtractor

logger = logging.getLogger(__name__)

# 论证指令上位、禁令压缩收尾（机械规则枚举已删——linter 兜底，对模型无用）。
# 注：勿过度精简「培养规格」章型细结构——style_guide 散文不足以稳住，模型会漂移到 bullet。
_SYSTEM_INSTRUCTIONS = """\
你是教务公文写作子智能体，负责把章节骨架逐章展开成正文。

最重要的写作要求：充分论证每个论点——结合本章假说展开，给出依据与技术路径，不得只列结论、不得堆砌术语；每个论点须落到具体方法、措施与实施路径，展开到位后再收束。
你必须严格遵循下方「风格指南」：公文范式、对应层次（本科/高职）的子风格、术语与 _Avoid_、boilerplate、few-shot 范例章。

1. 衔接与开篇：承上启下经 chapter_summary 传递；若须衔接，以公文短语织入正文首句。
2. 事实忠实：证书名等结构化事实值**必须取自素材池摘录中的精确值**；表内证书列头可用「职业技能等级证书举例」「职业资格证书」等通用泛称（属列头非事实）。
3. 章型细结构（风格指南散文未细化、且 linter 不强制、模型易漂移者）：
   - 本科「培养目标与培养规格」合章的 `### （二）培养规格` = `#### 1.思政要求` `#### 2.素质要求` `#### 3.知识要求` `#### 4.能力要求` 四子节（思政独立成项）；`#### 1.思政要求` = **密集长段**（boilerplate 政治理论串整段论述，逐字串织入正文长句，不以 `1)2)3)` 编号条目拆散、亦不以无序 bullet 替）；`#### 4.能力要求` = `|职业能力|职业能力解构|` 式表（思政素质/通用基础/专业核心能力合并入表综合解构，多值单元格 `<br>` 编号密集列表）；素质/知识子节以密集长段或 `1)xxx<br>2)xxx` 编号条目展开。**本科培养规格不用无序 bullet 列表**（bullet 是高职合章口径）。
   - 高职「培养目标与培养规格」合章的 `### （二）培养规格` = `#### 1.素质` `#### 2.知识` `#### 3.能力` 三子节（思政并入素质、不设独立思政子项），各以无序密集 bullet 列表（`- xxx；`）展开，不以表格压缩、不以 `1)2)` 有序编号列表精简、不以叙述段替。
   - 独立「培养规格」章（无培养目标子节，不分本/高职）按黄金章型用单一指标表呈现，不适用上述列表/分节约束、须含表。
   - 课程设置章：课程设置列表用 `1.`/`（1）` 中文数字分点结构（公共课程/专业课程 → 通识必修/通识选修等），**不以简略表替**；学时学分总表用 markdown 表；专业核心课程教学内容与要求用 `#### N 课程名` 条目式（每条含共建企业/典型工作任务/教学内容等结构化散文，**不以表替**）；通识必修思政课须列课程全名逐字串（「习近平新时代中国特色社会主义思想概论」「思想道德与法治」「马克思主义基本原理」等），不以「思政类」「思政课程」等概括。
4. 引用角标（仅当传入了素材池时）：支撑论点/数据/观点/结论的句子须附 `[素材id]` 原位角标（单方括号、内为素材池中稳定 id），把角标放在支撑其 hypothesis_id 所指论点的句子处；多论据可并列叠加 `[m1][m2]`，同一素材复用同 id；仅可使用池内 id。引文终审门禁三条硬规则（违者整章返工）：
   - 量化断言（数值/百分比/倍数的提升、降低、对比）必须**同句**挂角标，且数值须来自素材摘录，禁止无据数值；
   - 角标所在句须如实转述所引素材摘录的核心观点与关键数值/对比结论，不得弱化或省略（素材称「高出20%」，正文须写出「20%」，不得改写成「显著提升」）；
   - 凡照抄或改写自素材摘录的句子必须挂对应 `[素材id]` 角标，不得漏标。
   - 素材池按佐证强度分【强支撑素材】与【弱佐证素材】两组：量化断言、精确数值与确定性结论只能由**强支撑素材**支撑；**弱佐证素材**（近似命中/补充）仅可作背景或趋势提示，其角标所在句须以留有余地的措辞转述（如「有资料显示」「部分数据表明」「或与……相关」），不得据其下精确数值或确定性结论；无强支撑素材时该论点宁可定性陈述、不得据弱佐证杜撰数值。
5. 禁令：
   - 禁 AI 空泛总结词与无据定性断言（「旨在构建」「确保符合」「致力于」「助力」「显著提升」「有效解决」等），公文用密集列举与事实陈述。
   - 本章开头不加独立前言段、不加「如下表所示」式导引句——直接 `## 标题` / `### 子节` → 正文或表；禁讲客体。
   - 素材无具体证书时不得写具体证书名，亦不得写「依据相关职业技能等级证书」「不限定为唯一证书」等无依据泛化句；数据格无依据则留空，不得以泛化散文句凑充。
   - 禁止新增/篡改/杜撰素材来源；正文不得夹带章内参考文献列表，也不得自行生成可见 `[n]` 序号（书目由下游渲染层统一处理）。

输出为一个 JSON 对象，字段如下：
- chapter_text：markdown 正文（含 ## 标题与 [素材id] 角标）；
- chapter_summary：供下一章承上启下用的一行公文摘要。
""" + JSON_ONLY_RULE


_AUDIT_TAG = "【章节自审】"


def _build_audit_system(items: Sequence[AuditItem]) -> str:
    """按适用裁决项拼装自审 system 提示词（裁决项按 doc_type 分派，ADR-0005）。

    裁决项判定准则来自 ssot-config ``audit_items``（与 lint 同源）；本函数只
    负责固定框架：角色、逐项裁决口径、输出 JSON 契约与「不臆造」通用准则。
    """
    blocks = "\n".join(
        f"{idx}. 【{item.label}】（item={item.id}）判定准则：\n{item.criteria}"
        for idx, item in enumerate(items, 1)
    )
    material_ids = [item.id for item in items if item.requires_materials]
    material_field_rule = (
        f"；item 为 {'、'.join(material_ids)} 的条目另须含 material_id（池内素材 id）"
        if material_ids
        else ""
    )
    return (
        "你是章节质检自审员，按下列裁决项逐项判断本章正文是否违规，"
        "只裁决下列各项，不扩大范围。\n\n"
        f"裁决项：\n{blocks}\n\n"
        "通用准则：无违规时返回空数组，不要臆造违规。\n\n"
        "输出为一个 JSON 对象，字段如下：\n"
        "- issues：每条含 item（上列裁决项 id 之一）与 excerpt"
        f"（正文中违规位置的片段或问题说明）{material_field_rule}；无违规时为空数组。\n"
    ) + JSON_ONLY_RULE


def _material_line(m: MaterialPayload) -> str:
    return f"- {m['id']}（支撑假说 {m['hypothesis_id']}，来源：{m['source']}）：{m['excerpt']}"


def _format_materials(materials: Sequence[MaterialPayload]) -> str:
    """按佐证强度分组渲染素材池：强支撑（pass）与弱佐证（inconclusive）分列标注。

    分组让模型据佐证强度调节措辞：强支撑可作量化断言与结论的直接依据；
    弱佐证仅近似命中/补充，只能作背景或趋势提示、措辞须留余地（详见系统提示词）。
    仅一类时不加另一类的空分组标题；两类皆空返回「（无）」。
    """
    if not materials:
        return "（无）"
    strong = [m for m in materials if m["verdict"] == "pass"]
    weak = [m for m in materials if m["verdict"] == "inconclusive"]
    blocks: list[str] = []
    if strong:
        blocks.append(
            "【强支撑素材】（可作为量化断言、数据与结论的直接依据）：\n"
            + "\n".join(_material_line(m) for m in strong)
        )
    if weak:
        blocks.append(
            "【弱佐证素材】（近似命中/补充，仅可作背景或趋势提示，不得据以下量化断言，措辞须留余地）：\n"
            + "\n".join(_material_line(m) for m in weak)
        )
    return "\n".join(blocks)


def _format_hypotheses(hypotheses: list[dict[str, Any]]) -> str:
    if not hypotheses:
        return "（无）"
    return "\n".join(f"- {h['id']}：{h['text']}" for h in hypotheses)


def _format_violations(violations: Sequence[Violation]) -> str:
    return "\n".join(f"- [{v.rule}] {v.message}" for v in violations)


def _format_note_entries(entries: Sequence[dict[str, Any]]) -> str:
    """把分区式修订说明的规则违规条目逐条渲染：规则名 + 位置摘录 + 修改指导。"""
    lines: list[str] = []
    for entry in entries:
        location = entry.get("location_excerpt") or ""
        loc = f"（位置：{location}）" if location.strip() else ""
        lines.append(f"- [{entry['rule']}]{loc} {entry['guidance']}")
    return "\n".join(lines)


def _format_revision_note(note: dict[str, Any]) -> str:
    """按优先级渲染分区式修订说明：用户指令 > error 违规 > warn 违规 > 冲突提示。

    评审产出的 ``RevisionNotePayload``（ADR-0006）渲染为分区块：用户指令区逐字
    落实（最高优先）、error 级违规必须修复、warn 级违规建议修复、冲突提示处以
    用户指令为准。任一区为空则整段省略，不产出空标题。
    """
    sections: list[str] = []
    user_directives = (note.get("user_directives") or "").strip()
    if user_directives:
        sections.append(f"【用户指令（最高优先，逐字落实）】\n{user_directives}")
    violations = note.get("rule_violations") or []
    errors = [entry for entry in violations if entry.get("severity") == "error"]
    warns = [entry for entry in violations if entry.get("severity") == "warn"]
    if errors:
        sections.append("【必须修复的违规（error 级）】\n" + _format_note_entries(errors))
    if warns:
        sections.append("【建议修复的违规（warn 级）】\n" + _format_note_entries(warns))
    conflicts = note.get("conflict_hints") or []
    if conflicts:
        sections.append(
            "【冲突提示（与上方用户指令抵触处，以用户指令为准）】\n"
            + "\n".join(f"- {c['description']}" for c in conflicts)
        )
    return "\n".join(sections)


def _build_context_block(task: dict[str, Any]) -> str:
    """draft / revise 共用的上下文块（章节骨架 / 字数目标 / 素材池 / 假说 / 衔接），不含尾部指令。

    文种与变体逐任务取自任务包（ADR-0005）；「层次」行取变体推导的 tier，
    与 lint 内部的变体键兑底同源——模型被告知的层次与校验执行的层次永远一致
    （无变体回落缺省「本科」）。字数目标块按文种取两层合并配置。
    """
    doc_type, doc_variant = carried_doc_facts(task)
    tier = tier_from_variant(doc_variant)
    spec = task["chapter_spec"]
    points = "\n".join(f"- {p['text']}" for p in spec["points"]) or "（无）"
    word_count_block = word_count_prompt_block(
        spec["title"], doc_type, chapter_type=spec.get("chapter_type")
    )
    prev = (
        f"上一章摘要（本章开头须公文风格承上启下衔接）：{task['prev_chapter_summary']}"
        if task["prev_chapter_summary"]
        else "（首章，无上一章摘要，无需承上启下）"
    )
    materials = citable_materials(task)
    if materials:
        material_block = (
            f"\n假说列表（每条有稳定 id，角标须落在支撑对应假说的句子处）：\n"
            f"{_format_hypotheses(spec['hypotheses'])}\n"
            f"素材池（仅可引用池内 id，禁止杜撰/篡改来源）：\n{_format_materials(materials)}\n"
        )
    else:
        # 素材池为空时仍须显式告知 writer：本章无可引素材，正文不得出现任何
        # 角标。否则系统提示词「引用角标（仅当传入了素材池时）」的条件句留
        # 下语义缝隙，真实模型会在无池时臆造 [素材id-N] 等占位角标——既无来源
        # 又对 reconcile 的 ASCII 角标模式隐形（正文残留角注、书目为空、无警告）。
        # 根因修复：堵住源头，让模型在无可引素材时以定性陈述展开、不臆造角标。
        material_block = (
            f"\n{EMPTY_CITABLE_MATERIALS_INSTRUCTION}"
            "无可引数据的论点按定性陈述展开，不得杜撰数值。\n"
        )
    word_count_section = f"{word_count_block}\n" if word_count_block else ""
    return f"""文种：{doc_type}
层次：{tier}
本章标题与要点：
- 标题：{spec["title"]}
- 要点：
{points}
{word_count_section}{prev}{material_block}"""


def _system_content(instructions: str, style_prose: str) -> str:
    """system = 指令 + 风格指南散文。"""
    return instructions + "\n\n### 风格指南\n" + style_prose


def _build_draft_user(
    task: dict[str, Any],
    *,
    fix_violations: Sequence[Violation] | None,
) -> str:
    """draft 的 user 提示词；``fix_violations`` 置位时切到修正口径（含违规清单）。

    修一次口径不回灌上一稿正文——用「完整上下文 + 违规清单」让模型按风格指南
    重写并规避违规。违规 message 已含规则名与片段，足供模型定位规避点。
    """
    base = _build_context_block(task)
    if fix_violations:
        return (
            f"{base}\n上一轮产出检出以下违规，请在重写本章正文与一行摘要时全部规避：\n"
            f"{_format_violations(fix_violations)}\n"
        )
    return f"{base}\n请按风格指南写出本章正文与一行摘要。"


def _build_revise_user(
    task: dict[str, Any],
    *,
    fix_violations: Sequence[Violation] | None,
) -> str:
    """revise 的 user 提示词：同一上下文块 + 现有正文 + 分区式修订说明。

    修订说明（``revision_note``，评审产物 ADR-0006）是 revise 模式唯一的修订
    驱动：按用户指令 > error 违规 > warn 违规 > 冲突提示的优先级分区呈现，三条
    修订链路（写作循环 / 终审打回 / 人工修订）统一消费同一结构（ADR-0007）。
    未被覆盖的内容与角标一律保持原样；``fix_violations`` 为旧的既存违规清单
    （ADR-0004，纯写作链路下已不再从 rewriter 传入，保留仅供调测脚本复用）。
    """
    base = _build_context_block(task)
    prompt = f"{base}\n现有正文：\n{task.get('current_text', '')}\n"
    note = task.get("revision_note")
    if note:
        prompt += (
            "修订说明（按优先级落实：用户指令 > error 违规 > warn 违规；"
            "未被覆盖的内容与 [素材id] 角标一律保持原样）：\n"
            f"{_format_revision_note(note)}\n"
        )
    if fix_violations:
        prompt += (
            f"现有正文另检出以下违规，请在本次修订中一并规避"
            f"（修复违规优先于「保持原样」，但不得扩大改动范围）：\n"
            f"{_format_violations(fix_violations)}\n"
        )
    return prompt


def _build_audit_user(chapter_text: str, task: dict[str, Any]) -> str:
    """自审 user 提示词：给素材池 + 正文，要可机器判读的违规列表。"""
    materials = citable_materials(task)
    material_block = (
        f"素材池（仅可引用池内 id）：\n{_format_materials(materials)}"
        if materials
        else EMPTY_CITABLE_MATERIALS_INSTRUCTION
    )
    return (
        f"{_AUDIT_TAG}按 system 中的裁决项逐项判断下面的本章正文。\n"
        f"{material_block}\n\n"
        f"本章正文：\n{chapter_text}\n\n"
        "判断并返回 issues（无违规时为空数组，不要臆造）。"
    )


class LlmWriterClient:
    """写作 LLM 注入点的真实适配器：纯文本 JSON-in-text 调用注入的 LLM 协议。

    draft / revise 经 ``llm.stream`` 逐片段消费：正文片段经 ``JsonFieldExtractor``
    从 JSON-in-text 增量抽出 chapter_text 纯正文、推理片段按 ``thinking`` kind
    原样外发，两 kind 经 ``DeltaMerger`` 合并后通过 ``EventHook`` 上网线为
    ``CONTENT_DELTA`` 事件（可丢级、不入可视化信封）；终态 ``WriterEnvelope``
    契约不变（``attempts`` / ``degraded`` 口径与旧非流式版一致）。
    audit 仍走 ``llm.invoke``（逐字流只覆盖写作 draft/revise 两路）。
    """

    def __init__(
        self,
        llm: LLM,
        *,
        max_attempts: int = 3,
        flush_chars: int = 64,
        flush_ms: int = 50,
        event_hook: EventHook = noop_hook,
    ) -> None:
        self._llm = llm
        self._max_attempts = max_attempts
        self._flush_chars = flush_chars
        self._flush_ms = flush_ms
        self._event_hook = event_hook

    def draft(
        self,
        task: dict[str, Any],
        style_prose: str,
        *,
        fix_violations: Sequence[Violation] | None = None,
    ) -> WriterEnvelope:
        chapter_id = task["chapter_spec"]["id"]
        user = _build_draft_user(task, fix_violations=fix_violations)
        return self._write_with_retry("draft", style_prose, user, chapter_id)

    def revise(
        self,
        task: dict[str, Any],
        style_prose: str,
        *,
        fix_violations: Sequence[Violation] | None = None,
    ) -> WriterEnvelope:
        chapter_id = task["chapter_spec"]["id"]
        user = _build_revise_user(task, fix_violations=fix_violations)
        return self._write_with_retry("revise", style_prose, user, chapter_id)

    def _write_with_retry(
        self,
        step: str,
        style_prose: str,
        user: str,
        chapter_id: str,
    ) -> WriterEnvelope:
        """draft / revise 共用的退化重试：合法信封 = dict 且正文/摘要皆为 str。

        退化（异常 / 解析失败 / 结构非法 / 空正文）→ 重试至 ``max_attempts``；
        拿到过合法信封（哪怕正文为空）→ 返回最后一次诚实结果（正文为空时
        ``degraded=True``）；从未拿到信封但抛过异常 → 重抛最后一个异常；
        否则返回空信封 ``degraded=True``。``attempts`` 回填实际执行的总轮次
        （末次退化可能发生在拿到信封之后，故以循环终止时的轮次为准）。

        每轮经 ``_stream_once`` 逐片段消费流式应答：流中途或 ``finish`` 抛
        ``FieldExtractionError`` 即本轮退化（与旧非流式 ``parse_json`` 失败
        等价）；已 flush 的逐字帧仍带本轮 attempt 号，调用方在更高 attempt
        须丢弃重建（契约见 ``CONTENT_DELTA``）。
        """
        messages = [
            {"role": "system", "content": _system_content(_SYSTEM_INSTRUCTIONS, style_prose)},
            {"role": "user", "content": user},
        ]
        last_envelope: WriterEnvelope | None = None
        last_exc: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._stream_once(step, messages, attempt, chapter_id)
                payload = parse_json(raw, step)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "rewriter_loop[%s] 尝试 %d/%d 退化（异常）：%s: %s",
                    step, attempt, self._max_attempts, type(exc).__name__, exc,
                )
                continue
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("chapter_text"), str)
                or not isinstance(payload.get("chapter_summary"), str)
            ):
                logger.warning(
                    "rewriter_loop[%s] 尝试 %d/%d 退化：应答结构非法",
                    step, attempt, self._max_attempts,
                )
                continue
            chapter_text: str = payload["chapter_text"]
            envelope = WriterEnvelope(
                chapter_text=chapter_text,
                chapter_summary=payload["chapter_summary"],
                attempts=attempt,
                degraded=not chapter_text.strip(),
            )
            if chapter_text.strip():
                return envelope
            last_envelope = envelope
            logger.warning(
                "rewriter_loop[%s] 尝试 %d/%d 退化：空 chapter_text",
                step, attempt, self._max_attempts,
            )
        if last_envelope is not None:
            # 全部尝试退化但拿到过合法信封：返回最后一次诚实结果（空正文如实标退化），
            # attempts 回填实际执行的总轮次而非拿到信封的那一轮。
            return last_envelope.model_copy(update={"attempts": self._max_attempts})
        if last_exc is not None:
            # 从未拿到信封但抛过异常：重抛最后一个，交由编排层兜底裁决。
            raise last_exc
        # 从未拿到信封也从未抛异常（纯结构非法）：诚实空稿，不抛。
        return WriterEnvelope(
            chapter_text="", chapter_summary="", attempts=self._max_attempts, degraded=True
        )

    def _stream_once(
        self,
        step: str,
        messages: list[dict[str, str]],
        attempt: int,
        chapter_id: str,
    ) -> str:
        """单轮流式消费：逐片段抽正文/推理，合并后上网线，返回完整原始 JSON 文本。

        - 正文片段（``kind=content``）原样累积到 ``raw_parts``，并喂入
          ``JsonFieldExtractor`` 增量抽出 chapter_text 纯正文外发逐字流；
          抽取失败（非法转义 / 非字符串值）即抛 ``FieldExtractionError``，
          由 ``_write_with_retry`` 的 try/except 捕获判本轮退化。
        - 推理片段（``kind=thinking``）直接喂入合并器外发，不进 JSON 抽取
          （推理 CoT 不在 chapter_text 字段内）。
        - 流结束调 ``extractor.finish`` 校验目标字段完整闭合；未闭合 / 未找到
          抛 ``FieldExtractionError``（同样判本轮退化）。
        - 错误路径不调 ``flush_remaining``：残余片段随本轮 attempt 丢弃，
          与「更高 attempt 须从零重建」契约一致；已 flush 帧仍带本轮 attempt 号。
        - ``sequence`` 在本轮内单调递增（content / thinking 共享同一计数器），
          新 attempt 复位为 0（每次 ``_stream_once`` 独立 seq 列表）。
        """
        extractor = JsonFieldExtractor("chapter_text")
        # 单元素列表当可变计数器：闭包内自增免 nonlocal，本流内 content /
        # thinking 共享同一 sequence，新 attempt 自然从 0 复位（本函数局部）。
        seq = [0]

        def on_flush(kind: str, text: str) -> None:
            payload = {
                "unit": UNIT,
                "chapter_id": chapter_id,
                "mode": step,
                "kind": kind,
                "delta": text,
                "attempt": attempt,
                "sequence": seq[0],
            }
            seq[0] += 1
            self._event_hook(CONTENT_DELTA, payload)

        merger = DeltaMerger(
            on_flush,
            flush_chars=self._flush_chars,
            flush_ms=self._flush_ms,
        )
        raw_parts: list[str] = []
        for chunk in self._llm.stream(messages):
            if chunk.kind == "thinking":
                merger.feed("thinking", chunk.text)
                continue
            raw_parts.append(chunk.text)
            # 抽取器吃原始 JSON 片段（非纯正文）：可能抛 FieldExtractionError，
            # 任其传播到 _write_with_retry 的 try/except 判本轮退化。
            prose = extractor.feed(chunk.text)
            if prose:
                merger.feed("content", prose)
        # finish 在目标字段未闭合/未找到时抛错：错误路径不 flush_remaining，
        # 残余缓冲随 attempt 丢弃；已 flush 帧仍属本轮 attempt 号。
        extractor.finish()
        merger.flush_remaining()
        return "".join(raw_parts)

    def audit(self, chapter_text: str, task: dict[str, Any]) -> AuditEnvelope:
        """自审裁决；``issues: []`` 合法非退化（不重试）。

        裁决项按任务包文种加载并按素材池适用性过滤（与编排层跳过口径同源），
        system 提示词逐任务拼装。异常 / 解析失败 / 结构非法 → 重试；
        耗尽 → 返回空裁决 ``degraded=True``——自审永不阻断主链。
        非法条目（item 不在适用裁决项内、依赖素材的裁决项缺 material_id 等）
        防御性丢弃。
        """
        doc_type, _ = carried_doc_facts(task)
        items = audit_items_for(doc_type, has_materials=bool(citable_materials(task)))
        if not items:
            # 无适用裁决项（编排层通常已跳过）：防御性返回合法空裁决，不发无意义调用。
            return AuditEnvelope()
        by_id: dict[str, AuditItem] = {item.id: item for item in items}
        messages = [
            {"role": "system", "content": _build_audit_system(items)},
            {"role": "user", "content": _build_audit_user(chapter_text, task)},
        ]
        for attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._llm.invoke(messages)
                payload = parse_json(raw, "audit")
            except Exception as exc:
                logger.warning(
                    "rewriter_loop[audit] 尝试 %d/%d 退化（异常）：%s: %s",
                    attempt, self._max_attempts, type(exc).__name__, exc,
                )
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("issues"), list):
                logger.warning(
                    "rewriter_loop[audit] 尝试 %d/%d 退化：应答结构非法",
                    attempt, self._max_attempts,
                )
                continue
            issues: list[AuditIssue] = []
            for entry in payload["issues"]:
                # 非法条目防御性丢弃：自审是尽力而为的辅助裁决，不因单条脏数据整体退化。
                if not isinstance(entry, dict):
                    logger.warning("rewriter_loop[audit] 丢弃非法自审条目：%r", entry)
                    continue
                item_id = entry.get("item")
                spec = by_id.get(item_id) if isinstance(item_id, str) else None
                if spec is None:
                    logger.warning("rewriter_loop[audit] 丢弃裁决项不明的自审条目：%r", entry)
                    continue
                material_id = entry.get("material_id")
                if spec.requires_materials and not isinstance(material_id, str):
                    logger.warning("rewriter_loop[audit] 丢弃缺 material_id 的自审条目：%r", entry)
                    continue
                excerpt = entry.get("excerpt")
                issues.append(
                    AuditIssue(
                        item=spec.id,
                        label=spec.label,
                        material_id=material_id if isinstance(material_id, str) else "",
                        excerpt=excerpt if isinstance(excerpt, str) else "",
                    )
                )
            return AuditEnvelope(issues=issues, attempts=attempt)
        logger.warning(
            "rewriter_loop[audit] 全部 %d 次尝试退化；降级为空裁决（自审不阻断）",
            self._max_attempts,
        )
        return AuditEnvelope(attempts=self._max_attempts, degraded=True)
