"""上下文装配：每次 LLM 调用的输入由此从 State 现场装配，禁止透传原始累积历史。

核心概念（与 CONTEXT.md「上下文装配」一节一致）：
- 内容段（Segment）：装配产出的最小单位，段名 + 文本。
- 提取器（Extractor）：State → 内容段列表的纯函数，可跨配方复用；
  禁止读取 State 之外的全局可变状态。
- 装配配方（Recipe）：按运行单元注册的一组提取器；差异收敛于配方，
  阈值与保留策略缺省全部取 AssemblerConfig。
- 统一入口 assemble(state, unit, **params)：调用点局部参数经 params 注入提取器。

压缩只在超阈值时发生：摘要链过长做「摘要的摘要」；修订台账保最近 K 轮原文 +
更早轮次一句话摘要。定位类参数（如 chapter_id）缺失时提取器返回空段列表而不抛错，
便于同一配方服务多场景。
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from typing import cast

from assembly.assembler_config import AssemblerConfig, BudgetOverride, load_assembler_config
from domain.citation_reconciler import MARKER_PATTERN
from llm.llm_config import RUNTIME_UNITS
from domain.state import RevisionRound, WritingAgentState

# 切首句的句末标点：中文句号/叹号/问号与对应英文标点。
_SENTENCE_ENDINGS = "。！？.!?"


@dataclass(frozen=True)
class Segment:
    """内容段：装配产出的最小单位。"""

    name: str
    text: str


# 提取器签名：纯函数，State + 调用点局部参数 + 装配配置 → 内容段列表。
Extractor = Callable[
    [WritingAgentState, Mapping[str, object], AssemblerConfig], list[Segment]
]


@dataclass(frozen=True)
class Recipe:
    """装配配方：一个运行单元的提取器组合 + 可选专属 token 预算覆盖。

    budget 为 None 时阈值全部取 AssemblerConfig 全局配置；
    否则装配时把覆盖中非 None 的字段合并进生效配置。
    """

    extractors: tuple[Extractor, ...]
    budget: BudgetOverride | None = None


@dataclass(frozen=True)
class AssembledContext:
    """装配结果：段名到文本的映射，供节点按名取用。"""

    segments: dict[str, str]

    def text(self, name: str, default: str = "") -> str:
        """按段名读取文本，缺失时返回 default。"""
        return self.segments.get(name, default)


def _first_sentence_digest(text: str, max_chars: int) -> str:
    """确定性一句话摘要：切首句后限长，超长截断并加省略号。"""
    stripped = text.strip()
    first = stripped
    for index, char in enumerate(stripped):
        if char in _SENTENCE_ENDINGS:
            first = stripped[: index + 1]
            break
    if len(first) > max_chars:
        first = first[:max_chars] + "…"
    return first


def digest_of_round(round_: RevisionRound, config: AssemblerConfig) -> str:
    """修订轮次的确定性一句话摘要：优先用已有 digest，否则取 raw_feedback 首句限长。

    纯函数，供装配提取器与 human_review_gate 落库共用，保证两处摘要一致。
    """
    if round_.digest is not None:
        return round_.digest
    return _first_sentence_digest(round_.raw_feedback, config.ledger_digest_max_chars)


def extract_user_intent(
    state: WritingAgentState,
    params: Mapping[str, object],
    config: AssemblerConfig,
) -> list[Segment]:
    """提取用户意图与用户身份两段。"""
    return [
        Segment("user_intent", state.get("user_intent", "")),
        Segment("user_identity", state.get("user_identity", "")),
    ]


def _compressed_chain_text(
    entries: list[tuple[str, str]], config: AssemblerConfig
) -> str:
    """（标题, 摘要）序列 → 摘要链文本，超阈值时做「摘要的摘要」压缩。

    压缩策略：总字符数超 summary_chain_max_chars 时，除最后一章保留原文外，
    更早各章截为首句摘要；仍超阈值则从最早章节起丢弃并在段首标注省略数量。
    实际摘要链与规划摘要链共用此行格式与压缩策略。
    """

    def _line(title: str, summary: str) -> str:
        return f"【{title}】{summary}"

    lines = [_line(title, summary) for title, summary in entries]
    text = "\n".join(lines)
    if len(text) > config.summary_chain_max_chars and lines:
        compressed = [
            _line(
                title,
                _first_sentence_digest(summary, config.summary_digest_max_chars),
            )
            for title, summary in entries[:-1]
        ]
        compressed.append(lines[-1])
        dropped = 0
        while (
            len("\n".join(compressed)) > config.summary_chain_max_chars
            and len(compressed) > 1
        ):
            compressed.pop(0)
            dropped += 1
        if dropped:
            compressed.insert(0, f"（更早 {dropped} 章摘要已省略）")
        text = "\n".join(compressed)
    return text


def extract_summary_chain(
    state: WritingAgentState,
    params: Mapping[str, object],
    config: AssemblerConfig,
) -> list[Segment]:
    """提取前文摘要链（实际写成的章节摘要），超阈值时压缩。

    params 支持可选 chapter_id：装配到该章之前的所有前章摘要；缺省取全部草稿。
    """
    drafts = list(state.get("chapter_drafts", []))
    chapter_id = params.get("chapter_id")
    if chapter_id is not None:
        index = next(
            (i for i, draft in enumerate(drafts) if draft.chapter_id == chapter_id),
            None,
        )
        if index is not None:
            drafts = drafts[:index]

    titles = {chapter.id: chapter.title for chapter in state.get("outline", [])}
    prev_summary = drafts[-1].summary if drafts else ""
    entries = [
        (titles.get(draft.chapter_id, draft.chapter_id), draft.summary)
        for draft in drafts
    ]
    return [
        Segment("summary_chain", _compressed_chain_text(entries, config)),
        Segment("prev_chapter_summary", prev_summary),
    ]


def extract_planned_summary_chain(
    state: WritingAgentState,
    params: Mapping[str, object],
    config: AssemblerConfig,
) -> list[Segment]:
    """提取规划摘要链：目标章之前各章的规划摘要拼链，供并行首写承接前文。

    并行首写时前章草稿尚未写成，衔接依据改为框架生成时的规划摘要
    （ChapterSpec.planned_summary）；段名沿用 summary_chain，
    行格式与压缩策略与实际摘要链一致，首章为空串。
    params 需带 chapter_id；缺失或不在大纲中时返回空段列表。
    """
    chapter_id = params.get("chapter_id")
    if chapter_id is None:
        return []
    outline = state.get("outline", [])
    index = next(
        (i for i, chapter in enumerate(outline) if chapter.id == chapter_id), None
    )
    if index is None:
        return []
    entries = [
        (chapter.title, chapter.planned_summary) for chapter in outline[:index]
    ]
    return [Segment("summary_chain", _compressed_chain_text(entries, config))]


def extract_revision_ledger(
    state: WritingAgentState,
    params: Mapping[str, object],
    config: AssemblerConfig,
) -> list[Segment]:
    """提取修订台账，保留策略：最近 K 轮原文 + 更早轮次一句话摘要。

    更早轮次优先用已有 digest 字段，为 None 时按 raw_feedback 首句确定性生成。
    台账为空时段文本为空串。多轮意见全量持久化在 State 中，装配时按需注入，
    保证多轮迭代不失忆。
    """
    ledger = list(state.get("revision_ledger", []))
    if not ledger:
        return [Segment("revision_ledger", "")]

    keep = config.ledger_keep_rounds
    earlier, recent = ledger[:-keep], ledger[-keep:]
    lines: list[str] = []
    for round_ in earlier:
        lines.append(f"第{round_.round_no}轮（摘要）：{digest_of_round(round_, config)}")
    for round_ in recent:
        lines.append(f"第{round_.round_no}轮：{round_.raw_feedback}")
        for directive in round_.directives:
            lines.append(
                f"  - [{directive.type}] {directive.target_chapter_id}：{directive.instruction}"
            )
    return [Segment("revision_ledger", "\n".join(lines))]


def extract_citation_digest(
    state: WritingAgentState,
    params: Mapping[str, object],
    config: AssemblerConfig,
) -> list[Segment]:
    """提取引文库摘要：总条数 + 按章节分组的通过/未通过计数。"""
    library = state.get("citation_library", [])
    counts: dict[str, list[int]] = {}
    for material in library:
        entry = counts.setdefault(material.chapter_id, [0, 0])
        entry[0 if material.verdict == "pass" else 1] += 1
    lines = [f"引文库共 {len(library)} 条素材。"]
    for chapter_id, (passed, failed) in counts.items():
        lines.append(f"章节 {chapter_id}：通过 {passed} 条，未通过 {failed} 条")
    return [Segment("citation_digest", "\n".join(lines))]


def extract_chapter_list(
    state: WritingAgentState,
    params: Mapping[str, object],
    config: AssemblerConfig,
) -> list[Segment]:
    """提取大纲章节清单：每行「id 标题」。"""
    lines = [f"{chapter.id} {chapter.title}" for chapter in state.get("outline", [])]
    return [Segment("chapter_list", "\n".join(lines))]


def extract_chapter_materials(
    state: WritingAgentState,
    params: Mapping[str, object],
    config: AssemblerConfig,
) -> list[Segment]:
    """提取指定章节通过校验的素材，JSON 序列化。

    params 需带 chapter_id；缺失时返回空段列表（同一配方服务多场景）。
    """
    chapter_id = params.get("chapter_id")
    if chapter_id is None:
        return []
    materials = [
        material.model_dump()
        for material in state.get("citation_library", [])
        if material.chapter_id == chapter_id and material.verdict == "pass"
    ]
    return [Segment("chapter_materials", json.dumps(materials, ensure_ascii=False))]


def extract_chapter_draft(
    state: WritingAgentState,
    params: Mapping[str, object],
    config: AssemblerConfig,
) -> list[Segment]:
    """提取指定章节正文全文与被引素材（正文角标命中的素材 id/excerpt）。

    params 需带 chapter_id；缺失或草稿不存在时返回空段列表。
    角标解析复用 citation_reconciler.MARKER_PATTERN，与对账逻辑保持一致。
    """
    chapter_id = params.get("chapter_id")
    if chapter_id is None:
        return []
    draft = next(
        (
            draft
            for draft in state.get("chapter_drafts", [])
            if draft.chapter_id == chapter_id
        ),
        None,
    )
    if draft is None:
        return []
    marker_ids = set(MARKER_PATTERN.findall(draft.text))
    cited = [
        {"id": material.id, "excerpt": material.excerpt}
        for material in state.get("citation_library", [])
        if material.id in marker_ids
    ]
    return [
        Segment("chapter_text", draft.text),
        Segment("cited_materials", json.dumps(cited, ensure_ascii=False)),
    ]


def extract_user_feedback(
    state: WritingAgentState,
    params: Mapping[str, object],
    config: AssemblerConfig,
) -> list[Segment]:
    """透传调用点注入的本轮用户意见。params 带 feedback；缺失时返回空段列表。"""
    feedback = params.get("feedback")
    if feedback is None:
        return []
    return [Segment("user_feedback", str(feedback))]


# 写作单元（首写与改写）需要更完整的前文摘要链承接文风与叙事，
# 两份配方共享一个较宽的摘要链预算覆盖；其余阈值沿用全局配置。
_WRITING_BUDGET = BudgetOverride(summary_chain_max_chars=1200)

# 按运行单元注册的装配配方；键覆盖 llm_config.RUNTIME_UNITS 全部 8 个单元。
# 写作各配方只留 RewriteTask 实际消费的段（摘要链与章节素材）；
# chapter_drafter（并行首写）用规划摘要链替代实际摘要链，其余与写作配方一致。
RECIPES: dict[str, Recipe] = {
    "framework_orchestrator": Recipe(extractors=(extract_user_intent,)),
    "reference_orchestrator": Recipe(
        extractors=(extract_user_intent, extract_citation_digest)
    ),
    "search_agent": Recipe(extractors=(extract_citation_digest,)),
    "chapter_drafter": Recipe(
        extractors=(extract_planned_summary_chain, extract_chapter_materials),
        budget=_WRITING_BUDGET,
    ),
    "writing_orchestrator": Recipe(
        extractors=(extract_summary_chain, extract_chapter_materials),
        budget=_WRITING_BUDGET,
    ),
    "rewriter_loop": Recipe(
        extractors=(extract_summary_chain, extract_chapter_materials),
        budget=_WRITING_BUDGET,
    ),
    "citation_validator": Recipe(extractors=(extract_chapter_draft,)),
    "human_review_gate": Recipe(
        extractors=(extract_chapter_list, extract_revision_ledger, extract_user_feedback)
    ),
}

if set(RECIPES) != set(RUNTIME_UNITS):
    raise RuntimeError(
        "装配配方必须覆盖全部运行单元：RECIPES 的键与 llm_config.RUNTIME_UNITS 不一致"
    )


def _effective_config(
    config: AssemblerConfig, budget: BudgetOverride | None
) -> AssemblerConfig:
    """把配方专属预算覆盖合并进全局配置：值为 None 的字段沿用全局值。"""
    if budget is None:
        return config
    overrides = {
        field.name: value
        for field in fields(BudgetOverride)
        if (value := getattr(budget, field.name)) is not None
    }
    return replace(config, **overrides)


def assemble(
    state: WritingAgentState,
    unit: str,
    *,
    config: AssemblerConfig | None = None,
    **params: object,
) -> AssembledContext:
    """统一装配入口：按运行单元的配方依次执行提取器，汇总为装配结果。

    未知单元抛 ValueError；config 缺省从环境变量读取；配方带专属预算覆盖时
    先合并进生效配置再传入提取器；调用点局部参数经 **params 注入各提取器，
    提取器缺参时返回空段而不抛错。
    """
    recipe = RECIPES.get(unit)
    if recipe is None:
        raise ValueError(f"未知运行单元：{unit}，合法取值：{tuple(RECIPES)}")
    if config is None:
        config = load_assembler_config()
    config = _effective_config(config, recipe.budget)

    segments: dict[str, str] = {}
    for extractor in recipe.extractors:
        for segment in extractor(state, params, config):
            segments[segment.name] = segment.text
    return AssembledContext(segments=segments)


def assemble_with(
    state: WritingAgentState,
    overrides: Mapping[str, object],
    unit: str,
    *,
    config: AssemblerConfig | None = None,
    **params: object,
) -> AssembledContext:
    """先以 overrides 浅覆盖 State 再装配：服务「现场覆盖后装配」惯用法。

    不修改原 State；用于以调用现场累积的中间结果（如逐章增长的引文库、
    已完成的章节草稿）覆盖对应字段后按配方装配。
    """
    merged = cast(WritingAgentState, {**state, **overrides})
    return assemble(merged, unit, config=config, **params)
