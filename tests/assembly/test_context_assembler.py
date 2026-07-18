"""context_assembler 上下文装配的单元测试。

覆盖：配置读取与回落、未知单元报错、提取器纯函数性质、
摘要链压缩三档策略、修订台账保留策略与多轮不失忆、
引文库摘要分章计数、章节草稿被引素材提取、提取器跨配方复用、
7 个运行单元全部可装配。
"""

import copy
import json

import pytest

from assembly.assembler_config import AssemblerConfig, load_assembler_config
from assembly.context_assembler import (
    RECIPES,
    assemble,
    assemble_with,
    extract_chapter_draft,
    extract_chapter_materials,
    extract_citation_digest,
    extract_summary_chain,
)
from llm.llm_config import RUNTIME_UNITS
from domain.state import (
    ChapterDraft,
    ChapterSpec,
    Material,
    RevisionDirective,
    RevisionRound,
    WritingAgentState,
)

_DEFAULT_CONFIG = AssemblerConfig(
    summary_chain_max_chars=800,
    summary_digest_max_chars=60,
    ledger_keep_rounds=2,
    ledger_digest_max_chars=60,
)


def _material(mat_id: str, chapter_id: str, verdict: str) -> Material:
    return Material(
        id=mat_id,
        hypothesis_id=f"{chapter_id}-p1-h1",
        chapter_id=chapter_id,
        source=f"来源 {mat_id}",
        excerpt=f"摘录 {mat_id}",
        relevance_score=0.8,
        verdict=verdict,  # type: ignore[arg-type]
    )


def _make_state() -> WritingAgentState:
    """三章大纲 + 两章草稿 + 含通过与未通过素材的引文库 + 两轮修订台账。"""
    return WritingAgentState(
        user_intent="写一篇产业分析",
        user_identity="行业研究员",
        outline=[
            ChapterSpec(id="ch1", title="第一章"),
            ChapterSpec(id="ch2", title="第二章"),
            ChapterSpec(id="ch3", title="第三章"),
        ],
        citation_library=[
            _material("m-1", "ch1", "pass"),
            _material("m-2", "ch1", "fail"),
            _material("m-3", "ch2", "pass"),
        ],
        chapter_drafts=[
            ChapterDraft(chapter_id="ch1", text="第一章正文 [m-1]", summary="第一章摘要。补充说明。"),
            ChapterDraft(chapter_id="ch2", text="第二章正文 [m-3] [m-1]", summary="第二章摘要。"),
        ],
        revision_ledger=[
            RevisionRound(
                round_no=1,
                raw_feedback="第一轮意见原文。后半句。",
                directives=[
                    RevisionDirective(
                        target_chapter_id="ch1", type="rewrite_only", instruction="收紧语气"
                    )
                ],
            ),
            RevisionRound(round_no=2, raw_feedback="第二轮意见原文。"),
        ],
    )


# ---------- 配置读取 ----------


def test_配置未设置时全部取缺省值():
    assert load_assembler_config({}) == _DEFAULT_CONFIG


def test_配置独立覆盖单个变量其余取缺省():
    config = load_assembler_config({"ASSEMBLER_LEDGER_KEEP_ROUNDS": "3"})
    assert config.ledger_keep_rounds == 3
    assert config.summary_chain_max_chars == 800
    assert config.summary_digest_max_chars == 60
    assert config.ledger_digest_max_chars == 60


@pytest.mark.parametrize(
    "name",
    [
        "ASSEMBLER_SUMMARY_CHAIN_MAX_CHARS",
        "ASSEMBLER_SUMMARY_DIGEST_MAX_CHARS",
        "ASSEMBLER_LEDGER_KEEP_ROUNDS",
        "ASSEMBLER_LEDGER_DIGEST_MAX_CHARS",
    ],
)
def test_配置非法值抛错并指明变量名(name: str):
    with pytest.raises(ValueError, match=name):
        load_assembler_config({name: "0"})


# ---------- 装配入口 ----------


def test_未知单元抛ValueError():
    with pytest.raises(ValueError, match="未知运行单元"):
        assemble(_make_state(), "no_such_unit", config=_DEFAULT_CONFIG)


def test_七个运行单元都能装配不抛错():
    state = _make_state()
    for unit in RUNTIME_UNITS:
        context = assemble(state, unit, config=_DEFAULT_CONFIG)
        assert isinstance(context.segments, dict)


def test_便捷读取text缺失段返回default():
    context = assemble(_make_state(), "framework_orchestrator", config=_DEFAULT_CONFIG)
    assert context.text("user_intent") == "写一篇产业分析"
    assert context.text("不存在的段", "缺省值") == "缺省值"


# ---------- 提取器纯函数性质 ----------


def test_提取器纯函数_两次调用产出相等且不修改state():
    state = _make_state()
    snapshot = copy.deepcopy(state)
    first = assemble(state, "writing_orchestrator", config=_DEFAULT_CONFIG, chapter_id="ch2")
    second = assemble(state, "writing_orchestrator", config=_DEFAULT_CONFIG, chapter_id="ch2")
    assert first.segments == second.segments
    assert state == snapshot


def test_提取器跨配方复用_同一对象出现在两个配方中():
    from assembly.context_assembler import extract_user_intent

    assert extract_user_intent in RECIPES["framework_orchestrator"].extractors
    assert any(
        extractor is extract_user_intent
        for extractor in RECIPES["reference_orchestrator"].extractors
    )


# ---------- 摘要链 ----------


def _chain_state(summaries: list[str]) -> WritingAgentState:
    return WritingAgentState(
        outline=[ChapterSpec(id=f"ch{i}", title=f"第{i}章") for i in range(1, len(summaries) + 2)],
        chapter_drafts=[
            ChapterDraft(chapter_id=f"ch{i}", text=f"ch{i} 正文", summary=summary)
            for i, summary in enumerate(summaries, start=1)
        ],
    )


def test_摘要链未超阈值原样注入():
    state = _chain_state(["第一章摘要。", "第二章摘要。"])
    segments = extract_summary_chain(state, {}, _DEFAULT_CONFIG)
    by_name = {segment.name: segment.text for segment in segments}
    assert by_name["summary_chain"] == "【第1章】第一章摘要。\n【第2章】第二章摘要。"
    assert by_name["prev_chapter_summary"] == "第二章摘要。"


def test_摘要链chapter_id只取该章之前的前章():
    state = _chain_state(["第一章摘要。", "第二章摘要。"])
    segments = extract_summary_chain(state, {"chapter_id": "ch2"}, _DEFAULT_CONFIG)
    by_name = {segment.name: segment.text for segment in segments}
    assert by_name["summary_chain"] == "【第1章】第一章摘要。"
    assert by_name["prev_chapter_summary"] == "第一章摘要。"


def test_摘要链超阈值_最后一章保留原文_更早章截为首句摘要():
    config = AssemblerConfig(
        summary_chain_max_chars=30,
        summary_digest_max_chars=10,
        ledger_keep_rounds=2,
        ledger_digest_max_chars=60,
    )
    state = _chain_state(["早章首句。这些尾巴内容不该出现。", "末章摘要原文完整保留。"])
    segments = extract_summary_chain(state, {}, config)
    chain = {segment.name: segment.text for segment in segments}["summary_chain"]
    assert "早章首句。" in chain
    assert "尾巴内容" not in chain
    assert "末章摘要原文完整保留。" in chain


def test_摘要链首句超长截断加省略号():
    config = AssemblerConfig(
        summary_chain_max_chars=30,
        summary_digest_max_chars=5,
        ledger_keep_rounds=2,
        ledger_digest_max_chars=60,
    )
    state = _chain_state(["这是一个没有断句标点的超长摘要文本", "末章摘要。"])
    segments = extract_summary_chain(state, {}, config)
    chain = {segment.name: segment.text for segment in segments}["summary_chain"]
    assert "这是一个没…" in chain


def test_摘要链极端超长_最早章被丢弃且有省略标注():
    config = AssemblerConfig(
        summary_chain_max_chars=40,
        summary_digest_max_chars=30,
        ledger_keep_rounds=2,
        ledger_digest_max_chars=60,
    )
    state = _chain_state(
        ["第一章很长的摘要句子甲。", "第二章很长的摘要句子乙。", "第三章末尾摘要保留原文。"]
    )
    segments = extract_summary_chain(state, {}, config)
    chain = {segment.name: segment.text for segment in segments}["summary_chain"]
    assert "（更早 " in chain and "章摘要已省略）" in chain
    assert "第一章很长的摘要句子甲" not in chain
    assert "第三章末尾摘要保留原文。" in chain


def test_摘要链无前章时两段皆空():
    state = _chain_state(["第一章摘要。"])
    segments = extract_summary_chain(state, {"chapter_id": "ch1"}, _DEFAULT_CONFIG)
    by_name = {segment.name: segment.text for segment in segments}
    assert by_name["summary_chain"] == ""
    assert by_name["prev_chapter_summary"] == ""


# ---------- 修订台账 ----------


def _ledger_state(rounds: list[RevisionRound]) -> WritingAgentState:
    return WritingAgentState(revision_ledger=rounds)


def test_台账不足K轮全原文():
    state = _make_state()
    context = assemble(state, "human_review_gate", config=_DEFAULT_CONFIG)
    ledger = context.text("revision_ledger")
    assert "第1轮：第一轮意见原文。后半句。" in ledger
    assert "第2轮：第二轮意见原文。" in ledger
    assert "[rewrite_only] ch1：收紧语气" in ledger


def test_台账超过K轮_早期轮次为一句话摘要且优先用已有digest():
    rounds = [
        RevisionRound(round_no=1, raw_feedback="第一轮原文。多余部分。", digest="第一轮既有摘要"),
        RevisionRound(round_no=2, raw_feedback="第二轮原文很长。多余部分。"),
        RevisionRound(round_no=3, raw_feedback="第三轮原文。"),
        RevisionRound(round_no=4, raw_feedback="第四轮原文。"),
    ]
    context = assemble(_ledger_state(rounds), "human_review_gate", config=_DEFAULT_CONFIG)
    ledger = context.text("revision_ledger")
    # 第 1 轮优先用既有 digest；第 2 轮 digest 为 None，确定性生成首句摘要。
    assert "第1轮（摘要）：第一轮既有摘要" in ledger
    assert "第一轮原文" not in ledger
    assert "第2轮（摘要）：第二轮原文很长。" in ledger
    assert "多余部分" not in ledger
    # 最近 2 轮原文注入。
    assert "第3轮：第三轮原文。" in ledger
    assert "第4轮：第四轮原文。" in ledger


def test_台账为空时段文本为空串():
    context = assemble(_ledger_state([]), "human_review_gate", config=_DEFAULT_CONFIG)
    assert context.text("revision_ledger") == ""


def test_多轮迭代不失忆_第1轮意见摘要仍在装配结果中():
    rounds = [
        RevisionRound(round_no=i, raw_feedback=f"第{i}轮独特意见内容。补充细节。")
        for i in range(1, 5)
    ]
    context = assemble(
        _ledger_state(rounds),
        "human_review_gate",
        config=_DEFAULT_CONFIG,
        feedback="第5轮新意见",
    )
    ledger = context.text("revision_ledger")
    assert "第1轮独特意见内容。" in ledger
    assert context.text("user_feedback") == "第5轮新意见"


# ---------- 引文库摘要与章节素材 ----------


def test_引文库摘要_总条数与分章通过未通过计数():
    segments = extract_citation_digest(_make_state(), {}, _DEFAULT_CONFIG)
    digest = segments[0].text
    assert "引文库共 3 条素材。" in digest
    assert "章节 ch1：通过 1 条，未通过 1 条" in digest
    assert "章节 ch2：通过 1 条，未通过 0 条" in digest


def test_章节素材_只取该章pass素材并JSON序列化():
    context = assemble(
        _make_state(), "writing_orchestrator", config=_DEFAULT_CONFIG, chapter_id="ch1"
    )
    materials = json.loads(context.text("chapter_materials"))
    assert [material["id"] for material in materials] == ["m-1"]
    assert materials[0]["excerpt"] == "摘录 m-1"


def test_章节素材_缺chapter_id时无该段且装配不抛错():
    context = assemble(_make_state(), "writing_orchestrator", config=_DEFAULT_CONFIG)
    assert "chapter_materials" not in context.segments
    assert context.text("chapter_materials", "无") == "无"


# ---------- 章节草稿与被引素材 ----------


def test_章节草稿_正文全文与角标命中的被引素材():
    segments = extract_chapter_draft(_make_state(), {"chapter_id": "ch2"}, _DEFAULT_CONFIG)
    by_name = {segment.name: segment.text for segment in segments}
    assert by_name["chapter_text"] == "第二章正文 [m-3] [m-1]"
    cited = json.loads(by_name["cited_materials"])
    assert {item["id"] for item in cited} == {"m-1", "m-3"}
    assert all("excerpt" in item for item in cited)


def test_章节草稿_缺chapter_id或草稿不存在返回空段列表():
    state = _make_state()
    assert extract_chapter_draft(state, {}, _DEFAULT_CONFIG) == []
    assert extract_chapter_draft(state, {"chapter_id": "ch9"}, _DEFAULT_CONFIG) == []


# ---------- 章节清单 ----------


def test_章节清单_id与标题逐行():
    context = assemble(_make_state(), "human_review_gate", config=_DEFAULT_CONFIG)
    assert context.text("chapter_list") == "ch1 第一章\nch2 第二章\nch3 第三章"


# ---------- 覆盖 State 后装配 ----------


def test_assemble_with_覆盖字段生效且不修改原state():
    state = _make_state()
    snapshot = copy.deepcopy(state)
    context = assemble_with(
        state, {"citation_library": []}, "search_agent", config=_DEFAULT_CONFIG
    )
    assert "引文库共 0 条素材。" in context.text("citation_digest")
    assert state == snapshot


# ---------- 配方覆盖 ----------


def test_配方注册表键与运行单元一致():
    assert set(RECIPES) == set(RUNTIME_UNITS)


def test_写作配方只留实际消费的段且共享预算覆盖():
    for unit in ("writing_orchestrator", "rewriter_loop"):
        assert RECIPES[unit].extractors == (
            extract_summary_chain,
            extract_chapter_materials,
        )
    assert RECIPES["writing_orchestrator"].budget is not None
    assert RECIPES["writing_orchestrator"].budget is RECIPES["rewriter_loop"].budget


# ---------- 每配方专属 token 预算 ----------


def test_配方预算覆盖生效_写作单元摘要链阈值放宽():
    # 全局阈值收得很小，直接调提取器必触发压缩。
    config = AssemblerConfig(
        summary_chain_max_chars=10,
        summary_digest_max_chars=60,
        ledger_keep_rounds=2,
        ledger_digest_max_chars=60,
    )
    state = _chain_state(["第一章摘要。补充说明。", "第二章摘要。"])
    direct = {s.name: s.text for s in extract_summary_chain(state, {}, config)}
    assert "补充说明" not in direct["summary_chain"]
    # 经写作配方装配时，专属预算覆盖把阈值放宽到 1200，不触发压缩。
    context = assemble(state, "writing_orchestrator", config=config)
    assert "补充说明。" in context.text("summary_chain")


def test_配方预算未覆盖字段回落全局配置():
    # 摘要链超过配方覆盖的 1200 阈值触发压缩；每章摘要保留字符数
    # 未被覆盖，回落全局配置的 5。
    config = AssemblerConfig(
        summary_chain_max_chars=10,
        summary_digest_max_chars=5,
        ledger_keep_rounds=2,
        ledger_digest_max_chars=60,
    )
    state = _chain_state(["长" * 1300, "末章摘要。"])
    context = assemble(state, "writing_orchestrator", config=config)
    chain = context.text("summary_chain")
    assert "长长长长长…" in chain
    assert "长" * 6 not in chain
    assert "末章摘要。" in chain
