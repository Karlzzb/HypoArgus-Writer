"""LlmWriterClient 真实适配器的契约测试：FakeLLM 预置文本应答驱动。"""

import json
from typing import Any

import pytest

from agents.rewriter_loop import LlmWriterClient, Violation
from domain.events import CONTENT_DELTA
from llm.llm_client import FakeLLM, Message, StreamChunk
from llm.llm_json import JSON_ONLY_RULE

_STYLE_PROSE = "风格指南散文片段：公文范式与子风格约束。"


def _make_client(llm: Any, **kwargs: Any) -> LlmWriterClient:
    """构造客户端：默认无事件钩子；逐字流测试传 event_hook / flush_chars。"""
    return LlmWriterClient(llm, **kwargs)


def _writer_json(text: str, summary: str = "一行摘要") -> str:
    return json.dumps({"chapter_text": text, "chapter_summary": summary}, ensure_ascii=False)


class _RaisingLLM:
    """每次调用都抛异常的假 LLM：验证异常重试与重抛路径。

    draft/revise 经 stream 消费，audit 经 invoke 消费——两路都抛同款异常，
    保证全异常路径在流式形态下仍触发退化重试与重抛。
    """

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def metadata(self) -> dict[str, str]:
        return {"model": "raising-llm", "base_url": "fake://"}

    def invoke(self, messages: list[Message]) -> str:
        self.call_count += 1
        raise RuntimeError(f"模拟网络故障 #{self.call_count}")

    def stream(self, messages: list[Message]):  # type: ignore[no-untyped-def]
        self.call_count += 1
        raise RuntimeError(f"模拟网络故障 #{self.call_count}")


class _ThinkingLLM:
    """确定性吐思考 + 正文的假 LLM：思考片段不进 JSON 抽取、正文片段进。

    用于验证逐字流能区分 content / thinking 两 kind：思考由 DeltaMerger
    单独缓冲外发、正文经 JsonFieldExtractor 抽出纯值后外发。
    """

    def __init__(self, content_json: str, thinking: str = "推理片段。") -> None:
        self._content_json = content_json
        self._thinking = thinking

    @property
    def metadata(self) -> dict[str, str]:
        return {"model": "thinking-llm", "base_url": "fake://"}

    def stream(self, messages: list[Message]):  # type: ignore[no-untyped-def]
        # 思考整段先吐、再吐正文 JSON（与真实思考开启模型的片段顺序一致）。
        if self._thinking:
            yield StreamChunk("thinking", self._thinking)
        # 正文按 8 字符切片吐，模拟真实流式分块。
        for start in range(0, len(self._content_json), 8):
            yield StreamChunk("content", self._content_json[start : start + 8])

    def invoke(self, messages: list[Message]) -> str:
        # audit 路径不进本类；保留以满足 LLM 协议形态，本测试不调用。
        return self._content_json


def test_真实适配器_draft正常_信封字段与提示词内容合规(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=[_writer_json("## 一、示例章节\n正文。[m-h-1]")])
    envelope = _make_client(llm).draft(draft_task, _STYLE_PROSE)

    assert envelope.chapter_text == "## 一、示例章节\n正文。[m-h-1]"
    assert envelope.chapter_summary == "一行摘要"
    assert envelope.attempts == 1
    assert envelope.degraded is False

    [messages] = llm.calls
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert messages[0]["role"] == "system"
    # system = 指令 + 风格指南散文 + JSON-only 规则。
    assert _STYLE_PROSE in system
    assert JSON_ONLY_RULE in system
    # user 上下文块含文种/层次/素材 id/假说 id/上一章摘要。
    assert "文种：人才培养方案" in user
    assert "层次：本科" in user
    assert "m-h-1" in user
    assert "h-1" in user
    assert "上一章摘要：已完成背景铺陈。" in user
    assert "【强支撑素材】" in user
    assert "可作为量化断言、数据与结论的直接依据" in user
    # 只喂 pass 素材：fail 素材 id 不出现在提示词。
    assert "m-fail-x" not in user


def test_真实适配器_只渲染当前章节假说可引用素材(draft_task: dict[str, Any]) -> None:
    draft_task["materials"].append(
        {
            "id": "m-other-chapter",
            "hypothesis_id": "other-chapter-h1",
            "source": "其他章来源",
            "url": "https://example.com/other",
            "source_kind": "web",
            "source_ref": {"url": "https://example.com/other"},
            "excerpt": "其他章摘录",
            "relevance_score": 0.95,
            "verdict": "pass",
        }
    )
    llm = FakeLLM(responses=[_writer_json("## 一、示例章节\n正文。[m-h-1]")])
    _make_client(llm).draft(draft_task, _STYLE_PROSE)

    user = llm.calls[0][1]["content"]
    assert "m-h-1" in user
    assert "m-other-chapter" not in user
    assert "other-chapter-h1" not in user


def test_真实适配器_当前章节无假说时可引用池失败关闭(draft_task: dict[str, Any]) -> None:
    draft_task["chapter_spec"]["hypotheses"] = []
    llm = FakeLLM(responses=[_writer_json("## 一、示例章节\n定性陈述正文。")])
    _make_client(llm).draft(draft_task, _STYLE_PROSE)

    user = llm.calls[0][1]["content"]
    assert "本章无可引素材" in user
    assert "不得出现任何" in user and "角标" in user
    assert "m-h-1" not in user
    assert "m-h-2" not in user


def test_真实适配器_弱佐证素材分组渲染并进提示词(draft_task: dict[str, Any]) -> None:
    """杠杆②：inconclusive 弱佐证进写作池但与 pass 强支撑分组标注；fail 仍排除；
    系统提示词含弱佐证措辞规则。"""
    draft_task["materials"].append(
        {
            "id": "m-weak-1",
            "hypothesis_id": "h-1",
            "source": "弱来源",
            "url": "https://example.com/m-weak-1",
            "source_kind": "web",
            "source_ref": {"url": "https://example.com/m-weak-1"},
            "excerpt": "弱摘录",
            "relevance_score": 0.3,
            "verdict": "inconclusive",
        }
    )
    llm = FakeLLM(responses=[_writer_json("## 一、示例章节\n正文。[m-h-1]")])
    _make_client(llm).draft(draft_task, _STYLE_PROSE)

    [messages] = llm.calls
    system = messages[0]["content"]
    user = messages[1]["content"]
    # 素材池按佐证强度分两节，强支撑在弱佐证之前。
    assert "【强支撑素材】" in user and "【弱佐证素材】" in user
    assert user.index("【强支撑素材】") < user.index("【弱佐证素材】")
    # 弱佐证素材进写作池（放宽过滤），且排在弱佐证节内。
    assert "m-weak-1" in user
    assert user.index("m-h-1") < user.index("m-weak-1")
    # fail 素材仍不进池。
    assert "m-fail-x" not in user
    # 系统提示词含弱佐证措辞规则（强支撑支撑量化断言、弱佐证只作背景提示）。
    assert "弱佐证素材" in system


def test_真实适配器_素材池为空_注入禁角标令且不渲染素材池块(
    draft_task: dict[str, Any],
) -> None:
    """根因修复（issue #62 收口发现）：素材池为空时上下文块显式禁角标。

    旧实现 materials 为空时整段省略 material_block，系统提示词「引用角标
    （仅当传入了素材池时）」留下语义缝隙，真实模型在无池时臆造 [素材id-N]
    占位角标——既无来源又对 reconcile 的 ASCII 角标模式隐形。根因修复：无池
    时显式告知 writer 不得出现任何角标、不臆造素材 id。
    """
    draft_task["materials"] = []
    llm = FakeLLM(responses=[_writer_json("## 一、示例章节\n定性陈述正文。")])
    _make_client(llm).draft(draft_task, _STYLE_PROSE)

    [messages] = llm.calls
    user = messages[1]["content"]
    # 显式禁令到位：不得出现角标、不臆造素材 id。
    assert "不得出现任何" in user and "角标" in user
    assert "不得臆造素材 id" in user
    # 无池时不渲染素材池块与假说列表块（避免给模型一堆无 id 可引的假说）。
    assert "仅可引用池内 id" not in user
    assert "假说列表" not in user
    # 不残留任何 ASCII 素材 id（无池本就无 id）。
    assert "m-" not in user


def test_真实适配器_draft解析失败_重试后成功(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=["这不是 JSON", _writer_json("正文。")])
    envelope = _make_client(llm).draft(draft_task, _STYLE_PROSE)

    assert envelope.chapter_text == "正文。"
    assert envelope.attempts == 2
    assert len(llm.calls) == 2


def test_真实适配器_draft空正文耗尽_返回最后一次诚实结果(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=[_writer_json(""), _writer_json(""), _writer_json("", "末次摘要")])
    envelope = _make_client(llm).draft(draft_task, _STYLE_PROSE)

    assert envelope.chapter_text == ""
    assert envelope.chapter_summary == "末次摘要"
    assert envelope.degraded is True
    assert envelope.attempts == 3
    assert len(llm.calls) == 3


def test_真实适配器_draft末轮退化在信封之后_attempts按总轮次回填(
    draft_task: dict[str, Any],
) -> None:
    # 第 1 轮解析失败、第 2 轮拿到空正文信封、第 3 轮结构非法：
    # 返回第 2 轮的诚实信封，但 attempts 须为实际执行的总轮次 3。
    llm = FakeLLM(
        responses=[
            "这不是 JSON",
            _writer_json("", "第二轮摘要"),
            json.dumps({"chapter_text": 123}),
        ]
    )
    envelope = _make_client(llm).draft(draft_task, _STYLE_PROSE)

    assert envelope.chapter_text == ""
    assert envelope.chapter_summary == "第二轮摘要"
    assert envelope.degraded is True
    assert envelope.attempts == 3
    assert len(llm.calls) == 3


def test_真实适配器_draft全异常_重抛最后一个异常(draft_task: dict[str, Any]) -> None:
    llm = _RaisingLLM()
    with pytest.raises(RuntimeError) as excinfo:
        _make_client(llm).draft(draft_task, _STYLE_PROSE)
    assert "模拟网络故障 #3" in str(excinfo.value)
    assert llm.call_count == 3


def test_真实适配器_修一次口径_提示词含违规清单(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=[_writer_json("修后正文。")])
    violations = [Violation(rule="oral_blacklist", message="出现口语化表达「我们」。")]
    _make_client(llm).draft(draft_task, _STYLE_PROSE, fix_violations=violations)

    user = llm.calls[0][1]["content"]
    assert "[oral_blacklist] 出现口语化表达「我们」。" in user
    assert "重写本章正文与一行摘要时全部规避" in user


def test_真实适配器_revise_提示词含现有正文与分区修订说明(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=[_writer_json("改后正文。")])
    draft_task["mode"] = "revise"
    draft_task["revision_note"] = {
        "user_directives": "精简第一段",
        "rule_violations": [
            {
                "rule": "hypothesis_no_support",
                "location_excerpt": "论点乙段落",
                "guidance": "为论点乙补充数据佐证",
                "severity": "error",
            }
        ],
        "conflict_hints": [{"description": "用户要求精简与字数下限冲突，以用户指令为准"}],
        "passed": False,
    }
    draft_task["current_text"] = "现有正文初稿。[m-h-1]"
    _make_client(llm).revise(draft_task, _STYLE_PROSE)

    user = llm.calls[0][1]["content"]
    assert "现有正文：\n现有正文初稿。[m-h-1]" in user
    # 分区式修订说明按优先级渲染：用户指令区逐字呈现，error 违规带位置与指导。
    assert "【用户指令（最高优先，逐字落实）】\n精简第一段" in user
    assert "[hypothesis_no_support]（位置：论点乙段落） 为论点乙补充数据佐证" in user
    assert "用户要求精简与字数下限冲突，以用户指令为准" in user
    assert "保持原样" in user


def test_真实适配器_revise修一次_同时含修订说明与违规清单(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=[_writer_json("再改正文。")])
    draft_task["mode"] = "revise"
    draft_task["revision_note"] = {
        "user_directives": "精简第一段",
        "rule_violations": [],
        "conflict_hints": [],
        "passed": True,
    }
    draft_task["current_text"] = "现有正文初稿。"
    violations = [Violation(rule="numbering", message="行起手「1、」命中禁用编号式。")]
    _make_client(llm).revise(draft_task, _STYLE_PROSE, fix_violations=violations)

    user = llm.calls[0][1]["content"]
    assert "现有正文：\n现有正文初稿。" in user
    assert "【用户指令（最高优先，逐字落实）】\n精简第一段" in user
    assert "[numbering] 行起手「1、」命中禁用编号式。" in user


def test_真实适配器_audit正常_条目转换正确(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "issues": [
                        {
                            "item": "unmarked_derived_content",
                            "material_id": "m-h-1",
                            "excerpt": "疑似片段",
                        }
                    ]
                }
            )
        ]
    )
    envelope = _make_client(llm).audit("本章正文。", draft_task)

    assert len(envelope.issues) == 1
    assert envelope.issues[0].item == "unmarked_derived_content"
    assert envelope.issues[0].label == "派生未标"
    assert envelope.issues[0].material_id == "m-h-1"
    assert envelope.issues[0].excerpt == "疑似片段"
    assert envelope.degraded is False
    # 自审指令置于 system（修复源仓库漏挂 system 的缺陷）；user 含素材池与正文。
    [messages] = llm.calls
    assert messages[0]["role"] == "system"
    assert "质检自审员" in messages[0]["content"]
    assert "【派生未标】" in messages[0]["content"]
    user = messages[1]["content"]
    assert "m-h-1" in user
    assert "本章正文。" in user


def test_真实适配器_audit空裁决_合法不重试(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=[json.dumps({"issues": []})])
    envelope = _make_client(llm).audit("本章正文。", draft_task)

    assert envelope.issues == []
    assert envelope.degraded is False
    assert len(llm.calls) == 1


def test_真实适配器_audit连续垃圾应答_降级空裁决不抛(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=["垃圾", "垃圾", "垃圾"])
    envelope = _make_client(llm).audit("本章正文。", draft_task)

    assert envelope.issues == []
    assert envelope.degraded is True
    assert len(llm.calls) == 3


def test_真实适配器_audit非法条目_防御性丢弃(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "issues": [
                        {"excerpt": "缺裁决项id的条目"},
                        {"item": "臆造的裁决项", "excerpt": "裁决项不在适用集内"},
                        {"item": "unmarked_derived_content", "excerpt": "派生未标缺素材id"},
                        {"item": "unmarked_derived_content", "material_id": "m-h-2"},
                        "不是对象的条目",
                    ]
                }
            )
        ]
    )
    envelope = _make_client(llm).audit("本章正文。", draft_task)

    assert len(envelope.issues) == 1
    assert envelope.issues[0].material_id == "m-h-2"
    assert envelope.issues[0].excerpt == ""


def test_真实适配器_audit按文种分派_调研报告拿到语义裁决项(draft_task: dict[str, Any]) -> None:
    # 调研报告 + 有素材：通用层「派生未标」与文种层「对比叙事」「四步递进」并集拼装。
    draft_task["doc_type"] = "调研报告"
    draft_task["doc_variant"] = None
    llm = FakeLLM(responses=[json.dumps({"issues": []})])
    _make_client(llm).audit("本章正文。", draft_task)

    system = llm.calls[0][0]["content"]
    assert "【派生未标】" in system
    assert "【对比叙事】" in system
    assert "横向比" in system
    assert "【四步递进】" in system
    assert "归因分析" in system
    # 依赖素材的裁决项声明 material_id 字段要求。
    assert "material_id" in system


def test_真实适配器_audit按文种分派_人培方案不被问调研报告裁决项(
    draft_task: dict[str, Any],
) -> None:
    llm = FakeLLM(responses=[json.dumps({"issues": []})])
    _make_client(llm).audit("本章正文。", draft_task)

    system = llm.calls[0][0]["content"]
    assert "【派生未标】" in system
    assert "对比叙事" not in system
    assert "四步递进" not in system


def test_真实适配器_audit调研报告无素材_只问语义裁决项且条目可解析(
    draft_task: dict[str, Any],
) -> None:
    # 素材池为空：依赖素材的「派生未标」不适用，语义裁决项照常自审；
    # 语义条目不带 material_id 也合法。
    draft_task["doc_type"] = "调研报告"
    draft_task["doc_variant"] = None
    for material in draft_task["materials"]:
        material["verdict"] = "fail"
    llm = FakeLLM(
        responses=[
            json.dumps(
                {"issues": [{"item": "comparison_narrative", "excerpt": "孤立数值断言句"}]}
            )
        ]
    )
    envelope = _make_client(llm).audit("本章正文。", draft_task)

    system = llm.calls[0][0]["content"]
    assert "派生未标" not in system
    assert "【对比叙事】" in system
    assert "【四步递进】" in system
    assert len(envelope.issues) == 1
    assert envelope.issues[0].item == "comparison_narrative"
    assert envelope.issues[0].label == "对比叙事"
    assert envelope.issues[0].material_id == ""


def test_真实适配器_audit无pass素材_仅审非依赖素材裁决项(draft_task: dict[str, Any]) -> None:
    # 共用 audit_items（ADR-0006）后，空素材池仍有不依赖素材的 warn 裁决项适用：
    # 自审照常发起，但只问非依赖素材项（章内连贯 / 摘要链一致），依赖素材项被剔除。
    for material in draft_task["materials"]:
        material["verdict"] = "fail"
    llm = FakeLLM(responses=[json.dumps({"issues": []}, ensure_ascii=False)])
    envelope = _make_client(llm).audit("本章正文。", draft_task)

    assert envelope.issues == []
    assert envelope.degraded is False
    assert len(llm.calls) == 1
    system = llm.calls[0][0]["content"]
    # 依赖素材的裁决项（派生未标 / 论证质量）不出现在提示词；非依赖项出现。
    assert "intra_chapter_coherence" in system
    assert "summary_chain_consistency" in system
    assert "unmarked_derived_content" not in system
    assert "weak_material_assertion" not in system
    user = llm.calls[0][1]["content"]
    assert "本章无可引素材" in user
    assert "不得出现任何" in user and "角标" in user
    assert "不得生成 `[1]`" in user
    assert "参考文献列表" in user
    assert "仅可引用池内 id" not in user
    assert "m-h-1" not in user


def test_真实适配器_draft系统提示词_不含双方括号残留(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=[_writer_json("正文。")])
    _make_client(llm).draft(draft_task, _STYLE_PROSE)

    assert "[[" not in llm.calls[0][0]["content"]


def test_真实适配器_draft字数目标_叙述章型注入中上限提示(draft_task: dict[str, Any]) -> None:
    # 章标题非 table_required 章型 → 叙述章型 → 字数目标块含「中上限」提示。
    draft_task["chapter_spec"]["title"] = "一、总则"
    llm = FakeLLM(responses=[_writer_json("正文。")])
    _make_client(llm).draft(draft_task, _STYLE_PROSE)

    user = llm.calls[0][1]["content"]
    assert "本章目标字数" in user
    assert "2000～5000" in user
    assert "600～1500" in user
    assert "200～500" in user
    assert "中上限" in user
    assert "表型章" not in user


def test_真实适配器_draft字数目标_表章注入中下限且不得凑段(draft_task: dict[str, Any]) -> None:
    # 职业面向章为 table_required → 表章 → 字数目标块含「中下限」「不得表外堆砌」。
    # 章型经任务包携带并优先于标题解析（ADR-0005），须与标题一并同步。
    draft_task["chapter_spec"]["title"] = "五、职业面向"
    draft_task["chapter_spec"]["chapter_type"] = "职业面向"
    llm = FakeLLM(responses=[_writer_json("正文。")])
    _make_client(llm).draft(draft_task, _STYLE_PROSE)

    user = llm.calls[0][1]["content"]
    assert "本章目标字数" in user
    assert "表型章" in user
    assert "中下限" in user
    assert "不得在表外堆砌" in user


def test_真实适配器_revise字数目标_同样注入目标区间(draft_task: dict[str, Any]) -> None:
    # revise 与 draft 共用上下文块 → 修订提示词同样携带本章目标字数区间。
    draft_task["chapter_spec"]["title"] = "一、总则"
    draft_task["mode"] = "revise"
    draft_task["revision_note"] = {
        "user_directives": "精简第一段",
        "rule_violations": [],
        "conflict_hints": [],
        "passed": True,
    }
    draft_task["current_text"] = "现有正文初稿。"
    llm = FakeLLM(responses=[_writer_json("改后正文。")])
    _make_client(llm).revise(draft_task, _STYLE_PROSE)

    user = llm.calls[0][1]["content"]
    assert "本章目标字数" in user
    assert "2000～5000" in user


def test_真实适配器_系统提示词再平衡_论证指令上位(draft_task: dict[str, Any]) -> None:
    llm = FakeLLM(responses=[_writer_json("正文。")])
    _make_client(llm).draft(draft_task, _STYLE_PROSE)

    system = llm.calls[0][0]["content"]
    # 最重要的写作要求置于指令首行（论证指令上位）。
    lines = [line for line in system.split("\n") if line.strip()]
    first_instruction_line = next(
        (line for line in lines if "充分论证每个论点" in line or "最重要的写作要求" in line), None
    )
    assert first_instruction_line is not None
    # 空泛总结词禁令补充「显著提升」「有效解决」等定性断言词。
    assert "显著提升" in system or "有效解决" in system



def test_真实适配器_文种与变体逐任务取自任务包(draft_task: dict[str, Any]) -> None:
    """同一客户端连续服务不同文种的任务包：上下文块随任务切换，不固化构造期配置。

    「层次」行取变体推导的 tier，与 lint 的推导同源：高职变体注入 层次：高职；
    无变体文种回落缺省 层次：本科（lint 也按本科执行，两侧永远一致）。
    """
    llm = FakeLLM(responses=[_writer_json("正文一。"), _writer_json("正文二。")])
    client = _make_client(llm)

    draft_task["doc_variant"] = "高职"
    client.draft(draft_task, _STYLE_PROSE)
    draft_task["doc_type"] = "汇报材料"
    draft_task["doc_variant"] = None
    client.draft(draft_task, _STYLE_PROSE)

    first_user = llm.calls[0][1]["content"]
    second_user = llm.calls[1][1]["content"]
    assert "文种：人才培养方案" in first_user
    assert "层次：高职" in first_user
    assert "文种：汇报材料" in second_user
    assert "层次：本科" in second_user


def test_真实适配器_任务包缺文种字段_回落通用公文与缺省层次(draft_task: dict[str, Any]) -> None:
    del draft_task["doc_type"]
    del draft_task["doc_variant"]
    llm = FakeLLM(responses=[_writer_json("正文。")])
    _make_client(llm).draft(draft_task, _STYLE_PROSE)

    user = llm.calls[0][1]["content"]
    assert "文种：通用公文" in user
    assert "层次：本科" in user


# ---- 逐字流 content_delta（issue #59）：流式 draft/revise 上网线契约 ----


def _capture_hook() -> tuple[list[tuple[str, dict]], Any]:
    """构造捕获钩子与事件列表：(event_type, payload) 入列。"""
    events: list[tuple[str, dict]] = []
    return events, (lambda etype, payload: events.append((etype, dict(payload))))


def test_逐字流_draft上网线载荷字段齐全且content拼接为正文(
    draft_task: dict[str, Any],
) -> None:
    """draft 经 stream 消费：content_delta 帧载荷字段齐全、拼接 == chapter_text。

    FakeLLM 默认 chunk_size=8，故 8 字符一切；flush_chars 设小以保证多帧。
    """
    chapter_text = "## 一、示例章节\n正文。[m-h-1]"
    llm = FakeLLM(responses=[_writer_json(chapter_text)])
    events, hook = _capture_hook()
    envelope = _make_client(
        llm, flush_chars=4, flush_ms=0, event_hook=hook
    ).draft(draft_task, _STYLE_PROSE)

    deltas = [p for et, p in events if et == CONTENT_DELTA]
    assert deltas, "draft 调用须发 content_delta 事件"
    # 每帧载荷字段齐全。
    for payload in deltas:
        assert payload["unit"] == "rewriter_loop"
        assert payload["chapter_id"] == draft_task["chapter_spec"]["id"]
        assert payload["mode"] == "draft"
        assert payload["kind"] == "content"
        assert payload["attempt"] == 1
    # sequence 在 attempt 内单调递增、从 0 起。
    seqs = [p["sequence"] for p in deltas]
    assert seqs == sorted(seqs)
    assert seqs[0] == 0
    # 拼接所有 content 帧的 delta == 终态信封 chapter_text（与 chapter_ready 整块一致）。
    joined = "".join(p["delta"] for p in deltas)
    assert joined == envelope.chapter_text == chapter_text


def test_逐字流_revise同样上网线mode为revise(draft_task: dict[str, Any]) -> None:
    """revise 经 stream 消费：content_delta 的 mode==revise、拼接为改写后正文。"""
    draft_task["mode"] = "revise"
    draft_task["revision_note"] = {
        "user_directives": "精简第一段",
        "rule_violations": [],
        "conflict_hints": [],
        "passed": True,
    }
    draft_task["current_text"] = "现有正文初稿。[m-h-1]"
    llm = FakeLLM(responses=[_writer_json("改后正文。[m-h-1]")])
    events, hook = _capture_hook()
    envelope = _make_client(
        llm, flush_chars=4, flush_ms=0, event_hook=hook
    ).revise(draft_task, _STYLE_PROSE)

    deltas = [p for et, p in events if et == CONTENT_DELTA]
    assert deltas
    assert all(p["mode"] == "revise" for p in deltas)
    assert all(p["chapter_id"] == draft_task["chapter_spec"]["id"] for p in deltas)
    joined = "".join(p["delta"] for p in deltas)
    assert joined == envelope.chapter_text == "改后正文。[m-h-1]"


def test_逐字流_退化重试attempt递增且旧帧丢弃重建(draft_task: dict[str, Any]) -> None:
    """流中途 finish 抛 FieldExtractionError → 本轮退化 → 下轮 attempt+1。

    第 1 轮：chapter_text 值未闭合（``{"chapter_text": "部分``），finish 抛错；
    第 2 轮：合法 JSON，成功。两轮都发 content_delta，但 attempt 不同——
    调用方按更高 attempt 丢弃旧增量、从零重建：第 2 轮 sequence 从 0 复位、
    拼接 == 终态正文（丢弃重建语义）。
    """
    # 第 1 轮 JSON 不闭合 → extractor.finish 抛 FieldExtractionError；
    # 第 2 轮合法 → 成功。
    bad_response = '{"chapter_text": "部分未闭合'
    good_response = _writer_json("重建后正文。[m-h-1]")
    llm = FakeLLM(responses=[bad_response, good_response])
    events, hook = _capture_hook()
    envelope = _make_client(
        llm, max_attempts=2, flush_chars=4, flush_ms=0, event_hook=hook
    ).draft(draft_task, _STYLE_PROSE)

    deltas = [p for et, p in events if et == CONTENT_DELTA]
    # 两轮都发了帧（第 1 轮在 finish 抛错前已 flush 部分 content 帧）。
    attempt1_deltas = [p for p in deltas if p["attempt"] == 1]
    attempt2_deltas = [p for p in deltas if p["attempt"] == 2]
    assert attempt1_deltas, "第 1 轮退化前应有 content_delta 帧"
    assert attempt2_deltas, "第 2 轮成功应有 content_delta 帧"
    # 第 2 轮 sequence 从 0 复位（新 attempt 重建）。
    assert attempt2_deltas[0]["sequence"] == 0
    assert [p["sequence"] for p in attempt2_deltas] == sorted(
        p["sequence"] for p in attempt2_deltas
    )
    # 调用方按第 2 轮重建：拼接 attempt==2 的 content 帧 == 终态正文。
    joined = "".join(p["delta"] for p in attempt2_deltas)
    assert joined == envelope.chapter_text == "重建后正文。[m-h-1]"
    # attempts 回填总轮次 2。
    assert envelope.attempts == 2


def test_逐字流_思考kind可区分且正文仍拼接一致(draft_task: dict[str, Any]) -> None:
    """思考开启模型：thinking 与 content 两 kind 都上网线、可区分。

    思考片段不进 JsonFieldExtractor 抽取（推理 CoT 不在 chapter_text 字段内），
    故 thinking 帧与 content 帧分别由 DeltaMerger 攒帧外发；sequence 在
    同一 attempt 内跨 kind 共享、单调递增。
    """
    content_json = _writer_json("正文。", "摘要。")
    llm = _ThinkingLLM(content_json=content_json, thinking="推理 CoT 片段。")
    events, hook = _capture_hook()
    envelope = _make_client(
        llm, flush_chars=2, flush_ms=0, event_hook=hook
    ).draft(draft_task, _STYLE_PROSE)

    deltas = [p for et, p in events if et == CONTENT_DELTA]
    thinking_deltas = [p for p in deltas if p["kind"] == "thinking"]
    content_deltas = [p for p in deltas if p["kind"] == "content"]
    # 两 kind 都有帧且可区分。
    assert thinking_deltas, "思考帧须上网线"
    assert content_deltas, "正文帧须上网线"
    # 思考帧拼接 == 原始思考文本。
    assert "".join(p["delta"] for p in thinking_deltas) == "推理 CoT 片段。"
    # 正文帧拼接 == 终态信封 chapter_text。
    assert "".join(p["delta"] for p in content_deltas) == envelope.chapter_text == "正文。"
    # sequence 跨 kind 在同一 attempt 内单调递增（共享计数器）。
    seqs = [p["sequence"] for p in deltas]
    assert seqs == sorted(seqs)
    assert seqs[0] == 0


def test_逐字流_audit不上网线保持invoke(draft_task: dict[str, Any]) -> None:
    """audit 仍走 invoke、不发任何 content_delta（逐字流只覆盖 draft/revise）。"""
    llm = FakeLLM(responses=[json.dumps({"issues": []})])
    events, hook = _capture_hook()
    _make_client(llm, flush_chars=4, flush_ms=0, event_hook=hook).audit(
        "本章正文。", draft_task
    )
    assert [et for et, _ in events if et == CONTENT_DELTA] == []
    # audit 仍经 invoke（FakeLLM.calls 记录的是 _dispatch 调用，invoke/stream 共用）。
    assert len(llm.calls) == 1
