"""human_review_gate 主节点：人工中断点与用户意见解析器。

用 LangGraph interrupt 实现真实中断：中断载荷只含元数据（迭代轮次、章节 id
列表与未决引文警告），不携带正文全文。恢复值契约为 dict：
{"action": "finalize"} 定稿；{"action": "revise", "feedback": "自然语言意见"}
经 LLM 一次调用解析为修订指令列表，程序侧过滤非法条目并追加修订台账。

本节点是全流程唯一安全汇点：恢复值契约不符或意见解析不出任何有效指令时，
不抛异常终止，而是携错误说明重新中断等待人工重新提交，保证系统永不转死。
"""

from typing import Any, Protocol

from langgraph.types import interrupt

from llm_client import LLM, LLMFactory
from llm_json import JSON_ONLY_RULE
from llm_json import invoke_json
from state import (
    ChapterSpec,
    RevisionDirective,
    RevisionRound,
    WorkflowStatus,
    WritingAgentState,
)

# 修订指令类型的合法取值；类型不在其中的应答项被程序丢弃。
DIRECTIVE_TYPES: tuple[str, ...] = ("rewrite_only", "evidence_augmented")

_TYPE_GUIDE = (
    "两类修订类型定义：\n"
    "- rewrite_only 纯改写：不需要新证据、仅调整文字表达；\n"
    "- evidence_augmented 补充佐证：需要新素材支撑。"
)


class HumanReviewGateNode(Protocol):
    """节点函数类型：入参与返回均为图状态（state 具名，满足 LangGraph 节点协议）。"""

    def __call__(self, state: WritingAgentState) -> WritingAgentState: ...


def _validate_decision(decision: Any) -> tuple[str, str]:
    """校验恢复值契约，返回（action, feedback）；契约不符抛中文 ValueError。"""
    if not isinstance(decision, dict):
        raise ValueError("人工中断点的恢复值必须是 JSON 对象（dict）")
    action = decision.get("action")
    if action == "finalize":
        return "finalize", ""
    if action == "revise":
        feedback = decision.get("feedback")
        if not (isinstance(feedback, str) and feedback.strip()):
            raise ValueError("恢复值 action=revise 时必须携带非空的 feedback 意见文本")
        return "revise", feedback.strip()
    raise ValueError(f"人工中断点的恢复值 action 非法：{action!r}，只接受 finalize 或 revise")


def _parse_directives(
    llm: LLM, outline: list[ChapterSpec], feedback: str
) -> list[RevisionDirective]:
    """LLM 意见解析：把自然语言意见拆解为修订指令列表，程序侧过滤非法条目。"""
    chapter_ids = {chapter.id for chapter in outline}
    chapter_lines = "\n".join(
        f"- {chapter.id}：{chapter.title}" for chapter in outline
    )
    system = (
        "你是修订意见解析器。把用户一次提交的自然语言修改意见，"
        "拆解为逐章的结构化修订指令列表。\n"
        + _TYPE_GUIDE
        + "\n输出 JSON 数组，逐条一项："
        '{"target_chapter_id": "章节 id", '
        '"type": "rewrite_only" 或 "evidence_augmented", '
        '"instruction": "该章要做什么的一句话中文指令"}。'
        + JSON_ONLY_RULE
    )
    user = f"章节清单：\n{chapter_lines}\n\n用户修改意见：{feedback}"
    payload = invoke_json(llm, "修订意见解析", system, user, list)

    directives: list[RevisionDirective] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        target_chapter_id = item.get("target_chapter_id")
        directive_type = item.get("type")
        instruction = item.get("instruction")
        if target_chapter_id not in chapter_ids:
            continue
        if directive_type not in DIRECTIVE_TYPES:
            continue
        if not (isinstance(instruction, str) and instruction.strip()):
            continue
        directives.append(
            RevisionDirective(
                target_chapter_id=target_chapter_id,
                type=directive_type,
                instruction=instruction.strip(),
            )
        )
    if not directives:
        raise ValueError("用户意见解析不出任何有效修订指令，请人工确认意见内容")
    return directives


def make_human_review_gate_node(llm_factory: LLMFactory) -> HumanReviewGateNode:
    """构造 human_review_gate 节点函数。"""

    def node(state: WritingAgentState) -> WritingAgentState:
        outline = state.get("outline", [])
        error: str | None = None
        # 安全汇点循环：契约不符或解析失败都回到中断点重新等待人工，永不转死。
        while True:
            payload = {
                "iteration_round": state.get("iteration_round", 0),
                "chapter_ids": [chapter.id for chapter in outline],
                "citation_warnings": state.get("citation_warnings", []),
            }
            if error is not None:
                payload["error"] = error
            decision = interrupt(payload)
            try:
                action, feedback = _validate_decision(decision)
                if action == "finalize":
                    # 定稿不调 LLM，元数据无从获取，只记录单元名。
                    return WritingAgentState(
                        status=WorkflowStatus.FINISHED,
                        pending_directives=[],
                        citation_warnings=[],
                        current_node_llm_config={"unit": "human_review_gate"},
                    )
                llm = llm_factory("human_review_gate")
                directives = _parse_directives(llm, outline, feedback)
                break
            except ValueError as exc:
                error = str(exc)

        round_no = state.get("iteration_round", 0) + 1
        # 台账列表字段是整值覆盖语义，须带上既有旧轮次再追加新一轮。
        ledger = list(state.get("revision_ledger", [])) + [
            RevisionRound(
                round_no=round_no, raw_feedback=feedback, directives=directives
            )
        ]
        return WritingAgentState(
            pending_directives=directives,
            revision_ledger=ledger,
            iteration_round=round_no,
            citation_warnings=[],
            # 新一轮修订获得全新的终审重试预算。
            citation_retry_count=0,
            status=WorkflowStatus.AWAIT_USER_REVIEW,
            current_node_llm_config={"unit": "human_review_gate", **llm.metadata},
        )

    return node
