"""human_review_gate 主节点：人工中断点与用户意见解析器（含定位增强）。

用 LangGraph interrupt 实现真实中断：中断载荷只含元数据（迭代轮次、章节 id
列表、未决引文警告与篇级评审 warn 提示），不携带正文全文。恢复值契约为 dict：
{"action": "finalize"} 定稿；{"action": "revise", "feedback": "自然语言意见"}
经 LLM 一次调用解析为修订指令列表；{"action": "confirm"} 仅在大扇出确认
中断（载荷携 pending_confirmation 解析清单）时可用，确认后按清单执行。

定位增强（issue #49）：
- 两级定位——用户引用正文原文时，程序先对引文做确定性子串匹配（归一化去
  角标与空白），唯一命中直达目标章；未命中或多章命中时回退到 LLM 在解析
  调用中给出的章节判断（解析 prompt 携各章草稿摘要辅助判断）。
- 全局意见——LLM 标记 locate=global 后由程序侧确定性扇出为逐章指令。
- 大扇出确认——解析后受影响章数超过大纲一半时，携解析清单重新中断待人工
  confirm 后才执行（复用安全汇点重新中断模式）。
- 含混回问——意见含混或引文定位失败时，携 clarification_questions 重新
  中断回问用户，不猜测、不执行任何指令。

本节点是全流程唯一安全汇点：恢复值契约不符或意见解析不出任何有效指令时，
不抛异常终止，而是携错误说明重新中断等待人工重新提交，保证系统永不转死。
重新中断依赖 LangGraph 的节点重放语义：恢复时节点从头重放、既有 interrupt
按序返回历史恢复值。意见解析的 LLM 调用包在 LangGraph task（durable
execution）里，结果随任务写入落 checkpoint——重放时直接取缓存、不重复调用，
从而保证 confirm 执行的清单严格等于确认中断时回显给用户的那份解析结果。
"""

import re
from typing import Any, Protocol, cast, get_args

from langgraph.func import task
from langgraph.types import interrupt

from assembly.assembler_config import AssemblerConfig, load_assembler_config
from assembly.context_assembler import assemble, digest_of_round
from llm.llm_client import LLMFactory
from llm.llm_json import JSON_ONLY_RULE
from llm.llm_json import invoke_json
from domain.citation_reconciler import MARKER_PATTERN
from domain.state import (
    ChapterDraft,
    ChapterSpec,
    DirectiveType,
    RevisionDirective,
    RevisionRound,
    WorkflowStatus,
    WritingAgentState,
)

# 修订指令类型的合法取值（取自 DirectiveType 字面量单一事实源）；
# 类型不在其中的应答项被程序丢弃。
DIRECTIVE_TYPES: tuple[str, ...] = get_args(DirectiveType)

# 恢复值契约的合法动作全集：服务层与本节点共用的单一事实源（契约测试锚定）。
RESUME_ACTIONS: tuple[str, ...] = ("finalize", "revise", "confirm")

# 解析应答条目的定位方式合法取值。
_LOCATE_KINDS: tuple[str, ...] = ("chapter", "quote", "global", "unclear")

# 回问文本中引文片段的展示截断长度。
_QUOTE_PREVIEW_CHARS = 30

# 归一化匹配时剔除的空白字符。
_WHITESPACE_PATTERN = re.compile(r"\s+")

_TYPE_GUIDE = (
    "两类修订类型定义：\n"
    "- rewrite_only 纯改写：不需要新证据、仅调整文字表达；\n"
    "- evidence_augmented 补充佐证：需要新素材支撑。"
)

_LOCATE_GUIDE = (
    "四种定位方式（locate 字段）定义：\n"
    "- chapter：用户点名了具体章节，target_chapter_id 必填；\n"
    "- quote：用户引用了正文原文片段，quote 字段逐字照抄该引文，"
    "并在 target_chapter_id 给出你按章节摘要判断的最可能章节（判断不了则为 null）；\n"
    "- global：意见针对全篇各章普遍适用（如整体口吻、全篇格式），由系统扇出到每一章；\n"
    "- unclear：意见含混、定位不了或看不出要做什么——不要猜测，"
    "在 question 字段写出向用户回问的具体问题。"
)


class HumanReviewGateNode(Protocol):
    """节点函数类型：入参与返回均为图状态（state 具名，满足 LangGraph 节点协议）。"""

    def __call__(self, state: WritingAgentState) -> WritingAgentState: ...


def _normalized_for_match(text: str) -> str:
    """引文匹配归一化：剔除素材角标与全部空白，只留正文字符。"""
    return _WHITESPACE_PATTERN.sub("", MARKER_PATTERN.sub("", text))


def match_quote_chapters(quote: str, drafts: list[ChapterDraft]) -> list[str]:
    """确定性引文定位：归一化子串匹配，返回命中章节 id 列表（按草稿顺序）。

    纯函数：唯一命中（返回列表长度为 1）即直达目标章；空引文恒不命中。
    """
    needle = _normalized_for_match(quote)
    if not needle:
        return []
    return [
        draft.chapter_id
        for draft in drafts
        if needle in _normalized_for_match(draft.text)
    ]


def needs_fanout_confirmation(
    directives: list[RevisionDirective], outline: list[ChapterSpec]
) -> bool:
    """大扇出判定：受影响章数严格超过大纲一半时须先经人工确认。"""
    affected = {directive.target_chapter_id for directive in directives}
    return 2 * len(affected) > len(outline)


def _validate_decision(decision: Any, confirmable: bool) -> tuple[str, str]:
    """校验恢复值契约，返回（action, feedback）；契约不符抛中文 ValueError。

    confirm 仅在存在待确认的解析清单（confirmable=True）时合法。
    """
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
    if action == "confirm":
        if not confirmable:
            raise ValueError(
                "当前没有待确认的修订指令清单，confirm 不可用；请提交 finalize 或 revise"
            )
        return "confirm", ""
    raise ValueError(
        f"人工中断点的恢复值 action 非法：{action!r}，只接受 {'/'.join(RESUME_ACTIONS)}"
    )


def _quote_preview(quote: str) -> str:
    """回问文本中的引文预览：超长截断加省略号。"""
    stripped = quote.strip()
    if len(stripped) > _QUOTE_PREVIEW_CHARS:
        return stripped[:_QUOTE_PREVIEW_CHARS] + "…"
    return stripped


def resolve_directives(
    payload: list[Any],
    outline: list[ChapterSpec],
    drafts: list[ChapterDraft],
) -> tuple[list[RevisionDirective], list[str]]:
    """把 LLM 解析应答条目确定性归结为修订指令与回问问题，纯函数。

    逐条处理：chapter 直取（缺 locate 字段但带合法 target_chapter_id 的
    旧形态条目同此）；quote 先做确定性子串匹配（唯一命中直达），未命中回退
    LLM 给出的章节判断，两级都失败则生成回问；global 扇出为逐章指令；
    unclear 收集回问问题。类型非法、指令为空或章节 id 幻觉的条目按噪声丢弃。
    产出的指令按（章节, 类型, 指令）去重保序；只要存在回问问题，本轮不执行
    任何指令（不猜测），由调用方携问题重新中断。
    """
    chapter_ids = {chapter.id for chapter in outline}
    directives: list[RevisionDirective] = []
    questions: list[str] = []

    def _append(target: str, directive_type: str, instruction: str) -> None:
        # 调用点已按 DIRECTIVE_TYPES 校验过 directive_type，此处收窄为字面量类型。
        directives.append(
            RevisionDirective(
                target_chapter_id=target,
                type=cast(DirectiveType, directive_type),
                instruction=instruction,
            )
        )

    for item in payload:
        if not isinstance(item, dict):
            continue
        locate = item.get("locate")
        if locate is None:
            locate = "chapter"
        if locate not in _LOCATE_KINDS:
            continue
        if locate == "unclear":
            question = item.get("question")
            if isinstance(question, str) and question.strip():
                questions.append(question.strip())
            else:
                questions.append("该条意见含混，无法定位或解析，请补充说明后重新提交。")
            continue
        directive_type = item.get("type")
        instruction = item.get("instruction")
        if directive_type not in DIRECTIVE_TYPES:
            continue
        if not (isinstance(instruction, str) and instruction.strip()):
            continue
        instruction = instruction.strip()
        if locate == "global":
            for chapter in outline:
                _append(chapter.id, directive_type, instruction)
            continue
        if locate == "quote":
            quote = item.get("quote")
            if not (isinstance(quote, str) and quote.strip()):
                quote = ""
            hits = match_quote_chapters(quote, drafts) if quote else []
            if len(hits) == 1:
                _append(hits[0], directive_type, instruction)
                continue
            target = item.get("target_chapter_id")
            if target in chapter_ids:
                _append(str(target), directive_type, instruction)
                continue
            questions.append(
                f"无法定位引文「{_quote_preview(quote)}」所属的章节"
                f"（指令：{instruction}），请指明目标章节后重新提交。"
            )
            continue
        # locate == "chapter"：章节 id 幻觉按噪声丢弃（与既有过滤语义一致）。
        target = item.get("target_chapter_id")
        if target not in chapter_ids:
            continue
        _append(str(target), directive_type, instruction)

    deduped: list[RevisionDirective] = []
    seen: set[tuple[str, str, str]] = set()
    for directive in directives:
        key = (directive.target_chapter_id, directive.type, directive.instruction)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(directive)
    return deduped, questions


def _build_parse_messages(
    chapter_digest: str, revision_ledger: str, feedback: str
) -> tuple[str, str]:
    """构造意见解析 prompt（system, user），纯函数。

    章节清单（含各章草稿摘要，供 LLM 判断意见落章）、历史修订台账、本轮意见
    均取自装配段（chapter_list、revision_ledger、user_feedback）。
    历史台账仅作背景帮助 LLM 理解历次要求，只解析本轮意见。
    LLM 只产结构化条目，定位归结（引文匹配、扇出、回问）由程序确定性完成。
    """
    system = (
        "你是修订意见解析器。把用户一次提交的自然语言修改意见，"
        "拆解为结构化修订指令列表。\n"
        + _TYPE_GUIDE
        + "\n"
        + _LOCATE_GUIDE
        + "\n历史修订台账仅作背景帮助你理解历次要求，只解析本轮意见，"
        "不要为历史轮次生成指令。"
        "\n输出 JSON 数组，逐条一项："
        '{"locate": "chapter" 或 "quote" 或 "global" 或 "unclear", '
        '"target_chapter_id": "章节 id 或 null", '
        '"quote": "用户引用的正文原文片段或 null", '
        '"type": "rewrite_only" 或 "evidence_augmented", '
        '"instruction": "该章要做什么的一句话中文指令", '
        '"question": "locate=unclear 时向用户回问的问题，否则为 null"}。'
        + JSON_ONLY_RULE
    )
    ledger_block = f"历史修订台账（仅作背景）：\n{revision_ledger}\n\n" if revision_ledger else ""
    user = (
        f"章节清单（每行：id 标题：草稿摘要）：\n{chapter_digest}\n\n"
        f"{ledger_block}"
        f"本轮用户修改意见：{feedback}"
    )
    return system, user


def make_human_review_gate_node(
    llm_factory: LLMFactory, assembler_config: AssemblerConfig | None = None
) -> HumanReviewGateNode:
    """构造 human_review_gate 节点函数。

    assembler_config 为 None 时在节点执行时读取环境变量装配配置。
    """

    @task
    def _parse_feedback_task(
        chapter_digest: str, revision_ledger: str, feedback: str
    ) -> list[Any]:
        """意见解析的 LLM 调用（durable execution）。

        结果随任务写入落 checkpoint——confirm 恢复重放时直接取缓存、不重复
        调用，从而保证执行清单严格等于确认中断时回显给用户的那份解析结果，
        不因重放而漂移成另一份清单（真实 LLM 非确定，重放重解析会变）。
        闭包捕获 llm_factory 以保留测试桩注入点；返回纯标量/字典/列表的
        原始 JSON 负载，定位归结（resolve_directives）是确定性纯函数、在任务
        外重放执行，其结果随负载缓存天然一致。
        """
        llm = llm_factory("human_review_gate")
        system, user = _build_parse_messages(
            chapter_digest, revision_ledger, feedback
        )
        return invoke_json(llm, "修订意见解析", system, user, list)

    def node(state: WritingAgentState) -> WritingAgentState:
        config = assembler_config
        if config is None:
            config = load_assembler_config()
        outline = state.get("outline", [])
        drafts = state.get("chapter_drafts", [])
        outline_order = [chapter.id for chapter in outline]
        error: str | None = None
        questions: list[str] = []
        # 大扇出确认现场：待确认的解析清单与触发它的那轮意见原文。
        pending: list[RevisionDirective] | None = None
        pending_feedback = ""
        # 安全汇点循环：契约不符、解析失败、回问与大扇出确认都回到中断点
        # 重新等待人工，永不转死。
        while True:
            payload: dict[str, Any] = {
                "iteration_round": state.get("iteration_round", 0),
                "chapter_ids": list(outline_order),
                "citation_warnings": state.get("citation_warnings", []),
                "review_warnings": state.get("review_warnings", []),
            }
            if error is not None:
                payload["error"] = error
            if questions:
                payload["clarification_questions"] = list(questions)
            if pending is not None:
                affected = {directive.target_chapter_id for directive in pending}
                payload["pending_confirmation"] = {
                    "affected_chapter_ids": [
                        chapter_id
                        for chapter_id in outline_order
                        if chapter_id in affected
                    ],
                    "total_chapters": len(outline),
                    "directives": [directive.model_dump() for directive in pending],
                }
            decision = interrupt(payload)
            try:
                action, feedback = _validate_decision(
                    decision, confirmable=pending is not None
                )
            except ValueError as exc:
                # 契约不符：保留待确认清单与回问（若有），用户仍可 confirm 或改提意见。
                error = str(exc)
                continue
            error = None
            if action == "finalize":
                # 定稿不调 LLM，元数据无从获取，只记录单元名。
                return WritingAgentState(
                    status=WorkflowStatus.FINISHED,
                    pending_directives=[],
                    citation_warnings=[],
                    review_warnings=[],
                    current_node_llm_config={"unit": "human_review_gate"},
                )
            if action == "confirm":
                # confirmable 校验保证 pending 非 None。
                assert pending is not None
                directives = pending
                feedback = pending_feedback
                break
            # action == "revise"：重新解析本轮意见，作废既有待确认清单与回问。
            questions = []
            pending = None
            # 章节清单（含摘要）、历史台账、本轮意见经装配段现场取得（不失忆）。
            context = assemble(
                state,
                "human_review_gate",
                config=config,
                feedback=feedback,
            )
            try:
                # 解析的 LLM 调用包在 durable task 里（见 _parse_feedback_task
                # docstring）：confirm 恢复重放时取缓存不重复调用，执行清单
                # 严格等于确认时回显的那份，不因重放漂移而误执行。
                parsed_payload = _parse_feedback_task(
                    context.text("chapter_list"),
                    context.text("revision_ledger"),
                    feedback,
                ).result()
            except ValueError as exc:
                error = str(exc)
                continue
            directives, questions = resolve_directives(parsed_payload, outline, drafts)
            if not directives and not questions:
                # 解析不出任何有效指令：回问用户，不猜测、不转死。
                error = "用户意见解析不出任何有效修订指令，请人工确认意见内容"
                continue
            if questions:
                # 含混或定位失败：回问用户，不猜测、本轮不执行任何指令。
                continue
            if needs_fanout_confirmation(directives, outline):
                # 大扇出：携解析清单重新中断待人工确认后才执行。
                pending = directives
                pending_feedback = feedback
                continue
            break

        # 解析 LLM 由 durable task 内部构造；此处仅取其配置元数据上报，
        # 不触发真实调用（OpenAI 兼容客户端惰性建连，构造无网络开销）。
        llm = llm_factory("human_review_gate")
        round_no = state.get("iteration_round", 0) + 1
        # 台账列表字段是整值覆盖语义，须带上既有旧轮次再追加新一轮。
        ledger = list(state.get("revision_ledger", [])) + [
            RevisionRound(
                round_no=round_no, raw_feedback=feedback, directives=directives
            )
        ]
        # 滑出保留窗口（最近 K 轮之外）的更早轮次落库一句话摘要：
        # 摘要在写回 State 时一次生成并持久化，装配时直接取用，两处逻辑共用同一纯函数。
        keep = config.ledger_keep_rounds
        earlier, recent = ledger[:-keep], ledger[-keep:]
        ledger = [
            round_
            if round_.digest is not None
            else round_.model_copy(update={"digest": digest_of_round(round_, config)})
            for round_ in earlier
        ] + recent
        return WritingAgentState(
            pending_directives=directives,
            revision_ledger=ledger,
            iteration_round=round_no,
            citation_warnings=[],
            review_warnings=[],
            # 新一轮修订获得全新的终审重试预算。
            citation_retry_count=0,
            status=WorkflowStatus.AWAIT_USER_REVIEW,
            current_node_llm_config={"unit": "human_review_gate", **llm.metadata},
        )

    return node
