"""WritingAgentState 图状态定义与全局状态机枚举。

层级关系（见 CONTEXT.md）：章节 1—n 论点，论点 1—N 假说。
list 字段缺省采用整值覆盖语义；唯一例外是 chapter_drafts——
首写阶段经 Send 并行扇出后各分支只回写单章草稿，该字段使用
merge_chapter_drafts reducer 按 chapter_id 合并（同 id 替换、新 id 插入、
按 ch 编号排序），串行路径整值覆盖在该语义下逐项等价。
status 与 current_node_llm_config 使用 keep_last reducer：并行首写分支
在同一超步写入相同值时不再触发 LastValue 冲突，串行语义不变。
"""

import enum
import re
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel


class WorkflowStatus(str, enum.Enum):
    """全局状态机枚举，随节点流转。"""

    IDLE = "IDLE"
    FRAMEWORK_BUILDING = "FRAMEWORK_BUILDING"
    REFERENCE_FETCHING = "REFERENCE_FETCHING"
    ARTICLE_WRITING = "ARTICLE_WRITING"
    CITATION_CHECKING = "CITATION_CHECKING"
    AWAIT_USER_REVIEW = "AWAIT_USER_REVIEW"
    FINISHED = "FINISHED"
    ERROR_FAILED = "ERROR_FAILED"


def status_text(value: object, default: str = "") -> str:
    """把状态机值序列化为纯字符串：枚举取 value，None 取 default，其余 str()。"""
    if value is None:
        return default
    if isinstance(value, WorkflowStatus):
        return value.value
    return str(value)


class Hypothesis(BaseModel):
    """假说：从论点派生的可证伪、可检索验证的具体命题。"""

    id: str
    text: str
    refute_condition: str
    """证伪条件：每条假说必须声明。"""
    angle: str
    """六角度之一：假设 / 失效模式 / 边界条件 / 竞争解释 / 预言 / 反事实。"""


class ArgumentPoint(BaseModel):
    """论点：章节内的一个中心主张。"""

    id: str
    text: str
    hypotheses: list[Hypothesis] = []


class ChapterSpec(BaseModel):
    """章节骨架：写作与检索的基本执行粒度。"""

    id: str
    title: str
    subsections: list[str] = []
    """三级标题实例化文本；保留模板的一二级骨架层级。"""
    points: list[ArgumentPoint] = []
    chapter_type: str | None = None
    """章型：模板骨架章标题原文，实例化时由 framework_orchestrator 作为骨架事实
    随章写入（ADR-0005）——大纲逐章携带即「章序号 → 章型」映射，
    lint 直接消费、不从位置或标题反推；自由结构模式为 None。"""
    planned_summary: str = ""
    """规划摘要：框架生成时预判的本章一句话内容概要，供并行首写时
    后章用前章的规划摘要衔接（替代实际写成的摘要链）；
    LLM 应答缺失时由程序从标题与论点确定性兜底，默认空串兼容旧 checkpoint。"""


SourceKind = Literal["web", "knowledge_base", "structured_data"]
"""素材来源通道：联网搜索 / 知识库 / 结构化数据，三条检索通道各占一值。"""


class Material(BaseModel):
    """结构化引文库条目：正文只嵌其 id 作为轻量角标。"""

    id: str
    hypothesis_id: str
    chapter_id: str
    source: str
    url: str | None = None
    source_kind: SourceKind = "web"
    """来源通道：驱动书目渲染的类型标识；
    默认 web 兼容旧 checkpoint（与既有条目全按联网来源渲染的行为一致）。"""
    excerpt: str
    relevance_score: float
    verdict: Literal["pass", "fail"]


class SelfCheck(BaseModel):
    """rewriter_loop 的单章自检结果：双层校验的第一层。"""

    citations_ok: bool = True
    issues: list[str] = []


class ChapterDraft(BaseModel):
    """单章正文与摘要：摘要供下一章串行承接，构成摘要链。"""

    chapter_id: str
    text: str
    """章节正文，含原位角标（素材 id）。"""
    summary: str
    self_check: SelfCheck = SelfCheck()


class RevisionDirective(BaseModel):
    """修订指令：用户意见解析出的结构化最小修订单位。"""

    target_chapter_id: str
    type: Literal["rewrite_only", "evidence_augmented"]
    """rewrite_only 纯改写；evidence_augmented 补充佐证。"""
    instruction: str


class RevisionRound(BaseModel):
    """修订台账中的一轮记录：全量持久化，保证多轮迭代不失忆。"""

    round_no: int
    raw_feedback: str
    directives: list[RevisionDirective] = []
    digest: str | None = None
    """更早轮次压缩后的一句话摘要；最近轮次保留原文时为 None。"""


class CitationIssue(BaseModel):
    """引文对账发现的单条问题。"""

    kind: Literal[
        "orphan_marker",
        "unused_material",
        "cross_chapter",
        "semantic_mismatch",
        "self_check_failed",
        "numbering_broken",
    ]
    """orphan_marker 无来源的标注；unused_material 未被引用的素材；
    cross_chapter 跨章误引；semantic_mismatch 语义核查不通过；
    self_check_failed 单章自检（双层校验第一层）不通过；
    numbering_broken 跨章编号重复、断号或与大纲不一致。"""
    chapter_id: str
    material_id: str
    detail: str


class CitationReport(BaseModel):
    """citation_validator 全局终审结论。"""

    passed: bool
    issues: list[CitationIssue] = []
    failed_chapter_ids: list[str] = []
    """终审不合格、需定向回退重写的章节。"""


# 框架侧章节 id 恒为 ch{n}；排序按数字后缀，保证并行完成顺序不影响
# 摘要链、编号核查与成文顺序。不合形态的 id 排在其后按字典序稳定兜底。
_CHAPTER_ID_PATTERN = re.compile(r"^ch(\d+)$")


def _chapter_order_key(chapter_id: str) -> tuple[int, int, str]:
    """章节草稿排序键：ch{n} 按 n 数值升序，其余 id 靠后按字典序。"""
    match = _CHAPTER_ID_PATTERN.match(chapter_id)
    if match:
        return (0, int(match.group(1)), chapter_id)
    return (1, 0, chapter_id)


def merge_chapter_drafts(
    existing: list[ChapterDraft] | None, new: list[ChapterDraft] | None
) -> list[ChapterDraft]:
    """chapter_drafts 的 reducer：按 chapter_id 合并，同 id 新值替换、新 id 插入。

    结果按 ch 编号数字后缀排序，使并行首写分支的完成顺序不影响成文顺序；
    串行节点回写完整列表时逐项同 id 替换，与旧的整值覆盖语义等价。
    """
    merged = {draft.chapter_id: draft for draft in (existing or [])}
    for draft in new or []:
        merged[draft.chapter_id] = draft
    return sorted(
        merged.values(), key=lambda draft: _chapter_order_key(draft.chapter_id)
    )


def keep_last(existing: object, new: object) -> object:
    """标量字段的 keep_last reducer：并行分支写入相同值时不冲突，串行语义同 LastValue。

    并行首写各分支写的 status / current_node_llm_config 恒为相同值，
    取最后到达者即可；LangGraph 对无 reducer 的键在同一超步收到多次写入
    会抛 InvalidUpdateError，此 reducer 即为放行该场景而设。
    """
    return new


class WritingAgentState(TypedDict, total=False):
    """LangGraph 图状态：全流程唯一事实源，经 Postgres 存档器持久化。"""

    user_intent: str
    user_identity: str
    genre: str
    template_id: str | None
    """匹配到的模板标识；自由结构模式为 None。"""
    doc_type: str
    """文种：品类识别选中模板后经文种注册表确定性锚定（ADR-0005），
    无模板命中落「通用公文」兑底；由 framework_orchestrator 一次写入，全链路只读。"""
    doc_variant: str | None
    """文种内变体（目前仅人才培养方案声明本科/高职）；无变体为 None，与文种同刻写入、只读。"""
    outline: list[ChapterSpec]
    citation_library: list[Material]
    chapter_drafts: Annotated[list[ChapterDraft], merge_chapter_drafts]
    """章节草稿：唯一使用合并 reducer 的字段（并行首写各分支只回写单章）。"""
    revision_ledger: list[RevisionRound]
    pending_directives: list[RevisionDirective]
    """本轮待执行的修订指令；writing_orchestrator 执行完毕后清空。"""
    revised_chapter_ids: list[str]
    """本轮被修改章节；citation_validator 据此做增量核查，核查完毕后清空。"""
    citation_report: CitationReport | None
    """最近一次终审结论。"""
    citation_retry_count: int
    """终审失败定向回退的已重试次数；超限强制进入人工中断点。"""
    citation_warnings: list[str]
    """重试超限携带的未决引文警告，交人工裁决。"""
    status: Annotated[WorkflowStatus, keep_last]
    iteration_round: int
    execution_trace_id: str
    current_node_llm_config: Annotated[dict[str, str], keep_last]
    """当前节点生效的 LLM 配置元数据（不含密钥）。"""


def initial_state(
    user_intent: str, user_identity: str, execution_trace_id: str
) -> WritingAgentState:
    """构造 IDLE 起始状态。"""
    return WritingAgentState(
        user_intent=user_intent,
        user_identity=user_identity,
        genre="",
        template_id=None,
        doc_type="",
        doc_variant=None,
        outline=[],
        citation_library=[],
        chapter_drafts=[],
        revision_ledger=[],
        pending_directives=[],
        revised_chapter_ids=[],
        citation_report=None,
        citation_retry_count=0,
        citation_warnings=[],
        status=WorkflowStatus.IDLE,
        iteration_round=0,
        execution_trace_id=execution_trace_id,
        current_node_llm_config={},
    )
