"""WritingAgentState 图状态定义与全局状态机枚举。

层级关系（见 CONTEXT.md）：章节 1—n 论点，论点 1—N 假说。
list 字段缺省采用整值覆盖语义；两个例外是并行扇出阶段各分支只回写单章
产物的字段——chapter_drafts 使用 merge_chapter_drafts reducer 按 chapter_id
合并（同 id 替换、新 id 插入、按 ch 编号排序），citation_library 使用
merge_citation_library reducer 按素材 id 合并并跨章按 URL 去重；
串行路径整值覆盖在两者语义下逐项等价。
status 与 current_node_llm_config 使用 keep_last reducer：并行分支
在同一超步写入相同值时不再触发 LastValue 冲突，串行语义不变。
"""

import enum
import re
from typing import Annotated, Any, Literal, TypedDict

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
    source_ref: dict[str, Any] | None = None
    """真实来源定位：Material.id 只承担不透明引用身份，来源定位从本字段读取。
    默认为 None 兼容旧 checkpoint 与旧 payload。"""
    excerpt: str
    relevance_score: float
    verdict: Literal["pass", "fail", "inconclusive"]
    """佐证强度三值：pass 强支撑、inconclusive 弱佐证（近似命中/补充）、fail 反例或不可用。
    默认 checkpoint 兼容——旧条目只有 pass/fail 两值，inconclusive 为本轮新增。"""


CITABLE_VERDICTS: frozenset[str] = frozenset({"pass", "inconclusive"})
"""可进写作池的素材 verdict 集合：pass 强支撑 + inconclusive 弱佐证；fail 仅入库供审计。
素材过滤口径的唯一事实源，装配层与写作注入点共用，避免同形字面量多处漂移。"""


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


DirectiveType = Literal["rewrite_only", "evidence_augmented"]
"""修订指令类型字面量的单一事实源：rewrite_only 纯改写；evidence_augmented 补充佐证。
模型字段与意见解析归结的收窄断言共用。"""


class RevisionDirective(BaseModel):
    """修订指令：用户意见解析出的结构化最小修订单位。"""

    target_chapter_id: str
    type: DirectiveType
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
    """篇级终审 document_reviewer 发现的单条 error 级问题。

    模型名与字段名保留 Citation 前缀是检查点与契约稳定考虑（旧 checkpoint 兼容），
    实际承载篇级终审全部 error 级问题（引用四步 + 结构完整性 + 跨章硬事实冲突）。"""

    kind: Literal[
        "orphan_marker",
        "unused_material",
        "cross_chapter",
        "semantic_mismatch",
        "self_check_failed",
        "numbering_broken",
        "fact_conflict",
        "chapter_missing",
    ]
    """orphan_marker 无来源的标注；unused_material 未被引用的素材；
    cross_chapter 跨章误引；semantic_mismatch 语义核查不通过；
    self_check_failed 单章自检（双层校验第一层）不通过；
    numbering_broken 跨章编号重复、断号或与大纲不一致；
    fact_conflict 跨章硬事实冲突（篇级评审判定，严重级由代码固定为 error）；
    chapter_missing 大纲章节缺少成稿（结构完整性确定性判定）。"""
    chapter_id: str
    material_id: str
    detail: str


class CitationReport(BaseModel):
    """篇级终审 document_reviewer 的全篇终审结论（模型名保留 Citation 前缀兼容旧 checkpoint）。"""

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


def merge_citation_library(
    existing: list[Material] | None, new: list[Material] | None
) -> list[Material]:
    """citation_library 的 reducer：按素材 id 合并，再按章排序并跨章按 URL 去重。

    检索阶段经 Send 并行扇出后各分支只回写单章素材，合并语义：
    同 id 新值替换、新 id 插入，结果按章节编号排序（章内保持插入顺序），
    使并行分支的完成顺序不影响引文库顺序；同 URL 素材只保留章序最靠前的
    一条（跨章去重从检索编排收敛到此处），url 为 None 的素材不参与去重。
    修订轮增量检索回写完整列表时逐项同 id 替换，与旧的整值覆盖语义等价。
    """
    merged = {material.id: material for material in (existing or [])}
    for material in new or []:
        merged[material.id] = material
    ordered = sorted(
        merged.values(), key=lambda material: _chapter_order_key(material.chapter_id)
    )
    seen_urls: set[str] = set()
    library: list[Material] = []
    for material in ordered:
        if material.url is not None:
            if material.url in seen_urls:
                continue
            seen_urls.add(material.url)
        library.append(material)
    return library


def keep_last(existing: object, new: object) -> object:
    """标量字段的 keep_last reducer：并行分支写入相同值时不冲突，串行语义同 LastValue。

    并行首写各分支写的 status / current_node_llm_config 恒为相同值，
    取最后到达者即可；LangGraph 对无 reducer 的键在同一超步收到多次写入
    会抛 InvalidUpdateError，此 reducer 即为放行该场景而设。
    """
    return new


def merge_revised_ids(
    existing: list[str] | None, new: list[str] | None
) -> list[str]:
    """revised_chapter_ids 的 reducer：并行回退扇出各分支只回写本章节 id，
    并集汇入并按章序排序；空列表写入时清空本轮修订集。

    本字段语义为「本轮被修改章节集」：空列表即「本轮无人被修改」=清空，
    document_reviewer 在核查完毕后据此写空列表重置、使下一轮从空集起计；
    非空列表则并集累加。并行回退扇出各分支恒写非空单元素 ``[chapter_id]``，
    故空列表清空语义不会被并行分支意外触发——只有显式的轮次重置写空列表。
    串行节点回写完整累积列表时与整值覆盖语义等价（并集不增不减、去重）。
    """
    if not new:
        return []
    merged = set(existing or []) | set(new)
    return sorted(merged, key=_chapter_order_key)


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
    citation_library: Annotated[list[Material], merge_citation_library]
    """引文库：并行检索各分支只回写单章素材，经合并 reducer 汇入并跨章去重。"""
    chapter_drafts: Annotated[list[ChapterDraft], merge_chapter_drafts]
    """章节草稿：并行首写各分支只回写单章草稿，经合并 reducer 汇入。"""
    revision_ledger: list[RevisionRound]
    pending_directives: list[RevisionDirective]
    """本轮待执行的修订指令；全部 Send 分支汇合并经终审后统一清空。"""
    directive_chapter_id: Annotated[str | None, keep_last]
    """人工修订 Send 分支的运行态目标章；汇合后的终审节点会清空。"""
    revised_chapter_ids: Annotated[list[str], merge_revised_ids]
    """本轮被修改章节；并行回退扇出各分支只回写本章节 id，经合并 reducer 并集汇入；
    document_reviewer 据此做增量核查，核查完毕后写空列表清空（见 reducer）。"""
    citation_report: CitationReport | None
    """最近一次终审结论。"""
    citation_retry_count: int
    """终审失败定向回退的已重试次数；超限强制进入人工中断点。"""
    citation_warnings: list[str]
    """重试超限携带的未决引文警告，交人工裁决。"""
    review_warnings: list[str]
    """篇级评审的 warn 级提示（章间衔接/口径统一/跨章重复），每次终审都写入，
    不打回、不影响重试，随人工中断点呈现给人工，供其自行裁量。"""
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
        directive_chapter_id=None,
        revised_chapter_ids=[],
        citation_report=None,
        citation_retry_count=0,
        citation_warnings=[],
        review_warnings=[],
        status=WorkflowStatus.IDLE,
        iteration_round=0,
        execution_trace_id=execution_trace_id,
        current_node_llm_config={},
    )
