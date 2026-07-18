"""WritingAgentState 图状态定义与全局状态机枚举。

层级关系（见 CONTEXT.md）：章节 1—n 论点，论点 1—N 假说。
本期为刚性流水线，list 字段采用缺省整值覆盖语义，不使用累加 reducer。
"""

import enum
from typing import Literal, TypedDict

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


class Material(BaseModel):
    """结构化引文库条目：正文只嵌其 id 作为轻量角标。"""

    id: str
    hypothesis_id: str
    chapter_id: str
    source: str
    url: str | None = None
    excerpt: str
    relevance_score: float
    verdict: Literal["pass", "fail"]


class ChapterDraft(BaseModel):
    """单章正文与摘要：摘要供下一章串行承接，构成摘要链。"""

    chapter_id: str
    text: str
    """章节正文，含原位角标（素材 id）。"""
    summary: str


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


class WritingAgentState(TypedDict, total=False):
    """LangGraph 图状态：全流程唯一事实源，经 Postgres 存档器持久化。"""

    user_intent: str
    user_identity: str
    genre: str
    template_id: str | None
    """匹配到的模板标识；自由结构模式为 None。"""
    outline: list[ChapterSpec]
    citation_library: list[Material]
    chapter_drafts: list[ChapterDraft]
    revision_ledger: list[RevisionRound]
    status: WorkflowStatus
    iteration_round: int
    execution_trace_id: str
    current_node_llm_config: dict[str, str]
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
        outline=[],
        citation_library=[],
        chapter_drafts=[],
        revision_ledger=[],
        status=WorkflowStatus.IDLE,
        iteration_round=0,
        execution_trace_id=execution_trace_id,
        current_node_llm_config={},
    )
