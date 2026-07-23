"""mock 装配档场景库：FakeLLM 确定性应答计划。

供 ``service.mock_stack.build_mock_graph`` 装配 mock 栈，驱动真分支
（多章大纲解析 / 角标正文 / 篇级终审 warn / 退化重试）。应答序列直接
复用 ``tests/llm_response_plans`` 的共用编排，不在本模块重写，避免口径
漂移。新增场景在此单点登记，装配档只认 ``MockScenario`` 一个类型。
"""

from dataclasses import dataclass

from tests.llm_response_plans import (
    FRAMEWORK_KEYED_RESPONSES,
    FRAMEWORK_RESPONSES,
    SEMANTIC_PASS,
    WRITER_KEYED_RESPONSES,
)

# 篇级 transition warn 应答：document_reviewer 主节点在全篇语义核查之后
# 做一次四维评审（1 条 LLM 调用），其 _document_review 期望 JSON 数组，
# 逐项 {"dimension":"transition|consistency|duplication|fact_conflict",
# "chapter_ids":["ch1","ch2"],"detail":"..."}。
# transition/consistency/duplication 为 warn（进 review_warnings、不打回、
# 仍进人工中断点）；fact_conflict 为 error（打回重写，场景库不用它）。
# chapter_ids 必须是大纲真实存在的章 id（ch1/ch2），否则被剔除。
DOC_REVIEW_WARN = (
    '[{"dimension":"transition","chapter_ids":["ch1","ch2"],'
    '"detail":"章间衔接生硬，承接断裂"}]'
)


@dataclass(frozen=True)
class MockScenario:
    """mock 档场景：FakeLLM 顺序应答 + 键控应答 + 流式分块大小。

    ``responses`` 是 FakeLLM 的顺序应答列表（头部弹出，耗尽返回兜底串）；
    ``keyed`` 是键控应答字典，键出现在调用消息文本中时优先弹出该键的下
    一条应答；``chunk_size`` 控制 FakeLLM.stream 的定长分块大小。frozen
    使场景实例不可变，装配档可安全跨运行复用同一引用。
    """

    responses: list[str]
    keyed: dict[str, list[str]]
    chunk_size: int = 8


# 默认场景：秒级走完到审阅门，覆盖多章大纲解析 / 角标正文 / 篇级终审
# warn 呈现。顺序应答 = 框架 3 条 + 2 章语义核查 + 1 条篇级 warn；
# 键控 = 框架假说 + 写作信封。
DEFAULT_SCENARIO = MockScenario(
    responses=[
        *FRAMEWORK_RESPONSES,
        SEMANTIC_PASS,
        SEMANTIC_PASS,
        DOC_REVIEW_WARN,
    ],
    keyed={**FRAMEWORK_KEYED_RESPONSES, **WRITER_KEYED_RESPONSES},
)

# 退化重试场景：默认场景基础上覆盖 ch1 写作键控序列为
# [malformed, valid draft, valid revise]，配 WRITER_DELTA_FLUSH_CHARS=小值
# 时触发 attempt 1 失败（malformed JSON / 解析失败 / 空正文）、attempt 2
# 成功的退化重试。malformed 串沿用 #59 退化测试的未闭合 JSON 模板。
BAD_CH1_DRAFT = '{"chapter_text": "部分未闭合'
DEGRADATION_SCENARIO = MockScenario(
    responses=DEFAULT_SCENARIO.responses,
    keyed={
        **DEFAULT_SCENARIO.keyed,
        "- 标题：第一章": [
            BAD_CH1_DRAFT,
            WRITER_KEYED_RESPONSES["- 标题：第一章"][0],
            WRITER_KEYED_RESPONSES["- 标题：第一章"][1],
        ],
    },
)
