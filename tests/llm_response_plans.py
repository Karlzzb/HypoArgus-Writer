"""FakeLLM 响应编排计划：端到端测试共用的确定性应答序列。

test_graph_e2e / test_graph_event_stream / test_api_e2e 与
tests/agents/rewriter_loop/test_real_contract 共用，
避免同一份编排与信封拼装在多个测试文件里漂移。

framework 的假说生成按章节并发，调用顺序不确定，
假说应答放在 FRAMEWORK_KEYED_RESPONSES 里按论点提示词片段键控分派；
其余调用仍走顺序应答列表。
"""

import json

# framework_orchestrator 的顺序应答：品类识别（自由结构）→ 大纲（2 章）→
# 全文论点单次调用（每章 1 条论点）。
FRAMEWORK_RESPONSES = [
    '{"genre": "行业评论", "template_file": null}',
    '[{"title": "第一章", "subsections": []}, {"title": "第二章", "subsections": []}]',
    '[{"chapter_index": 1, "points": [{"text": "论点一"}]}, '
    '{"chapter_index": 2, "points": [{"text": "论点二"}]}]',
]

# framework 的键控应答：并发的逐论点假说调用按论点内容绑定应答。
FRAMEWORK_KEYED_RESPONSES = {
    "待发散的论点：论点一": [
        '[{"text": "假说一", "refute_condition": "出现公开反例即证伪", '
        '"angle": "假设", "evidence_retrievable": true}]',
    ],
    "待发散的论点：论点二": [
        '[{"text": "假说二", "refute_condition": "出现公开反例即证伪", '
        '"angle": "预言", "evidence_retrievable": true}]',
    ],
}

# framework 阶段的 LLM 调用总数：顺序应答 + 键控假说应答。
FRAMEWORK_LLM_CALLS = len(FRAMEWORK_RESPONSES) + sum(
    len(values) for values in FRAMEWORK_KEYED_RESPONSES.values()
)

# 语义核查全部对应（无问题）的应答：每个受审章节一条。
SEMANTIC_PASS = "[]"

# 篇级评审「无任何发现」的放行应答：document_reviewer 每次运行在全部
# 语义核查之后恰好多消费一条（一次全篇四维评审调用）。
DOCUMENT_REVIEW_PASS = "[]"

# 首轮全量终审通过所需的完整顺序应答序列（2 章各一条语义核查 + 一条篇级评审）；
# 配套的假说应答固定取 FRAMEWORK_KEYED_RESPONSES。
FIRST_PASS_RESPONSES = [
    *FRAMEWORK_RESPONSES,
    SEMANTIC_PASS,
    SEMANTIC_PASS,
    DOCUMENT_REVIEW_PASS,
]

# 首轮全量终审通过的 LLM 调用总数（含键控假说调用）。
FIRST_PASS_LLM_CALLS = FRAMEWORK_LLM_CALLS + 3

# 一轮 revise 的意见解析应答：ch2 纯改写；随后增量核查只重审 ch2 一条，
# 篇级评审始终全量再一条。
DIRECTIVE_RESPONSE = json.dumps(
    [
        {
            "target_chapter_id": "ch2",
            "type": "rewrite_only",
            "instruction": "口吻更克制",
        }
    ],
    ensure_ascii=False,
)
REVISE_ROUND_RESPONSES = [DIRECTIVE_RESPONSE, SEMANTIC_PASS, DOCUMENT_REVIEW_PASS]

# 一次意见混合两类分支且落在不同章节：ch1 纯改写、ch2 补充佐证。
MIXED_DIRECTIVE_RESPONSE = json.dumps(
    [
        {
            "target_chapter_id": "ch1",
            "type": "rewrite_only",
            "instruction": "引言口吻更克制",
        },
        {
            "target_chapter_id": "ch2",
            "type": "evidence_augmented",
            "instruction": "补充行业数据佐证",
        },
    ],
    ensure_ascii=False,
)

# 端到端主干完整编排：首轮全量终审 + 混合意见解析 + 增量核查重审两章各一条
# + 修订轮的篇级评审一条。混合意见影响 2/2 章（超过大纲一半），触发大扇出
# 确认重新中断；confirm 恢复时 human_review_gate 节点从头重放、意见解析
# LLM 调用重复执行一次，故解析应答备两份。
TRUNK_RESPONSES = [
    *FIRST_PASS_RESPONSES,
    MIXED_DIRECTIVE_RESPONSE,
    MIXED_DIRECTIVE_RESPONSE,
    SEMANTIC_PASS,
    SEMANTIC_PASS,
    DOCUMENT_REVIEW_PASS,
]


def writer_envelope(chapter_text: str, chapter_summary: str) -> str:
    """拼 rewriter_loop 真实现所需的写作信封 JSON-in-text 应答。

    公开导出：真实现契约测试与图级 E2E 共用同一信封拼装，避免各测试文件
    重复实现导致口径漂移。
    """
    return json.dumps(
        {"chapter_text": chapter_text, "chapter_summary": chapter_summary},
        ensure_ascii=False,
    )


def joined_prompt(messages: list[dict[str, str]]) -> str:
    """把一次 LLM 调用的全部消息文本拼接，供提示词内容断言。"""
    return "\n".join(message.get("content", "") for message in messages)


# 自审空裁决：真实现每次写作后自审一次，空转编排一律判无违规。
AUDIT_EMPTY_RESPONSE = '{"issues": []}'

# 章级评审空裁决：chapter_reviewer 真实现单次调用的合法非退化应答
# （issues 与 conflicts 均为空数组，评审通过、循环短路）。
REVIEW_EMPTY_RESPONSE = '{"issues": [], "conflicts": []}'

# rewriter_loop 真实现的键控应答（demo 空转与端到端主干等真链路场景与
# TRUNK_RESPONSES 合用）：写作调用按上下文块的「- 标题：<章名>」行键控
# （同章 draft 在前、revise 在后，按调用时间顺序弹出）；章级评审调用按
# 【章节评审】标签键控（demo 空转走 chapter_reviewer 真实现：首写两章 +
# 修订两章共 4 次评审，一律空裁决短路）。正文刻意规避全部 lint 规则
# （角标在素材池内、无口语化/编号/意识形态违规、无 ## 标题不落章型），
# 保证每章恰好一次写作调用、不触发修订，应答计划保持确定性。revise 产物
# 落实 MIXED_DIRECTIVE_RESPONSE 的两条指令（正文含指令原文，供落实断言）。
WRITER_KEYED_RESPONSES = {
    "【章节评审】": [REVIEW_EMPTY_RESPONSE] * 4,
    "- 标题：第一章": [
        writer_envelope(
            "本专业面向智能制造领域培养高素质人才，课程体系对接行业标准。[m-ch1-p1-h1]",
            "第一章完成培养定位与背景铺陈。",
        ),
        writer_envelope(
            "本专业以克制口吻阐明培养定位，课程体系对接行业标准。"
            "（修订落实：引言口吻更克制）[m-ch1-p1-h1]",
            "第一章按修订意见收束引言口吻。",
        ),
    ],
    "- 标题：第二章": [
        writer_envelope(
            # 正文承接前章摘要原文（供摘要链断言 drafts[0].summary in drafts[1].text）。
            "承接前章——第一章完成培养定位与背景铺陈。"
            "在培养定位基础上，本专业构建产教融合的课程实施路径。[m-ch2-p1-h1]",
            "第二章完成课程实施路径论述。",
        ),
        writer_envelope(
            "在培养定位基础上，本专业以行业数据论证课程实施路径成效。"
            "（修订落实：补充行业数据佐证）[m-ch2-p1-h1]",
            "第二章按修订意见补充行业数据佐证。",
        ),
    ],
}
