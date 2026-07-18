"""FakeLLM 响应编排计划：端到端测试共用的确定性应答序列。

test_graph_e2e / test_graph_event_stream / test_api_e2e 三处共用，
避免同一份编排在多个测试文件里漂移。
"""

import json

# framework_orchestrator 的最小应答序列：
# 品类识别（自由结构）→ 大纲（2 章）→ 逐章论点 → 逐论点假说。
FRAMEWORK_RESPONSES = [
    '{"genre": "行业评论", "template_file": null}',
    '[{"title": "第一章", "subsections": []}, {"title": "第二章", "subsections": []}]',
    '[{"text": "论点一"}]',
    '[{"text": "假说一", "refute_condition": "出现公开反例即证伪", '
    '"angle": "假设", "evidence_retrievable": true}]',
    '[{"text": "论点二"}]',
    '[{"text": "假说二", "refute_condition": "出现公开反例即证伪", '
    '"angle": "预言", "evidence_retrievable": true}]',
]

# 语义核查全部对应（无问题）的应答：每个受审章节一条。
SEMANTIC_PASS = "[]"

# 首轮全量核查通过所需的完整应答序列（2 章各一条语义核查）。
FIRST_PASS_RESPONSES = [*FRAMEWORK_RESPONSES, SEMANTIC_PASS, SEMANTIC_PASS]

# 一轮 revise 的意见解析应答：ch2 纯改写；随后增量核查只重审 ch2 一条。
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
REVISE_ROUND_RESPONSES = [DIRECTIVE_RESPONSE, SEMANTIC_PASS]

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

# 端到端主干完整编排：首轮全量核查 + 混合意见解析 + 增量核查重审两章各一条。
TRUNK_RESPONSES = [
    *FIRST_PASS_RESPONSES,
    MIXED_DIRECTIVE_RESPONSE,
    SEMANTIC_PASS,
    SEMANTIC_PASS,
]
