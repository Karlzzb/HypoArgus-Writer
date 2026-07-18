"""端到端闭环测试：注入假 LLM 跑整张图，验证状态机流转、人工中断点与迭代闭环。

framework_orchestrator 预置最小 JSON 应答序列（自由结构、2 章、每章 1 论点
1 假说）；reference_orchestrator 与 writing_orchestrator 走打桩子智能体、
不调 LLM；citation_validator 每个受审章节消费一条语义核查 JSON 应答；
human_review_gate 经 LangGraph interrupt 真实中断，仅在 revise 恢复时消费
一条意见解析 JSON 应答。

人工中断点依赖存档器：非持久化用例用 InMemorySaver。
Postgres 连接串取环境变量 HYPOARGUS_TEST_PG_DSN，缺省指向本地测试库；
库不可达时跳过持久化用例（其余用例仍必须全绿）。
"""

import json
import os
import socket
import uuid
from urllib.parse import urlparse

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from citation_reconciler import MARKER_PATTERN
from graph import MAIN_NODES, build_graph, postgres_checkpointer
from llm_client import FakeLLM
from llm_config import RUNTIME_UNITS
from state import WorkflowStatus, initial_state

TEST_PG_DSN = os.environ.get(
    "HYPOARGUS_TEST_PG_DSN",
    "postgresql://postgres:postgres@127.0.0.1:15432/postgres",
)

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
FRAMEWORK_LLM_CALLS = len(FRAMEWORK_RESPONSES)

# 语义核查全部对应（无问题）的应答：每个受审章节一条。
SEMANTIC_PASS = "[]"

# 首轮全量核查通过所需的完整应答序列（2 章各一条语义核查）。
FIRST_PASS_RESPONSES = [*FRAMEWORK_RESPONSES, SEMANTIC_PASS, SEMANTIC_PASS]

FINALIZE = {"action": "finalize"}


def _build(responses: list[str], **kwargs):
    """带 InMemorySaver 与共享假 LLM 构图，返回（graph, fake, config）。"""
    fake = FakeLLM(list(responses))
    graph = build_graph(
        llm_factory=lambda unit: fake, checkpointer=InMemorySaver(), **kwargs
    )
    config = {"configurable": {"thread_id": f"e2e-{uuid.uuid4()}"}}
    return graph, fake, config


def _assert_framework_state(values: dict) -> None:
    """framework 节点之后 State 必须含合规的大纲、论点与假说。"""
    assert values["template_id"] is None
    assert values["genre"] == "行业评论"
    outline = values["outline"]
    assert [chapter.id for chapter in outline] == ["ch1", "ch2"]
    assert outline[0].points[0].id == "ch1-p1"
    assert outline[0].points[0].hypotheses[0].id == "ch1-p1-h1"
    assert outline[1].points[0].hypotheses[0].id == "ch2-p1-h1"


def _assert_full_draft(result: dict) -> None:
    """空转产物必须是全文草稿：角标可溯源、摘要链承接、自检入 State。"""
    outline = result["outline"]
    drafts = result["chapter_drafts"]
    library = result["citation_library"]

    # 每条素材回链到大纲中的假说 ID，且标注所属章节。
    hypothesis_ids = {
        hypothesis.id
        for chapter in outline
        for point in chapter.points
        for hypothesis in point.hypotheses
    }
    assert {material.hypothesis_id for material in library} == hypothesis_ids
    assert {material.chapter_id for material in library} == {"ch1", "ch2"}

    # 逐章有正文与摘要，正文含角标，角标全部可在引文库中查到。
    material_ids = {material.id for material in library}
    assert [draft.chapter_id for draft in drafts] == ["ch1", "ch2"]
    for draft in drafts:
        assert draft.summary
        markers = MARKER_PATTERN.findall(draft.text)
        assert markers, f"章节 {draft.chapter_id} 正文缺少角标"
        assert set(markers) <= material_ids
        assert draft.self_check.citations_ok is True

    # 摘要链逐章承接：第二章正文承接第一章摘要。
    assert drafts[0].summary in drafts[1].text


def _pg_reachable(dsn: str) -> bool:
    parsed = urlparse(dsn)
    try:
        with socket.create_connection(
            (parsed.hostname or "127.0.0.1", parsed.port or 5432), timeout=2
        ):
            return True
    except OSError:
        return False


def test_主节点清单与运行单元清单一致():
    # 5 个主节点必须都是合法运行单元，防止两处常量清单漂移。
    assert set(MAIN_NODES) <= set(RUNTIME_UNITS)
    assert len(MAIN_NODES) == 5


def test_假LLM端到端_状态机按序流转至人工中断点():
    graph, _, config = _build(FIRST_PASS_RESPONSES)

    observed: list[tuple[str, WorkflowStatus]] = []
    interrupted = False
    for update in graph.stream(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-e2e"),
        config,
        stream_mode="updates",
    ):
        for node_name, node_update in update.items():
            if node_name == "__interrupt__":
                interrupted = True
                continue
            observed.append((node_name, node_update["status"]))
            if node_name == MAIN_NODES[0]:
                _assert_framework_state(node_update)

    # 终审通过即进入等待人工状态，图停在中断点。
    assert observed == [
        ("framework_orchestrator", WorkflowStatus.FRAMEWORK_BUILDING),
        ("reference_orchestrator", WorkflowStatus.REFERENCE_FETCHING),
        ("writing_orchestrator", WorkflowStatus.ARTICLE_WRITING),
        ("citation_validator", WorkflowStatus.AWAIT_USER_REVIEW),
    ]
    assert interrupted

    # 定稿恢复：human_review_gate 收束到 FINISHED。
    result = graph.invoke(Command(resume=FINALIZE), config)
    assert result["status"] == WorkflowStatus.FINISHED


def test_假LLM端到端_产出带角标全文草稿并可定稿():
    graph, _, config = _build(FIRST_PASS_RESPONSES)
    result = graph.invoke(
        initial_state("写一篇人才培养方案", "专业撰稿人", "trace-draft"), config
    )

    # 停在中断点：载荷只含元数据，不含正文全文。
    payload = result["__interrupt__"][0].value
    assert payload["chapter_ids"] == ["ch1", "ch2"]
    assert payload["citation_warnings"] == []
    assert all(
        draft.text not in json.dumps(payload, ensure_ascii=False)
        for draft in result["chapter_drafts"]
    )
    _assert_framework_state(result)
    _assert_full_draft(result)
    assert result["citation_report"].passed is True

    result = graph.invoke(Command(resume=FINALIZE), config)
    assert result["status"] == WorkflowStatus.FINISHED
    _assert_full_draft(result)


def test_混合修订意见_两类分支同轮执行且仅指定章节被修改():
    # 一次意见混合两类诉求，都落在 ch2：纯改写 + 补充佐证。
    directive_response = json.dumps(
        [
            {
                "target_chapter_id": "ch2",
                "type": "rewrite_only",
                "instruction": "口吻更克制",
            },
            {
                "target_chapter_id": "ch2",
                "type": "evidence_augmented",
                "instruction": "补充行业数据佐证",
            },
        ],
        ensure_ascii=False,
    )
    # 首轮 2 条语义核查 + 意见解析 + 增量核查只重审 ch2 的 1 条语义核查。
    graph, fake, config = _build(
        [*FIRST_PASS_RESPONSES, directive_response, SEMANTIC_PASS]
    )
    result = graph.invoke(initial_state("意图", "身份", "trace-loop"), config)
    text_before = {
        draft.chapter_id: draft.text for draft in result["chapter_drafts"]
    }

    result = graph.invoke(
        Command(resume={"action": "revise", "feedback": "第二章口吻克制些，再补数据"}),
        config,
    )

    # 仅指定章节被修改：ch1 原样，ch2 落实了两条修订指令。
    drafts = {draft.chapter_id: draft for draft in result["chapter_drafts"]}
    assert drafts["ch1"].text == text_before["ch1"]
    assert drafts["ch2"].text != text_before["ch2"]
    assert "口吻更克制" in drafts["ch2"].text
    assert "补充行业数据佐证" in drafts["ch2"].text

    # 台账追加一轮、轮次递增，两类指令同轮混合。
    assert result["iteration_round"] == 1
    (revision_round,) = result["revision_ledger"]
    assert [d.type for d in revision_round.directives] == [
        "rewrite_only",
        "evidence_augmented",
    ]

    # 修订后自动增量核查：只重审被修改章节（语义核查只多调一次），再回中断点。
    assert len(fake.calls) == FRAMEWORK_LLM_CALLS + 2 + 1 + 1
    assert result["citation_report"].passed is True
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    assert "__interrupt__" in result

    result = graph.invoke(Command(resume=FINALIZE), config)
    assert result["status"] == WorkflowStatus.FINISHED


def _parse_call_user_contents(fake: FakeLLM) -> list[str]:
    """从 FakeLLM 调用记录中筛出 human_review_gate 意见解析调用的 user 文本。"""
    contents: list[str] = []
    for messages in fake.calls:
        roles = {message["role"]: message["content"] for message in messages}
        if "修订意见解析器" in roles.get("system", ""):
            contents.append(roles.get("user", ""))
    return contents


def test_多轮迭代human_review_gate不失忆_第2轮解析prompt含第1轮意见():
    # 两轮修订：第 1 轮意见含独特可检索串；第 2 轮解析时历史台账须带上第 1 轮意见。
    round1_directive = json.dumps(
        [{"target_chapter_id": "ch1", "type": "rewrite_only", "instruction": "精炼引言"}],
        ensure_ascii=False,
    )
    round2_directive = json.dumps(
        [{"target_chapter_id": "ch2", "type": "rewrite_only", "instruction": "收束结论"}],
        ensure_ascii=False,
    )
    # 首轮 2 条语义核查 + 第1轮解析 + 增量核查1条 + 第2轮解析 + 增量核查1条。
    graph, fake, config = _build(
        [
            *FIRST_PASS_RESPONSES,
            round1_directive,
            SEMANTIC_PASS,
            round2_directive,
            SEMANTIC_PASS,
        ]
    )
    round1_feedback = "第一轮独特意见：引言部分务必更精炼有力"
    graph.invoke(initial_state("意图", "身份", "trace-memory"), config)
    graph.invoke(
        Command(resume={"action": "revise", "feedback": round1_feedback}), config
    )
    graph.invoke(
        Command(resume={"action": "revise", "feedback": "第二轮意见：结论再收束"}), config
    )

    parse_users = _parse_call_user_contents(fake)
    assert len(parse_users) == 2
    # 第 1 轮解析时台账尚无历史轮次，第 2 轮解析 prompt 必含第 1 轮意见（不失忆）。
    assert round1_feedback not in parse_users[0].split("本轮用户修改意见")[0]
    assert round1_feedback in parse_users[1]

    graph.invoke(Command(resume=FINALIZE), config)


def test_rewriter任务包prev_chapter_summary含多个前章摘要链():
    # 三章首写：末章任务包的 prev_chapter_summary 须含前两章摘要（摘要链验收）。
    from subagents import SubagentAdapter, stub_rewriter_loop_run

    framework_3ch = [
        '{"genre": "行业评论", "template_file": null}',
        '[{"title": "第一章", "subsections": []}, '
        '{"title": "第二章", "subsections": []}, '
        '{"title": "第三章", "subsections": []}]',
        '[{"text": "论点一"}]',
        '[{"text": "假说一", "refute_condition": "出现公开反例即证伪", '
        '"angle": "假设", "evidence_retrievable": true}]',
        '[{"text": "论点二"}]',
        '[{"text": "假说二", "refute_condition": "出现公开反例即证伪", '
        '"angle": "预言", "evidence_retrievable": true}]',
        '[{"text": "论点三"}]',
        '[{"text": "假说三", "refute_condition": "出现公开反例即证伪", '
        '"angle": "边界条件", "evidence_retrievable": true}]',
    ]

    tasks: list[dict] = []

    async def _recording_run(task: dict) -> dict:
        tasks.append(task)
        return await stub_rewriter_loop_run(task)

    recorder = SubagentAdapter("rewriter_loop", _recording_run)
    graph, _, config = _build(
        [*framework_3ch, SEMANTIC_PASS, SEMANTIC_PASS, SEMANTIC_PASS],
        rewriter_loop=recorder,
    )
    graph.invoke(initial_state("意图", "身份", "trace-chain"), config)

    draft_tasks = [task for task in tasks if task["mode"] == "draft"]
    assert [task["chapter_spec"]["id"] for task in draft_tasks] == ["ch1", "ch2", "ch3"]
    # 首章为空；末章摘要链含前两章各自摘要（带章节标题前缀、逐行拼接）。
    assert draft_tasks[0]["prev_chapter_summary"] == ""
    chain = draft_tasks[2]["prev_chapter_summary"]
    assert "【第一章】" in chain and "【第二章】" in chain
    # 末章摘要链是逐行拼接的多章摘要，而非仅紧邻一章。
    assert chain.count("\n") >= 1


def test_终审失败只重写不合格章节_超限携警告进入中断点():
    semantic_fail = json.dumps(
        [{"material_id": "m-ch1-p1-h1", "aligned": False, "reason": "观点不对应"}],
        ensure_ascii=False,
    )
    # 首轮：ch1 语义失败、ch2 通过 → 定向回退只重写 ch1；
    # 增量核查只重审 ch1，再次失败 → 超过上限（1）携警告进入中断点。
    graph, fake, config = _build(
        [*FRAMEWORK_RESPONSES, semantic_fail, SEMANTIC_PASS, semantic_fail],
        citation_max_retries=1,
    )
    result = graph.invoke(initial_state("意图", "身份", "trace-retry"), config)

    drafts = {draft.chapter_id: draft.text for draft in result["chapter_drafts"]}
    # 只重写不合格章节：ch1 落实了修复指令，ch2 未被重写。
    assert "根据引文终审发现的问题修复本章" in drafts["ch1"]
    assert "根据引文终审发现的问题修复本章" not in drafts["ch2"]
    # 超限不死循环：携未决引文警告强制进入人工中断点。
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    assert result["citation_retry_count"] == 2
    warnings = result["__interrupt__"][0].value["citation_warnings"]
    assert warnings and all("semantic_mismatch" in w for w in warnings)
    # 语义核查调用：首轮 2 章 + 增量重审 ch1 一次。
    assert len(fake.calls) == FRAMEWORK_LLM_CALLS + 2 + 1

    # 人工裁决仍可定稿收束，流程永不卡死。
    result = graph.invoke(Command(resume=FINALIZE), config)
    assert result["status"] == WorkflowStatus.FINISHED


def test_LLM调用次数与单元归属():
    fake = FakeLLM(list(FIRST_PASS_RESPONSES))
    units_seen: list[str] = []

    def factory(unit: str) -> FakeLLM:
        units_seen.append(unit)
        return fake

    graph = build_graph(llm_factory=factory, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "e2e-llm-count"}}
    graph.invoke(initial_state("意图", "身份", "trace-llm"), config)
    result = graph.invoke(Command(resume=FINALIZE), config)

    # 检索与写作由打桩子智能体承担；定稿分支不调 LLM。
    assert units_seen == ["framework_orchestrator", "citation_validator"]
    assert len(fake.calls) == len(FIRST_PASS_RESPONSES)
    # 终态记录的是最后一个节点（human_review_gate）的配置元数据，且不含密钥。
    assert result["current_node_llm_config"]["unit"] == "human_review_gate"
    assert "api_key" not in result["current_node_llm_config"]


@pytest.mark.skipif(
    not _pg_reachable(TEST_PG_DSN), reason="测试 Postgres 不可达"
)
def test_状态经Postgres存档器持久化():
    import psycopg

    thread_id = f"e2e-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    with postgres_checkpointer(TEST_PG_DSN) as saver:
        fake = FakeLLM(list(FIRST_PASS_RESPONSES))
        graph = build_graph(llm_factory=lambda unit: fake, checkpointer=saver)
        result = graph.invoke(
            initial_state("持久化测试", "专业撰稿人", "trace-pg"), config
        )
        # 中断等待人工期间，存档的状态机值是 AWAIT_USER_REVIEW（断点续跑的前提）。
        assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW
        _assert_full_draft(result)
        snapshot = graph.get_state(config)
        assert snapshot.values["status"] == WorkflowStatus.AWAIT_USER_REVIEW

        # 恢复定稿后终态同样入档。
        result = graph.invoke(Command(resume=FINALIZE), config)
        assert result["status"] == WorkflowStatus.FINISHED
        assert graph.get_state(config).values["status"] == WorkflowStatus.FINISHED

    # 直接查库：该 thread_id 下确有 checkpoint 记录（每个节点一步 + 起始步）。
    with psycopg.connect(TEST_PG_DSN) as conn:
        row = conn.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = %s", (thread_id,)
        ).fetchone()
        assert row is not None and row[0] >= len(MAIN_NODES)
