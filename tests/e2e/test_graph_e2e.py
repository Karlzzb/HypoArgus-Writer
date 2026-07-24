"""端到端闭环测试：注入假 LLM 跑整张图，验证状态机流转、人工中断点与迭代闭环。

framework_orchestrator 预置最小 JSON 应答序列（自由结构、2 章、每章 1 论点
1 假说）；reference_orchestrator 与 writing_orchestrator 缺省走打桩子智能体、
不调 LLM（中断恢复用例例外：走 rewriter_loop 真实现链路，写作与自审经
键控应答分派）；document_reviewer 每个受审章节消费一条语义核查 JSON 应答，
再消费一条篇级评审 JSON 应答（始终全量、在全部语义核查之后）；
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
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.types import Command

from agents.chapter_reviewer import make_stub_chapter_reviewer
from agents.rewriter_loop import make_stub_rewriter_loop
from agents.search_agent import make_stub_search_agent, material_id_from_source_ref
from domain.citation_reconciler import MARKER_PATTERN
from domain.units import MAIN_NODES
from graph import build_graph, checkpoint_serializer, postgres_checkpointer
from llm.llm_client import FakeLLM
from llm.llm_config import RUNTIME_UNITS
from domain.state import SourceKind, WorkflowStatus, initial_state
from service.llm_response_plans import (
    DOCUMENT_REVIEW_PASS,
    FIRST_PASS_LLM_CALLS,
    FIRST_PASS_RESPONSES,
    FRAMEWORK_KEYED_RESPONSES,
    FRAMEWORK_LLM_CALLS,
    FRAMEWORK_RESPONSES,
    SEMANTIC_PASS,
)

TEST_PG_DSN = os.environ.get(
    "HYPOARGUS_TEST_PG_DSN",
    "postgresql://postgres:postgres@127.0.0.1:15432/postgres",
)

FINALIZE = {"action": "finalize"}


def _stub_material_id(chapter_id: str, hypothesis_id: str) -> str:
    source_kinds: tuple[SourceKind, ...] = ("web", "knowledge_base", "structured_data")
    source_kind = source_kinds[sum(hypothesis_id.encode()) % len(source_kinds)]
    source_ref = {
        "stub_source": "search_agent",
        "chapter_id": chapter_id,
        "hypothesis_id": hypothesis_id,
    }
    if source_kind == "web":
        source_ref["url"] = f"https://stub.example/{hypothesis_id}"
    return material_id_from_source_ref(source_kind, source_ref)


def _build(responses: list[str], keyed: dict[str, list[str]] | None = None, **kwargs):
    """带 InMemorySaver 与共享假 LLM 构图，返回（graph, fake, config）。

    本文件验收的是图编排（状态机、路由、中断闭环），写作单元默认注入
    打桩改写器（真实现契约在 tests/agents/rewriter_loop/ 单独覆盖）；
    需要记录器等定制时经 kwargs 显式覆盖。keyed 用于按提示词内容键控的
    额外应答（并发调用场景，与框架假说键控应答合并）。
    """
    fake = FakeLLM(
        list(responses),
        keyed_responses={**FRAMEWORK_KEYED_RESPONSES, **(keyed or {})},
    )
    kwargs.setdefault("rewriter_loop", make_stub_rewriter_loop())
    kwargs.setdefault("search_agent", make_stub_search_agent())
    # 章级评审默认走打桩（本文件验收图编排；真实现评审在 tests/agents/chapter_reviewer/
    # 单独覆盖）：桩恒判「零违规、通过」，首写循环 write→review 后即短路，不调 LLM。
    kwargs.setdefault("chapter_reviewer", make_stub_chapter_reviewer())
    graph = build_graph(
        llm_factory=lambda unit: fake, checkpointer=InMemorySaver(serde=checkpoint_serializer()), **kwargs
    )
    config: RunnableConfig = {"configurable": {"thread_id": f"e2e-{uuid.uuid4()}"}}
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
    # 6 个主节点必须都是合法运行单元，防止两处常量清单漂移。
    assert set(MAIN_NODES) <= set(RUNTIME_UNITS)
    assert len(MAIN_NODES) == 6


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
            if node_update is None:
                # reference_join 是无操作汇合节点，不产生状态更新。
                assert node_name == "reference_join"
                continue
            observed.append((node_name, node_update["status"]))
            if node_name == MAIN_NODES[0]:
                _assert_framework_state(node_update)

    # 终审通过即进入等待人工状态，图停在中断点。
    # 检索与首写两段并行扇出：2 章各一个任务、各回写一次单章更新
    #（同段两条更新同名同值，到达顺序不影响断言），章级产物落 checkpoint。
    assert observed == [
        ("framework_orchestrator", WorkflowStatus.FRAMEWORK_BUILDING),
        ("reference_orchestrator", WorkflowStatus.ARTICLE_WRITING),
        ("reference_orchestrator", WorkflowStatus.ARTICLE_WRITING),
        ("document_reviewer", WorkflowStatus.AWAIT_USER_REVIEW),
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

    # 并行首写的前章衔接：第二章正文承接第一章的规划摘要（规划摘要链），
    # 而非实际写成的摘要——各章因此不依赖前章草稿、可以并行。
    drafts = {draft.chapter_id: draft for draft in result["chapter_drafts"]}
    planned_ch1 = result["outline"][0].planned_summary
    assert planned_ch1
    assert planned_ch1 in drafts["ch2"].text

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
    # 首轮全量终审 + 意见解析 + 增量核查只重审 ch2 的 1 条语义核查 + 篇级评审 1 条。
    graph, fake, config = _build(
        [*FIRST_PASS_RESPONSES, directive_response, SEMANTIC_PASS, DOCUMENT_REVIEW_PASS]
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

    # 修订后自动增量核查：只重审被修改章节（语义核查只多调一次），
    # 篇级评审全量再调一次，再回中断点。
    assert len(fake.calls) == FIRST_PASS_LLM_CALLS + 1 + 1 + 1
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
    # 首轮全量终审 + 每轮（解析 + 增量核查 1 条 + 篇级评审 1 条）× 2。
    graph, fake, config = _build(
        [
            *FIRST_PASS_RESPONSES,
            round1_directive,
            SEMANTIC_PASS,
            DOCUMENT_REVIEW_PASS,
            round2_directive,
            SEMANTIC_PASS,
            DOCUMENT_REVIEW_PASS,
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


@pytest.mark.parametrize(
    "backend",
    [
        "memory",
        pytest.param(
            "postgres",
            marks=pytest.mark.skipif(
                not _pg_reachable(TEST_PG_DSN), reason="测试 Postgres 不可达"
            ),
        ),
    ],
)
def test_写作自环中断恢复_已完成章节零重复调用且产物与不中断路径等价(backend: str):
    """章级 checkpoint 验收（ADR-0001 约束 1 与 4，写→评→重写循环版）。

    验收口径按 issue #46：**stub 崩溃注入、不调真 LLM**（全量真实链路验证留 T6）。
    故障注入在子智能体桩层：第二章首写（rewriter mode=draft）抛致命异常，图调用
    在写第二章的并行分支内崩溃并向外抛出，模拟进程死于该章执行中途；首章分支的
    checkpoint 已落盘，同 thread_id 二次驱动恢复。断言：已完成章节零重复写作/评审
    调用、续跑后该章重走「写→评」、subagent_start/end 成对且带 chapter_id 与 mode、
    最终产物与不中断路径完全等价。
    """
    import asyncio as _asyncio
    from contextlib import contextmanager

    from agents.chapter_reviewer import stub_chapter_reviewer_run
    from agents.contracts import SubagentAdapter
    from agents.rewriter_loop import stub_rewriter_loop_run
    from domain.events import SUBAGENT_END, SUBAGENT_START

    def _recorder(events: list[tuple[str, dict]]):
        def _hook(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        return _hook

    def _crashing_rewriter(
        calls: list[tuple[str, str]],
        events: list[tuple[str, dict]],
        crash_chapter: str | None,
    ) -> SubagentAdapter:
        """故障注入桩改写器：对 crash_chapter 的首写抛致命异常，其余走确定性桩。

        崩溃前先让位给并行的其他分支（asyncio.sleep），使「某章完成、崩溃章
        未完成」的死亡现场确定性成立。
        """

        async def _run(task: dict) -> dict:
            chapter_id = task["chapter_spec"]["id"]
            mode = task["mode"]
            calls.append((chapter_id, mode))
            if chapter_id == crash_chapter and mode == "draft":
                await _asyncio.sleep(0.2)
                raise RuntimeError("故障注入：进程死于第二章首写")
            return await stub_rewriter_loop_run(task)

        return SubagentAdapter("rewriter_loop", _run, _recorder(events))

    def _recording_reviewer(
        calls: list[tuple[str, str]], events: list[tuple[str, dict]]
    ) -> SubagentAdapter:
        """记录调用的桩评审器：恒判「零违规、通过」，供零重复与写→评断言。"""

        async def _run(task: dict) -> dict:
            calls.append((task["chapter_spec"]["id"], task["mode"]))
            return await stub_chapter_reviewer_run(task)

        return SubagentAdapter("chapter_reviewer", _run, _recorder(events))

    def _pair(events: list[tuple[str, dict]], unit: str, chapter_id: str) -> list[str]:
        return [
            etype
            for etype, payload in events
            if payload["unit"] == unit and payload["chapter_id"] == chapter_id
        ]

    @contextmanager
    def _checkpoint_backend(kind: str):
        if kind == "postgres":
            with postgres_checkpointer(TEST_PG_DSN) as saver:
                yield saver
        else:
            yield InMemorySaver(serde=checkpoint_serializer())

    with _checkpoint_backend(backend) as saver:
        thread_id = f"e2e-crash-{uuid.uuid4()}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # 第一个「进程」：桩改写器对第二章首写抛致命异常，图调用在写第二章的
        # 并行分支内崩溃并向外抛出；首章分支的 checkpoint 已落盘，等价于
        # 「某章完成、下一章未完成」时进程被 kill。
        rw_before: list[tuple[str, str]] = []
        rv_before: list[tuple[str, str]] = []
        events_before: list[tuple[str, dict]] = []
        fake = FakeLLM(
            list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
        )
        graph = build_graph(
            llm_factory=lambda unit: fake,
            checkpointer=saver,
            search_agent=make_stub_search_agent(),
            rewriter_loop=_crashing_rewriter(rw_before, events_before, "ch2"),
            chapter_reviewer=_recording_reviewer(rv_before, events_before),
        )
        with pytest.raises(RuntimeError, match="故障注入：进程死于第二章首写"):
            graph.invoke(
                initial_state("写一篇人才培养方案", "专业撰稿人", "trace-crash"),
                config,
            )
        # 死亡现场：并行首写超步崩溃时，已完成的 ch1 分支写入作为 pending write
        # 被 checkpoint 保留，待执行任务只剩失败的 ch2 分支（chapter_drafter）；
        # ch2 分支自身的写入被丢弃，不留半成品。
        snapshot = graph.get_state(config)
        assert snapshot.next == ("reference_orchestrator",)
        assert [d.chapter_id for d in snapshot.values["chapter_drafts"]] == ["ch1"]
        # 崩溃前：ch1 走完写→评（各一次），ch2 首写被注入异常（评审未触达）。
        assert sorted(rw_before) == [("ch1", "draft"), ("ch2", "draft")]
        assert rv_before == [("ch1", "review")]

        # 第二个「进程」：同 thread_id 恢复，只备剩余阶段应答（2 章语义核查 + 篇级评审）。
        rw_after: list[tuple[str, str]] = []
        rv_after: list[tuple[str, str]] = []
        events_after: list[tuple[str, dict]] = []
        fake2 = FakeLLM([SEMANTIC_PASS, SEMANTIC_PASS, DOCUMENT_REVIEW_PASS])
        graph2 = build_graph(
            llm_factory=lambda unit: fake2,
            checkpointer=saver,
            search_agent=make_stub_search_agent(),
            rewriter_loop=_crashing_rewriter(rw_after, events_after, None),
            chapter_reviewer=_recording_reviewer(rv_after, events_after),
        )
        resumed = graph2.invoke(None, config)

        # 已完成章节零重复调用：恢复进程只重跑第二章的写→评，首章零重复。
        assert rw_after == [("ch2", "draft")]
        assert rv_after == [("ch2", "review")]

        # 事件成对且带业务上下文（ADR-0001 约束 2 与 4）。崩溃进程按章拆分：
        # 首章的写作/评审均 start/end 成对；被杀的第二章首写留「有 start 无 end」
        # 残链——异常沿桩层上抛后 subagent_end 不再发出，如实反映死亡现场。
        assert _pair(events_before, "rewriter_loop", "ch1") == [
            SUBAGENT_START,
            SUBAGENT_END,
        ]
        assert _pair(events_before, "chapter_reviewer", "ch1") == [
            SUBAGENT_START,
            SUBAGENT_END,
        ]
        assert _pair(events_before, "rewriter_loop", "ch2") == [SUBAGENT_START]
        # 续跑：第二章写→评均成对。
        assert _pair(events_after, "rewriter_loop", "ch2") == [
            SUBAGENT_START,
            SUBAGENT_END,
        ]
        assert _pair(events_after, "chapter_reviewer", "ch2") == [
            SUBAGENT_START,
            SUBAGENT_END,
        ]
        # 事件均带 chapter_id 与 mode 业务上下文。
        for _etype, payload in events_before + events_after:
            assert payload["chapter_id"] in {"ch1", "ch2"}
            assert payload["mode"] in {"draft", "review"}
        assert resumed["status"] == WorkflowStatus.AWAIT_USER_REVIEW
        _assert_full_draft(resumed)

        # 与不中断路径的产物完全等价（同一桩编排保证可逐字段比对）。
        baseline_fake = FakeLLM(
            list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
        )
        baseline_graph = build_graph(
            llm_factory=lambda unit: baseline_fake,
            checkpointer=InMemorySaver(serde=checkpoint_serializer()),
            search_agent=make_stub_search_agent(),
            rewriter_loop=_crashing_rewriter([], [], None),
            chapter_reviewer=_recording_reviewer([], []),
        )
        baseline_config: RunnableConfig = {
            "configurable": {"thread_id": f"e2e-crash-base-{uuid.uuid4()}"}
        }
        baseline = baseline_graph.invoke(
            initial_state("写一篇人才培养方案", "专业撰稿人", "trace-crash-base"),
            baseline_config,
        )
        assert resumed["chapter_drafts"] == baseline["chapter_drafts"]
        assert resumed["citation_library"] == baseline["citation_library"]
        assert resumed["citation_report"] == baseline["citation_report"]

        # 恢复后仍可定稿收束。
        result = graph2.invoke(Command(resume=FINALIZE), config)
        assert result["status"] == WorkflowStatus.FINISHED


def test_修订超步中断恢复_评审后改写崩溃则该章评审与改写整体重跑():
    """修订超步原子性验收（ADR-0001 约束 1 与 4，issue #47 评审前置版）。

    修订分支现为「评审 + 改写」两次子智能体调用。故障注入在 ch2 的改写侧，
    并先让 ch1 并行分支完成：其写入作为 pending write 落 checkpoint。断言：
    同 thread_id 恢复后 ch2 从头重跑整个修订分支（评审与改写都重来），已完成
    的 ch1 修订零重复子智能体调用。
    """
    import asyncio as _asyncio

    from agents.chapter_reviewer import stub_chapter_reviewer_run
    from agents.contracts import SubagentAdapter
    from agents.rewriter_loop import stub_rewriter_loop_run

    def _crashing_revise_rewriter(
        calls: list[tuple[str, str]], crash_chapter: str | None
    ) -> SubagentAdapter:
        """故障注入桩改写器：对 crash_chapter 的定向改写抛致命异常，其余走确定性桩。"""

        async def _run(task: dict) -> dict:
            chapter_id = task["chapter_spec"]["id"]
            mode = task["mode"]
            calls.append((chapter_id, mode))
            if chapter_id == crash_chapter and mode == "revise":
                await _asyncio.sleep(0.2)
                raise RuntimeError("故障注入：进程死于第二章定向改写")
            return await stub_rewriter_loop_run(task)

        return SubagentAdapter("rewriter_loop", _run)

    def _recording_reviewer(calls: list[tuple[str, str]]) -> SubagentAdapter:
        """记录调用的桩评审器：恒判「零违规、通过」，供零重复与整超步重跑断言。"""

        async def _run(task: dict) -> dict:
            calls.append((task["chapter_spec"]["id"], task["mode"]))
            return await stub_chapter_reviewer_run(task)

        return SubagentAdapter("chapter_reviewer", _run)

    # 一轮意见落两章各一条纯改写指令：两章并行，ch2 分支中途崩溃。
    directive_response = json.dumps(
        [
            {
                "target_chapter_id": "ch1",
                "type": "rewrite_only",
                "instruction": "收紧第一章语气",
            },
            {
                "target_chapter_id": "ch2",
                "type": "rewrite_only",
                "instruction": "收束第二章结论",
            },
        ],
        ensure_ascii=False,
    )
    saver = InMemorySaver(serde=checkpoint_serializer())
    thread_id = f"e2e-revise-crash-{uuid.uuid4()}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 第一个「进程」：首轮走到中断点后按意见开修订轮，ch2 改写抛致命异常。
    rw_before: list[tuple[str, str]] = []
    rv_before: list[tuple[str, str]] = []
    # 意见影响 2/2 章（超过大纲一半），触发大扇出确认；confirm 恢复时
    # 节点重放、解析走 durable task 缓存不重复调用，故解析应答只备一份。
    fake = FakeLLM(
        [*FIRST_PASS_RESPONSES, directive_response],
        keyed_responses=FRAMEWORK_KEYED_RESPONSES,
    )
    graph = build_graph(
        llm_factory=lambda unit: fake,
        checkpointer=saver,
        search_agent=make_stub_search_agent(),
        rewriter_loop=_crashing_revise_rewriter(rw_before, "ch2"),
        chapter_reviewer=_recording_reviewer(rv_before),
    )
    graph.invoke(initial_state("意图", "身份", "trace-revise-crash"), config)
    confirm_request = graph.invoke(
        Command(resume={"action": "revise", "feedback": "两章都按意见修改"}),
        config,
    )
    assert "pending_confirmation" in confirm_request["__interrupt__"][0].value
    with pytest.raises(RuntimeError, match="故障注入：进程死于第二章定向改写"):
        graph.invoke(Command(resume={"action": "confirm"}), config)

    # 死亡现场：ch2 修订分支的评审虽已跑完，但该分支写入整体丢弃；ch1
    # 分支产物已作为 pending write 落 checkpoint，待执行任务只剩 ch2。
    assert ("ch2", "revise") in rv_before  # 崩溃前该章评审确已执行。
    assert sorted(rw_before) == [
        ("ch1", "draft"),
        ("ch1", "revise"),
        ("ch2", "draft"),
        ("ch2", "revise"),
    ]
    snapshot = graph.get_state(config)
    assert snapshot.next == ("writing_orchestrator",)
    drafts = {d.chapter_id: d for d in snapshot.values["chapter_drafts"]}
    assert "收紧第一章语气" in drafts["ch1"].text
    assert "收束第二章结论" not in drafts["ch2"].text
    assert [
        d.target_chapter_id for d in snapshot.values["pending_directives"]
    ] == ["ch1", "ch2"]
    assert snapshot.values["revised_chapter_ids"] == ["ch1"]

    # 第二个「进程」：同 thread_id 恢复，只备增量核查应答（两受审章各一条 + 篇级评审一条）。
    rw_after: list[tuple[str, str]] = []
    rv_after: list[tuple[str, str]] = []
    fake2 = FakeLLM([SEMANTIC_PASS, SEMANTIC_PASS, DOCUMENT_REVIEW_PASS])
    graph2 = build_graph(
        llm_factory=lambda unit: fake2,
        checkpointer=saver,
        search_agent=make_stub_search_agent(),
        rewriter_loop=_crashing_revise_rewriter(rw_after, None),
        chapter_reviewer=_recording_reviewer(rv_after),
    )
    resumed = graph2.invoke(None, config)

    # 分支原子性：恢复进程把 ch2 的修订分支从头重跑——评审与改写各恰一次；
    # 已完成的 ch1 修订零重复调用（评审与改写都不再触达）。
    assert rv_after == [("ch2", "revise")]
    assert rw_after == [("ch2", "revise")]

    # 恢复后两章意见均已落实，回到人工中断点并可定稿收束。
    drafts = {d.chapter_id: d for d in resumed["chapter_drafts"]}
    assert "收紧第一章语气" in drafts["ch1"].text
    assert "收束第二章结论" in drafts["ch2"].text
    assert resumed["pending_directives"] == []
    assert resumed["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    result = graph2.invoke(Command(resume=FINALIZE), config)
    assert result["status"] == WorkflowStatus.FINISHED


def test_人工修订多章在同一并行波次改写():
    """Issue #76：同轮不同章节的人工修订必须并发，不能退化为串行自环。

    两章全局意见经确认后各自进入一个 Send 分支。改写桩以屏障要求两个
    revise 调用在同一波次抵达：旧的串行路由会使首个调用超时，本用例因此
    能直接捕获用户可见的修订阶段墙钟退化。
    """
    import asyncio as _asyncio
    import threading

    from agents.contracts import SubagentAdapter
    from agents.rewriter_loop import stub_rewriter_loop_run

    barrier = threading.Barrier(2)
    revise_calls: list[str] = []
    revision_notes: dict[str, str] = {}

    async def _run(task: dict) -> dict:
        if task["mode"] == "revise" and task["revision_note"]["user_directives"]:
            chapter_id = task["chapter_spec"]["id"]
            revise_calls.append(chapter_id)
            revision_notes[chapter_id] = task["revision_note"]["user_directives"]
            await _asyncio.to_thread(barrier.wait, 1.0)
        return await stub_rewriter_loop_run(task)

    directive_response = json.dumps(
        [
            {
                "target_chapter_id": "ch1",
                "type": "rewrite_only",
                "instruction": "收紧第一章论证",
            },
            {
                "target_chapter_id": "ch2",
                "type": "rewrite_only",
                "instruction": "收束第二章结论",
            },
        ],
        ensure_ascii=False,
    )
    graph, _, config = _build(
        [
            *FIRST_PASS_RESPONSES,
            directive_response,
            SEMANTIC_PASS,
            SEMANTIC_PASS,
            DOCUMENT_REVIEW_PASS,
        ],
        rewriter_loop=SubagentAdapter("rewriter_loop", _run),
    )

    graph.invoke(initial_state("意图", "身份", "trace-revise-parallel"), config)
    confirmation = graph.invoke(
        Command(resume={"action": "revise", "feedback": "全篇口吻更克制"}),
        config,
    )
    assert "pending_confirmation" in confirmation["__interrupt__"][0].value

    result = graph.invoke(Command(resume={"action": "confirm"}), config)

    assert sorted(revise_calls) == ["ch1", "ch2"]
    assert revision_notes == {"ch1": "收紧第一章论证", "ch2": "收束第二章结论"}
    assert [draft.chapter_id for draft in result["chapter_drafts"]] == ["ch1", "ch2"]
    assert result["pending_directives"] == []
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW


def test_检索并行扇出中断恢复_已完成分支素材保留且只重跑失败分支():
    """检索阶段章级 checkpoint 验收（ADR-0001 约束 1 的 Send 并行扇出表述）。

    故障注入：search_agent 对 ch2 的调用抛致命异常，图调用在检索超步内
    崩溃并向外抛出；已完成的 ch1 分支写入作为 pending write 被保留，
    同 thread_id 二次驱动只重跑 ch2 分支——ch1 的检索调用零重复，
    续跑走完剩余流程到人工中断点且引文库两章齐全。
    """
    import asyncio as _asyncio

    class 记录检索适配器:
        """按任务包假说返回打桩素材；可对指定章节注入致命异常。"""

        def __init__(self, crash_chapter: str | None) -> None:
            self.unit = "search_agent"
            self.calls: list[str] = []
            self._crash_chapter = crash_chapter

        async def run(self, task: dict) -> dict:
            chapter_id = task["chapter_id"]
            self.calls.append(chapter_id)
            if chapter_id == self._crash_chapter:
                # 让位给并行的 ch1 分支先完成，使「某章完成、另一章未完成」
                # 的死亡现场确定性成立。
                await _asyncio.sleep(0.2)
                raise RuntimeError("故障注入：进程死于 ch2 检索")
            return {
                "materials": [
                    {
                        "id": f"m-{hypothesis['id']}",
                        "hypothesis_id": hypothesis["id"],
                        "source": f"来源 {hypothesis['id']}",
                        "url": None,
                        "source_kind": "knowledge_base",
                        "excerpt": f"摘录 {hypothesis['id']}",
                        "relevance_score": 0.9,
                        "verdict": "pass",
                    }
                    for hypothesis in task["hypotheses"]
                ]
            }

    saver = InMemorySaver(serde=checkpoint_serializer())
    thread_id = f"e2e-ref-crash-{uuid.uuid4()}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 第一个「进程」：ch2 检索分支抛致命异常，检索超步内崩溃。
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    crashing = 记录检索适配器(crash_chapter="ch2")
    graph = build_graph(
        llm_factory=lambda unit: fake,
        checkpointer=saver,
        search_agent=crashing,
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
    with pytest.raises(RuntimeError, match="故障注入：进程死于 ch2 检索"):
        graph.invoke(
            initial_state("写一篇人才培养方案", "专业撰稿人", "trace-ref-crash"),
            config,
        )
    assert sorted(crashing.calls) == ["ch1", "ch2"]
    # 死亡现场：已完成的 ch1 分支素材作为 pending write 被 checkpoint 保留，
    # 待执行任务只剩失败的 ch2 检索分支（章级素材独立落 checkpoint）。
    snapshot = graph.get_state(config)
    assert snapshot.next == ("reference_orchestrator",)
    assert [
        material.chapter_id for material in snapshot.values["citation_library"]
    ] == ["ch1"]

    # 第二个「进程」：同 thread_id 恢复，只备剩余阶段应答（2 章语义核查 + 篇级评审）。
    fake2 = FakeLLM([SEMANTIC_PASS, SEMANTIC_PASS, DOCUMENT_REVIEW_PASS])
    healthy = 记录检索适配器(crash_chapter=None)
    graph2 = build_graph(
        llm_factory=lambda unit: fake2,
        checkpointer=saver,
        search_agent=healthy,
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
    resumed = graph2.invoke(None, config)

    # 已完成分支零重复检索调用：恢复进程只重跑失败的 ch2 分支。
    assert healthy.calls == ["ch2"]
    assert resumed["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    assert {
        material.chapter_id for material in resumed["citation_library"]
    } == {"ch1", "ch2"}
    assert [draft.chapter_id for draft in resumed["chapter_drafts"]] == ["ch1", "ch2"]


def test_rewriter任务包prev_chapter_summary含多个前章摘要链():
    # 三章首写：末章任务包的 prev_chapter_summary 须含前两章摘要（摘要链验收）。
    from agents.contracts import SubagentAdapter
    from agents.rewriter_loop import stub_rewriter_loop_run

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
        [
            *framework_3ch,
            SEMANTIC_PASS,
            SEMANTIC_PASS,
            SEMANTIC_PASS,
            DOCUMENT_REVIEW_PASS,
        ],
        rewriter_loop=recorder,
    )
    graph.invoke(initial_state("意图", "身份", "trace-chain"), config)

    # 并行首写各分支的执行完成顺序不定：按目标章 id 排序后断言。
    draft_tasks = sorted(
        (task for task in tasks if task["mode"] == "draft"),
        key=lambda task: task["chapter_spec"]["id"],
    )
    assert [task["chapter_spec"]["id"] for task in draft_tasks] == ["ch1", "ch2", "ch3"]
    # 首章为空；末章规划摘要链含前两章各自的规划摘要（带章节标题前缀、逐行拼接）。
    assert draft_tasks[0]["prev_chapter_summary"] == ""
    chain = draft_tasks[2]["prev_chapter_summary"]
    assert "【第一章】" in chain and "【第二章】" in chain
    # 末章摘要链是逐行拼接的多章摘要，而非仅紧邻一章。
    assert chain.count("\n") >= 1


def test_终审失败只重写不合格章节_超限携警告进入中断点():
    ch1_material_id = _stub_material_id("ch1", "ch1-p1-h1")
    semantic_fail = json.dumps(
        [{"material_id": ch1_material_id, "aligned": False, "reason": "观点不对应"}],
        ensure_ascii=False,
    )
    # 首轮：ch1 语义失败、ch2 通过 → 定向回退只重写 ch1；
    # 增量核查只重审 ch1，再次失败 → 超过上限（1）携警告进入中断点。
    # 语义核查逐章并发、应答内容不同，按提示词中的章节标识键控分派；
    # 篇级评审两轮各放行一条（顺序应答，恒在该轮语义核查之后消费）。
    graph, fake, config = _build(
        [*FRAMEWORK_RESPONSES, DOCUMENT_REVIEW_PASS, DOCUMENT_REVIEW_PASS],
        keyed={
            "章节 ch1 正文": [semantic_fail, semantic_fail],
            "章节 ch2 正文": [SEMANTIC_PASS],
        },
        document_review_max_retries=1,
    )
    result = graph.invoke(initial_state("意图", "身份", "trace-retry"), config)

    drafts = {draft.chapter_id: draft.text for draft in result["chapter_drafts"]}
    # 只重写不合格章节：ch1 落实了终审报告折成的修订说明（打桩附注修改指导），
    # ch2 未被重写。
    assert "（修订落实：" in drafts["ch1"]
    assert "观点不对应" in drafts["ch1"]
    assert "（修订落实：" not in drafts["ch2"]
    # 超限不死循环：携未决引文警告强制进入人工中断点。
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    assert result["citation_retry_count"] == 2
    warnings = result["__interrupt__"][0].value["citation_warnings"]
    assert warnings and all("semantic_mismatch" in w for w in warnings)
    # 语义核查调用：首轮 2 章 + 增量重审 ch1 一次；篇级评审每轮 1 次共 2 次。
    assert len(fake.calls) == FRAMEWORK_LLM_CALLS + 2 + 1 + 2

    # 人工裁决仍可定稿收束，流程永不卡死。
    result = graph.invoke(Command(resume=FINALIZE), config)
    assert result["status"] == WorkflowStatus.FINISHED


def test_篇级评审fact_conflict打回_只重写涉事章节且修订规则可溯源():
    """篇级硬伤（跨章硬事实冲突）触发定向打回（issue #48）。

    首轮语义核查全放行、篇级评审报 ch2 fact_conflict → 只重写 ch2；
    重写任务的修订说明规则名是 document_review.fact_conflict（终审报告
    折成的分区式修订说明，打桩改写器把修改指导附注进正文）；增量核查
    与二次篇级评审放行后回到人工中断点。
    """
    from agents.contracts import SubagentAdapter
    from agents.rewriter_loop import stub_rewriter_loop_run

    tasks: list[dict] = []

    async def _recording_run(task: dict) -> dict:
        tasks.append(task)
        return await stub_rewriter_loop_run(task)

    fact_conflict_response = json.dumps(
        [
            {
                "dimension": "fact_conflict",
                "chapter_ids": ["ch2"],
                "detail": "两章对同一指标结论相反",
            }
        ],
        ensure_ascii=False,
    )
    # 首轮：2 条语义核查放行 + 篇级评审报硬伤；回退重写 ch2 后
    # 增量核查 1 条 + 篇级评审放行。
    graph, _, config = _build(
        [
            *FRAMEWORK_RESPONSES,
            SEMANTIC_PASS,
            SEMANTIC_PASS,
            fact_conflict_response,
            SEMANTIC_PASS,
            DOCUMENT_REVIEW_PASS,
        ],
        rewriter_loop=SubagentAdapter("rewriter_loop", _recording_run),
    )
    result = graph.invoke(initial_state("意图", "身份", "trace-fact"), config)

    # 定向回退：只有 ch2 被 revise，且修订说明规则名带 document_review 前缀。
    revise_tasks = [task for task in tasks if task["mode"] == "revise"]
    assert [task["chapter_spec"]["id"] for task in revise_tasks] == ["ch2"]
    rules = [
        entry["rule"]
        for entry in revise_tasks[0]["revision_note"]["rule_violations"]
    ]
    assert rules == ["document_review.fact_conflict"]

    # 打桩改写器把修改指导附注进正文：ch2 落实、ch1 未被重写。
    drafts = {draft.chapter_id: draft.text for draft in result["chapter_drafts"]}
    # 问题明细点名全部涉及章节后随修订说明附注进正文。
    assert (
        "（修订落实：跨章硬事实冲突（涉及章节 ch2）：两章对同一指标结论相反）"
        in drafts["ch2"]
    )
    assert "（修订落实：" not in drafts["ch1"]

    # 重写后终审通过，回到人工中断点且无未决警告。
    assert result["citation_report"].passed is True
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    payload = result["__interrupt__"][0].value
    assert payload["citation_warnings"] == []
    assert payload["review_warnings"] == []

    result = graph.invoke(Command(resume=FINALIZE), config)
    assert result["status"] == WorkflowStatus.FINISHED


@pytest.mark.parametrize(
    "backend",
    [
        "memory",
        pytest.param(
            "postgres",
            marks=pytest.mark.skipif(
                not _pg_reachable(TEST_PG_DSN), reason="测试 Postgres 不可达"
            ),
        ),
    ],
)
def test_回退并行扇出中断恢复_已完成分支保留且只重跑崩溃分支(backend: str):
    """回退并行扇出验收（ADR-0001 约束 1 与 4，回退扇出版）。

    篇级评审报 ch1+ch2 fact_conflict → route_after_document_reviewer 为两章各发
    一个 Send 并行回退（issue #64：回退改回 Send 扇出，对齐首写）。崩溃注入于
    ch1 回退改写中途：ch1 分支内抛致命异常，ch2 分支的写入作为 pending write
    被 checkpoint 保留。同 thread_id 二次驱动恢复：只重跑崩溃的 ch1 分支，ch2
    零重复——这正是并行扇出 + 超步事务语义成立的可观测证据（串行自环下 ch1
    崩溃则 ch2 根本未曾执行）。断言：已完成分支零重复、subagent 事件成对且带
    chapter_id/mode、最终产物与不中断路径等价。
    """
    import asyncio as _asyncio
    from contextlib import contextmanager

    from agents.chapter_reviewer import make_stub_chapter_reviewer
    from agents.contracts import SubagentAdapter
    from agents.rewriter_loop import stub_rewriter_loop_run
    from domain.events import SUBAGENT_END, SUBAGENT_START

    def _recorder(events: list[tuple[str, dict]]):
        def _hook(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        return _hook

    def _crashing_rewriter(
        calls: list[tuple[str, str]],
        events: list[tuple[str, dict]],
        crash_chapter: str | None,
    ) -> SubagentAdapter:
        """故障注入桩改写器：对 crash_chapter 的回退改写（mode=revise）抛致命异常，
        其余走确定性桩。崩溃前先让位给并行兄弟分支，使「某章完成、崩溃章未完成」
        的死亡现场确定性成立。"""

        async def _run(task: dict) -> dict:
            chapter_id = task["chapter_spec"]["id"]
            mode = task["mode"]
            calls.append((chapter_id, mode))
            if chapter_id == crash_chapter and mode == "revise":
                await _asyncio.sleep(0.2)
                raise RuntimeError("故障注入：进程死于 ch1 回退改写")
            return await stub_rewriter_loop_run(task)

        return SubagentAdapter("rewriter_loop", _run, _recorder(events))

    def _pair(events: list[tuple[str, dict]], unit: str, chapter_id: str) -> list[str]:
        return [
            etype
            for etype, payload in events
            if payload["unit"] == unit and payload["chapter_id"] == chapter_id
        ]

    @contextmanager
    def _checkpoint_backend(kind: str):
        if kind == "postgres":
            with postgres_checkpointer(TEST_PG_DSN) as saver:
                yield saver
        else:
            yield InMemorySaver(serde=checkpoint_serializer())

    fact_conflict_response = json.dumps(
        [
            {
                "dimension": "fact_conflict",
                "chapter_ids": ["ch1", "ch2"],
                "detail": "两章对同一指标结论相反",
            }
        ],
        ensure_ascii=False,
    )

    with _checkpoint_backend(backend) as saver:
        thread_id = f"e2e-fallback-crash-{uuid.uuid4()}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # 第一个「进程」：首写两章 + 篇级评审报硬伤（ch1+ch2）→ 并行回退扇出，
        # ch1 分支注入异常崩溃，ch2 分支先完成、其写入作为 pending write 落盘。
        rw_before: list[tuple[str, str]] = []
        events_before: list[tuple[str, dict]] = []
        fake = FakeLLM(
            [
                *FRAMEWORK_RESPONSES,
                SEMANTIC_PASS,
                SEMANTIC_PASS,
                fact_conflict_response,
            ],
            keyed_responses=FRAMEWORK_KEYED_RESPONSES,
        )
        graph = build_graph(
            llm_factory=lambda unit: fake,
            checkpointer=saver,
            search_agent=make_stub_search_agent(),
            rewriter_loop=_crashing_rewriter(rw_before, events_before, "ch1"),
            chapter_reviewer=make_stub_chapter_reviewer(),
        )
        with pytest.raises(RuntimeError, match="故障注入：进程死于 ch1 回退改写"):
            graph.invoke(
                initial_state("写一篇人才培养方案", "专业撰稿人", "trace-fb-crash"),
                config,
            )

        # 死亡现场：并行回退扇出崩溃时，已完成的 ch2 分支写入作为 pending write
        # 被 checkpoint 保留——草稿已是修订版（含修订落实）、revised_chapter_ids
        # 含 ch2；ch1 分支自身写入被丢弃，仍为首写原稿。待执行任务只剩 ch1 分支。
        snapshot = graph.get_state(config)
        assert snapshot.next == ("writing_orchestrator",)
        assert set(snapshot.values["revised_chapter_ids"]) == {"ch2"}
        drafts_before = {d.chapter_id: d.text for d in snapshot.values["chapter_drafts"]}
        assert "修订落实" in drafts_before["ch2"]
        assert "修订落实" not in drafts_before["ch1"]
        # 崩溃前：两章首写各一次 draft；回退扇出中 ch2 revise 完成、ch1 revise 崩溃。
        assert sorted(rw_before) == [
            ("ch1", "draft"),
            ("ch1", "revise"),
            ("ch2", "draft"),
            ("ch2", "revise"),
        ]

        # 第二个「进程」：同 thread_id 恢复，只备二次终审应答（2 章增量语义核查 + 篇级评审）。
        rw_after: list[tuple[str, str]] = []
        events_after: list[tuple[str, dict]] = []
        fake2 = FakeLLM([SEMANTIC_PASS, SEMANTIC_PASS, DOCUMENT_REVIEW_PASS])
        graph2 = build_graph(
            llm_factory=lambda unit: fake2,
            checkpointer=saver,
            search_agent=make_stub_search_agent(),
            rewriter_loop=_crashing_rewriter(rw_after, events_after, None),
            chapter_reviewer=make_stub_chapter_reviewer(),
        )
        resumed = graph2.invoke(None, config)

        # 已完成分支零重复：恢复进程只重跑崩溃的 ch1 回退改写，ch2 零重复。
        # 这是并行扇出 + 超步事务语义成立的可观测证据（串行自环下 ch1 崩溃则
        # ch2 根本未曾执行，无从保留其写入）。
        assert rw_after == [("ch1", "revise")]

        # 事件成对且带业务上下文（ADR-0001 约束 2 与 4）。崩溃进程：ch2 回退改写
        # start/end 成对；被杀的 ch1 回退改写留「有 start 无 end」残链。
        ch1_before = _pair(events_before, "rewriter_loop", "ch1")
        assert ch1_before[-1] == SUBAGENT_START  # ch1 revise 崩溃，无 end
        ch2_before = _pair(events_before, "rewriter_loop", "ch2")
        assert ch2_before[-2:] == [SUBAGENT_START, SUBAGENT_END]  # ch2 revise 成对
        # 续跑：ch1 回退改写成对。
        assert _pair(events_after, "rewriter_loop", "ch1") == [
            SUBAGENT_START,
            SUBAGENT_END,
        ]
        # 事件均带 chapter_id 与 mode 业务上下文。
        for _etype, payload in events_before + events_after:
            assert payload["chapter_id"] in {"ch1", "ch2"}
            assert payload["mode"] in {"draft", "revise"}
        assert resumed["status"] == WorkflowStatus.AWAIT_USER_REVIEW
        assert resumed["citation_report"].passed is True

        # 两章均被回退改写、落实修订说明。
        drafts = {d.chapter_id: d.text for d in resumed["chapter_drafts"]}
        assert "修订落实" in drafts["ch1"]
        assert "修订落实" in drafts["ch2"]

        # 与不中断路径的产物完全等价（同一桩编排保证可逐字段比对）。
        baseline_fake = FakeLLM(
            [
                *FRAMEWORK_RESPONSES,
                SEMANTIC_PASS,
                SEMANTIC_PASS,
                fact_conflict_response,
                SEMANTIC_PASS,
                SEMANTIC_PASS,
                DOCUMENT_REVIEW_PASS,
            ],
            keyed_responses=FRAMEWORK_KEYED_RESPONSES,
        )
        baseline_graph = build_graph(
            llm_factory=lambda unit: baseline_fake,
            checkpointer=InMemorySaver(serde=checkpoint_serializer()),
            search_agent=make_stub_search_agent(),
            rewriter_loop=_crashing_rewriter([], [], None),
            chapter_reviewer=make_stub_chapter_reviewer(),
        )
        baseline_config: RunnableConfig = {
            "configurable": {"thread_id": f"e2e-fallback-base-{uuid.uuid4()}"}
        }
        baseline = baseline_graph.invoke(
            initial_state("写一篇人才培养方案", "专业撰稿人", "trace-fb-base"),
            baseline_config,
        )
        assert resumed["chapter_drafts"] == baseline["chapter_drafts"]
        assert resumed["citation_library"] == baseline["citation_library"]
        assert resumed["citation_report"] == baseline["citation_report"]

        # 恢复后仍可定稿收束。
        result = graph2.invoke(Command(resume=FINALIZE), config)
        assert result["status"] == WorkflowStatus.FINISHED


def test_篇级评审warn不打回_警告呈人工且不触发重写():
    """篇级 warn 三维（此处跨章重复）呈人工不打回（issue #48）。

    篇级评审只报 warn 维度：不触发任何 revise 重写、不消耗重试预算，
    流程直进人工中断点，review_warnings 随载荷呈人工且载荷仍只含元数据。
    """
    from agents.contracts import SubagentAdapter
    from agents.rewriter_loop import stub_rewriter_loop_run

    tasks: list[dict] = []

    async def _recording_run(task: dict) -> dict:
        tasks.append(task)
        return await stub_rewriter_loop_run(task)

    warn_response = json.dumps(
        [
            {
                "dimension": "duplication",
                "chapter_ids": ["ch1", "ch2"],
                "detail": "两章大段重复论述",
            }
        ],
        ensure_ascii=False,
    )
    graph, _, config = _build(
        [*FRAMEWORK_RESPONSES, SEMANTIC_PASS, SEMANTIC_PASS, warn_response],
        rewriter_loop=SubagentAdapter("rewriter_loop", _recording_run),
    )
    result = graph.invoke(initial_state("意图", "身份", "trace-warn"), config)

    # 不打回：只有两次首写，没有任何 revise 调用，重试预算未消耗。
    assert [task["mode"] for task in tasks] == ["draft", "draft"]
    assert result["citation_report"].passed is True
    assert result["citation_retry_count"] == 0
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW

    # warn 随中断载荷呈人工，载荷仍只含元数据（不携带正文全文）。
    payload = result["__interrupt__"][0].value
    warnings = payload["review_warnings"]
    assert len(warnings) == 1
    assert "跨章重复" in warnings[0] and "两章大段重复论述" in warnings[0]
    assert all(
        draft.text not in json.dumps(payload, ensure_ascii=False)
        for draft in result["chapter_drafts"]
    )

    result = graph.invoke(Command(resume=FINALIZE), config)
    assert result["status"] == WorkflowStatus.FINISHED


@pytest.mark.parametrize(
    ("标题一", "标题二", "断言片段"),
    [
        ("一、第一章", "一、第二章", "重复"),
        ("一、第一章", "三、第二章", "断号"),
    ],
)
def test_章节编号重复与断号_图层检出并超限进入中断点(
    标题一: str, 标题二: str, 断言片段: str
):
    """issue #18 图层验收：坏编号从大纲实例化贯穿到终审检出。

    打桩改写器把 spec.title 原样写进正文 ## 标题，改写重试无法修复编号，
    超限后 numbering_broken 警告进入人工中断点，流程不卡死。
    """
    outline_response = json.dumps(
        [
            {"title": 标题一, "subsections": []},
            {"title": 标题二, "subsections": []},
        ],
        ensure_ascii=False,
    )
    framework_responses = [
        FRAMEWORK_RESPONSES[0],
        outline_response,
        FRAMEWORK_RESPONSES[2],
    ]
    # 首轮全量核查 2 章 + 增量重审 ch2 一次，语义核查与篇级评审全部放行，
    # 保证失败仅由编号校验产生。
    graph, _, config = _build(
        [
            *framework_responses,
            SEMANTIC_PASS,
            SEMANTIC_PASS,
            DOCUMENT_REVIEW_PASS,
            SEMANTIC_PASS,
            DOCUMENT_REVIEW_PASS,
        ],
        document_review_max_retries=1,
    )
    result = graph.invoke(initial_state("意图", "身份", "trace-numbering"), config)

    report = result["citation_report"]
    assert report.passed is False
    numbering_issues = [
        issue for issue in report.issues if issue.kind == "numbering_broken"
    ]
    assert numbering_issues
    assert all(issue.chapter_id == "ch2" for issue in numbering_issues)
    assert any(断言片段 in issue.detail for issue in numbering_issues)
    # 超限携编号警告进入人工中断点。
    assert result["status"] == WorkflowStatus.AWAIT_USER_REVIEW
    warnings = result["__interrupt__"][0].value["citation_warnings"]
    assert any("numbering_broken" in warning for warning in warnings)

    # 人工裁决仍可定稿收束，流程永不卡死。
    result = graph.invoke(Command(resume=FINALIZE), config)
    assert result["status"] == WorkflowStatus.FINISHED


def test_LLM调用次数与单元归属():
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    units_seen: list[str] = []

    def factory(unit: str) -> FakeLLM:
        units_seen.append(unit)
        return fake

    graph = build_graph(
        llm_factory=factory,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        search_agent=make_stub_search_agent(),
        rewriter_loop=make_stub_rewriter_loop(),
        chapter_reviewer=make_stub_chapter_reviewer(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "e2e-llm-count"}}
    graph.invoke(initial_state("意图", "身份", "trace-llm"), config)
    result = graph.invoke(Command(resume=FINALIZE), config)

    # 检索与写作由打桩子智能体承担；定稿分支不调 LLM。
    assert units_seen == ["framework_orchestrator", "document_reviewer"]
    assert len(fake.calls) == FIRST_PASS_LLM_CALLS
    # 终态记录的是最后一个节点（human_review_gate）的配置元数据，且不含密钥。
    assert result["current_node_llm_config"]["unit"] == "human_review_gate"
    assert "api_key" not in result["current_node_llm_config"]


@pytest.mark.skipif(
    not _pg_reachable(TEST_PG_DSN), reason="测试 Postgres 不可达"
)
def test_状态经Postgres存档器持久化():
    import psycopg

    thread_id = f"e2e-{uuid.uuid4()}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    with postgres_checkpointer(TEST_PG_DSN) as saver:
        fake = FakeLLM(
            list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
        )
        graph = build_graph(
            llm_factory=lambda unit: fake,
            checkpointer=saver,
            search_agent=make_stub_search_agent(),
            rewriter_loop=make_stub_rewriter_loop(),
        )
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
