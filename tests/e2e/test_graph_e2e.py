"""端到端闭环测试：注入假 LLM 跑整张图，验证状态机流转、人工中断点与迭代闭环。

framework_orchestrator 预置最小 JSON 应答序列（自由结构、2 章、每章 1 论点
1 假说）；reference_orchestrator 与 writing_orchestrator 缺省走打桩子智能体、
不调 LLM（中断恢复用例例外：走 rewriter_loop 真实现链路，写作与自审经
键控应答分派）；citation_validator 每个受审章节消费一条语义核查 JSON 应答；
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

from agents.rewriter_loop import make_stub_rewriter_loop
from domain.citation_reconciler import MARKER_PATTERN
from domain.units import MAIN_NODES
from graph import build_graph, postgres_checkpointer
from llm.llm_client import FakeLLM
from llm.llm_config import RUNTIME_UNITS
from domain.state import WorkflowStatus, initial_state
from tests.llm_response_plans import (
    AUDIT_EMPTY_RESPONSE,
    FIRST_PASS_LLM_CALLS,
    FIRST_PASS_RESPONSES,
    FRAMEWORK_KEYED_RESPONSES,
    FRAMEWORK_LLM_CALLS,
    FRAMEWORK_RESPONSES,
    SEMANTIC_PASS,
    WRITER_KEYED_RESPONSES,
    joined_prompt,
)

TEST_PG_DSN = os.environ.get(
    "HYPOARGUS_TEST_PG_DSN",
    "postgresql://postgres:postgres@127.0.0.1:15432/postgres",
)

FINALIZE = {"action": "finalize"}


def _build(responses: list[str], **kwargs):
    """带 InMemorySaver 与共享假 LLM 构图，返回（graph, fake, config）。

    本文件验收的是图编排（状态机、路由、中断闭环），写作单元默认注入
    打桩改写器（真实现契约在 tests/agents/rewriter_loop/ 单独覆盖）；
    需要记录器等定制时经 kwargs 显式覆盖。
    """
    fake = FakeLLM(list(responses), keyed_responses=FRAMEWORK_KEYED_RESPONSES)
    kwargs.setdefault("rewriter_loop", make_stub_rewriter_loop())
    graph = build_graph(
        llm_factory=lambda unit: fake, checkpointer=InMemorySaver(), **kwargs
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
    # writing_orchestrator 图内自环：每超步只写一章，2 章即逐章进入两次，
    # 章级产物按超步落 checkpoint 正是本切片的验收点。
    assert observed == [
        ("framework_orchestrator", WorkflowStatus.FRAMEWORK_BUILDING),
        ("reference_orchestrator", WorkflowStatus.REFERENCE_FETCHING),
        ("writing_orchestrator", WorkflowStatus.ARTICLE_WRITING),
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


def _writer_keyed(chapter_titles: list[str]) -> dict[str, list[str]]:
    """按剩余章节切片真实现写作/自审的键控应答（每章一次首写 + 一次自审）。

    自审键置于首位：自审提示词内嵌全文正文，须先于「- 标题：」键匹配，
    与 WRITER_KEYED_RESPONSES 的键序约定一致。
    """
    keyed: dict[str, list[str]] = {
        "【引用自审】": [AUDIT_EMPTY_RESPONSE] * len(chapter_titles)
    }
    for title in chapter_titles:
        key = f"- 标题：{title}"
        keyed[key] = WRITER_KEYED_RESPONSES[key][:1]
    return keyed


def _writer_call_counts(fake: FakeLLM) -> dict[str, int]:
    """按章节标题统计写作模型调用次数（不含自审），供零重复断言。"""
    counts = {"第一章": 0, "第二章": 0}
    for messages in fake.calls:
        text = joined_prompt(messages)
        if "【引用自审】" in text:
            continue
        for title in counts:
            if f"- 标题：{title}" in text:
                counts[title] += 1
    return counts


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
    """章级 checkpoint 验收（ADR-0001 约束 1 与 4，真实现链路版）。

    链路口径：真编排（make_rewriter_loop → make_writer_run）+ 真 style linter +
    真 JSON 解析路径（LlmWriterClient），仅最底层模型调用用 FakeLLM 替身——
    仅在桩上通过的验收不算通过（约束 4）。
    故障注入（issue #8 Testing Decisions）：包一层在第二章首写调用上抛致命
    异常的 LLM，图调用在写第二章的超步内崩溃并向外抛出，模拟进程死于该章
    执行中途；首章超步的 checkpoint 已落盘，同 thread_id 二次驱动恢复。
    断言：已完成章节零重复模型调用（逐章统计写作与自审调用）、
    subagent_start/end 成对带 chapter_id 与 mode 且 progress 事件全部落在
    成对区间内并携同一业务上下文、最终产物与不中断路径完全等价。
    """
    from contextlib import contextmanager

    from agents.rewriter_loop import make_rewriter_loop
    from domain.events import SUBAGENT_END, SUBAGENT_PROGRESS, SUBAGENT_START

    class _CrashOnSecondChapterWrite:
        """故障注入 LLM：第二章首写调用抛致命异常，其余调用透传底层 FakeLLM。

        写作缝（LlmWriterClient）的退化重试会把每轮异常都打到这里，
        从未拿到信封则重抛最后一个异常——``crash_calls`` 记录实际重试轮次。
        """

        def __init__(self, inner: FakeLLM) -> None:
            self._inner = inner
            self.crash_calls = 0

        @property
        def metadata(self) -> dict[str, str]:
            return self._inner.metadata

        def invoke(self, messages: list[dict[str, str]]) -> str:
            text = joined_prompt(messages)
            if "- 标题：第二章" in text and "【引用自审】" not in text:
                self.crash_calls += 1
                raise RuntimeError("故障注入：进程死于第二章首写")
            return self._inner.invoke(messages)

    @contextmanager
    def _checkpoint_backend(kind: str):
        if kind == "postgres":
            with postgres_checkpointer(TEST_PG_DSN) as saver:
                yield saver
        else:
            yield InMemorySaver()

    def _recorder(events: list[tuple[str, dict]]):
        def _hook(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        return _hook

    def _assert_chapter_events(
        events: list[tuple[str, dict]], chapter_id: str
    ) -> None:
        """单章事件链：start/end 成对带业务上下文，progress 夹在区间内且上下文一致。

        钩子层的「父子链」即此序：progress 全部发生在本章 start 之后、end 之前，
        并携同一 unit/chapter_id/mode（信封侧 parent_id 挂接在发射器测试中覆盖）。
        """
        assert events, f"章节 {chapter_id} 未记录到任何子智能体事件"
        types = [etype for etype, _ in events]
        assert types[0] == SUBAGENT_START and types[-1] == SUBAGENT_END
        assert all(etype == SUBAGENT_PROGRESS for etype in types[1:-1])
        for _, payload in events:
            assert payload["unit"] == "rewriter_loop"
            assert payload["chapter_id"] == chapter_id
            assert payload["mode"] == "draft"
        # 真实现首写链路的关键步骤序：写作调用对 → lint → 自审调用对 → 自审收束。
        assert [payload["step"] for etype, payload in events[1:-1]] == [
            "llm_call_start",
            "llm_call_end",
            "lint_done",
            "llm_call_start",
            "llm_call_end",
            "audit_done",
        ]

    with _checkpoint_backend(backend) as saver:
        thread_id = f"e2e-crash-{uuid.uuid4()}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # 第一个「进程」：故障注入——第二章首写调用抛致命异常，图调用在写
        # 第二章的超步内崩溃并向外抛出；首章超步的 checkpoint 已落盘，等价于
        # 「某章完成、下一章未完成」时进程被 kill。只备首章的写作/自审键控应答：
        # 第二章的调用在进入底层 FakeLLM 前即被注入异常拦截。
        events_before: list[tuple[str, dict]] = []
        fake = FakeLLM(
            list(FIRST_PASS_RESPONSES),
            keyed_responses={
                **FRAMEWORK_KEYED_RESPONSES,
                **_writer_keyed(["第一章"]),
            },
        )
        crashing = _CrashOnSecondChapterWrite(fake)
        graph = build_graph(
            llm_factory=lambda unit: fake,
            checkpointer=saver,
            rewriter_loop=make_rewriter_loop(
                lambda unit: crashing, _recorder(events_before)
            ),
        )
        with pytest.raises(RuntimeError, match="故障注入：进程死于第二章首写"):
            graph.invoke(
                initial_state("写一篇人才培养方案", "专业撰稿人", "trace-crash"),
                config,
            )
        # 写作缝退化重试 3 轮全部命中注入异常、从未拿到信封后重抛。
        assert crashing.crash_calls == 3
        # 死亡现场：首章已入档，待执行节点仍是 writing_orchestrator（写第二章）；
        # 第二章超步崩溃，其写入被整体丢弃，不留半成品。
        snapshot = graph.get_state(config)
        assert snapshot.next == ("writing_orchestrator",)
        assert [d.chapter_id for d in snapshot.values["chapter_drafts"]] == ["ch1"]
        # 底层 FakeLLM 只见过首章的一次写作与一次自审（框架阶段调用另计）：
        # 第二章的三轮重试均在进入底层模型前被拦截、未产生任何应答，
        # 故调用计数与「进程死于两章之间」完全等价。
        assert _writer_call_counts(fake) == {"第一章": 1, "第二章": 0}
        assert len(fake.calls) == FRAMEWORK_LLM_CALLS + 2

        # 第二个「进程」：同 thread_id 恢复，只备剩余章节的写作/自审应答
        # （键控切片到未完成章）与 2 章语义核查的顺序应答。
        events_after: list[tuple[str, dict]] = []
        fake2 = FakeLLM(
            [SEMANTIC_PASS, SEMANTIC_PASS],
            keyed_responses=_writer_keyed(["第二章"]),
        )
        graph2 = build_graph(
            llm_factory=lambda unit: fake2,
            checkpointer=saver,
            rewriter_loop=make_rewriter_loop(
                lambda unit: fake2, _recorder(events_after)
            ),
        )
        resumed = graph2.invoke(None, config)

        # 已完成章节零重复模型调用：恢复进程只写了第二章，且总调用数恰为
        # 第二章写作 1 + 自审 1 + 语义核查 2——首章的写作与自审花费为零。
        assert _writer_call_counts(fake2) == {"第一章": 0, "第二章": 1}
        assert len(fake2.calls) == 4
        # 恢复进程的自审只审第二章素材池，未触碰首章素材。
        audit_prompts = [
            joined_prompt(messages)
            for messages in fake2.calls
            if "【引用自审】" in joined_prompt(messages)
        ]
        assert len(audit_prompts) == 1
        assert "m-ch2-p1-h1" in audit_prompts[0]
        assert "m-ch1-p1-h1" not in audit_prompts[0]
        # 摘要链跨崩溃存续：恢复进程写第二章的提示词携带 checkpoint 里
        # 首章的摘要（而非重写首章得来）。
        (ch2_write_prompt,) = [
            joined_prompt(messages)
            for messages in fake2.calls
            if "- 标题：第二章" in joined_prompt(messages)
            and "【引用自审】" not in joined_prompt(messages)
        ]
        assert "第一章完成培养定位与背景铺陈。" in ch2_write_prompt

        # 事件成对且带业务上下文、progress 父子链正确（ADR-0001 约束 2 与 4）。
        # 崩溃进程的事件流按章拆分断言：首章链完整成对；被杀的第二章尝试
        # 留下「有 start 无 end」的残链——SubagentAdapter 在 run 之前发
        # subagent_start、编排在写作调用前发 llm_call_start，异常沿缝上抛后
        # llm_call_end 与 subagent_end 均不再发出，事件流如实反映死亡现场。
        ch1_before = [(t, p) for t, p in events_before if p["chapter_id"] == "ch1"]
        ch2_before = [(t, p) for t, p in events_before if p["chapter_id"] == "ch2"]
        _assert_chapter_events(ch1_before, "ch1")
        assert [t for t, _ in ch2_before] == [SUBAGENT_START, SUBAGENT_PROGRESS]
        assert all(p["mode"] == "draft" for _, p in ch2_before)
        assert ch2_before[1][1]["step"] == "llm_call_start"
        _assert_chapter_events(events_after, "ch2")
        assert resumed["status"] == WorkflowStatus.AWAIT_USER_REVIEW
        _assert_full_draft(resumed)

        # 与不中断路径的产物完全等价（同一键控应答计划保证真链路可逐字段比对）。
        baseline_fake = FakeLLM(
            list(FIRST_PASS_RESPONSES),
            keyed_responses={
                **FRAMEWORK_KEYED_RESPONSES,
                **_writer_keyed(["第一章", "第二章"]),
            },
        )
        baseline_graph = build_graph(
            llm_factory=lambda unit: baseline_fake, checkpointer=InMemorySaver()
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
    fake = FakeLLM(
        list(FIRST_PASS_RESPONSES), keyed_responses=FRAMEWORK_KEYED_RESPONSES
    )
    units_seen: list[str] = []

    def factory(unit: str) -> FakeLLM:
        units_seen.append(unit)
        return fake

    graph = build_graph(
        llm_factory=factory,
        checkpointer=InMemorySaver(),
        rewriter_loop=make_stub_rewriter_loop(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "e2e-llm-count"}}
    graph.invoke(initial_state("意图", "身份", "trace-llm"), config)
    result = graph.invoke(Command(resume=FINALIZE), config)

    # 检索与写作由打桩子智能体承担；定稿分支不调 LLM。
    assert units_seen == ["framework_orchestrator", "citation_validator"]
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
