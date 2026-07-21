"""检查点序列化器测试：domain.state 类型显式注册进 msgpack 允许清单。

背景（issue #14）：LangGraph 对未注册类型的反序列化会告警并将在未来版本
阻断。graph.checkpoint_serializer 必须把 domain.state 全部状态模型注册进
allowed_msgpack_modules，保证严格模式（LANGGRAPH_STRICT_MSGPACK=true）下
检查点往返依然成立。
"""

import os
import subprocess
import sys
from pathlib import Path

from graph import CHECKPOINT_MSGPACK_TYPES, checkpoint_serializer
from domain.state import (
    ArgumentPoint,
    ChapterDraft,
    ChapterSpec,
    CitationIssue,
    CitationReport,
    Hypothesis,
    Material,
    RevisionDirective,
    RevisionRound,
    SelfCheck,
    WorkflowStatus,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sample_objects() -> list[object]:
    """覆盖 domain.state 全部会进入检查点的模型实例。"""
    hypothesis = Hypothesis(
        id="ch1-p1-h1", text="假说", refute_condition="证伪条件", angle="预言"
    )
    point = ArgumentPoint(id="ch1-p1", text="论点", hypotheses=[hypothesis])
    return [
        ChapterSpec(id="ch1", title="标题", subsections=["子标题"], points=[point]),
        Material(
            id="m1",
            hypothesis_id="ch1-p1-h1",
            chapter_id="ch1",
            source="来源",
            url=None,
            source_kind="knowledge_base",
            excerpt="摘录",
            relevance_score=0.9,
            verdict="pass",
        ),
        ChapterDraft(
            chapter_id="ch1",
            text="正文",
            summary="摘要",
            self_check=SelfCheck(citations_ok=False, issues=["问题"]),
        ),
        RevisionRound(
            round_no=1,
            raw_feedback="意见",
            directives=[
                RevisionDirective(
                    target_chapter_id="ch1",
                    type="rewrite_only",
                    instruction="改写",
                )
            ],
        ),
        CitationReport(
            passed=False,
            issues=[
                CitationIssue(
                    kind="orphan_marker",
                    chapter_id="ch1",
                    material_id="m1",
                    detail="明细",
                )
            ],
            failed_chapter_ids=["ch1"],
        ),
        WorkflowStatus.ARTICLE_WRITING,
    ]


def test_注册清单覆盖domain_state全部模型类型() -> None:
    """新增状态模型时清单自动纳入，不依赖手工维护。"""
    registered = set(CHECKPOINT_MSGPACK_TYPES)
    for expected in (
        Hypothesis,
        ArgumentPoint,
        ChapterSpec,
        Material,
        SelfCheck,
        ChapterDraft,
        RevisionDirective,
        RevisionRound,
        CitationIssue,
        CitationReport,
        WorkflowStatus,
    ):
        assert expected in registered, f"{expected.__name__} 未注册进检查点允许清单"


def test_注册后的往返不再触发未注册类型告警(caplog) -> None:
    from langgraph.checkpoint.serde import jsonplus

    # 该告警按进程去重：先清空去重集合，避免其他用例已触发过
    # 同类型告警导致本断言空洞通过。
    jsonplus._warned_unregistered_types.clear()
    serde = checkpoint_serializer()
    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        for obj in _sample_objects():
            assert serde.loads_typed(serde.dumps_typed(obj)) == obj
    assert "Deserializing unregistered type" not in caplog.text


def test_严格模式下注册类型往返成立() -> None:
    """LANGGRAPH_STRICT_MSGPACK 在导入期读取，须以子进程加环境变量做真实验证。"""
    script = (
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "from graph import checkpoint_serializer\n"
        "from domain.state import ChapterSpec, WorkflowStatus\n"
        "serde = checkpoint_serializer()\n"
        "spec = ChapterSpec(id='ch1', title='标题')\n"
        "assert serde.loads_typed(serde.dumps_typed(spec)) == spec\n"
        "status = WorkflowStatus.FINISHED\n"
        "assert serde.loads_typed(serde.dumps_typed(status)) == status\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
