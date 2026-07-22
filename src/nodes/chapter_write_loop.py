"""章级「写→评→重写」循环：首写路径的共享编排（ADR-0006 T3）。

首写不再是单次 rewriter 调用：write(draft) 后经 chapter_reviewer 章级评审，
评审不过则按分区式修订说明 rewrite(revise) 一次，如此至多 ``max_rewrites`` 轮。
**评审在顶、无终态复审**——最后一次 rewrite 之后不再评审（章级不做终态复审，
交由全局 citation_validator 兜底）。缺省 ``max_rewrites=1``：

    write(draft)                       # 1 次写作调用
    if 空稿: 短路返回（不评审、不重写）
    attempts = 0
    while attempts < max_rewrites:     # 缺省 1
        review                          # chapter_reviewer
        if note.passed: break           # 干净初稿：write+review = 2 次调用
        rewrite(revise, revision_note)  # rewriter mode=revise
        attempts += 1                   # 脏初稿：write+review+rewrite = 3 次调用
    # 循环退出后不再评审

发出的 ChapterDraft 的 ``self_check`` 来源分三种（对应上面三条退出路径）：
- 空稿短路：rewriter 的退化自检（citations_ok=False + 退化说明）——首写与重写
  退化为空稿同口径（重写空稿不 re-lint，避免空文本零违规被误判通过）。
- 评审通过：评审的 self_check（评审对该文本已做确定性 lint + 四维自审）。
- 重写后退出：对**重写后正文**跑纯函数 re-lint（零 LLM、非复审）折出的修后终态
  自检（ADR-0004「修后终态」口径）——唯一那次评审跑在重写前的文本上，故以确定性
  re-lint 给出重写后正文的引用类终态，避免带着评审前的陈旧失败自检误伤人工门禁。

循环整体在单个编排超步内同步跑完（chapter_drafter 的一个 Send 分支、或
writing_orchestrator 的一个超步），不新增图内边、不越非子图边界（ADR-0001 约束 3）；
崩溃重跑只损失进行中的该章/分支（约束 1）。rewriter 与 reviewer 各自经
SubagentAdapter 发成对启停事件并携 chapter_id / mode（约束 2）。
"""

from __future__ import annotations

import os

from agents.contracts import (
    ChapterSpecPayload,
    MaterialPayload,
    ReviewTask,
    RewriteTask,
    SelfCheckPayload,
    Subagent,
)
from agents.rewriter_loop.style_linter import CITATION_RULES, lint
from domain.env_config import read_nonnegative_int
from domain.state import ChapterDraft, SelfCheck

MAX_REWRITES_ENV = "CHAPTER_MAX_REWRITES"
"""章级重写次数上限的环境变量名。"""

DEFAULT_MAX_REWRITES = 1
"""章级重写次数上限缺省值：缺省 write+review+至多一次 rewrite（干净初稿短路为 2 次调用）。"""


def resolve_max_rewrites(max_rewrites: int | None) -> int:
    """解析章级重写次数上限：显式传入优先，否则读环境变量（缺省 1，允许 0 关闭重写）。"""
    if max_rewrites is not None:
        return max_rewrites
    return read_nonnegative_int(os.environ, MAX_REWRITES_ENV, DEFAULT_MAX_REWRITES)


def _relint_self_check(
    text: str,
    doc_type: str,
    doc_variant: str | None,
    chapter_spec: ChapterSpecPayload,
    materials: list[MaterialPayload],
) -> SelfCheckPayload:
    """对重写后正文跑纯函数 re-lint，折出修后终态自检（零 LLM，非复审）。

    引用类规则与 rewriter / reviewer 同源 ``CITATION_RULES``：终态正文仍存
    引用类违规则 ``citations_ok=False``，交全局终审裁决。
    """
    violations = lint(
        text,
        doc_type,
        doc_variant,
        chapter_type=chapter_spec.get("chapter_type"),
        materials=materials or None,
        hypotheses=chapter_spec["hypotheses"],
    )
    issues = [f"[{v.rule}] {v.message}" for v in violations]
    citations_ok = not any(v.rule in CITATION_RULES for v in violations)
    return SelfCheckPayload(citations_ok=citations_ok, issues=issues)


async def run_chapter_write_loop(
    *,
    rewriter_loop: Subagent,
    chapter_reviewer: Subagent,
    max_rewrites: int,
    doc_type: str,
    doc_variant: str | None,
    chapter_spec: ChapterSpecPayload,
    materials: list[MaterialPayload],
    prev_chapter_summary: str,
) -> ChapterDraft:
    """跑一章的写→评→重写循环，返回该章成稿 ChapterDraft（见模块文档的三条退出路径）。"""
    chapter_id = chapter_spec["id"]
    draft_task = RewriteTask(
        mode="draft",
        doc_type=doc_type,
        doc_variant=doc_variant,
        chapter_spec=chapter_spec,
        materials=materials,
        prev_chapter_summary=prev_chapter_summary,
    )
    result = await rewriter_loop.run(dict(draft_task))
    text = result["chapter_text"]
    summary = result["chapter_summary"]
    self_check: SelfCheckPayload = result["self_check"]

    if text.strip():
        # 非空稿才进评审循环；空稿诚实短路（沿用 rewriter 的退化自检，不评审、不重写）。
        attempts = 0
        while attempts < max_rewrites:
            review_task = ReviewTask(
                mode="review",
                doc_type=doc_type,
                doc_variant=doc_variant,
                chapter_spec=chapter_spec,
                chapter_text=text,
                materials=materials,
                prev_chapter_summary=prev_chapter_summary,
            )
            review = await chapter_reviewer.run(dict(review_task))
            note = review["revision_note"]
            self_check = review["self_check"]
            if note["passed"]:
                break
            revise_task = RewriteTask(
                mode="revise",
                doc_type=doc_type,
                doc_variant=doc_variant,
                chapter_spec=chapter_spec,
                materials=materials,
                prev_chapter_summary=prev_chapter_summary,
                revision_note=note,
                current_text=text,
            )
            revised = await rewriter_loop.run(dict(revise_task))
            text = revised["chapter_text"]
            summary = revised["chapter_summary"]
            attempts += 1
            if not text.strip():
                # 重写退化为空稿：沿用 rewriter 的退化自检诚实短路——空文本 re-lint
                # 会零违规「洗白」成 citations_ok=True，绕过全局终审兜底。
                self_check = revised["self_check"]
                break
            # 无终态复审：以确定性 re-lint 给出重写后正文的修后终态自检（ADR-0004）。
            self_check = _relint_self_check(
                text, doc_type, doc_variant, chapter_spec, materials
            )

    return ChapterDraft(
        chapter_id=chapter_id,
        text=text,
        summary=summary,
        self_check=SelfCheck(
            citations_ok=self_check["citations_ok"],
            issues=self_check["issues"],
        ),
    )
