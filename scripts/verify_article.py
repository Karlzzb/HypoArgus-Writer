#!/usr/bin/env python
"""成品文档五维验证脚本（issue #19 真实 E2E 复跑验收配套）。

对 demo 产出的成品文档（-article 文件）做程序化复查：
按二级标题切章后逐章跑 style_linter 全量规则（字数区间、量化查臆造、
口语黑名单等），并复用 domain 编号校验器检查跨章编号连续唯一。

内容单薄、论证空心化两个维度无法纯程序化判定，仍需人工通读；
本脚本只覆盖可机检的维度并输出逐章字数报告辅助人工复查。

成品口径的章型：成品文档不携带 State 章型事实，缺省经 lint 的标题回落解析，
固定标题章型正常命中；自由观点标题的可重复章（如调研报告维度章）不命中
章型专属规则（含表章豁免）。验收者可用 --chapter-types 按章序显式提供
生成期 State 携带的章型（骨架事实，ADR-0005），使章型规则与生成期同口径。

用法：uv run python scripts/verify_article.py var/demo_archive/<thread_id>-article.md \
    --doc-type 调研报告 \
    --chapter-types "监测概述与数据说明,维度章,维度章,维度章,主要发现与问题诊断,结论与对策建议"
（章型项数须与成品实切章数一致，不含参考文献章。）
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents.rewriter_loop.style_linter import (  # noqa: E402
    _LintContext,
    _quantitative_violations,
    check_word_count,
    count_prose_words,
    lint,
    load_config,
    normalize_cjk_ws,
    resolve_chapter_type,
)
from domain.chapter_numbering_validator import (  # noqa: E402
    validate_chapter_numbering,
)
from domain.doc_types import tier_from_variant  # noqa: E402
from domain.state import ChapterDraft  # noqa: E402

# 与 domain 校验器一致的二级标题行（非三级），用于整篇切章。
_H2_LINE = re.compile(r"^##\s+(?!#)")

# 参考文献属书目渲染产物，不按正文章节口径校验。
_SKIP_TITLES = ("参考文献",)


def split_chapters(text: str) -> list[tuple[str, str]]:
    """按二级标题把整篇成品切成（标题, 章正文）列表；忽略围栏代码块内的 ##。"""
    chapters: list[tuple[str, list[str]]] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        if not in_fence and _H2_LINE.match(stripped):
            chapters.append((stripped.removeprefix("##").strip(), [line]))
            continue
        if chapters:
            chapters[-1][1].append(line)
    return [(title, "\n".join(lines)) for title, lines in chapters]


def quantitative_violations_strict(
    body: str,
    cfg: dict,
    doc_variant: str | None = None,
    chapter_type: str | None = None,
) -> list:
    """成品口径的量化断言复查：无同句角标即违规。

    成品文档已脱离生成期 references 上下文，lint() 会整组跳过查臆造；
    这里直接复用内部量化规则（references 置空 = 无任何依据数值兜底），
    即成品中所有量化断言必须携带同句引文角标。属验收脚本对内部规则的
    有意复用，规则本体演进时以 style_linter 为准。
    ``doc_variant`` 与 lint() 同口径经 ``tier_from_variant`` 解析，
    保证变体分键规则在成品复查与生成期取同一键；``chapter_type``
    显式指定章型（--chapter-types 提供的 State 事实），未指定回落标题解析。
    """
    normalized = normalize_cjk_ws(body)
    ctx = _LintContext(
        text=normalized,
        cfg=cfg,
        variant=tier_from_variant(doc_variant),
        template=resolve_chapter_type(normalized, cfg, chapter_type),
        domain=None,
        references=[],
        materials=None,
        hypotheses=None,
    )
    return _quantitative_violations(ctx, cfg.get("fabrication") or {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("article", type=Path, help="成品文档路径（-article.md）")
    parser.add_argument("--doc-type", default="人才培养方案", help="文种，与生成时一致")
    parser.add_argument(
        "--doc-variant",
        default=None,
        help="文种变体（人培方案的 本科/高职），与生成时一致；缺省无变体，人培经兑底取本科",
    )
    parser.add_argument(
        "--chapter-types",
        default=None,
        help=(
            "按章序逗号分隔的章型列表（生成期 State 携带的骨架事实，"
            "不含参考文献章）；缺省经标题回落解析"
        ),
    )
    args = parser.parse_args()

    text = args.article.read_text(encoding="utf-8")
    chapters = [
        (title, body)
        for title, body in split_chapters(text)
        if not any(title.startswith(skip) for skip in _SKIP_TITLES)
    ]
    if not chapters:
        print("未切出任何章节，请确认成品文档格式。")
        return 1

    chapter_types: list[str | None] = [None] * len(chapters)
    if args.chapter_types:
        provided = [
            item.strip() for item in args.chapter_types.replace("，", ",").split(",")
        ]
        if len(provided) != len(chapters):
            print(
                f"--chapter-types 提供 {len(provided)} 项，"
                f"与切出的 {len(chapters)} 章不符。"
            )
            return 1
        chapter_types = [item or None for item in provided]

    cfg = load_config(args.doc_type)
    total_violations = 0

    print(f"共 {len(chapters)} 章（不含参考文献）\n")
    print("== 逐章字数与 lint ==")
    for (title, body), chapter_type in zip(chapters, chapter_types):
        words = count_prose_words(body)
        wc_violations = check_word_count(body, cfg, template_title=chapter_type)
        lint_violations = lint(
            body, args.doc_type, args.doc_variant, chapter_type=chapter_type
        )
        lint_violations.extend(
            quantitative_violations_strict(
                body, cfg, args.doc_variant, chapter_type
            )
        )
        total_violations += len(lint_violations)
        status = "通过" if not lint_violations else f"{len(lint_violations)} 条违规"
        print(f"- {title}：{words} 字，{status}")
        for violation in lint_violations:
            print(f"    [{violation.rule}] {violation.message}")
        # check_word_count 已含于 lint；单列仅为标注维度归属。
        assert all(v in lint_violations for v in wc_violations)

    print("\n== 跨章编号连续唯一 ==")
    drafts = [
        ChapterDraft(chapter_id=f"ch{i}", text=body, summary="")
        for i, (_, body) in enumerate(chapters, start=1)
    ]
    numbering_issues = validate_chapter_numbering(drafts, outline=[])
    if numbering_issues:
        total_violations += len(numbering_issues)
        for issue in numbering_issues:
            print(f"- [{issue.chapter_id}] {issue.message}")
    else:
        print("- 通过：正文编号从「一」起连续递增且不重复")

    print(f"\n合计违规：{total_violations}")
    return 0 if total_violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
