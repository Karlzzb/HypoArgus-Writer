"""引文对账纯程序逻辑：提取全文角标与引文库互查，不调 LLM。

三类程序性错误（见 PRD「引用体系」）：
- orphan_marker 无来源的标注：角标 id 在引文库中不存在；
- cross_chapter 跨章误引：角标指向的素材存在但归属其他章节；
- unused_material 未被引用的素材：verdict=pass 的素材未在所属章节正文出现。
"""

import re

from state import ChapterDraft, CitationIssue, Material

# 正文角标形如 [素材id]；id 只含字母数字、下划线、连字符，避免误匹配中文方括号内容。
MARKER_PATTERN = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def reconcile(
    drafts: list[ChapterDraft],
    library: list[Material],
    scope_chapter_ids: set[str] | None = None,
) -> list[CitationIssue]:
    """对账指定范围内的章节草稿与引文库，返回确定性顺序的问题列表。

    scope_chapter_ids 为 None 时全量对账；否则只检查这些章节（增量对账）。
    输出顺序：按章节草稿顺序，章内先角标类问题再未引用素材，均按素材 id 升序。
    """
    materials_by_id = {material.id: material for material in library}
    issues: list[CitationIssue] = []

    for draft in drafts:
        if scope_chapter_ids is not None and draft.chapter_id not in scope_chapter_ids:
            continue
        markers = set(MARKER_PATTERN.findall(draft.text))

        # 角标类问题：无来源的标注与跨章误引。
        for marker in sorted(markers):
            material = materials_by_id.get(marker)
            if material is None:
                issues.append(
                    CitationIssue(
                        kind="orphan_marker",
                        chapter_id=draft.chapter_id,
                        material_id=marker,
                        detail=f"章节 {draft.chapter_id} 正文角标 [{marker}] 在引文库中不存在。",
                    )
                )
            elif material.chapter_id != draft.chapter_id:
                issues.append(
                    CitationIssue(
                        kind="cross_chapter",
                        chapter_id=draft.chapter_id,
                        material_id=marker,
                        detail=(
                            f"章节 {draft.chapter_id} 引用了归属章节 "
                            f"{material.chapter_id} 的素材 {marker}。"
                        ),
                    )
                )

        # 未被引用的素材：只统计 verdict=pass 者（fail 的素材本就不给写作用）。
        unused = [
            material
            for material in library
            if material.chapter_id == draft.chapter_id
            and material.verdict == "pass"
            and material.id not in markers
        ]
        for material in sorted(unused, key=lambda item: item.id):
            issues.append(
                CitationIssue(
                    kind="unused_material",
                    chapter_id=draft.chapter_id,
                    material_id=material.id,
                    detail=(
                        f"素材 {material.id} 通过校验但未在章节 "
                        f"{draft.chapter_id} 正文中被引用。"
                    ),
                )
            )
    return issues
