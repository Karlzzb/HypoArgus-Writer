"""契约映射：SearchTask/SearchResult 与检索引擎公开出入参的相互转换。

薄适配层的唯一职责（issue #35）：
把整章假说列表映射为引擎的正向检索项（假说本文）与反向检索项
（refute_condition 驱动，关系固定为 oppose），
把引擎的逐项证据裁决与引文记录映射为逐条回链假说 ID 的素材条目
（回填 url 与 source_kind，裁决折算为 pass/fail）。
全部为纯函数且只操作 dict，不导入引擎实现，离线测试无需引擎依赖。
"""

from typing import Any, Literal

from agents.contracts import MaterialPayload, SearchResult, SourceKind

ENGINE_DOCUMENT_ID = "hypoargus-writer"
"""引擎入参的文档标识：本项目按章一次调用，文档粒度无业务语义，取固定值。"""

_ITEM_ID_SEPARATOR = "::"
"""检索项 id 的编码分隔符：``<假说id>::<forward|reverse>``，出参按此回链假说。"""

_SOURCE_KIND_BY_TYPE: dict[str, SourceKind] = {
    "WEB": "web",
    "KNOWLEDGE_BASE": "knowledge_base",
    "STRUCTURED_DATA": "structured_data",
}
"""引擎引文来源类型 → 契约三通道标识：两侧恰好三值一一对应。"""


def forward_item_id(hypothesis_id: str) -> str:
    """假说的正向检索项 id。"""
    return f"{hypothesis_id}{_ITEM_ID_SEPARATOR}forward"


def reverse_item_id(hypothesis_id: str) -> str:
    """假说的反向检索项 id。"""
    return f"{hypothesis_id}{_ITEM_ID_SEPARATOR}reverse"


def split_item_id(item_id: str) -> tuple[str, str] | None:
    """把检索项 id 解回（假说 id, 线别）；不合编码约定返回 None。"""
    head, separator, line = item_id.rpartition(_ITEM_ID_SEPARATOR)
    if not separator or not head or line not in ("forward", "reverse"):
        return None
    return head, line


def engine_payload_from_task(task: dict[str, Any]) -> dict[str, Any]:
    """SearchTask → 引擎公开入参（search-agent-input/v1 的 dict 形态）。

    每条假说产生一个正向检索项（claim，目标文本为假说本文）；
    refute_condition 非空白的假说另产生一个反向检索项（oppose，
    目标文本为反驳条件），落实"可证伪"的产品设计。
    既有引文库摘要作为每个正向项的既有证据文本，供引擎规避重复素材；
    品类进论证边界字段，作为检索范围提示。
    """
    chapter_id = task["chapter_id"]
    digest = task["existing_materials_digest"].strip()
    forward_items: list[dict[str, Any]] = []
    reverse_items: list[dict[str, Any]] = []
    for hypothesis in task["hypotheses"]:
        forward_items.append(
            {
                "item_id": forward_item_id(hypothesis["id"]),
                "item_type": "claim",
                "target_text": hypothesis["text"],
                "existing_evidence_text": digest or None,
            }
        )
        refute_condition = hypothesis["refute_condition"].strip()
        if refute_condition:
            reverse_items.append(
                {
                    "item_id": reverse_item_id(hypothesis["id"]),
                    "target_text": refute_condition,
                    "relation_to_original": "oppose",
                }
            )

    paragraph: dict[str, Any] = {
        "paragraph_id": chapter_id,
        "paragraph_text": "\n".join(
            hypothesis["text"] for hypothesis in task["hypotheses"]
        ),
        "forward_items": forward_items,
        "reverse_items": reverse_items,
    }
    genre = task["genre"].strip()
    if genre:
        paragraph["argument_context"] = {"boundary": genre}
    return {
        "request_id": f"chapter-{chapter_id}",
        "document_id": ENGINE_DOCUMENT_ID,
        "paragraph": paragraph,
    }


def search_result_from_engine_output(
    output: dict[str, Any], task: dict[str, Any]
) -> SearchResult:
    """引擎公开出参（search-agent-output/v1 的 dict 形态）→ SearchResult。

    契约 verdict 的语义是"该素材可否作为支撑假说的通过证据"：
    正向线裁决中被列为支撑引文的素材为 pass，其余（补充、反例）为 fail；
    反向线（反驳条件检索）命中的素材一律 fail 入库——它们是对假说的
    削弱证据，供后续环节筛选与审计，不得进入写作素材池。
    同一（假说, 引文）在正反两线都出现时取 pass 优先（正向支撑事实成立）。
    回链不上任务包假说的裁决项（编码不合约定或假说未知）按脏数据丢弃。
    """
    chapter_id = task["chapter_id"]
    hypothesis_ids = {hypothesis["id"] for hypothesis in task["hypotheses"]}
    citations = {
        citation["citation_id"]: citation for citation in output.get("citations", [])
    }
    picked: dict[tuple[str, str], MaterialPayload] = {}
    for decision in output.get("results", []):
        linked = split_item_id(decision["item_id"])
        if linked is None or linked[0] not in hypothesis_ids:
            continue
        hypothesis_id, line = linked
        supporting = set(decision.get("supporting_citation_ids", []))
        for citation_id in decision.get("citation_ids", []):
            citation = citations.get(citation_id)
            if citation is None:
                continue
            source_kind = _SOURCE_KIND_BY_TYPE.get(citation.get("source_type", ""))
            if source_kind is None:
                continue
            verdict: Literal["pass", "fail"] = (
                "pass" if line == "forward" and citation_id in supporting else "fail"
            )
            key = (hypothesis_id, citation_id)
            existing = picked.get(key)
            if existing is not None and (
                existing["verdict"] == "pass" or verdict == "fail"
            ):
                continue
            picked[key] = MaterialPayload(
                id=f"m-{chapter_id}-{hypothesis_id}-{citation_id}",
                hypothesis_id=hypothesis_id,
                source=citation.get("source_name")
                or citation.get("title")
                or "未知来源",
                url=citation.get("url"),
                source_kind=source_kind,
                excerpt=citation["summary"],
                relevance_score=float(citation["judgment"]["confidence"]),
                verdict=verdict,
            )
    return SearchResult(materials=list(picked.values()))
