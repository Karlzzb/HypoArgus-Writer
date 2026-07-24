"""契约映射：SearchTask/SearchResult 与检索引擎公开出入参的相互转换。

薄适配层的唯一职责（issue #35）：
把整章假说列表映射为引擎的正向检索项（假说本文）与反向检索项
（refute_condition 驱动，关系固定为 oppose），
把引擎的逐项证据裁决与引文记录映射为逐条回链假说 ID 的素材条目
（回填 source_ref、url 与 source_kind，裁决折算为 pass/fail）。
全部为纯函数且只操作 dict，不导入引擎实现，离线测试无需引擎依赖。
"""

import hashlib
import json
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

_CROCKFORD_BASE32_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
"""不透明 Material ID 使用 Crockford Base32，避开 I/L/O/U 易混字符。"""


def _compact_source_ref(value: dict[str, Any]) -> dict[str, Any]:
    """递归剔除 None / 空容器，保证同一语义输入有唯一 JSON 表示。"""
    compacted: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        if isinstance(item, dict):
            nested = _compact_source_ref(item)
            if nested:
                compacted[key] = nested
        elif item not in (None, "", [], {}):
            compacted[key] = item
    return compacted


def _crockford_base32_130_bits(payload: bytes) -> str:
    """把 SHA-256 摘要的高 130 bit 编为固定 26 位 Crockford Base32。"""
    value = int.from_bytes(hashlib.sha256(payload).digest(), "big") >> (256 - 130)
    chars: list[str] = []
    for shift in range(125, -1, -5):
        chars.append(_CROCKFORD_BASE32_ALPHABET[(value >> shift) & 0b11111])
    return "".join(chars)


def material_id_from_source_ref(source_kind: SourceKind, source_ref: dict[str, Any]) -> str:
    """由稳定来源定位确定性派生正文可见 Material ID。

    输出形态固定为 ``m_<26位CrockfordBase32>``。输入 JSON 排序并剔除空值，
    因而不受 dict 构造顺序影响；原始章节、假说、citation 或 locator 文本只进入
    哈希前镜像，不出现在正文可见 id 中。
    """
    identity = {
        "source_kind": source_kind,
        "source_ref": _compact_source_ref(source_ref),
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"m_{_crockford_base32_130_bits(payload.encode())}"


def _source_ref_from_citation(
    source_kind: SourceKind, citation: dict[str, Any]
) -> dict[str, Any]:
    """从引擎公开 CitationRecord 字段构造真实来源定位。

    简化/旧记录可能缺少知识库或结构化定位字段；此时落到可复现的公开记录摘要，
    只作为兼容性身份材料，真实来源仍优先来自 url / knowledge / dataset 字段。
    """
    provenance = citation.get("provenance") or {}
    if source_kind == "web":
        source_ref = {
            "url": citation.get("url"),
            "content_fingerprint": provenance.get("content_fingerprint"),
            "source_evidence_fingerprint": provenance.get(
                "source_evidence_fingerprint"
            ),
        }
    elif source_kind == "knowledge_base":
        source_ref = {
            "knowledge_id": citation.get("knowledge_id"),
            "file_id": citation.get("file_id"),
            "chunk_id": citation.get("chunk_id"),
            "page": citation.get("page"),
            "content_fingerprint": provenance.get("content_fingerprint"),
            "source_evidence_fingerprint": provenance.get(
                "source_evidence_fingerprint"
            ),
        }
    else:
        source_ref = {
            "scenario_key": provenance.get("scenario_key"),
            "dataset_id": provenance.get("dataset_id"),
            "query_execution_id": provenance.get("query_execution_id"),
            "content_fingerprint": provenance.get("content_fingerprint"),
            "source_evidence_fingerprint": provenance.get(
                "source_evidence_fingerprint"
            ),
        }
    compacted = _compact_source_ref(source_ref)
    if compacted:
        return compacted
    fallback = {
        "source_name": citation.get("source_name"),
        "title": citation.get("title"),
        "summary_sha256": hashlib.sha256(
            str(citation.get("summary", "")).encode()
        ).hexdigest(),
    }
    return _compact_source_ref(fallback)


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
    章节论点与假说一并聚合进段落文本（查询构造的素材来源），论点另经
    argument_context.argument_path 给引擎论证层级上下文（杠杆①：查询聚合论点+假说）。
    """
    chapter_id = task["chapter_id"]
    digest = task["existing_materials_digest"].strip()
    points = task.get("points", [])
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
            [point["text"] for point in points]
            + [hypothesis["text"] for hypothesis in task["hypotheses"]]
        ),
        "forward_items": forward_items,
        "reverse_items": reverse_items,
    }
    argument_context: dict[str, Any] = {}
    if points:
        # 论点是章下第一论证层：argument_path 的 level 从 1 起（章为根、不入路径）。
        argument_context["argument_path"] = [
            {"level": 1, "node_id": point["id"], "text": point["text"]}
            for point in points
        ]
    genre = task["genre"].strip()
    if genre:
        argument_context["boundary"] = genre
    if argument_context:
        paragraph["argument_context"] = argument_context
    return {
        "request_id": f"chapter-{chapter_id}",
        "document_id": ENGINE_DOCUMENT_ID,
        "paragraph": paragraph,
    }


_VERDICT_RANK: dict[str, int] = {"fail": 0, "inconclusive": 1, "pass": 2}
"""三值 verdict 优先级：同（假说, 引文）在多条裁决出现时取强者（pass > inconclusive > fail）。"""


def search_result_from_engine_output(
    output: dict[str, Any], task: dict[str, Any]
) -> SearchResult:
    """引擎公开出参（search-agent-output/v1 的 dict 形态）→ SearchResult。

    契约 verdict 三值化（杠杆②，经 T0 诊断放行、收窄口径）：
    - 正向线被列为支撑引文（supporting）的素材为 pass——强支撑，进写作池；
    - 正向线被列为补充引文（supplementary，即近似命中/SUPPLEMENT）的素材为
      inconclusive——弱佐证，进写作池但按弱佐证渲染；
    - 其余（正向线反例、反向线命中）一律 fail——供审计不进写作池。
    IRRELEVANT/NEUTRAL 噪声在引擎裁决层已丢弃、根本不进公开出参，此处无需再判。
    反向线（反驳条件检索）命中的素材一律 fail：它们是对假说的削弱证据，
    即便被引擎列为补充也不得进写作池。
    同一（假说, 引文）在多线出现时按 _VERDICT_RANK 取强者。
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
        supplementary = set(decision.get("supplementary_citation_ids", []))
        for citation_id in decision.get("citation_ids", []):
            citation = citations.get(citation_id)
            if citation is None:
                continue
            source_kind = _SOURCE_KIND_BY_TYPE.get(citation.get("source_type", ""))
            if source_kind is None:
                continue
            verdict: Literal["pass", "fail", "inconclusive"]
            if line == "forward" and citation_id in supporting:
                verdict = "pass"
            elif line == "forward" and citation_id in supplementary:
                verdict = "inconclusive"
            else:
                verdict = "fail"
            key = (hypothesis_id, citation_id)
            existing = picked.get(key)
            if existing is not None and (
                _VERDICT_RANK[verdict] <= _VERDICT_RANK[existing["verdict"]]
            ):
                continue
            source_ref = _source_ref_from_citation(source_kind, citation)
            picked[key] = MaterialPayload(
                id=material_id_from_source_ref(source_kind, source_ref),
                hypothesis_id=hypothesis_id,
                source=citation.get("source_name")
                or citation.get("title")
                or "未知来源",
                url=citation.get("url"),
                source_kind=source_kind,
                source_ref=source_ref,
                excerpt=citation["summary"],
                relevance_score=float(citation["judgment"]["confidence"]),
                verdict=verdict,
            )
    return SearchResult(materials=list(picked.values()))
