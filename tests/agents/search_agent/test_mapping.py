"""契约映射离线测试：SearchTask/SearchResult 与引擎公开出入参的相互转换。

入参侧对照引擎冻结的公开契约模型逐项校验（映射产物必须能过引擎的
Pydantic 严格校验），出参侧断言假说回链、裁决折算、url 与 source_kind 回填。
"""

from typing import Any

from agents.search_agent import (
    ENGINE_DOCUMENT_ID,
    engine_payload_from_task,
    fake_engine_output,
    forward_item_id,
    material_id_from_source_ref,
    reverse_item_id,
    search_result_from_engine_output,
    split_item_id,
)

TASK: dict[str, Any] = {
    "chapter_id": "ch-1",
    "hypotheses": [
        {"id": "h-1", "text": "示例假说一", "refute_condition": "若出现反例则证伪"},
        {"id": "h-2", "text": "示例假说二", "refute_condition": "   "},
    ],
    "genre": "行业白皮书",
    "existing_materials_digest": "既有素材摘要",
}


def _citation(citation_id: str, source_type: str, url: str | None) -> dict[str, Any]:
    """构造引文记录：只含适配层消费的字段。"""
    return {
        "citation_id": citation_id,
        "source_type": source_type,
        "source_name": f"来源（{citation_id}）",
        "title": f"标题（{citation_id}）",
        "url": url,
        "knowledge_id": f"kb-{citation_id}" if source_type == "KNOWLEDGE_BASE" else None,
        "file_id": f"file-{citation_id}" if source_type == "KNOWLEDGE_BASE" else None,
        "chunk_id": f"chunk-{citation_id}" if source_type == "KNOWLEDGE_BASE" else None,
        "summary": f"摘录（{citation_id}）",
        "judgment": {"confidence": 0.8},
        "provenance": {
            "scenario_key": (
                f"scenario-{citation_id}"
                if source_type == "STRUCTURED_DATA"
                else None
            ),
            "dataset_id": (
                f"dataset-{citation_id}" if source_type == "STRUCTURED_DATA" else None
            ),
            "query_execution_id": (
                f"query-{citation_id}" if source_type == "STRUCTURED_DATA" else None
            ),
            "content_fingerprint": f"fp-{citation_id}",
        },
    }


def _material_by_source_kind(output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    materials = search_result_from_engine_output(output, TASK)["materials"]
    return {material["source_kind"]: material for material in materials}


def test_稳定来源输入映射为确定性不透明material_id() -> None:
    source_ref = {"url": "https://example.com/source/path?x=1"}

    first = material_id_from_source_ref("web", source_ref)
    second = material_id_from_source_ref("web", dict(reversed(source_ref.items())))

    assert first == second
    assert first.startswith("m_")
    assert len(first) == 28
    assert set(first.removeprefix("m_")) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


def test_三类来源稳定输入_id形状确定且不泄漏来源语义() -> None:
    cases = [
        (
            "web",
            {
                "url": "https://example.com/ch-1/h-1/c-web",
                "content_fingerprint": "fp-web",
            },
        ),
        (
            "knowledge_base",
            {
                "knowledge_id": "kb-ch-1",
                "file_id": "file-h-1",
                "chunk_id": "chunk-c-kb",
                "content_fingerprint": "fp-kb",
            },
        ),
        (
            "structured_data",
            {
                "scenario_key": "scenario-ch-1",
                "dataset_id": "dataset-h-1",
                "query_execution_id": "query-c-doris",
                "content_fingerprint": "fp-structured",
            },
        ),
    ]

    ids = [material_id_from_source_ref(kind, source_ref) for kind, source_ref in cases]
    repeated = [
        material_id_from_source_ref(kind, dict(reversed(source_ref.items())))
        for kind, source_ref in cases
    ]

    assert len(set(ids)) == len(cases)
    assert ids == repeated
    for material_id in ids:
        assert material_id.startswith("m_")
        assert len(material_id) == 28
        assert "ch-1" not in material_id
        assert "h-1" not in material_id
        assert "c-" not in material_id
        assert "example" not in material_id
        assert "kb" not in material_id.lower()
        assert "dataset" not in material_id.lower()


def test_任务包映射为引擎入参_正反向检索项与既有证据摘要齐备() -> None:
    payload = engine_payload_from_task(TASK)

    assert payload["request_id"] == "chapter-ch-1"
    assert payload["document_id"] == ENGINE_DOCUMENT_ID
    paragraph = payload["paragraph"]
    assert paragraph["paragraph_id"] == "ch-1"
    assert "示例假说一" in paragraph["paragraph_text"]
    assert paragraph["argument_context"] == {"boundary": "行业白皮书"}

    # 每条假说恰好一个正向项；既有引文库摘要作为既有证据文本。
    forward = paragraph["forward_items"]
    assert [item["item_id"] for item in forward] == [
        forward_item_id("h-1"),
        forward_item_id("h-2"),
    ]
    assert all(item["item_type"] == "claim" for item in forward)
    assert forward[0]["target_text"] == "示例假说一"
    assert all(item["existing_evidence_text"] == "既有素材摘要" for item in forward)

    # 反驳条件驱动反向检索：仅非空白 refute_condition 产生 oppose 反向项。
    reverse = paragraph["reverse_items"]
    assert [item["item_id"] for item in reverse] == [reverse_item_id("h-1")]
    assert reverse[0]["target_text"] == "若出现反例则证伪"
    assert reverse[0]["relation_to_original"] == "oppose"


def test_入参映射产物通过引擎公开契约校验() -> None:
    from search_agent.evidence_retrieval.public_contracts import SearchAgentInputState

    SearchAgentInputState.model_validate(engine_payload_from_task(TASK))


def test_空白摘要与空白品类映射为缺省形态() -> None:
    task = dict(TASK, existing_materials_digest="  ", genre="")
    paragraph = engine_payload_from_task(task)["paragraph"]
    assert all(
        item["existing_evidence_text"] is None for item in paragraph["forward_items"]
    )
    assert "argument_context" not in paragraph


def test_检索项id编码往返_不合约定返回None() -> None:
    assert split_item_id(forward_item_id("h-9")) == ("h-9", "forward")
    assert split_item_id(reverse_item_id("h-9")) == ("h-9", "reverse")
    assert split_item_id("没有分隔符") is None
    assert split_item_id("h-9::sideways") is None


def test_引擎出参映射_回链裁决url与来源通道逐项回填() -> None:
    output = {
        "results": [
            {
                "item_id": forward_item_id("h-1"),
                "citation_ids": ["c-web", "c-kb"],
                "supporting_citation_ids": ["c-web"],
            },
            {
                "item_id": reverse_item_id("h-1"),
                "citation_ids": ["c-doris"],
                "supporting_citation_ids": ["c-doris"],
            },
        ],
        "citations": [
            _citation("c-web", "WEB", "https://example.com/a"),
            _citation("c-kb", "KNOWLEDGE_BASE", None),
            _citation("c-doris", "STRUCTURED_DATA", None),
        ],
    }

    materials = search_result_from_engine_output(output, TASK)["materials"]
    by_kind = {material["source_kind"]: material for material in materials}
    assert set(by_kind) == {"web", "knowledge_base", "structured_data"}

    # 全部素材回链发起检索项的假说。
    assert all(material["hypothesis_id"] == "h-1" for material in materials)

    # 正向线仅支撑引文为 pass；补充引文与反向线（即便引擎列为支撑）一律 fail。
    assert by_kind["web"]["verdict"] == "pass"
    assert by_kind["knowledge_base"]["verdict"] == "fail"
    assert by_kind["structured_data"]["verdict"] == "fail"

    # url 与 source_kind 按引文回填：三通道类型标识一一对应，仅联网带链接。
    assert by_kind["web"]["source_kind"] == "web"
    assert by_kind["web"]["url"] == "https://example.com/a"
    assert by_kind["web"]["source_ref"]["url"] == "https://example.com/a"
    assert by_kind["knowledge_base"]["source_kind"] == "knowledge_base"
    assert by_kind["knowledge_base"]["url"] is None
    assert by_kind["knowledge_base"]["source_ref"] == {
        "chunk_id": "chunk-c-kb",
        "content_fingerprint": "fp-c-kb",
        "file_id": "file-c-kb",
        "knowledge_id": "kb-c-kb",
    }
    assert by_kind["structured_data"]["source_kind"] == "structured_data"
    assert by_kind["structured_data"]["source_ref"] == {
        "content_fingerprint": "fp-c-doris",
        "dataset_id": "dataset-c-doris",
        "query_execution_id": "query-c-doris",
        "scenario_key": "scenario-c-doris",
    }

    # 其余字段：来源名、摘录与相关度分从引文记录携带。
    assert by_kind["web"]["source"] == "来源（c-web）"
    assert by_kind["web"]["excerpt"] == "摘录（c-web）"
    assert by_kind["web"]["relevance_score"] == 0.8


def test_同假说同引文正反两线并存时pass优先() -> None:
    output = {
        "results": [
            {
                "item_id": reverse_item_id("h-1"),
                "citation_ids": ["c-1"],
                "supporting_citation_ids": [],
            },
            {
                "item_id": forward_item_id("h-1"),
                "citation_ids": ["c-1"],
                "supporting_citation_ids": ["c-1"],
            },
        ],
        "citations": [_citation("c-1", "WEB", "https://example.com/1")],
    }
    materials = search_result_from_engine_output(output, TASK)["materials"]
    assert len(materials) == 1
    assert materials[0]["verdict"] == "pass"


def test_脏数据丢弃_未知假说未知引文与未知来源类型不入结果() -> None:
    output = {
        "results": [
            {
                "item_id": forward_item_id("h-999"),
                "citation_ids": ["c-1"],
                "supporting_citation_ids": ["c-1"],
            },
            {
                "item_id": "编码不合约定",
                "citation_ids": ["c-1"],
                "supporting_citation_ids": [],
            },
            {
                "item_id": forward_item_id("h-1"),
                "citation_ids": ["c-missing", "c-unknown-type"],
                "supporting_citation_ids": [],
            },
        ],
        "citations": [
            _citation("c-1", "WEB", "https://example.com/1"),
            _citation("c-unknown-type", "CARRIER_PIGEON", None),
        ],
    }
    assert search_result_from_engine_output(output, TASK)["materials"] == []


def test_无证据假说不产生素材_据此被下游过滤() -> None:
    """h-1 检回支撑证据、h-2 一无所获：h-2 无任何素材条目（更无 pass），
    下游按 pass 素材筛选时该假说被正确过滤。"""
    output = {
        "results": [
            {
                "item_id": forward_item_id("h-1"),
                "citation_ids": ["c-1"],
                "supporting_citation_ids": ["c-1"],
            },
            {
                "item_id": forward_item_id("h-2"),
                "citation_ids": [],
                "supporting_citation_ids": [],
            },
        ],
        "citations": [_citation("c-1", "WEB", "https://example.com/1")],
    }
    materials = search_result_from_engine_output(output, TASK)["materials"]
    assert {material["hypothesis_id"] for material in materials} == {"h-1"}


def test_假引擎出参经映射产出契约合规素材() -> None:
    payload = engine_payload_from_task(TASK)
    materials = search_result_from_engine_output(fake_engine_output(payload), TASK)[
        "materials"
    ]

    # 2 条假说 → 2 正向 pass；1 条非空反驳条件 → 1 反向 fail。
    verdicts = sorted(
        (material["hypothesis_id"], material["verdict"]) for material in materials
    )
    assert verdicts == [("h-1", "fail"), ("h-1", "pass"), ("h-2", "pass")]
    for material in materials:
        assert material["source_kind"] in ("web", "knowledge_base", "structured_data")
        assert (material["url"] is not None) == (material["source_kind"] == "web")


def test_假引擎出参通过引擎公开契约校验() -> None:
    from search_agent.evidence_retrieval.public_contracts import SearchAgentOutputState

    SearchAgentOutputState.model_validate(
        fake_engine_output(engine_payload_from_task(TASK))
    )


# ---------- 杠杆①：查询构造聚合论点 + 假说 ----------

_TASK_WITH_POINTS: dict[str, Any] = dict(
    TASK,
    points=[
        {"id": "p-1", "text": "论点甲的论证方向"},
        {"id": "p-2", "text": "论点乙的论证方向"},
    ],
)


def test_任务包聚合论点与假说进查询构造() -> None:
    paragraph = engine_payload_from_task(_TASK_WITH_POINTS)["paragraph"]

    # 段落文本聚合论点在前、假说在后，供引擎查询构造取用（杠杆①）。
    assert paragraph["paragraph_text"] == "\n".join(
        ["论点甲的论证方向", "论点乙的论证方向", "示例假说一", "示例假说二"]
    )
    # 论点另经 argument_context.argument_path 给引擎论证层级上下文，品类仍进 boundary。
    context = paragraph["argument_context"]
    assert context["boundary"] == "行业白皮书"
    assert context["argument_path"] == [
        {"level": 1, "node_id": "p-1", "text": "论点甲的论证方向"},
        {"level": 1, "node_id": "p-2", "text": "论点乙的论证方向"},
    ]


def test_任务包无论点时段落文本仅假说且无argument_path() -> None:
    paragraph = engine_payload_from_task(TASK)["paragraph"]
    assert paragraph["paragraph_text"] == "示例假说一\n示例假说二"
    assert paragraph["argument_context"] == {"boundary": "行业白皮书"}


def test_带论点入参映射产物通过引擎公开契约校验() -> None:
    from search_agent.evidence_retrieval.public_contracts import SearchAgentInputState

    SearchAgentInputState.model_validate(engine_payload_from_task(_TASK_WITH_POINTS))


# ---------- 杠杆②：INCONCLUSIVE（补充引文）降级落库 ----------


def test_引擎出参_正向补充引文降级为inconclusive落库() -> None:
    """正向线支撑引文 → pass；正向线补充引文（SUPPLEMENT/近似命中）→ inconclusive；
    正向线反例与反向线命中 → fail。IRRELEVANT 噪声在引擎裁决层已丢弃、不入出参。"""
    output = {
        "results": [
            {
                "item_id": forward_item_id("h-1"),
                "citation_ids": ["c-pass", "c-weak", "c-refute"],
                "supporting_citation_ids": ["c-pass"],
                "supplementary_citation_ids": ["c-weak"],
            },
            {
                "item_id": reverse_item_id("h-1"),
                "citation_ids": ["c-rev"],
                "supporting_citation_ids": [],
                "supplementary_citation_ids": ["c-rev"],
            },
        ],
        "citations": [
            _citation("c-pass", "WEB", "https://example.com/pass"),
            _citation("c-weak", "KNOWLEDGE_BASE", None),
            _citation("c-refute", "WEB", "https://example.com/refute"),
            _citation("c-rev", "STRUCTURED_DATA", None),
        ],
    }
    by_source = {
        material["source"]: material
        for material in search_result_from_engine_output(output, TASK)["materials"]
    }
    assert by_source["来源（c-pass）"]["verdict"] == "pass"
    assert by_source["来源（c-weak）"]["verdict"] == "inconclusive"
    assert by_source["来源（c-refute）"]["verdict"] == "fail"
    # 反向线即便被引擎列为补充也一律 fail（对假说的削弱证据不进写作池）。
    assert by_source["来源（c-rev）"]["verdict"] == "fail"


def test_同假说同引文多线并存时取强者_pass优于inconclusive优于fail() -> None:
    output = {
        "results": [
            {
                "item_id": reverse_item_id("h-1"),
                "citation_ids": ["c-1"],
                "supporting_citation_ids": [],
                "supplementary_citation_ids": [],
            },
            {
                "item_id": forward_item_id("h-1"),
                "citation_ids": ["c-1"],
                "supporting_citation_ids": [],
                "supplementary_citation_ids": ["c-1"],
            },
        ],
        "citations": [_citation("c-1", "WEB", "https://example.com/1")],
    }
    materials = search_result_from_engine_output(output, TASK)["materials"]
    # 反向线判 fail、正向线判 inconclusive：取强者 inconclusive。
    assert len(materials) == 1
    assert materials[0]["verdict"] == "inconclusive"
