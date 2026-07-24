"""公开检索输出适配层契约测试。"""

from search_agent.evidence_retrieval.output_adapter import build_public_output
from search_agent.evidence_retrieval.public_contracts import SearchAgentInputState


def test_public_web_citation_source_ref_uses_canonical_url() -> None:
    public_input = SearchAgentInputState.model_validate(
        {
            "request_id": "req-web-source-ref",
            "document_id": "doc-1",
            "paragraph": {
                "paragraph_id": "ch1",
                "paragraph_text": "公开网页可校验事实。",
                "forward_items": [
                    {"item_id": "h1::forward", "target_text": "公开网页事实。"}
                ],
            },
        }
    )
    diagnostic = {
        "request_id": "req-web-source-ref",
        "paragraph_results": [
            {
                "paragraph_id": "ch1",
                "results": [
                    {
                        "task_id": "req-web-source-ref:task:1",
                        "item_id": "h1::forward",
                        "line_type": "forward",
                        "node_id": "h1::forward",
                        "target_text": "公开网页事实。",
                        "execution_status": "SUCCESS",
                        "termination_reason": "SUFFICIENT",
                        "verification": {
                            "verdict": "INCONCLUSIVE",
                            "upstream_status": "doubtful",
                            "confidence": 0.5,
                            "reason": "passthrough",
                        },
                        "evidence_items": [
                            {
                                "evidence_id": "ev-web-1",
                                "task_id": "req-web-source-ref:task:1",
                                "source_type": "web",
                                "source_name": "web",
                                "source_ref": {
                                    "url": "HTTPS://Example.COM:443/a/?utm_source=x&b=2&a=1#frag"
                                },
                                "title": "网页",
                                "content": "公开网页事实。",
                                "quoted_spans": ["公开网页事实。"],
                                "relation": "SUPPLEMENT",
                                "judge_confidence": 0.0,
                                "content_fingerprint": "fp-web",
                                "source_evidence_fingerprint": "sfp-web",
                                "metadata": {
                                    "retrieval_candidate_passthrough": True,
                                    "query_ids": ["q-1"],
                                },
                            }
                        ],
                        "evidence_quality": {},
                    }
                ],
            }
        ],
    }

    output = build_public_output(public_input, diagnostic)

    assert output.citations[0].url == "HTTPS://Example.COM:443/a/?utm_source=x&b=2&a=1#frag"
    assert output.citations[0].source_ref == {
        "content_fingerprint": "fp-web",
        "query_ids": ["q-1"],
        "source_evidence_fingerprint": "sfp-web",
        "url": "https://example.com/a?a=1&b=2",
    }


def test_public_citation_preserves_source_ref_from_evidence_item() -> None:
    public_input = SearchAgentInputState.model_validate(
        {
            "request_id": "req-source-ref",
            "document_id": "doc-1",
            "paragraph": {
                "paragraph_id": "ch1",
                "paragraph_text": "专业薪资可由结构化数据校验。",
                "forward_items": [
                    {
                        "item_id": "h1::forward",
                        "target_text": "软件技术专业平均实习薪资为 5000 元。",
                    }
                ],
            },
        }
    )
    diagnostic = {
        "request_id": "req-source-ref",
        "paragraph_results": [
            {
                "paragraph_id": "ch1",
                "results": [
                    {
                        "task_id": "req-source-ref:task:1",
                        "item_id": "h1::forward",
                        "line_type": "forward",
                        "node_id": "h1::forward",
                        "target_text": "软件技术专业平均实习薪资为 5000 元。",
                        "execution_status": "SUCCESS",
                        "termination_reason": "SUFFICIENT",
                        "verification": {
                            "verdict": "INCONCLUSIVE",
                            "upstream_status": "doubtful",
                            "confidence": 0.5,
                            "reason": "passthrough",
                        },
                        "evidence_items": [
                            {
                                "evidence_id": "ev-structured-1",
                                "task_id": "req-source-ref:task:1",
                                "source_type": "structured",
                                "source_name": "structured_tool",
                                "source_ref": {
                                    "scenario_name": "salary_scenario",
                                    "record_id": "tool-call-1",
                                    "dataset_id": "doris-main",
                                    "query_execution_id": "exec-123",
                                    "query_params_hash": "params-hash",
                                },
                                "title": "薪资查询",
                                "content": "记录1：专业=软件技术；平均薪资=5000 元。",
                                "quoted_spans": ["记录1：专业=软件技术；平均薪资=5000 元。"],
                                "relation": "SUPPLEMENT",
                                "judge_confidence": 0.0,
                                "content_fingerprint": "fp-structured",
                                "source_evidence_fingerprint": "sfp-structured",
                                "metadata": {
                                    "retrieval_candidate_passthrough": True,
                                    "scenario_key": "salary_scenario",
                                    "tool_call_id": "tool-call-1",
                                    "columns": ["专业", "平均薪资"],
                                    "row_count": 1,
                                    "query_ids": ["q-1"],
                                },
                            }
                        ],
                        "evidence_quality": {},
                    }
                ],
            }
        ],
    }

    output = build_public_output(public_input, diagnostic)

    assert len(output.citations) == 1
    assert output.citations[0].source_ref == {
        "columns": ["专业", "平均薪资"],
        "content_fingerprint": "fp-structured",
        "dataset_id": "doris-main",
        "query_execution_id": "exec-123",
        "query_ids": ["q-1"],
        "query_params_hash": "params-hash",
        "record_id": "tool-call-1",
        "row_count": 1,
        "scenario_key": "salary_scenario",
        "source_evidence_fingerprint": "sfp-structured",
    }
