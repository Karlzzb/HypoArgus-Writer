"""Configuration with default < environment < explicit override precedence."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class EvidenceRetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Safe default for upstream/downstream integration. Shadow mode still runs
    # the complete graph and tracing, but forbids propagation and writeback.
    shadow_mode: bool = True

    # === 外部服务地址与凭据（只能由系统环境配置，请求不可覆盖） ===
    volcano_base_url: str = "https://open.volcengineapi.com"
    volcano_search_path: str = "/api/v1/search"
    volcano_api_key: SecretStr | None = Field(default=None, repr=False)
    bisheng_base_url: str | None = None
    # Deployments without a dedicated access endpoint must inject an
    # access_checker/auth_headers_provider; selected knowledge fails closed.
    bisheng_access_path: str | None = None
    bisheng_token: SecretStr | None = Field(default=None, repr=False)
    bisheng_retrieve_base_url: str | None = None
    bisheng_retrieve_path: str = "/api/v1/knowledge/retrieve"
    bisheng_retrieve_timeout_ms: int = Field(default=12000, ge=100)
    bisheng_retrieve_connect_timeout_ms: int = Field(default=3000, ge=100)
    bisheng_retrieve_read_timeout_ms: int = Field(default=10000, ge=100)
    bisheng_retrieve_retry_count: int = Field(default=1, ge=0, le=5)
    bisheng_retrieve_top_k: int = Field(default=5, ge=1, le=50)
    bisheng_retrieve_score_threshold: float | None = Field(default=None, ge=0, le=1)
    bisheng_retrieve_max_text_chars: int = Field(default=4000, ge=200, le=20000)
    # Bisheng retrieval is request-bound and all 24 tasks must still run. A
    # bounded eight-worker pool removes an avoidable four-wave tail while
    # remaining below the provider's rate-limit ceiling.
    public_kb_concurrency: int = Field(default=4, ge=1, le=32)
    bisheng_max_connections: int = Field(default=16, ge=1, le=128)
    bisheng_keepalive_connections: int = Field(default=12, ge=1, le=128)
    # Doris FE MySQL protocol.  SearchAgent owns the read-only scenario SQL;
    # no SSH tunnel and no external Structured HTTP service are involved.
    doris_host: str | None = None
    doris_port: int = Field(default=9030, ge=1, le=65535)
    doris_database: str | None = None
    doris_username: str | None = None
    doris_password: SecretStr | None = Field(default=None, repr=False)
    doris_charset: str = "utf8mb4"
    doris_pool_size: int = Field(default=5, ge=1, le=50)
    doris_max_overflow: int = Field(default=5, ge=0, le=50)
    doris_pool_timeout_seconds: int = Field(default=5, ge=1, le=120)
    doris_pool_recycle_seconds: int = Field(default=1500, ge=60)
    doris_connect_timeout_seconds: int = Field(default=5, ge=1, le=120)
    doris_read_timeout_seconds: int = Field(default=30, ge=1, le=300)
    doris_query_timeout_seconds: int = Field(default=30, ge=1, le=900)
    doris_dataset_id: str = "hypoargus-doris-structured"
    public_knowledge_ids: list[str] = Field(default_factory=list)

    # === Batch/单任务并发、总轮次与硬超时预算 ===
    batch_concurrency: int = Field(default=4, ge=1)
    web_search_concurrency: int = Field(default=4, ge=1, le=32)
    web_fetch_concurrency: int = Field(default=4, ge=1, le=32)
    kb_retrieve_concurrency: int = Field(default=4, ge=1, le=32)
    structured_concurrency: int = Field(default=6, ge=1, le=16)
    max_tasks_per_request: int | None = Field(default=None, ge=1)
    batch_hard_timeout_ms: int | None = Field(default=None, ge=1)
    task_hard_timeout_ms: int = Field(default=90000, ge=100)
    max_total_rounds_per_item: int = Field(default=3, ge=1)
    max_web_rounds_per_item: int = Field(default=2, ge=1)
    max_public_kb_calls_per_item: int = Field(default=1, ge=0)
    max_selected_kb_calls_per_item: int = Field(default=1, ge=0)
    max_structured_calls_per_item: int = Field(default=2, ge=0)
    no_new_evidence_round_limit: int = Field(default=2, ge=1)

    # === Web Query、搜索候选、正文抓取、代理与 SSRF 策略 ===
    initial_query_count: int = Field(default=2, ge=1, le=10)
    web_doc_count: int = Field(default=10, ge=1, le=50)
    web_max_snippet_length: int = Field(default=300, ge=50, le=2000)
    web_query_max_length: int = Field(default=500, ge=50, le=1000)
    web_max_image_count_per_doc: int = Field(default=1, ge=0, le=10)
    web_keep_top_k_urls: int = Field(default=5, ge=1)
    web_keep_top_k_chunks: int = Field(default=5, ge=1)
    web_fetch_top_n: int = Field(default=3, ge=1, le=10)
    web_candidates_per_task: int = Field(default=3, ge=1, le=10)
    kb_candidates_per_task: int = Field(default=3, ge=1, le=12)
    total_candidates_per_task: int = Field(default=8, ge=3, le=24)
    web_search_timeout_ms: int = Field(default=1500, ge=100)
    web_fetch_timeout_ms: int = Field(default=3000, ge=100)
    web_retry_count: int = Field(default=1, ge=0, le=5)
    web_max_response_bytes: int = Field(default=2_000_000, ge=1024)
    # PDF 头与最小长度双重校验，避免把网关 HTML 错当 PDF 交给解析器。
    web_pdf_min_bytes: int = Field(default=64, ge=5)
    web_chunk_chars: int = Field(default=1800, ge=200)
    web_whitelist_enabled: bool = False
    web_allowed_domains: list[str] = Field(default_factory=list)
    web_blocked_domains: list[str] = Field(default_factory=list)
    web_trusted_proxy_enabled: bool = False
    snippet_only_score_cap: float = Field(default=0.45, ge=0, le=1)

    # === Structured 场景注册表、请求超时与重试 ===
    structured_scenarios_cache_ttl_seconds: int = Field(default=3600, ge=0)
    structured_query_cache_ttl_seconds: float = Field(default=30.0, ge=0)
    structured_health_cache_ttl_seconds: float = Field(default=60.0, ge=0)
    # A three-paragraph production run spans roughly 20 seconds between the
    # first and third prefetch; 30 seconds collapses that repeated failure I/O
    # while remaining short enough to recover promptly.
    structured_health_failure_cache_ttl_seconds: float = Field(default=30.0, ge=0)
    structured_timeout_ms: int = Field(default=35000, ge=100)
    structured_retry_count: int = Field(default=1, ge=0, le=5)

    # === parallel_sources 独立预算 ===
    parallel_flow_timeout_ms: int = Field(default=95000, ge=100)
    parallel_finalize_reserve_ms: int = Field(default=100, ge=0, le=1000)
    parallel_web_search_timeout_ms: int = Field(default=5000, ge=100)
    parallel_web_fetch_timeout_ms: int = Field(default=8000, ge=100)
    parallel_kb_timeout_ms: int = Field(default=12000, ge=100)
    parallel_structured_timeout_ms: int = Field(default=8000, ge=100)
    parallel_batch_judge_timeout_ms: int = Field(default=70000, ge=100)
    gap_retrieval_reserved_ms: int = Field(default=12000, ge=1000, le=30000)
    judge_model: str | None = None
    # Judge 调用成功但格式不可解析时，仅做一次 JSON 格式修复，不重新判断事实。
    batch_judge_parse_retry_enabled: bool = True
    batch_judge_parse_retry_count: int = Field(default=1, ge=0, le=1)
    batch_judge_parse_retry_timeout_ms: int = Field(default=30000, ge=100)
    # 原始响应仅进入内部诊断/脱敏 Trace，不进入精简业务返回。
    batch_judge_raw_preview_chars: int = Field(default=6000, ge=400, le=8000)
    # Compact neutral-only output makes four task groups safe while amortizing
    # provider round trips. The token planner remains the final hard bound.
    judge_batch_max_tasks: int = Field(default=4, ge=1, le=24)
    judge_batch_max_candidates: int = Field(default=24, ge=1, le=60)
    judge_batch_max_input_tokens: int = Field(default=30000, ge=1000, le=200000)
    judge_batch_concurrency: int = Field(default=8, ge=1, le=16)
    judge_expected_output_tokens_per_candidate: int = Field(default=180, ge=20, le=500)
    judge_model_context_limit: int = Field(default=65536, ge=4096)
    parallel_judge_candidate_max_chars: int = Field(default=800, ge=100, le=10000)
    denylist_domains: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)
    max_results_per_domain: int = Field(default=1, ge=0)
    structured_max_rows: int = Field(default=20, ge=1, le=200)
    structured_max_candidates: int = Field(default=3, ge=1, le=10)
    # Structured candidate grouping: small result sets collapse to a single
    # candidate carrying the full row set + query context.
    structured_max_rows_per_candidate: int = Field(default=20, ge=1, le=200)
    structured_max_candidates_per_task: int = Field(default=2, ge=1, le=10)

    # === Structured Tool Calling 配置 ===
    # LLM 必须调用的最少真实场景工具数（不含 no_structured_query）
    # Three audited scenarios have fully defaultable parameters, so 0..3 can
    # be guaranteed even when both LLM routing attempts fail.
    structured_min_tool_calls: int = Field(default=1, ge=0, le=3)
    # LLM 最多可调用的工具数（含 no_structured_query）
    structured_max_tool_calls: int = Field(default=5, ge=1, le=12)
    # 参数校验失败后的修复重试次数
    structured_repair_count: int = Field(default=1, ge=0, le=3)

    # === 证据充分性、安全闸门与最终判定阈值（请求不可放宽） ===
    min_effective_evidence_count: int = Field(default=2, ge=1)
    min_direct_evidence_count: int = Field(default=1, ge=1)
    min_directness_score: float = Field(default=0.60, ge=0, le=1)
    min_independent_source_count: int = Field(default=1, ge=1)
    min_independent_document_count: int = Field(default=2, ge=1)
    min_claim_coverage_score: float = Field(default=0.65, ge=0, le=1)
    min_final_evidence_score: float = Field(default=0.70, ge=0, le=1)
    high_authority_single_evidence_score: float = Field(default=0.88, ge=0, le=1)
    max_noise_ratio: float = Field(default=0.60, ge=0, le=1)
    conflict_weight_threshold: float = Field(default=0.70, ge=0)
    verdict_margin: float = Field(default=0.25, ge=0)
    authority_threshold: float = Field(default=0.75, ge=0, le=1)
    # Authoritative Structured single-source sufficiency override. A single
    # Structured result with full slot coverage, top authority and directness
    # can satisfy a fact-retrieval task without the generic multi-source rule.
    # Disabled by default; flip to True when product confirms the rule.
    authoritative_structured_override_enabled: bool = Field(default=False)
    authoritative_structured_min_authority: float = Field(default=0.85, ge=0, le=1)
    authoritative_structured_min_directness: float = Field(default=0.80, ge=0, le=1)
    authoritative_structured_require_full_slot_coverage: bool = Field(default=True)

    # === V12 downstream citation admission gate ===
    # ``candidate_passthrough`` is the HypoArgus integration mode: retrieval
    # candidates are packaged as citations without any Evidence Judge call or
    # SUPPORT/REFUTE/NEUTRAL decision.  The parent Judgment node is the sole
    # adjudicator.  ``judged`` preserves standalone SearchAgent behaviour.
    evidence_output_mode: Literal["judged", "candidate_passthrough"] = "judged"
    public_citation_min_confidence: float = Field(default=0.60, ge=0, le=1)
    public_citation_min_directness: float = Field(default=0.60, ge=0, le=1)
    public_supplement_min_confidence: float = Field(default=0.40, ge=0, le=1)

    # === 去重、时效性和 Trace 脱敏 ===
    near_duplicate_threshold: float = Field(default=0.88, ge=0, le=1)
    freshness_half_life_days: int = Field(default=365, ge=1)
    trace_content: bool = False
    trace_max_chars: int = Field(default=500, ge=0)
    # Parallel 流程追踪事件的进程内非阻塞队列上限。
    trace_queue_max: int = Field(default=1000, ge=1)
    quality_weights: dict[str, float] = Field(default_factory=lambda: {
        "relevance": 0.28, "authority": 0.22, "directness": 0.20,
        "slot_coverage": 0.15, "freshness": 0.10, "traceability": 0.05,
    })

    @field_validator("public_knowledge_ids", "web_allowed_domains", "web_blocked_domains", "denylist_domains", "preferred_domains", mode="before")
    @classmethod
    def parse_lists(cls, value):
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return value

    @model_validator(mode="after")
    def validate_weights(self):
        expected = {"relevance", "authority", "directness", "slot_coverage", "freshness", "traceability"}
        if set(self.quality_weights) != expected or abs(sum(self.quality_weights.values()) - 1.0) > 1e-6:
            raise ValueError("quality_weights must contain all six dimensions and sum to 1")
        doris_values = (self.doris_host, self.doris_username, self.doris_password)
        if any(doris_values) and not all(doris_values):
            raise ValueError("doris_host, doris_username and doris_password must be configured together")
        if self.structured_min_tool_calls > self.structured_max_tool_calls:
            raise ValueError("structured_min_tool_calls cannot exceed structured_max_tool_calls")
        return self

    @classmethod
    def from_env(cls, **overrides: Any) -> EvidenceRetrievalConfig:
        # HypoArgus owns configuration in the repository root .env.  Keep the
        # package-local file as a standalone SearchAgent fallback only.
        try:
            from pathlib import Path

            from dotenv import load_dotenv
            if os.environ.get("SEARCH_AGENT_DISABLE_DOTENV") != "1":
                load_dotenv(Path.cwd() / ".env", override=False)
                load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        except Exception:
            pass
        names = {
            "volcano_base_url": "VOLCANO_SEARCH_BASE_URL", "volcano_api_key": "VOLCANO_SEARCH_API_KEY",
            "volcano_search_path": "VOLCANO_SEARCH_PATH",
            "bisheng_base_url": "BISHENG_BASE_URL", "bisheng_retrieve_base_url": "BISHENG_RETRIEVE_BASE_URL",
            "bisheng_token": "BISHENG_TOKEN",
            "doris_host": "DORIS_HOST", "doris_port": "DORIS_PORT",
            "doris_database": "DORIS_DATABASE", "doris_username": "DORIS_USERNAME",
            "doris_password": "DORIS_PASSWORD",
            "judge_model": "JUDGE_MODEL",
            "shadow_mode": "SEARCH_AGENT_SHADOW_MODE",
            "public_knowledge_ids": "PUBLIC_KNOWLEDGE_IDS", "web_allowed_domains": "WEB_ALLOWED_DOMAINS",
        }
        env_values: dict[str, Any] = {}
        for field in cls.model_fields:
            generic = f"EVIDENCE_RETRIEVAL_{field.upper()}"
            if os.environ.get(generic) is not None:
                env_values[field] = os.environ[generic]
        env_values.update({field: os.environ[name] for field, name in names.items() if os.environ.get(name)})
        if isinstance(env_values.get("quality_weights"), str):
            env_values["quality_weights"] = json.loads(env_values["quality_weights"])
        env_values.update(overrides)
        return cls(**env_values)
