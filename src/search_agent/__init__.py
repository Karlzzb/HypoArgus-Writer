"""检索引擎包（SearchAgent V12，自源项目 HypoArgus 一次性 fork，归本项目所有）。

保留三条检索通道：火山联网搜索、Bisheng 知识库、Doris 结构化数据。
公开入口为 ``SearchAgentRuntime`` / ``ainvoke_search_agent``（见 ``api.py``），
底层图构建为 ``build_evidence_retrieval_graph`` / ``build_search_agent_graph``。

包级 ``.env``（``LLM_*`` / 通道类变量 / ``LANGFUSE_*``）在导入时自动加载；
进程环境变量始终优先。
"""

from .env import load_env

load_env()

from .api import (  # noqa: E402
    SearchAgentClosedError,
    SearchAgentConfigurationError,
    SearchAgentContractError,
    SearchAgentRuntime,
    ainvoke_search_agent,
)
from .evidence_retrieval import (  # noqa: E402
    AtomicClaim,
    AtomicClaimGroup,
    ClaimLogicOperator,
    EvidenceRetrievalConfig,
    EvidenceRetrievalDependencies,
    NumericRelationVerifier,
    apply_claim_logic,
    atomize_claim,
    build_evidence_retrieval_graph,
    build_search_agent_graph,
    normalize_reverse_hypothesis,
)
from .evidence_retrieval.public_contracts import (  # noqa: E402
    SearchAgentInputState,
    SearchAgentOutputState,
)
from .tracing import get_langfuse_callback  # noqa: E402

__all__ = [
    "AtomicClaim",
    "AtomicClaimGroup",
    "ClaimLogicOperator",
    "EvidenceRetrievalConfig",
    "EvidenceRetrievalDependencies",
    "NumericRelationVerifier",
    "SearchAgentClosedError",
    "SearchAgentConfigurationError",
    "SearchAgentContractError",
    "SearchAgentInputState",
    "SearchAgentOutputState",
    "SearchAgentRuntime",
    "ainvoke_search_agent",
    "apply_claim_logic",
    "atomize_claim",
    "build_evidence_retrieval_graph",
    "build_search_agent_graph",
    "get_langfuse_callback",
    "normalize_reverse_hypothesis",
]
