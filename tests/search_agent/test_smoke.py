"""search_agent fork 冒烟测试：包可导入、三条通道代码在、检索图可离线构建。

只断言外部可见的构建行为，不触发任何网络调用。
"""

import search_agent
from search_agent.evidence_retrieval.config import EvidenceRetrievalConfig
from search_agent.evidence_retrieval.dependencies import EvidenceRetrievalDependencies


def test_public_api_importable() -> None:
    for name in search_agent.__all__:
        assert getattr(search_agent, name) is not None


def test_three_channel_clients_importable() -> None:
    from search_agent.evidence_retrieval.providers.bisheng_retrieve import BishengRetrieveClient
    from search_agent.evidence_retrieval.providers.doris_query import DorisQueryClient
    from search_agent.evidence_retrieval.providers.volcano_web import VolcanoWebSearchClient

    config = EvidenceRetrievalConfig()
    deps = EvidenceRetrievalDependencies.defaults(config)
    assert isinstance(deps.web_search, VolcanoWebSearchClient)
    assert isinstance(deps.kb_client, BishengRetrieveClient)
    assert isinstance(deps.structured_client, DorisQueryClient)


def test_modules_resolve_inside_this_repo() -> None:
    # 开发环境与源项目共用 conda 环境，源项目 editable 安装的同名包会为缺失子模块兜底；
    # 这里断言实际加载的模块都来自本仓库，防止悄悄用到源仓库代码。
    from pathlib import Path

    from search_agent.evidence_retrieval import batch_graph
    from search_agent.evidence_retrieval.providers import volcano_web

    pkg_root = Path(__file__).resolve().parents[2] / "src" / "search_agent"
    for module in (search_agent, batch_graph, volcano_web):
        assert module.__file__ is not None
        assert Path(module.__file__).is_relative_to(pkg_root)


def test_legacy_providers_pruned_from_repo() -> None:
    from pathlib import Path

    pkg_root = Path(search_agent.__file__).resolve().parent
    for pruned in ("providers", "edre", "subgraph.py", "retrieval.py", "consolidation.py", "llm.py"):
        assert not (pkg_root / pruned).exists()


def test_graphs_buildable_offline() -> None:
    from search_agent import build_evidence_retrieval_graph, build_search_agent_graph

    config = EvidenceRetrievalConfig()
    deps = EvidenceRetrievalDependencies.defaults(config)
    batch_graph = build_evidence_retrieval_graph(config, deps)
    assert batch_graph is not None
    public_graph = build_search_agent_graph(config, deps, batch_graph=batch_graph)
    assert public_graph is not None
