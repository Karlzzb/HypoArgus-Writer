"""Deprecated compatibility shim. Use search_agent_graph instead."""
from .search_agent_graph import build_search_agent_graph as build_search_agent_v11_graph

__all__ = ["build_search_agent_v11_graph"]
