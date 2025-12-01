"""LangChain integration for NewsCatcher CatchAll API.

This package provides LangChain-compatible tools and retrievers for the
CatchAll API, enabling natural language web search in AI applications.

This integration wraps the official newscatcher-catchall SDK and adds
LangChain-specific functionality for agents and RAG patterns.
"""

from langchain_catchall.client import (
    CatchAllClient,
    AsyncCatchAllClient,
    Record,
    PullJobResponseDto,
)
from langchain_catchall.tools import CatchAllTools
from langchain_catchall.helpers import format_results_for_llm, query_with_llm
from langchain_catchall.prompts import CATCHALL_AGENT_PROMPT

__version__ = "0.1.0"
__all__ = [
    "CatchAllClient",
    "AsyncCatchAllClient",
    "CatchAllTools",
    "Record",
    "PullJobResponseDto",
    "format_results_for_llm",
    "query_with_llm",
    "CATCHALL_AGENT_PROMPT",
]
