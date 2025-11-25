"""Helper functions for using CatchAll results with LLMs.

This module provides simple utility functions to format CatchAll results
and query them with LLMs, following LangChain's invoke pattern.
"""

from typing import Optional, List
from langchain_core.language_models import BaseLanguageModel
from newscatcher_catchall.types import PullJobResponseDto, Record


def format_results_for_llm(
    result: PullJobResponseDto,
    max_records: Optional[int] = None,
    include_citations: bool = True,
) -> str:
    """Format CatchAll results as a context string for LLM.
    
    This creates a readable text representation of your CatchAll results
    that can be sent to any LLM (GPT-4, Claude, etc.).
    
    Args:
        result: PullJobResponseDto from CatchAll search
        max_records: Maximum number of records to include (None = all)
                    Recommended: None for GPT-4-turbo/Claude (128k+ context)
                               50-100 for smaller context models
        include_citations: Whether to include source article info
        
    Returns:
        Formatted string with all record data
        
    Example:
        >>> result = client.search("Tech company earnings")
        >>> context = format_results_for_llm(result, max_records=20)
        >>> # Now send context to LLM
        
    """
    if not result.all_records:
        return "No records found in the search results."

    records_to_format = (
        result.all_records[:max_records] if max_records else result.all_records
    )

    context_parts = [
        "# Search Results",
        f"Query: {result.query or 'N/A'}",
        f"Total Records Available: {result.valid_records}",
        f"Records Included in Analysis: {len(records_to_format)}",
        "---"
    ]
    
    if result.valid_records > len(records_to_format):
        context_parts.append(
            f"⚠️ WARNING: This is a partial view! You are only seeing the top {len(records_to_format)} "
            f"out of {result.valid_records} records. "
            "Your answers must reflect that this is a subset."
        )
        context_parts.append("---")

    for i, record in enumerate(records_to_format, 1):
        context_parts.append(f"\n## Record {i}: {record.record_title}")

        context_parts.append("\n### Extracted Data:")
        for key, value in record.enrichment.items():
            if key != "record_title":
                context_parts.append(f"- **{key}**: {value}")

        if include_citations and record.citations:
            context_parts.append(f"\n### Sources: {len(record.citations)} articles")
            for cite in record.citations[:2]:
                context_parts.append(
                    f"- {cite.title} "
                    f"({cite.published_date})"
                )
        
        context_parts.append("")
    
    return "\n".join(context_parts)


def query_with_llm(
    result: PullJobResponseDto,
    question: str,
    llm: BaseLanguageModel,
    max_records: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Query CatchAll results with an LLM using LangChain's invoke pattern.
    
    This is the simple, recommended way to analyze CatchAll results with an LLM.
    It follows LangChain's standard invoke() pattern from their docs.
    
    Args:
        result: PullJobResponseDto from CatchAll search (can reuse same result!)
        question: Question to ask about the data
        llm: Any LangChain LLM (ChatOpenAI, ChatAnthropic, etc.)
        max_records: Limit context size (None = all records)
        system_prompt: Optional system prompt (default: analysis prompt)
        
    Returns:
        LLM's answer as a string
        
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_catchall import CatchAllClient, query_with_llm
        >>> 
        >>> # Do search once
        >>> client = CatchAllClient(api_key="...")
        >>> result = client.search("Q4 tech earnings")
        >>> 
        >>> # Save result, query many times
        >>> llm = ChatOpenAI(model="gpt-4-turbo")
        >>> 
        >>> answer1 = query_with_llm(result, "Highest revenue?", llm)
        >>> answer2 = query_with_llm(result, "Compare margins", llm)
        >>> answer3 = query_with_llm(result, "Key trends?", llm)
        
    Note:
        You can reuse the same 'result' object for multiple questions!
        No need to re-fetch from CatchAll - just keep querying the same result.
    """
    context = format_results_for_llm(result, max_records=max_records)

    if system_prompt is None:
        system_prompt = (
            "You are an AI assistant analyzing news data. "
            "Answer questions based ONLY on the provided search results. "
            "If the user asks for counts or top lists (e.g. 'top 10 states'), "
            "you MUST count them manually from the provided text context. "
            "If the context is truncated (check 'Showing X of Y records'), "
            "explicitly state: 'Based on the top [X] records provided...'. "
        "Do not halluncinate numbers not present in the context."
    )

    prompt = f"""{system_prompt}

{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    if hasattr(response, 'content'):
        return response.content
    return str(response)


def format_record(record: Record, include_citations: bool = True) -> str:
    """Format a single record for display or LLM context.
    
    Useful if you want to work with individual records.
    
    Args:
        record: Single Record from CatchAll results
        include_citations: Whether to include source info
        
    Returns:
        Formatted string representation of the record
        
    Example:
        >>> for record in result.all_records[:5]:
        ...     print(format_record(record))
    """
    parts = [f"Title: {record.record_title}", "\nData:"]
    
    for key, value in record.enrichment.items():
        if key != "record_title":
            parts.append(f"  - {key}: {value}")
    
    if include_citations and record.citations:
        parts.append(f"\nSources: {len(record.citations)} articles")
        if record.citations:
            parts.append(f"  First: {record.citations[0].title}")
    
    return "\n".join(parts)


__all__ = [
    "format_results_for_llm",
    "query_with_llm",
    "format_record",
]

