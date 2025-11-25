"""LangChain Tool integration for CatchAll API.

This module provides a simplified toolkit pattern for accessing CatchAll.
It exposes two distinct tools:
1. `catchall_search`: For finding NEW data.
2. `catchall_analyze`: For analyzing EXISTING data.
"""

import time
import sys
from typing import Optional, Type, List
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.language_models import BaseLanguageModel

from langchain_catchall.client import CatchAllClient, PullJobResponseDto
from langchain_catchall.helpers import query_with_llm


class CatchAllSearchInput(BaseModel):
    """Input for searching NEW data."""
    query: str = Field(
        description="What you want to find. Example: 'Find articles about AI developments in US'"
    )


class CatchAllAnalysisInput(BaseModel):
    """Input for analyzing EXISTING data."""
    question: str = Field(
        description="Analytical question about the cached data. Example: 'Summarize key findings'"
    )


class CatchAllTools:
    """Manages CatchAll API interaction and shared state (cache)."""

    def __init__(
        self,
        api_key: str,
        llm: BaseLanguageModel,
        max_results: int = 100,
        default_date_range_days: int = 14,
        base_url: Optional[str] = None,
        poll_interval: int = 30,
        max_wait_time: int = 1800,
        verbose: bool = True,
    ):
        self.api_key = api_key
        self.llm = llm
        self.max_results = max_results
        self.default_date_range_days = default_date_range_days
        self.verbose = verbose

        self._client = CatchAllClient(
            api_key=api_key,
            base_url=base_url,
            poll_interval=poll_interval,
            max_wait_time=max_wait_time,
        )

        self._cached_result: Optional[PullJobResponseDto] = None

    def _log(self, message: str, end: str = "\n"):
        """Helper to print logs if verbose is True."""
        if self.verbose:
            print(f"[CatchAll] {message}", end=end)
            if end != "\n":
                sys.stdout.flush()

    def get_tools(self) -> List[BaseTool]:
        """Return the list of tools for the Agent."""
        return [
            StructuredTool.from_function(
                func=self.search_news,
                name="catchall_search_news",
                description=(
                    "Use this tool to find NEW articles. "
                    "Input should be a broad topic like 'Find articles about companies opening offices'. "
                    "WARNING: This triggers a new 15-minute search. "
                    "NEVER use this for filtering or narrowing down existing results."
                ),
                args_schema=CatchAllSearchInput,
            ),
            StructuredTool.from_function(
                func=self.analyze_news,
                name="catchall_analyze_news",
                description=(
                    "Use this tool for ANY follow-up interaction with the search results. "
                    "Capabilities: "
                    "1. Filtering & Sorting ('Show only Florida', 'Sort by date') "
                    "2. Aggregation ('Group by company', 'Count by state') "
                    "3. QA ('What are the top trends?', 'Summarize key findings') "
                    "ALWAYS use this tool if you already have data. NEVER search again."
                ),
                args_schema=CatchAllAnalysisInput,
            ),
        ]

    def search_news(self, query: str) -> str:
        """Perform a new search on CatchAll."""
        if self._is_query_good(query):
            catchall_query = query
        else:
            catchall_query = self._transform_query(query)

        self._log(f"Starting NEW Search for: {catchall_query}")

        self._log("Submitting job...")
        job_id = self._client.submit_job(query=catchall_query)
        self._log(f"Job submitted. Job ID: {job_id}")

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > self._client.max_wait_time:
                raise TimeoutError(f"Job {job_id} timed out")
                
            status_info = self._client.get_status(job_id)
            status = status_info.status

            if self.verbose:
                time_str = f"{int(elapsed)}s"
                sys.stdout.write(f"\r[CatchAll] Search performing: {job_id}, Status: {status}, Time: {time_str}")
                sys.stdout.flush()
            
            if status == "job_completed":
                if self.verbose:
                    sys.stdout.write("\n")
                break
            elif status == "failed":
                if self.verbose:
                    sys.stdout.write("\n")
                return f"Search failed for job {job_id}"
                
            time.sleep(self._client.poll_interval)

        self._log("Retrieving results...")
        result = self._client.get_all_results(job_id)

        self._cached_result = result

        self._log(f"Cached {self.max_results} out of {result.valid_records} results")

        if not result.all_records:
            return f"No results found for query: {query}"
            
        return self._format_search_results(result)

    def analyze_news(self, question: str) -> str:
        """Analyze the cached search results."""
        self._log(f"Analyzing cache for: '{question}'")
        
        if self._cached_result is None:
            return (
                "ERROR: No news data available to analyze yet. "
                "Please call 'catchall_search_news' first to find articles."
            )

        answer = query_with_llm(
            result=self._cached_result,
            question=question,
            llm=self.llm,
            max_records=self.max_results 
        )
        
        return answer

    def _is_query_good(self, query: str) -> bool:
        """Check if query is already well-formed."""
        query_lower = query.lower()
        has_good_start = any(query_lower.startswith(kw) for kw in ["find all articles", "search for articles"])
        has_date_range = "between" in query_lower and ("november" in query_lower or "20" in query)
        return has_good_start and has_date_range

    def _transform_query(self, user_query: str) -> str:
        """Transform user question into proper CatchAll query with dates."""
        today = datetime.now()
        prompt = f"""Transform this user question into a specific CatchAll search query with explicit dates.

User question: "{user_query}"
Today's date: {today.strftime("%B %d, %Y")}

Rules:
1. Start with "Find all articles about..."
2. Add date range "between [Date1] and [Date2]"
3. Default range (if not specified): {self.default_date_range_days} days ago to today.

Example: "AI news" -> "Find all articles about AI technology developments between November 5 and November 19, 2024"

Return ONLY the transformed query string."""
        
        response = self.llm.invoke(prompt)
        transformed = str(response.content if hasattr(response, 'content') else response).strip()
        return transformed.strip('"').strip("'")

    def _format_search_results(self, result: PullJobResponseDto) -> str:
        """Format initial search results summary."""
        output = [f"Found {result.valid_records} records (Showing top {self.max_results}).\n"]
        
        for i, record in enumerate(result.all_records[:self.max_results], 1):
            output.append(f"{i}. {record.record_title}")
            if record.enrichment:
                details = ", ".join(f"{k}: {v}" for k, v in record.enrichment.items() if k != "record_title")
                if details:
                    output.append(f"   ({details})")
        
        output.append("\nData successfully cached!")
        output.append("You can now use 'catchall_analyze_news' to filter, group, or summarize this data.")
        return "\n".join(output)


__all__ = ["CatchAllTools"]
