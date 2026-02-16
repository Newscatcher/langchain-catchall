"""LangChain Tool integration for CatchAll API.

This module provides a simplified toolkit pattern for accessing CatchAll.
It exposes two distinct tools:
1. `catchall_search`: For finding NEW data.
2. `catchall_analyze`: For analyzing EXISTING data.
"""

import time
import sys
import re
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.language_models import BaseLanguageModel

from langchain_catchall.client import CatchAllClient, PullJobResponseDto
from langchain_catchall.helpers import query_with_llm, evaluate_job_steps


class CatchAllSearchInput(BaseModel):
    """Input for searching NEW data."""
    query: str = Field(
        description="What you want to find. Example: 'Find articles about AI developments in US'"
    )
    limit: Optional[int] = Field(
        default=None,
        description="Optional limit for number of results to retrieve"
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
        limit: int = 100,
        default_date_range_days: int = 5,
        base_url: str = "https://catchall.newscatcherapi.com",
        poll_interval: int = 30,
        max_wait_time: int = 2400,
        verbose: bool = True,
        initialize_query: bool = True,
    ):
        self.api_key = api_key
        self.llm = llm
        self.limit = limit
        self.default_date_range_days = default_date_range_days
        self.verbose = verbose
        self.initialize_query = initialize_query

        self._client = CatchAllClient(
            api_key=api_key,
            base_url=base_url,
            poll_interval=poll_interval,
            max_wait_time=max_wait_time,
        )

        self._cached_result: Optional[PullJobResponseDto] = None
        self._cached_job_id: Optional[str] = None
        self._cached_limit: Optional[int] = None

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
                func=self.search_data,
                name="catchall_search_data",
                description=(
                    "Use this tool ONLY to find NEW articles when the user explicitly requests a search. "
                    "After searching, report the results count and STOP. Do not analyze automatically. "
                    "Example: 'Find all X' → Use this tool, then STOP. "
                    "Input should be a complete search query like 'Find all companies opening offices'. "
                    "Include 'Find all' and specify the topic with dates if needed. "
                    "WARNING: This triggers a new 15-minute search. "
                    "NEVER use this for filtering or narrowing down existing results."
                ),
                args_schema=CatchAllSearchInput,
            ),
            StructuredTool.from_function(
                func=self.analyze_data,
                name="catchall_analyze_data",
                description=(
                    "Use this tool ONLY when the user asks a follow-up question about EXISTING search results. "
                    "DO NOT use this immediately after a search unless explicitly requested by the user. "
                    "Wait for the user to ask a follow-up question like: "
                    "'Show only Florida', 'What are the trends?', 'Summarize the findings'. "
                    "Capabilities: "
                    "1. Filtering & Sorting ('Show only Florida', 'Sort by date') "
                    "2. Aggregation ('Group by company', 'Count by state') "
                    "3. QA ('What are the top trends?', 'Summarize key findings') "
                    "This tool is for filtering, sorting, aggregation, and Q&A on cached data."
                ),
                args_schema=CatchAllAnalysisInput,
            ),
        ]

    def search_data(self, query: str, limit: Optional[int] = None) -> str:
        """Perform a new search on CatchAll."""
        explicit_limit = self._extract_limit_from_query(query)
        if limit is not None:
            explicit_limit = limit

        if explicit_limit is not None and explicit_limit <= 0:
            return "ERROR: Limit must be a positive integer."

        catchall_query = query

        self._log(f"Starting NEW Search for: {catchall_query}")

        effective_limit: Optional[int]
        if explicit_limit is not None:
            effective_limit = int(explicit_limit)
        elif self._has_exhaustive_intent(query):
            effective_limit = None
        else:
            effective_limit = self.limit

        if effective_limit is not None and effective_limit > self.limit:
            self.limit = effective_limit

        initialize_payload: Optional[Dict[str, Any]] = None
        if self.initialize_query:
            self._log("Initializing job...")
            initialize_payload = self._extract_initialize_payload(
                self._client.initialize_job(query=catchall_query)
            )

        self._log("Submitting job...")
        job_id = self._client.submit_job(
            query=catchall_query,
            validators=initialize_payload.get("validators") if initialize_payload else None,
            enrichments=initialize_payload.get("enrichments") if initialize_payload else None,
            start_date=initialize_payload.get("start_date") if initialize_payload else None,
            end_date=initialize_payload.get("end_date") if initialize_payload else None,
            limit=effective_limit,
        )
        self._log(f"Job submitted. Job ID: {job_id}")

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > self._client.max_wait_time:
                raise TimeoutError(f"Job {job_id} timed out")

            status_info = self._client.get_status(job_id)
            completed_step, failed_step = evaluate_job_steps(status_info)
            status = self._get_display_status(status_info, completed_step=completed_step, failed_step=failed_step)

            if self.verbose:
                time_str = f"{int(elapsed)}s"
                sys.stdout.write(f"\r[CatchAll] Search performing: {job_id}, Status: {status}, Time: {time_str}")
                sys.stdout.flush()

            if completed_step:
                if self.verbose:
                    sys.stdout.write("\n")
                break
            elif failed_step:
                if self.verbose:
                    sys.stdout.write("\n")
                return f"Search failed for job {job_id}"

            time.sleep(self._client.poll_interval)

        self._log("Retrieving results...")
        result = self._client.get_all_results(job_id)

        self._cached_result = result
        self._cached_job_id = job_id
        self._cached_limit = effective_limit

        cached_records = min(int(result.valid_records), self.limit)
        self._log(f"Cached {cached_records} out of {result.valid_records} results")

        if not result.all_records:
            return f"No results found for query: {query}"

        return self._format_search_results(result)


    def analyze_data(self, question: str) -> str:
        """Analyze the cached search results."""
        self._log(f"Analyzing cache for: '{question}'")

        if self._cached_result is None:
            return (
                "ERROR: No data available to analyze yet. "
                "Please call 'catchall_search_data' first to find data."
            )

        answer = query_with_llm(
            result=self._cached_result,
            question=question,
            llm=self.llm,
            max_records=self.limit
        )

        return answer

    @staticmethod
    def _extract_initialize_payload(payload: Any) -> Dict[str, Any]:
        """Normalize initialize response into a simple dict."""
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return payload
        data: Dict[str, Any] = {}
        for key in ("validators", "enrichments", "start_date", "end_date"):
            value = getattr(payload, key, None)
            if value is not None:
                data[key] = value
        return data

    @staticmethod
    def _has_exhaustive_intent(query: str) -> bool:
        """Detect if the user requests exhaustive results."""
        return bool(
            re.search(
                r"\b(all|every|complete list|comprehensive|catch all)\b",
                query.lower(),
            )
        )

    @staticmethod
    def _extract_limit_from_query(query: str) -> Optional[int]:
        """Extract an explicit limit from the query text."""
        patterns = [
            r"\btop\s+(\d+)\b",
            r"\bfirst\s+(\d+)\b",
            r"\blimit(?:\s+to)?\s+(\d+)\b",
            r"\bshow\s+(\d+)\s+results?\b",
            r"\b(\d+)\s+results?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        return None


    def _format_search_results(self, result: PullJobResponseDto) -> str:
        """Format initial search results summary."""
        total_records = int(getattr(result, "valid_records", 0) or 0)
        displayed_records = min(self.limit, len(result.all_records or []))

        output = [f"Found {total_records} records (Showing top {displayed_records}).\n"]

        for i, record in enumerate(result.all_records[:self.limit], 1):
            output.append(f"{i}. {record.record_title}")
            if record.enrichment:
                details = ", ".join(f"{k}: {v}" for k, v in record.enrichment.items() if k != "record_title")
                if details:
                    output.append(f"   ({details})")

        output.append("\nData successfully cached!")
        output.append("\nIMPORTANT: Report these results to the user and STOP.")
        output.append("WAIT for the user's next question. Do NOT automatically analyze or summarize.")
        output.append("If the user asks a follow-up question, you can use 'catchall_analyze_data' to filter, group, or summarize this data.")
        return "\n".join(output)

    @staticmethod
    def _get_display_status(status_info, completed_step=None, failed_step=None) -> str:
        """
        Get a user-friendly "current" job status for progress output.
        """
        overall_status = str(getattr(status_info, "status", "") or "").strip().lower()
        if overall_status in {"completed", "failed"}:
            return overall_status

        if failed_step is not None:
            return "failed"
        if completed_step is not None:
            return "completed"

        steps = list(getattr(status_info, "steps", None) or [])
        if not steps:
            return overall_status or "submitted"

        try:
            steps_sorted = sorted(steps, key=lambda s: getattr(s, "order", 0) or 0)
        except Exception:
            steps_sorted = steps

        for step in steps_sorted:
            step_status = str(getattr(step, "status", "") or "").strip().lower()
            step_completed = bool(getattr(step, "completed", False))

            if step_status in {"completed", "failed"}:
                continue
            if not step_completed:
                return step_status or (overall_status or "submitted")

        return overall_status or "completed"


__all__ = ["CatchAllTools"]
