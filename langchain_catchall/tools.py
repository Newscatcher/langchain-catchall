"""LangChain Tool integration for CatchAll API.

This module provides a simplified toolkit pattern for accessing CatchAll.
It exposes two distinct tools:
1. `catchall_search`: For finding NEW data.
2. `catchall_analyze`: For analyzing EXISTING data.
"""

import time
import sys
import re
from typing import Optional, Type, List, Tuple
from datetime import datetime, timedelta
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
        base_url: str = "https://catchall.newscatcherapi.com",
        poll_interval: int = 30,
        max_wait_time: int = 2400,
        verbose: bool = True,
        transform_query: bool = True,
    ):
        self.api_key = api_key
        self.llm = llm
        self.max_results = max_results
        self.default_date_range_days = default_date_range_days
        self.verbose = verbose
        self.transform_query = transform_query

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
                func=self.search_data,
                name="catchall_search_data",
                description=(
                    "Use this tool ONLY to find NEW articles when the user explicitly requests a search. "
                    "After searching, report the results count and STOP. Do not analyze automatically. "
                    "Example: 'Find all articles about X' → Use this tool, then STOP. "
                    "Input should be a complete search query like 'Find all articles about companies opening offices'. "
                    "Include 'Find all articles about' and specify the topic with dates if needed. "
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

    def search_data(self, query: str) -> str:
        """Perform a new search on CatchAll."""
        if self.transform_query:
            if self._is_query_good(query):
                catchall_query = query
            else:
                catchall_query = self._transform_query(query)
        else:
            catchall_query = query

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
            completed_step, failed_step = evaluate_job_steps(status_info)
            status = status_info.status

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

        cached_records = min(int(result.valid_records), self.max_results)
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
            max_records=self.max_results
        )

        return answer

    def _is_query_good(self, query: str) -> bool:
        """Check if query is already well-formed."""
        query_lower = query.lower()

        has_good_start = any(query_lower.startswith(kw) for kw in ["find all", "find articles", "search", "catch all", "find"])
        has_date_range = "between" in query_lower

        if has_good_start and has_date_range:
            return True

        if has_date_range:
            between_pos = query_lower.find("between")
            if between_pos > 0:
                topic_part = query_lower[:between_pos].strip()
                return len(topic_part) > 4  # At least 5 characters for a meaningful topic

        return False

    def _transform_query(self, user_query: str) -> str:
        """Transform user question into proper CatchAll query with dates."""
        now = datetime.now().astimezone()
        relative_range = self._extract_relative_time_range(user_query, now)
        start_str = end_str = None
        if relative_range:
            start_dt, end_dt = relative_range
            start_str = self._format_datetime_with_minutes(start_dt)
            end_str = self._format_datetime_with_minutes(end_dt)

        prompt = f"""Transform this user question into a specific CatchAll search query with explicit dates.

User question: "{user_query}"
Current timestamp: {self._format_datetime_with_minutes(now)}

Rules:
1. Start with "Find all articles about..."
2. Add date range "between [Date1] and [Date2]"
3. Default range (if not specified): {self.default_date_range_days} days ago to today.
4. If the user requests a range shorter than a full day (e.g., last N hours/minutes), retain hours and minutes exactly as provided below.
"""

        if start_str and end_str:
            prompt += (
                "\nUse this exact window for the date range component:\n"
                f"- Start: {start_str}\n"
                f"- End: {end_str}\n"
                "Do not round to whole days.\n"
            )

        prompt += """
Example: "AI news" -> "Find all articles about AI technology developments between November 5 and November 19, 2025"

Return ONLY the transformed query string."""

        response = self.llm.invoke(prompt)
        transformed = str(response.content if hasattr(response, 'content') else response).strip()
        transformed = transformed.strip('"').strip("'")

        if start_str and end_str:
            transformed = self._apply_precise_time_range(transformed, start_str, end_str)

        return transformed

    def _format_datetime_with_minutes(self, value: datetime) -> str:
        """Format datetime with minute precision and timezone label."""
        localized = value.astimezone()
        formatted = localized.strftime("%B %d, %Y %H:%M %Z").strip()
        return formatted

    def _extract_relative_time_range(
        self, text: str, reference_time: datetime
    ) -> Optional[Tuple[datetime, datetime]]:
        """Return explicit start/end if query mentions last/past N hours/minutes."""
        pattern = re.compile(
            r"(last|past)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|hour|hr|minutes?|mins?|minute|min)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return None

        value = float(match.group(2))
        unit = match.group(3).lower()
        if value <= 0:
            return None

        if "hour" in unit or "hr" in unit:
            delta = timedelta(hours=value)
        else:
            delta = timedelta(minutes=value)

        end_time = reference_time
        start_time = end_time - delta
        return start_time, end_time

    def _apply_precise_time_range(self, query: str, start: str, end: str) -> str:
        """Overwrite/append the date range in the transformed query."""
        lower_query = query.lower()
        between_idx = lower_query.find("between")

        if between_idx != -1:
            and_idx = lower_query.find(" and ", between_idx)
            if and_idx != -1:
                end_idx = self._find_clause_end(query, and_idx + 5)
                prefix = query[:between_idx]
                suffix = query[end_idx:]
                return f"{prefix}between {start} and {end}{suffix}"

        query = query.rstrip(". ")
        spacer = "" if query.endswith("between") else " "
        return f"{query}{spacer}between {start} and {end}"

    @staticmethod
    def _find_clause_end(text: str, start_idx: int) -> int:
        """Find the natural end of the date clause."""
        for idx in range(start_idx, len(text)):
            if text[idx] in ".;!?":
                return idx
        return len(text)

    def _format_search_results(self, result: PullJobResponseDto) -> str:
        """Format initial search results summary."""
        output = [f"Found {result.valid_records} records (Showing top {self.max_results}).\n"]

        for i, record in enumerate(result.all_records[:self.max_results], 1):
            output.append(f"{i}. {record.record_title}")
            if record.enrichment:
                details = ", ".join(f"{k}: {v}" for k, v in record.enrichment.items() if k != "record_title")
                if details:
                    output.append(f"   ({details})")

        output.append("\n✅ Data successfully cached!")
        output.append("\n⚠️ IMPORTANT: Report these results to the user and STOP.")
        output.append("WAIT for the user's next question. Do NOT automatically analyze or summarize.")
        output.append("If the user asks a follow-up question, you can use 'catchall_analyze_data' to filter, group, or summarize this data.")
        return "\n".join(output)


__all__ = ["CatchAllTools"]
