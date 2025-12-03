"""Core client for interacting with the CatchAll API.

This module provides a LangChain-friendly wrapper around the official
newscatcher_catchall SDK, adding convenience methods for polling and
high-level web search operations.
"""

import time
from typing import Any, Dict, List, Optional

from newscatcher_catchall import CatchAllApi, AsyncCatchAllApi
from newscatcher_catchall.types import (
    Record,
    PullJobResponseDto,
    StatusResponseDto,
    ListUserJobsResponseDto,
)


class CatchAllClient:
    """LangChain-friendly wrapper for the CatchAll API.

    This client wraps the official newscatcher_catchall SDK and adds
    convenience methods for:
    - Automatic polling until job completion
    - High-level search method that handles the full workflow
    - Pagination handling

    Args:
        api_key: Your CatchAll API key
        base_url: API base URL (default: "https://catchall.newscatcherapi.xyz")
        poll_interval: Seconds to wait between status checks (default: 30)
        max_wait_time: Maximum seconds to wait for job completion (default: 1200)
        timeout: HTTP request timeout in seconds (default: 60)

    Example:
        >>> client = CatchAllClient(api_key="your_api_key")
        >>> result = client.search("Tech company earnings this quarter")
        >>> for record in result.all_records:
        ...     print(record.record_title)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://catchall.newscatcherapi.xyz",
        poll_interval: int = 30,
        max_wait_time: int = 1200,
        timeout: float = 60.0,
    ):
        """Initialize the CatchAll client."""
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time

        self._client = CatchAllApi(api_key=api_key, base_url=base_url, timeout=timeout)
    
    def submit_job(
        self,
        query: str,
        context: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> str:
        """Submit a new CatchAll job.
        
        Args:
            query: Natural language question describing what to find
            context: Additional context to focus the search
            schema: Template string to guide record formatting (e.g., "[COMPANY] earned [REVENUE]")
            
        Returns:
            Job ID for tracking the job
            
        Example:
            >>> job_id = client.submit_job(
            ...     query="Tech company earnings this quarter",
            ...     context="Focus on revenue and profit margins"
            ... )
        """
        response = self._client.jobs.create_job(
            query=query,
            context=context,
            schema=schema,
        )
        return response.job_id
    
    def get_status(self, job_id: str) -> StatusResponseDto:
        """Get the current status of a job.
        
        Args:
            job_id: The job identifier
            
        Returns:
            Status response with job_id and status fields
            
        Example:
            >>> status = client.get_status(job_id)
            >>> print(status.status)  # e.g., "data_fetched"
        """
        return self._client.jobs.get_job_status(job_id)
    
    def wait_for_completion(self, job_id: str) -> None:
        """Poll job status until completion or timeout.
        
        Args:
            job_id: The job identifier
            
        Raises:
            TimeoutError: If job doesn't complete within max_wait_time
            
        Example:
            >>> client.wait_for_completion(job_id)
        """
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.max_wait_time:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {self.max_wait_time} seconds"
                )
            
            status_info = self.get_status(job_id)
            
            if status_info.status == "job_completed":
                return
            
            time.sleep(self.poll_interval)
    
    def get_results(
        self, 
        job_id: str,
        page: int = 1,
        page_size: int = 100,
    ) -> PullJobResponseDto:
        """Retrieve results for a completed job.
        
        Args:
            job_id: The job identifier
            page: Page number to retrieve (default: 1)
            page_size: Number of records per page (default: 100, max: 1000)
            
        Returns:
            PullJobResponseDto containing all extracted records
            
        Example:
            >>> result = client.get_results(job_id)
            >>> print(f"Found {result.valid_records} records")
        """
        return self._client.jobs.get_job_results(
            job_id=job_id,
            page=page,
            page_size=page_size,
        )
    
    def get_all_results(self, job_id: str) -> PullJobResponseDto:
        """Retrieve all results for a job across all pages.
        
        Args:
            job_id: The job identifier
            
        Returns:
            PullJobResponseDto with all records from all pages
            
        Example:
            >>> result = client.get_all_results(job_id)
        """
        first_page = self.get_results(job_id, page=1, page_size=1000)
        
        if first_page.total_pages == 1:
            return first_page

        all_records = list(first_page.all_records or [])
        for page in range(2, first_page.total_pages + 1):
            page_result = self.get_results(job_id, page=page, page_size=1000)
            if page_result.all_records:
                all_records.extend(page_result.all_records)

        result_dict = first_page.dict() if hasattr(first_page, 'dict') else first_page.model_dump()
        result_dict['all_records'] = all_records
        
        return PullJobResponseDto(**result_dict)
    
    def search(
        self,
        query: str,
        context: Optional[str] = None,
        schema: Optional[str] = None,
        wait: bool = True,
    ) -> PullJobResponseDto:
        """Submit a query and optionally wait for results.
        
        This is the main convenience method that combines submit, wait, and retrieve.
        
        Args:
            query: Natural language question
            context: Additional context to focus the search
            schema: Template string for record formatting
            wait: If True, wait for completion and return results. If False, return immediately.
            
        Returns:
            PullJobResponseDto if wait=True, otherwise empty result with just job_id
            
        Example:
            >>> result = client.search(
            ...     query="Tech company earnings this quarter",
            ...     context="Focus on revenue growth"
            ... )
            >>> for record in result.all_records:
            ...     print(record.record_title)
        """
        job_id = self.submit_job(query=query, context=context, schema=schema)
        
        if not wait:
            # Return minimal response with just job_id
            return PullJobResponseDto(
                job_id=job_id,
                status="pending",
                page=1,
                total_pages=0,
                page_size=100,
            )
        
        self.wait_for_completion(job_id)
        return self.get_all_results(job_id)
    
    def list_jobs(self) -> List[ListUserJobsResponseDto]:
        """List all jobs for the authenticated user.
        
        Returns:
            List of jobs with job_id and query fields
            
        Example:
            >>> jobs = client.list_jobs()
            >>> for job in jobs:
            ...     print(f"{job.job_id}: {job.query}")
        """
        return self._client.jobs.get_user_jobs()


class AsyncCatchAllClient:
    """Async version of CatchAllClient.

    This client provides the same interface as CatchAllClient but with
    async/await support for better performance in async applications.

    Args:
        api_key: Your CatchAll API key
        base_url: API base URL (default: "https://catchall.newscatcherapi.xyz")
        poll_interval: Seconds to wait between status checks (default: 30)
        max_wait_time: Maximum seconds to wait for job completion (default: 1200)
        timeout: HTTP request timeout in seconds (default: 60)

    Example:
        >>> import asyncio
        >>> async def main():
        ...     client = AsyncCatchAllClient(api_key="your_api_key")
        ...     result = await client.search("Find all articles about warehouse or distribution center openings")
        ...     for record in result.all_records:
        ...         print(record.record_title)
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://catchall.newscatcherapi.xyz",
        poll_interval: int = 30,
        max_wait_time: int = 1200,
        timeout: float = 60.0,
    ):
        """Initialize the async CatchAll client."""
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time

        self._client = AsyncCatchAllApi(api_key=api_key, base_url=base_url, timeout=timeout)
    
    async def submit_job(
        self,
        query: str,
        context: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> str:
        """Submit a new CatchAll job (async).
        
        Args:
            query: Natural language question describing what to find
            context: Additional context to focus the search
            schema: Template string to guide record formatting
            
        Returns:
            Job ID for tracking the job
        """
        response = await self._client.jobs.create_job(
            query=query,
            context=context,
            schema=schema,
        )
        return response.job_id
    
    async def get_status(self, job_id: str) -> StatusResponseDto:
        """Get the current status of a job (async).
        
        Args:
            job_id: The job identifier
            
        Returns:
            Status response with job_id and status fields
        """
        return await self._client.jobs.get_job_status(job_id)
    
    async def wait_for_completion(self, job_id: str) -> None:
        """Poll job status until completion or timeout (async).
        
        Args:
            job_id: The job identifier
            
        Raises:
            TimeoutError: If job doesn't complete within max_wait_time
        """
        import asyncio
        
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.max_wait_time:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {self.max_wait_time} seconds"
                )
            
            status_info = await self.get_status(job_id)
            
            if status_info.status == "job_completed":
                return
            
            await asyncio.sleep(self.poll_interval)
    
    async def get_results(
        self, 
        job_id: str,
        page: int = 1,
        page_size: int = 100,
    ) -> PullJobResponseDto:
        """Retrieve results for a completed job (async).
        
        Args:
            job_id: The job identifier
            page: Page number to retrieve (default: 1)
            page_size: Number of records per page (default: 100, max: 1000)
            
        Returns:
            PullJobResponseDto containing all extracted records
        """
        return await self._client.jobs.get_job_results(
            job_id=job_id,
            page=page,
            page_size=page_size,
        )
    
    async def get_all_results(self, job_id: str) -> PullJobResponseDto:
        """Retrieve all results for a job across all pages (async).
        
        Args:
            job_id: The job identifier
            
        Returns:
            PullJobResponseDto with all records from all pages
        """
        first_page = await self.get_results(job_id, page=1, page_size=1000)
        
        if first_page.total_pages == 1:
            return first_page

        import asyncio

        all_records = list(first_page.all_records or [])
        remaining_pages = [
            self.get_results(job_id, page=page, page_size=1000)
            for page in range(2, first_page.total_pages + 1)
        ]
        
        page_results = await asyncio.gather(*remaining_pages)
        for page_result in page_results:
            if page_result.all_records:
                all_records.extend(page_result.all_records)

        result_dict = first_page.dict() if hasattr(first_page, 'dict') else first_page.model_dump()
        result_dict['all_records'] = all_records
        
        return PullJobResponseDto(**result_dict)
    
    async def search(
        self,
        query: str,
        context: Optional[str] = None,
        schema: Optional[str] = None,
        wait: bool = True,
    ) -> PullJobResponseDto:
        """Submit a query and optionally wait for results (async).
        
        This is the main convenience method that combines submit, wait, and retrieve.
        
        Args:
            query: Natural language question
            context: Additional context to focus the search
            schema: Template string for record formatting
            wait: If True, wait for completion and return results. If False, return immediately.
            
        Returns:
            PullJobResponseDto if wait=True, otherwise empty result with just job_id
        """
        job_id = await self.submit_job(query=query, context=context, schema=schema)
        
        if not wait:
            return PullJobResponseDto(
                job_id=job_id,
                status="pending",
                page=1,
                total_pages=0,
                page_size=100,
            )
        
        await self.wait_for_completion(job_id)
        return await self.get_all_results(job_id)
    
    async def list_jobs(self) -> List[ListUserJobsResponseDto]:
        """List all jobs for the authenticated user (async).
        
        Returns:
            List of jobs with job_id and query fields
        """
        return await self._client.jobs.get_user_jobs()


__all__ = [
    "CatchAllClient",
    "AsyncCatchAllClient",
    "Record",
    "PullJobResponseDto",
    "StatusResponseDto",
    "ListUserJobsResponseDto",
]
