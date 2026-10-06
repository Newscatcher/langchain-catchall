"""Core client for interacting with the CatchAll API.

This module provides a LangChain-friendly wrapper around the official
newscatcher_catchall SDK, adding convenience methods for polling and
high-level web search operations.
"""

import time
from typing import Any, Dict, List, Optional

from newscatcher_catchall import CatchAllApi, AsyncCatchAllApi
from newscatcher_catchall.types import (
    AdditionalAttributes,
    CompanyAttributes,
    Record,
    PullJobResponseDto,
    StatusResponseDto,
    ListUserJobsResponseDto,
)

from langchain_catchall.helpers import evaluate_job_steps


def _monitor_webhook_ids(
    webhook: Optional[Dict[str, Any]],
    webhook_ids: Optional[List[str]],
) -> Optional[List[str]]:
    """Translate the legacy webhook argument to the SDK's webhook ID list."""
    if webhook_ids is not None or webhook is None:
        return webhook_ids
    value = webhook.get("webhook_ids") or webhook.get("webhook_id") or webhook.get("id")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(
        "webhook must contain 'id', 'webhook_id', or a string list under 'webhook_ids'"
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
        base_url: API base URL (default: "https://catchall.newscatcherapi.com")
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
        base_url: str = "https://catchall.newscatcherapi.com",
        poll_interval: int = 30,
        max_wait_time: int = 2400,
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
        validators: Optional[List[Dict[str, Any]]] = None,
        enrichments: Optional[List[Dict[str, Any]]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        mode: Optional[str] = None,
        connected_dataset_ids: Optional[List[str]] = None,
        fetch_all_watchlist_news: Optional[bool] = None,
    ) -> str:
        """Submit a new CatchAll job.

        Args:
            query: Natural language question describing what to find
            context: Additional context to focus the search
            schema: Template string to guide record formatting (e.g., "[COMPANY] earned [REVENUE]")
            validators: Optional validators to apply
            enrichments: Optional enrichments to extract
            start_date: Optional ISO timestamp for start date
            end_date: Optional ISO timestamp for end date
            limit: Maximum number of results to retrieve from the API
            mode: Job processing mode ('lite' or 'base'). 'lite' returns titles and
                citations only with faster processing; 'base' runs the full pipeline.
            connected_dataset_ids: Dataset IDs to connect to the job for Company Monitors mode.
                The dataset must have latest_status=ready before submitting.
            fetch_all_watchlist_news: When True, retrieves all news for connected Company
                Monitors entities without topic filtering. Requires connected_dataset_ids.

        Returns:
            Job ID for tracking the job

        Example:
            >>> job_id = client.submit_job(
            ...     query="Tech company earnings this quarter",
            ...     context="Focus on revenue and profit margins",
            ...     mode="lite",
            ... )
        """
        payload: Dict[str, Any] = {"query": query}
        if context is not None:
            payload["context"] = context
        if schema is not None:
            payload["schema"] = schema
        if validators is not None:
            payload["validators"] = validators
        if enrichments is not None:
            payload["enrichments"] = enrichments
        if start_date is not None:
            payload["start_date"] = start_date
        if end_date is not None:
            payload["end_date"] = end_date
        if limit is not None:
            payload["limit"] = limit
        if mode is not None:
            payload["mode"] = mode
        if connected_dataset_ids is not None:
            payload["connected_dataset_ids"] = connected_dataset_ids
        if fetch_all_watchlist_news is not None:
            payload["fetch_all_watchlist_news"] = fetch_all_watchlist_news
        response = self._client.jobs.create_job(**payload)
        return response.job_id

    def initialize_job(self, query: str, context: Optional[str] = None) -> Any:
        """Initialize a job to get suggested parameters.

        Args:
            query: Natural language question describing what to find
            context: Additional context to focus the search

        Returns:
            Initialize response containing validators, enrichments, and date range
        """
        payload: Dict[str, Any] = {"query": query}
        if context is not None:
            payload["context"] = context
        return self._client.jobs.initialize(**payload)


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
            completed_step, failed_step = evaluate_job_steps(status_info)

            if completed_step:
                return
            if failed_step:
                raise RuntimeError(f"Job {job_id} failed to complete")

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
        validators: Optional[List[Dict[str, Any]]] = None,
        enrichments: Optional[List[Dict[str, Any]]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        wait: bool = True,
        mode: Optional[str] = None,
        connected_dataset_ids: Optional[List[str]] = None,
        fetch_all_watchlist_news: Optional[bool] = None,
    ) -> PullJobResponseDto:
        """Submit a query and optionally wait for results.

        This is the main convenience method that combines submit, wait, and retrieve.

        Args:
            query: Natural language question
            context: Additional context to focus the search
            schema: Template string for record formatting
            validators: Optional validators to apply
            enrichments: Optional enrichments to extract
            start_date: Optional ISO timestamp for start date
            end_date: Optional ISO timestamp for end date
            limit: Maximum number of results to retrieve from the API
            wait: If True, wait for completion and return results. If False, return immediately.
            mode: Job processing mode ('lite' or 'base').
            connected_dataset_ids: Dataset IDs to connect to the job for Company Monitors mode.
            fetch_all_watchlist_news: When True, retrieves all news for connected entities.

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
        job_id = self.submit_job(
            query=query,
            context=context,
            schema=schema,
            validators=validators,
            enrichments=enrichments,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            mode=mode,
            connected_dataset_ids=connected_dataset_ids,
            fetch_all_watchlist_news=fetch_all_watchlist_news,
        )

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

    def list_jobs(self, page: int = 1, page_size: int = 100) -> List[ListUserJobsResponseDto]:
        """List all jobs for the authenticated user.

        Returns:
            List of jobs with job_id and query fields

        Example:
            >>> jobs = client.list_jobs()
            >>> for job in jobs:
            ...     print(f"{job.job_id}: {job.query}")
        """
        return self._client.jobs.get_user_jobs(page=page, page_size=page_size)

    def delete_job(self, job_id: str) -> Any:
        """Delete a job.

        Args:
            job_id: The job identifier

        Returns:
            Delete job response with success and message fields
        """
        return self._client.jobs.delete_job(job_id)

    # ── Monitor lifecycle ─────────────────────────────────────────────────────

    def create_monitor(
        self,
        reference_job_id: str,
        schedule: str,
        webhook: Optional[Dict[str, Any]] = None,
        *,
        timezone: Optional[str] = None,
        limit: Optional[int] = None,
        backfill: Optional[bool] = None,
        webhook_ids: Optional[List[str]] = None,
    ) -> Any:
        """Create a scheduled monitor based on a reference job.

        Args:
            reference_job_id: Job ID to use as template for scheduled runs
            schedule: Event monitor schedule in plain text format (e.g. "every day at 9 AM")
            timezone: IANA timezone identifier (e.g. "America/New_York")
            limit: Maximum number of records per monitor run
            backfill: If True, fills data gap between reference job end_date and first run
            webhook_ids: IDs of centralized webhooks to notify on each run completion

        Returns:
            CreateMonitorResponseDto with monitor_id and schedule fields
        """
        payload: Dict[str, Any] = {
            "reference_job_id": reference_job_id,
            "schedule": schedule,
        }
        if timezone is not None:
            payload["timezone"] = timezone
        if limit is not None:
            payload["limit"] = limit
        if backfill is not None:
            payload["backfill"] = backfill
        resolved_webhook_ids = _monitor_webhook_ids(webhook, webhook_ids)
        if resolved_webhook_ids is not None:
            payload["webhook_ids"] = resolved_webhook_ids
        return self._client.event_monitors.create_monitor(**payload)

    def list_monitors(self) -> Any:
        """List all monitors for the authenticated user.

        Returns:
            ListMonitorsResponseDto with monitors list
        """
        return self._client.event_monitors.list_monitors()

    def pull_monitor_results(self, monitor_id: str) -> Any:
        """Retrieve aggregated results for a monitor.

        Args:
            monitor_id: The monitor identifier

        Returns:
            PullMonitorResponseDto with monitor_id, reference_job, validators,
            enrichments, all_records, and run_info fields
        """
        return self._client.event_monitors.pull_monitor_results(monitor_id)

    def list_monitor_jobs(self, monitor_id: str, sort: str = "asc") -> Any:
        """List jobs for a monitor.

        Args:
            monitor_id: The monitor identifier
            sort: Sort order for jobs by start_date ('asc' or 'desc')

        Returns:
            ListMonitorJobsResponse with list of monitor job entries
        """
        return self._client.event_monitors.list_monitor_jobs(monitor_id, sort=sort)

    def enable_monitor(self, monitor_id: str) -> Any:
        """Enable a monitor.

        Args:
            monitor_id: The monitor identifier

        Returns:
            EnableMonitorResponse with success and message fields
        """
        return self._client.event_monitors.enable_monitor(monitor_id)

    def disable_monitor(self, monitor_id: str) -> Any:
        """Disable a monitor.

        Args:
            monitor_id: The monitor identifier

        Returns:
            DisableMonitorResponse with success and message fields
        """
        return self._client.event_monitors.disable_monitor(monitor_id)

    def update_monitor(
        self,
        monitor_id: str,
        webhook: Optional[Dict[str, Any]] = None,
        *,
        webhook_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        schedule: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Any:
        """Update a monitor's webhook assignments, record limit, or schedule.

        Args:
            monitor_id: The monitor identifier
            webhook_ids: Updated list of webhook IDs. Pass [] to clear all assignments.
            limit: Updated maximum number of records per monitor run
            schedule: New natural-language schedule (e.g. "every day at 9 AM")
            timezone: IANA timezone for the new schedule

        Returns:
            UpdateMonitorResponseDto with updated monitor fields
        """
        payload: Dict[str, Any] = {}
        resolved_webhook_ids = _monitor_webhook_ids(webhook, webhook_ids)
        if resolved_webhook_ids is not None:
            payload["webhook_ids"] = resolved_webhook_ids
        if limit is not None:
            payload["limit"] = limit
        if schedule is not None:
            payload["schedule"] = schedule
        if timezone is not None:
            payload["timezone"] = timezone
        return self._client.event_monitors.update_monitor(monitor_id, **payload)

    def delete_monitor(self, monitor_id: str) -> Any:
        """Delete a monitor.

        Args:
            monitor_id: The monitor identifier

        Returns:
            DeleteMonitorResponseDto with success, message, and monitor_id fields
        """
        return self._client.event_monitors.delete_monitor(monitor_id)

    # ── Company watchlist ─────────────────────────────────────────────────────

    def create_entity(
        self,
        name: str,
        description: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a company entity for use in watchlist datasets.

        Args:
            name: The company or person name (required)
            description: Free-text description used for disambiguation
            domain: Company website domain (highest-signal identifier). When provided,
                it is packaged into additional_attributes.company_attributes.domain.
            **kwargs: Additional keyword arguments forwarded to entities.create_entity
                (e.g. entity_type, external_entity_id, additional_attributes)

        Returns:
            CreateEntityResponse with id and status fields
        """
        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if domain is not None and "additional_attributes" not in kwargs:
            payload["additional_attributes"] = AdditionalAttributes(
                company_attributes=CompanyAttributes(domain=domain)
            )
        payload.update(kwargs)
        return self._client.entities.create_entity(**payload)

    def delete_entity(self, entity_id: str) -> None:
        """Permanently delete an entity.

        Args:
            entity_id: Unique entity identifier
        """
        return self._client.entities.delete_entity(entity_id)

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> Any:
        """Create a watchlist dataset from a list of entity IDs.

        Args:
            name: Name for the dataset
            description: Optional description of the dataset
            entity_ids: IDs of existing entities to include in the dataset

        Returns:
            DatasetResponse with id, name, latest_status, and entity_count fields
        """
        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if entity_ids is not None:
            payload["entity_ids"] = entity_ids
        return self._client.datasets.create_dataset(**payload)

    def get_dataset(self, dataset_id: str) -> Any:
        """Retrieve a dataset by ID.

        Args:
            dataset_id: Unique dataset identifier

        Returns:
            DatasetResponse with id, name, latest_status, and entity_count fields
        """
        return self._client.datasets.get_dataset(dataset_id)

    def delete_dataset(self, dataset_id: str) -> None:
        """Permanently delete a dataset. Entities within the dataset are not deleted.

        Args:
            dataset_id: Unique dataset identifier
        """
        return self._client.datasets.delete_dataset(dataset_id)


class AsyncCatchAllClient:
    """Async version of CatchAllClient.

    This client provides the same interface as CatchAllClient but with
    async/await support for better performance in async applications.

    Args:
        api_key: Your CatchAll API key
        base_url: API base URL (default: "https://catchall.newscatcherapi.com")
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
        base_url: str = "https://catchall.newscatcherapi.com",
        poll_interval: int = 30,
        max_wait_time: int = 2400,
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
        validators: Optional[List[Dict[str, Any]]] = None,
        enrichments: Optional[List[Dict[str, Any]]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        mode: Optional[str] = None,
        connected_dataset_ids: Optional[List[str]] = None,
        fetch_all_watchlist_news: Optional[bool] = None,
    ) -> str:
        """Submit a new CatchAll job (async).

        Args:
            query: Natural language question describing what to find
            context: Additional context to focus the search
            schema: Template string to guide record formatting
            validators: Optional validators to apply
            enrichments: Optional enrichments to extract
            start_date: Optional ISO timestamp for start date
            end_date: Optional ISO timestamp for end date
            limit: Maximum number of results to retrieve from the API
            mode: Job processing mode ('lite' or 'base').
            connected_dataset_ids: Dataset IDs for Company Monitors mode.
            fetch_all_watchlist_news: When True, retrieves all news for connected entities.

        Returns:
            Job ID for tracking the job
        """
        payload: Dict[str, Any] = {"query": query}
        if context is not None:
            payload["context"] = context
        if schema is not None:
            payload["schema"] = schema
        if validators is not None:
            payload["validators"] = validators
        if enrichments is not None:
            payload["enrichments"] = enrichments
        if start_date is not None:
            payload["start_date"] = start_date
        if end_date is not None:
            payload["end_date"] = end_date
        if limit is not None:
            payload["limit"] = limit
        if mode is not None:
            payload["mode"] = mode
        if connected_dataset_ids is not None:
            payload["connected_dataset_ids"] = connected_dataset_ids
        if fetch_all_watchlist_news is not None:
            payload["fetch_all_watchlist_news"] = fetch_all_watchlist_news
        response = await self._client.jobs.create_job(**payload)
        return response.job_id

    async def initialize_job(self, query: str, context: Optional[str] = None) -> Any:
        """Initialize a job to get suggested parameters (async).

        Args:
            query: Natural language question describing what to find
            context: Additional context to focus the search

        Returns:
            Initialize response containing validators, enrichments, and date range
        """
        payload: Dict[str, Any] = {"query": query}
        if context is not None:
            payload["context"] = context
        return await self._client.jobs.initialize(**payload)


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
            completed_step, failed_step = evaluate_job_steps(status_info)

            if completed_step:
                return
            if failed_step:
                raise RuntimeError(f"Job {job_id} failed to complete")

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
        validators: Optional[List[Dict[str, Any]]] = None,
        enrichments: Optional[List[Dict[str, Any]]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        wait: bool = True,
        mode: Optional[str] = None,
        connected_dataset_ids: Optional[List[str]] = None,
        fetch_all_watchlist_news: Optional[bool] = None,
    ) -> PullJobResponseDto:
        """Submit a query and optionally wait for results (async).

        This is the main convenience method that combines submit, wait, and retrieve.

        Args:
            query: Natural language question
            context: Additional context to focus the search
            schema: Template string for record formatting
            validators: Optional validators to apply
            enrichments: Optional enrichments to extract
            start_date: Optional ISO timestamp for start date
            end_date: Optional ISO timestamp for end date
            limit: Maximum number of results to retrieve from the API
            wait: If True, wait for completion and return results. If False, return immediately.
            mode: Job processing mode ('lite' or 'base').
            connected_dataset_ids: Dataset IDs for Company Monitors mode.
            fetch_all_watchlist_news: When True, retrieves all news for connected entities.

        Returns:
            PullJobResponseDto if wait=True, otherwise empty result with just job_id
        """
        job_id = await self.submit_job(
            query=query,
            context=context,
            schema=schema,
            validators=validators,
            enrichments=enrichments,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            mode=mode,
            connected_dataset_ids=connected_dataset_ids,
            fetch_all_watchlist_news=fetch_all_watchlist_news,
        )

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

    async def list_jobs(self, page: int = 1, page_size: int = 100) -> List[ListUserJobsResponseDto]:
        """List all jobs for the authenticated user (async).

        Returns:
            List of jobs with job_id and query fields
        """
        return await self._client.jobs.get_user_jobs(page=page, page_size=page_size)

    async def delete_job(self, job_id: str) -> Any:
        """Delete a job (async).

        Args:
            job_id: The job identifier

        Returns:
            Delete job response with success and message fields
        """
        return await self._client.jobs.delete_job(job_id)

    # ── Monitor lifecycle ─────────────────────────────────────────────────────

    async def create_monitor(
        self,
        reference_job_id: str,
        schedule: str,
        webhook: Optional[Dict[str, Any]] = None,
        *,
        timezone: Optional[str] = None,
        limit: Optional[int] = None,
        backfill: Optional[bool] = None,
        webhook_ids: Optional[List[str]] = None,
    ) -> Any:
        """Create a scheduled monitor based on a reference job (async).

        Args:
            reference_job_id: Job ID to use as template for scheduled runs
            schedule: Event monitor schedule in plain text format
            timezone: IANA timezone identifier
            limit: Maximum number of records per monitor run
            backfill: If True, fills data gap between reference job end_date and first run
            webhook_ids: IDs of centralized webhooks to notify on each run completion

        Returns:
            CreateMonitorResponseDto with monitor_id and schedule fields
        """
        payload: Dict[str, Any] = {
            "reference_job_id": reference_job_id,
            "schedule": schedule,
        }
        if timezone is not None:
            payload["timezone"] = timezone
        if limit is not None:
            payload["limit"] = limit
        if backfill is not None:
            payload["backfill"] = backfill
        resolved_webhook_ids = _monitor_webhook_ids(webhook, webhook_ids)
        if resolved_webhook_ids is not None:
            payload["webhook_ids"] = resolved_webhook_ids
        return await self._client.event_monitors.create_monitor(**payload)

    async def list_monitors(self) -> Any:
        """List all monitors for the authenticated user (async).

        Returns:
            ListMonitorsResponseDto with monitors list
        """
        return await self._client.event_monitors.list_monitors()

    async def pull_monitor_results(self, monitor_id: str) -> Any:
        """Retrieve aggregated results for a monitor (async).

        Args:
            monitor_id: The monitor identifier

        Returns:
            PullMonitorResponseDto with monitor_id, reference_job, validators,
            enrichments, all_records, and run_info fields
        """
        return await self._client.event_monitors.pull_monitor_results(monitor_id)

    async def list_monitor_jobs(self, monitor_id: str, sort: str = "asc") -> Any:
        """List jobs for a monitor (async).

        Args:
            monitor_id: The monitor identifier
            sort: Sort order for jobs by start_date ('asc' or 'desc')

        Returns:
            ListMonitorJobsResponse with list of monitor job entries
        """
        return await self._client.event_monitors.list_monitor_jobs(monitor_id, sort=sort)

    async def enable_monitor(self, monitor_id: str) -> Any:
        """Enable a monitor (async).

        Args:
            monitor_id: The monitor identifier

        Returns:
            EnableMonitorResponse with success and message fields
        """
        return await self._client.event_monitors.enable_monitor(monitor_id)

    async def disable_monitor(self, monitor_id: str) -> Any:
        """Disable a monitor (async).

        Args:
            monitor_id: The monitor identifier

        Returns:
            DisableMonitorResponse with success and message fields
        """
        return await self._client.event_monitors.disable_monitor(monitor_id)

    async def update_monitor(
        self,
        monitor_id: str,
        webhook: Optional[Dict[str, Any]] = None,
        *,
        webhook_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        schedule: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Any:
        """Update a monitor's webhook assignments, record limit, or schedule (async).

        Args:
            monitor_id: The monitor identifier
            webhook_ids: Updated list of webhook IDs. Pass [] to clear all assignments.
            limit: Updated maximum number of records per monitor run
            schedule: New natural-language schedule
            timezone: IANA timezone for the new schedule

        Returns:
            UpdateMonitorResponseDto with updated monitor fields
        """
        payload: Dict[str, Any] = {}
        resolved_webhook_ids = _monitor_webhook_ids(webhook, webhook_ids)
        if resolved_webhook_ids is not None:
            payload["webhook_ids"] = resolved_webhook_ids
        if limit is not None:
            payload["limit"] = limit
        if schedule is not None:
            payload["schedule"] = schedule
        if timezone is not None:
            payload["timezone"] = timezone
        return await self._client.event_monitors.update_monitor(monitor_id, **payload)

    async def delete_monitor(self, monitor_id: str) -> Any:
        """Delete a monitor (async).

        Args:
            monitor_id: The monitor identifier

        Returns:
            DeleteMonitorResponseDto with success, message, and monitor_id fields
        """
        return await self._client.event_monitors.delete_monitor(monitor_id)

    # ── Company watchlist ─────────────────────────────────────────────────────

    async def create_entity(
        self,
        name: str,
        description: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a company entity for use in watchlist datasets (async).

        Args:
            name: The company or person name (required)
            description: Free-text description used for disambiguation
            domain: Company website domain. When provided, packaged into
                additional_attributes.company_attributes.domain.
            **kwargs: Additional keyword arguments forwarded to entities.create_entity

        Returns:
            CreateEntityResponse with id and status fields
        """
        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if domain is not None and "additional_attributes" not in kwargs:
            payload["additional_attributes"] = AdditionalAttributes(
                company_attributes=CompanyAttributes(domain=domain)
            )
        payload.update(kwargs)
        return await self._client.entities.create_entity(**payload)

    async def delete_entity(self, entity_id: str) -> None:
        """Permanently delete an entity (async).

        Args:
            entity_id: Unique entity identifier
        """
        return await self._client.entities.delete_entity(entity_id)

    async def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> Any:
        """Create a watchlist dataset from a list of entity IDs (async).

        Args:
            name: Name for the dataset
            description: Optional description of the dataset
            entity_ids: IDs of existing entities to include in the dataset

        Returns:
            DatasetResponse with id, name, latest_status, and entity_count fields
        """
        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if entity_ids is not None:
            payload["entity_ids"] = entity_ids
        return await self._client.datasets.create_dataset(**payload)

    async def get_dataset(self, dataset_id: str) -> Any:
        """Retrieve a dataset by ID (async).

        Args:
            dataset_id: Unique dataset identifier

        Returns:
            DatasetResponse with id, name, latest_status, and entity_count fields
        """
        return await self._client.datasets.get_dataset(dataset_id)

    async def delete_dataset(self, dataset_id: str) -> None:
        """Permanently delete a dataset (async). Entities within are not deleted.

        Args:
            dataset_id: Unique dataset identifier
        """
        return await self._client.datasets.delete_dataset(dataset_id)


__all__ = [
    "CatchAllClient",
    "AsyncCatchAllClient",
    "Record",
    "PullJobResponseDto",
    "StatusResponseDto",
    "ListUserJobsResponseDto",
]
