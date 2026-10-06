"""Run the Company Watchlist entity and dataset lifecycle."""

from __future__ import annotations

import os
import time

from langchain_catchall import CatchAllClient


def _wait_for_dataset(
    client: CatchAllClient,
    dataset_id: str,
    *,
    timeout: int = 600,
    poll_interval: int = 10,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        dataset = client.get_dataset(dataset_id)
        status = str(dataset.latest_status).lower()
        print(f"Dataset status: {status}")
        if status == "ready" or status.endswith(".ready"):
            return
        if status in {"failed", "error"} or status.endswith((".failed", ".error")):
            raise RuntimeError(f"Dataset entered terminal status: {status}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Dataset did not become ready within {timeout} seconds")


def run_company_watchlist(api_key: str) -> str:
    """Create watchlist resources, run one job, clean up, and return the job ID."""
    client = CatchAllClient(api_key=api_key)
    entity_id: str | None = None
    dataset_id: str | None = None
    job_id: str | None = None

    try:
        entity = client.create_entity(
            name="NewsCatcher",
            description="News intelligence API provider",
            domain="newscatcherapi.com",
        )
        entity_id = entity.id
        print(f"Created entity: {entity_id}")

        dataset = client.create_dataset(
            name=f"LangChain example {int(time.time())}",
            entity_ids=[entity_id],
        )
        dataset_id = dataset.id
        print(f"Created dataset: {dataset_id}")
        _wait_for_dataset(client, dataset_id)

        job_id = client.submit_job(
            query="Find recent news about this company",
            mode="lite",
            limit=10,
            connected_dataset_ids=[dataset_id],
            fetch_all_watchlist_news=True,
        )
        print(f"Submitted job: {job_id}")
        client.wait_for_completion(job_id)
        results = client.get_all_results(job_id)
        print(f"Found {results.valid_records} watchlist records")
        return job_id
    finally:
        if job_id:
            client.delete_job(job_id)
            print(f"Deleted job: {job_id}")
        if dataset_id:
            client.delete_dataset(dataset_id)
            print(f"Deleted dataset: {dataset_id}")
        if entity_id:
            client.delete_entity(entity_id)
            print(f"Deleted entity: {entity_id}")


if __name__ == "__main__":
    key = os.environ.get("CATCHALL_API_KEY", "")
    if not key:
        raise SystemExit("Set CATCHALL_API_KEY before running this example")
    run_company_watchlist(key)
