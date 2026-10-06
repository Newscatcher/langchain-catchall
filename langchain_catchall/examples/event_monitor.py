"""Run the Event Monitor lifecycle using an existing completed job."""

from __future__ import annotations

import os

from langchain_catchall import CatchAllClient


def run_event_monitor(api_key: str, completed_job_id: str) -> None:
    """Create, update, inspect, and clean up one event monitor."""
    client = CatchAllClient(api_key=api_key)
    monitor_id: str | None = None

    try:
        monitor = client.create_monitor(
            reference_job_id=completed_job_id,
            schedule="every day at 9 AM",
            timezone="UTC",
        )
        monitor_id = monitor.monitor_id
        print(f"Created monitor: {monitor_id}")

        print(f"Monitors: {client.list_monitors()}")
        client.disable_monitor(monitor_id)
        client.enable_monitor(monitor_id)
        client.update_monitor(monitor_id, schedule="every Monday at 8 AM")
        print(f"Monitor jobs: {client.list_monitor_jobs(monitor_id)}")
        print(f"Monitor results: {client.pull_monitor_results(monitor_id)}")
    finally:
        if monitor_id:
            client.delete_monitor(monitor_id)
            print(f"Deleted monitor: {monitor_id}")


if __name__ == "__main__":
    key = os.environ.get("CATCHALL_API_KEY", "")
    job_id = os.environ.get("CATCHALL_COMPLETED_JOB_ID", "")
    if not key or not job_id:
        raise SystemExit(
            "Set CATCHALL_API_KEY and CATCHALL_COMPLETED_JOB_ID before running this example"
        )
    run_event_monitor(key, job_id)
