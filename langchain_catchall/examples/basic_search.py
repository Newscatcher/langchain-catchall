"""Run a small one-shot CatchAll search."""

from __future__ import annotations

import os

from langchain_catchall import CatchAllClient

DEFAULT_QUERY = (
    "Find the top 10 articles about cybersecurity incidents disclosed "
    "during the last 3 days"
)


def run_basic_search(
    api_key: str,
    *,
    query: str = DEFAULT_QUERY,
    limit: int = 10,
) -> str:
    """Run a lite search, print a preview, and return its reusable job ID."""
    if limit < 10:
        raise ValueError("CatchAll search limit must be at least 10")

    client = CatchAllClient(api_key=api_key)
    result = client.search(query, mode="lite", limit=limit)

    print(f"Job ID: {result.job_id}")
    print(f"Found {result.valid_records} valid records")
    for record in (result.all_records or [])[:3]:
        print(f"- {record.record_title}")
    return result.job_id


if __name__ == "__main__":
    key = os.environ.get("CATCHALL_API_KEY", "")
    if not key:
        raise SystemExit("Set CATCHALL_API_KEY before running this example")
    run_basic_search(key)
