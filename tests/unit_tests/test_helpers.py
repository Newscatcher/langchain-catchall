from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_catchall.helpers import (
    format_record,
    format_results_for_llm,
    query_with_llm,
)
from newscatcher_catchall.types import BaseRecordEnrichment


def _record() -> SimpleNamespace:
    return SimpleNamespace(
        record_title="Example funding announcement",
        enrichment=BaseRecordEnrichment(enrichment_confidence=0.95),
        citations=[],
    )


def test_format_results_for_llm_supports_sdk_enrichment_model() -> None:
    result = SimpleNamespace(
        query="funding rounds",
        valid_records=1,
        all_records=[_record()],
    )

    context = format_results_for_llm(result)

    assert "Example funding announcement" in context
    assert "**enrichment_confidence**: 0.95" in context


def test_format_record_supports_sdk_enrichment_model() -> None:
    formatted = format_record(_record())

    assert "enrichment_confidence: 0.95" in formatted


def test_query_with_llm_reaches_model_with_sdk_enrichment() -> None:
    result = SimpleNamespace(
        query="funding rounds",
        valid_records=1,
        all_records=[_record()],
    )
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="Filtered answer")

    answer = query_with_llm(result, "Only rounds above $10M", llm)

    assert answer == "Filtered answer"
    assert "enrichment_confidence" in llm.invoke.call_args.args[0]
