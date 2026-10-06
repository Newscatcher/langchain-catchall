from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_catchall import CatchAllTools
from langchain_tests.unit_tests import ToolsUnitTests
from pydantic import BaseModel


class TestCatchAllSearchTool(ToolsUnitTests):
    @property
    def tool_constructor(self):
        def _create_tool(**kwargs):
            mock_llm = MagicMock()
            
            # Patch the CatchAllClient so it doesn't try to connect/init httpx
            with patch('langchain_catchall.tools.CatchAllClient'):
                toolkit = CatchAllTools(
                    api_key="test_key", 
                    llm=mock_llm, 
                    verbose=False
                )
                # Return the search tool (index 0)
                return toolkit.get_tools()[0]
        return _create_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "Find all articles about data breach announcements for last 5 days"}


class _SdkEnrichment(BaseModel):
    enrichment_confidence: float


@pytest.mark.parametrize(
    ("enrichment", "expected"),
    [
        ({"company": "NewsCatcher"}, "company: NewsCatcher"),
        (_SdkEnrichment(enrichment_confidence=0.95), "enrichment_confidence: 0.95"),
    ],
)
def test_format_search_results_supports_sdk_enrichment_models(
    enrichment: object,
    expected: str,
) -> None:
    toolkit = CatchAllTools.__new__(CatchAllTools)
    toolkit.limit = 10
    result = SimpleNamespace(
        valid_records=1,
        all_records=[
            SimpleNamespace(record_title="Example record", enrichment=enrichment)
        ],
    )

    formatted = toolkit._format_search_results(result)

    assert expected in formatted

