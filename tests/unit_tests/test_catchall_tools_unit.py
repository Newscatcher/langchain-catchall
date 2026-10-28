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


def _make_toolkit(initialize_query: bool = False):
    with patch("langchain_catchall.tools.CatchAllClient") as client_cls:
        toolkit = CatchAllTools(
            api_key="test_key",
            llm=MagicMock(),
            verbose=False,
            initialize_query=initialize_query,
        )
    return toolkit, client_cls.return_value


def test_toolkit_exposes_list_source_groups_tool():
    toolkit, _ = _make_toolkit()
    tools = {tool.name: tool for tool in toolkit.get_tools()}
    assert "catchall_list_source_groups" in tools
    assert "source_groups" in tools["catchall_search_data"].args


def test_list_source_groups_formats_response():
    toolkit, client = _make_toolkit()
    client.list_source_groups.return_value = SimpleNamespace(
        source_groups=[
            SimpleNamespace(slug="top-100-us-finance", name="Top 100 US Finance", description="Finance"),
            SimpleNamespace(slug="other", name="Other", description=None),
        ],
        total=2,
        page=1,
        page_size=100,
    )

    text = toolkit.list_source_groups()

    client.list_source_groups.assert_called_once_with(page=1, page_size=100)
    assert "top-100-us-finance: Top 100 US Finance - Finance" in text
    assert "- other: Other" in text


def test_search_data_passes_source_groups_to_submit():
    toolkit, client = _make_toolkit()
    client.submit_job.return_value = "job-1"
    client.max_wait_time = 10
    client.poll_interval = 0
    client.get_all_results.return_value = SimpleNamespace(valid_records=0, all_records=[])
    status = SimpleNamespace(status="completed", steps=[])

    with patch("langchain_catchall.tools.evaluate_job_steps", return_value=(object(), None)):
        client.get_status.return_value = status
        toolkit.search_data("Find news", source_groups=["a"])

    assert client.submit_job.call_args.kwargs["source_groups"] == ["a"]


def test_search_data_rejects_more_than_20_source_groups():
    toolkit, client = _make_toolkit()
    result = toolkit.search_data("Find news", source_groups=[str(i) for i in range(21)])
    assert result.startswith("ERROR")
    client.submit_job.assert_not_called()


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

