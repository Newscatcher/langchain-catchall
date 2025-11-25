import os
import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_catchall import CatchAllTools
from langchain_openai import ChatOpenAI

@pytest.mark.skipif(
    not os.environ.get("CATCHALL_API_KEY"), 
    reason="CATCHALL_API_KEY not set"
)
class TestCatchAllSearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self):
        def _create_tool(**kwargs):
            llm = ChatOpenAI(model="gpt-4-turbo")
            toolkit = CatchAllTools(
                api_key=os.environ["CATCHALL_API_KEY"],
                llm=llm,
                verbose=True
            )
            return toolkit.get_tools()[0]
        return _create_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "Find all articles about factory accidents or production stoppages reported in Asia for last 2 days"}

