from typing import Any, Dict, Type
import pytest
from unittest.mock import MagicMock, patch
from langchain_tests.unit_tests import ToolsUnitTests
from langchain_catchall import CatchAllTools
from langchain_core.tools import BaseTool

class TestCatchAllSearchTool(ToolsUnitTests):
    @property
    def tool_constructor(self):
        def _create_tool(**kwargs):
            mock_llm = MagicMock()
            
            # Patch the CatchAllClient so it doesn't try to connect/init httpx
            with patch('langchain_catchall.tools.CatchAllClient') as MockClient:
                toolkit = CatchAllTools(
                    api_key="test_key", 
                    llm=mock_llm, 
                    verbose=False
                )
                # Return the search tool (index 0)
                return toolkit.get_tools()[0]
        return _create_tool

    @property
    def tool_constructor_params(self) -> Dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> Dict:
        return {"query": "Find all articles about data breach announcements for last 5 days"}

