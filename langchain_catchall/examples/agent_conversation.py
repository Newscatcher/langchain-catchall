"""Run a realistic search-then-analyze LangGraph conversation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langchain_catchall import CATCHALL_AGENT_PROMPT, CatchAllTools

DEFAULT_SEARCH = (
    "Find the top 10 articles about startups that announced investment "
    "rounds during the last 5 days"
)
DEFAULT_FOLLOW_UPS = (
    "From those results, show only companies that raised at least $10 million. "
    "List each company, amount raised, funding round, lead investor, and announcement date.",
    "Group those qualifying companies by industry and calculate the total and average "
    "amount raised per industry. Do not run a new search.",
)


def _print_answer(response: dict[str, Any]) -> None:
    print(response["messages"][-1].content)


def run_agent_conversation(
    catchall_api_key: str,
    openai_api_key: str,
    *,
    search_question: str = DEFAULT_SEARCH,
    follow_up_questions: Sequence[str] = DEFAULT_FOLLOW_UPS,
    model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Search once, then filter and analyze the cached results in later turns."""
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI(model=model)
    toolkit = CatchAllTools(
        api_key=catchall_api_key,
        llm=llm,
        limit=10,
        verbose=True,
    )
    agent = create_agent(
        model=llm,
        tools=toolkit.get_tools(),
        system_prompt=CATCHALL_AGENT_PROMPT,
    )

    print(f"\nUSER: {search_question}")
    response = agent.invoke({"messages": [HumanMessage(content=search_question)]})
    _print_answer(response)
    responses = [response]

    for question in follow_up_questions:
        print(f"\nUSER: {question}")
        messages = list(response["messages"])
        messages.append(HumanMessage(content=question))
        response = agent.invoke({"messages": messages})
        _print_answer(response)
        responses.append(response)

    return responses


if __name__ == "__main__":
    catchall_key = os.environ.get("CATCHALL_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not catchall_key or not openai_key:
        raise SystemExit(
            "Set CATCHALL_API_KEY and OPENAI_API_KEY before running this example"
        )
    run_agent_conversation(catchall_key, openai_key)
