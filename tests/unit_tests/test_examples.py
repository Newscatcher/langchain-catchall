from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_catchall.examples.agent_conversation import run_agent_conversation
from langchain_catchall.examples.basic_search import run_basic_search
from langchain_catchall.examples.company_watchlist import run_company_watchlist
from langchain_catchall.examples.event_monitor import run_event_monitor


def test_basic_search_example_returns_reusable_job_id() -> None:
    result = SimpleNamespace(
        job_id="job-1",
        valid_records=1,
        all_records=[SimpleNamespace(record_title="Example")],
    )
    with patch(
        "langchain_catchall.examples.basic_search.CatchAllClient"
    ) as client_class:
        client_class.return_value.search.return_value = result

        assert run_basic_search("key") == "job-1"

    client_class.return_value.search.assert_called_once_with(
        "Find the top 10 articles about cybersecurity incidents disclosed "
        "during the last 3 days",
        mode="lite",
        limit=10,
    )


def test_agent_example_preserves_messages_for_cached_follow_ups() -> None:
    agent = MagicMock()

    def invoke(payload: dict) -> dict:
        messages = list(payload["messages"])
        messages.append(SimpleNamespace(content=f"answer-{len(messages)}"))
        return {"messages": messages}

    agent.invoke.side_effect = invoke
    toolkit = MagicMock()
    toolkit.get_tools.return_value = ["search", "analyze"]

    with (
        patch(
            "langchain_catchall.examples.agent_conversation.ChatOpenAI"
        ) as llm_class,
        patch(
            "langchain_catchall.examples.agent_conversation.CatchAllTools",
            return_value=toolkit,
        ) as toolkit_class,
        patch(
            "langchain_catchall.examples.agent_conversation.create_agent",
            return_value=agent,
        ),
    ):
        responses = run_agent_conversation(
            "catchall-key",
            "openai-key",
            follow_up_questions=("Filter the cached results", "Summarize them"),
        )

    assert len(responses) == 3
    assert agent.invoke.call_count == 3
    assert len(agent.invoke.call_args_list[0].args[0]["messages"]) == 1
    assert len(agent.invoke.call_args_list[1].args[0]["messages"]) == 3
    assert len(agent.invoke.call_args_list[2].args[0]["messages"]) == 5
    toolkit_class.assert_called_once_with(
        api_key="catchall-key",
        llm=llm_class.return_value,
        limit=10,
        verbose=True,
    )


def test_event_monitor_example_always_deletes_monitor() -> None:
    client = MagicMock()
    client.create_monitor.return_value = SimpleNamespace(monitor_id="monitor-1")

    with patch(
        "langchain_catchall.examples.event_monitor.CatchAllClient",
        return_value=client,
    ):
        run_event_monitor("key", "job-1")

    client.create_monitor.assert_called_once_with(
        reference_job_id="job-1",
        schedule="every day at 9 AM",
        timezone="UTC",
    )
    client.delete_monitor.assert_called_once_with("monitor-1")


def test_company_watchlist_example_cleans_up_in_reverse_order() -> None:
    client = MagicMock()
    client.create_entity.return_value = SimpleNamespace(id="entity-1")
    client.create_dataset.return_value = SimpleNamespace(id="dataset-1")
    client.get_dataset.return_value = SimpleNamespace(latest_status="ready")
    client.submit_job.return_value = "job-1"
    client.get_all_results.return_value = SimpleNamespace(valid_records=2)

    with patch(
        "langchain_catchall.examples.company_watchlist.CatchAllClient",
        return_value=client,
    ):
        assert run_company_watchlist("key") == "job-1"

    cleanup_calls = [
        call
        for call in client.method_calls
        if call[0] in {"delete_job", "delete_dataset", "delete_entity"}
    ]
    assert [call[0] for call in cleanup_calls] == [
        "delete_job",
        "delete_dataset",
        "delete_entity",
    ]
