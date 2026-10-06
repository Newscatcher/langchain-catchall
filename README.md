# 🦜🔗 LangChain CatchAll

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **The official LangChain integration for [CatchAll](https://www.newscatcherapi.com/docs/v3/catch-all/overview/introduction/) by NewsCatcher.**

Build autonomous web search agents, financial analysts, and research assistants that can find, read, and analyze millions of web pages.

---

## 🌟 Features

*   **Smart Caching:** "Fetch Once, Query Many." Search for a topic, then ask infinite follow-up questions instantly using the local cache.
*   **Agent Toolkit:** Ready-to-use `CatchAllTools` for LangGraph agents.
*   **Dual-Mode:** Supports both granular control (for scripts) and autonomous agents.
*   **LLM Agnostic:** Works with OpenAI, Gemini, Anthropic, or any LangChain-compatible model.

---

## 🚀 Quick Start

### Installation

```bash
python -m pip install --upgrade pip
pip install langchain-catchall
```

Set your CatchAll API key in the environment rather than embedding it in code:

```bash
export CATCHALL_API_KEY="your-key"
```

### Basic Usage (One-Shot Search)

```python
import os
from langchain_catchall import CatchAllClient

client = CatchAllClient(api_key=os.environ["CATCHALL_API_KEY"])
result = client.search(
    "Find the top 10 articles about cybersecurity incidents disclosed "
    "during the last 3 days",
    mode="lite",
    limit=10,
)

print(f"Job ID: {result.job_id}")
print(f"Found {result.valid_records} valid records")
for record in result.all_records[:3]:
    print(f"- {record.record_title}")
```

The job ID can later be reused to create an Event Monitor. This exact use case is
also available as an executable example:

```bash
python -m langchain_catchall.examples.basic_search
```

---

## 🤖 Search, Filter, and Analyze with an Agent

The agent performs one CatchAll search and keeps those results in its local toolkit
cache. Follow-up turns filter and analyze that cache instead of submitting another
long-running search.

```python
import os
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_catchall import CatchAllTools, CATCHALL_AGENT_PROMPT

llm = ChatOpenAI(model="gpt-4o")
toolkit = CatchAllTools(
    api_key=os.environ["CATCHALL_API_KEY"],
    llm=llm,
    limit=10,
    verbose=True,
)
agent = create_agent(
    model=llm,
    tools=toolkit.get_tools(),
    system_prompt=CATCHALL_AGENT_PROMPT,
)

# Turn 1 performs one new CatchAll search.
search_question = (
    "Find the top 10 articles about startups that announced investment "
    "rounds during the last 5 days"
)
print(f"\nUSER: {search_question}")
response = agent.invoke(
    {"messages": [HumanMessage(content=search_question)]}
)
print(f"AGENT: {response['messages'][-1].content}")

# Turn 2 filters the cached records. It must not submit another search.
filter_question = (
    "From those results, show only companies that raised at least $10 million. "
    "List each company, amount raised, funding round, lead investor, "
    "and announcement date."
)
print(f"\nUSER: {filter_question}")
response = agent.invoke(
    {
        "messages": [
            *response["messages"],
            HumanMessage(content=filter_question),
        ]
    }
)
print(f"AGENT: {response['messages'][-1].content}")

# Turn 3 analyzes the filtered records already held in the toolkit cache.
analysis_question = (
    "Group those qualifying companies by industry and calculate the total "
    "and average amount raised per industry. Do not run a new search."
)
print(f"\nUSER: {analysis_question}")
response = agent.invoke(
    {
        "messages": [
            *response["messages"],
            HumanMessage(content=analysis_question),
        ]
    }
)
print(f"AGENT: {response['messages'][-1].content}")
```

Install the optional model dependency before running the agent example:

```bash
python -m pip install langchain-openai
export OPENAI_API_KEY="your-key"
python -m langchain_catchall.examples.agent_conversation
```

Broad searches can take a while. To limit your search size and retrieve results faster we set limit=10 in our examples. CatchAll currently requires a search limit of at least 10.

---

## 📚 Advanced Patterns

### Fetch Once, Query Many (Financial Analyst Mode)

Perfect for deep dives where you don't want to re-run the search every time.

```python
import os
from langchain_catchall import CatchAllClient, query_with_llm
from langchain_openai import ChatOpenAI

# 1. Set up LLM
llm = ChatOpenAI(model="gpt-4o")

# 2. Grab needed data using CatchAllClient
client = CatchAllClient(api_key=os.environ["CATCHALL_API_KEY"])
result = client.search(
    "Find the top 10 articles about seed rounds over $5M announced this week",
    limit=10,
)

# 3. The Fast Analysis (Local Cache)
# Ask as many questions as you want
print(query_with_llm(result, "List top 3 deals", llm))
print(query_with_llm(result, "Who are the CEOs?", llm))
print(query_with_llm(result, "What is total amount of money raised in the US market", llm))
```

---

## 🔄 Event Monitors

Schedule recurring searches and retrieve their results.

```python
import os
from langchain_catchall import CatchAllClient

client = CatchAllClient(api_key=os.environ["CATCHALL_API_KEY"])
monitor_id = None

try:
    monitor = client.create_monitor(
        reference_job_id=os.environ["CATCHALL_COMPLETED_JOB_ID"],
        schedule="every day at 9 AM",
        timezone="UTC",
    )
    monitor_id = monitor.monitor_id
    print(f"Created monitor: {monitor_id}")

    print(client.list_monitors())
    client.disable_monitor(monitor_id)
    client.enable_monitor(monitor_id)
    client.update_monitor(monitor_id, schedule="every Monday at 8 AM")
    print(client.list_monitor_jobs(monitor_id))
    print(client.pull_monitor_results(monitor_id))
finally:
    if monitor_id:
        client.delete_monitor(monitor_id)
        print(f"Deleted monitor: {monitor_id}")
```

Use the job ID printed by the basic-search example:

```bash
export CATCHALL_COMPLETED_JOB_ID="your-completed-job-id"
python -m langchain_catchall.examples.event_monitor
```

The example deletes the monitor in a `finally` block, including after failures.

---

## 🏢 Company Watchlist (Company Monitors)

Track news for specific companies using entity datasets.

```python
import os
import time
from langchain_catchall import CatchAllClient

client = CatchAllClient(api_key=os.environ["CATCHALL_API_KEY"])
entity_id = dataset_id = job_id = None

try:
    # Create a company entity.
    entity = client.create_entity(
        name="Samsung",
        description="Electronincs brand and manufacturer from South Korea",
        domain="samsung.com",
    )
    entity_id = entity.id

    # Group the entity into a dataset.
    dataset = client.create_dataset(
        name=f"My company watchlist {int(time.time())}",
        entity_ids=[entity_id],
    )
    dataset_id = dataset.id

    # Wait up to 10 minutes for the dataset to become ready.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        dataset = client.get_dataset(dataset_id)
        status = str(dataset.latest_status).lower()
        print(f"Dataset status: {status}")
        if status == "ready" or status.endswith(".ready"):
            break
        if status in {"failed", "error"} or status.endswith((".failed", ".error")):
            raise RuntimeError(f"Dataset entered terminal status: {status}")
        time.sleep(10)
    else:
        raise TimeoutError("Dataset did not become ready within 10 minutes")

    # Search news connected to that watchlist.
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
finally:
    # Always remove production test resources in reverse order.
    if job_id:
        client.delete_job(job_id)
    if dataset_id:
        client.delete_dataset(dataset_id)
    if entity_id:
        client.delete_entity(entity_id)
```

Run the same example from the command line:

```bash
python -m langchain_catchall.examples.company_watchlist
```

The example uses a finite dataset timeout and always deletes its job, dataset, and
entity in reverse order.

---



## 📄 License

MIT License

