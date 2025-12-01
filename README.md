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
pip install langchain-catchall
```

### Basic Usage (One-Shot Search)

```python
import os
from langchain_catchall import CatchAllClient

os.environ["CATCHALL_API_KEY"] = "your_key"

client = CatchAllClient(api_key=os.environ["CATCHALL_API_KEY"])

# Search and wait for results
result = client.search("Find all articles about security incidents (data breaches, ransomware, hacks) disclosed between November 3 and November 5")

print(f"Found {result.valid_records} records.")
for record in result.all_records[:3]:
    print(f"- {record.record_title}")
```

---

## 🤖 Building an Autonomous Agent

The real power comes when you connect CatchAll to a LangGraph agent. The agent can decide when to search for new data and when to just analyze what it already found.

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_catchall import CatchAllTools, CATCHALL_AGENT_PROMPT

# 1. Setup Tools
llm = ChatOpenAI(model="gpt-4-turbo")
toolkit = CatchAllTools(api_key="...", llm=llm, verbose=True)
tools = toolkit.get_tools()

# 2. Create Agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"), 
    tools=tools
)

# 3. Run
messages = [SystemMessage(content=CATCHALL_AGENT_PROMPT)]
messages.append(("user", "Find all articles about corporate HQ relocations or office closures in the US for last 3 days"))

response = agent.invoke({"messages": messages})
print(response["messages"][-1].content)
```

---

## 📚 Advanced Patterns

### Fetch Once, Query Many (Financial Analyst Mode)

Perfect for deep dives where you don't want to re-run the search every time.

```python
from langchain_catchall import CatchAllClient, query_with_llm
from langchain_openai import ChatOpenAI

# 1. Set up LLM
llm = ChatOpenAI(model="gpt-4-turbo")

# 2. Grab needed data using CatchAllClient
client = CatchAllClient(api_key="...")
result = client.search("Find all articles about seed rounds over $5M announced this week")

# 3. The Fast Analysis (Local Cache)
# Ask as many questions as you want
print(query_with_llm(result, "List top 3 deals", llm))
print(query_with_llm(result, "Who are the CEOs?", llm))
print(query_with_llm(result, "What is total amount of money raised in the US market", llm))
```

---


## 📄 License

MIT License

