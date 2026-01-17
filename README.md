# LangGraph Agent Supervisor Demo

This repository demonstrates a multi‑agent architecture built with
[LangGraph](https://docs.langchain.com/oss/python/langgraph/) and exposed via a
FastAPI server.  A central **supervisor agent** routes user requests to
specialised worker agents:

* **Web search agent** – performs DuckDuckGo searches and returns
  formatted summaries.
* **Meetings agent** – manages an in‑memory calendar with list, create
  and edit commands.
* **RAG agent** – answers questions from a small document corpus using
  TF‑IDF similarity search.

The system uses the **supervisor pattern** described in the LangGraph
documentation, where a central agent coordinates specialists based on
task domain【806125048713531†L91-L103】.  Streaming is enabled to send incremental
updates to the client, improving responsiveness【998679072524577†L90-L103】.

## Structure

```
agent_project/
|
├── backend/             # FastAPI server and agent definitions
│   ├── agents.py        # Constructs the LangGraph with supervisor and workers
│   ├── main.py          # FastAPI app exposing streaming endpoint and static UI
│   ├── meetings.py      # In‑memory meetings manager
│   ├── rag.py           # Simple retrieval‑augmented search using TF‑IDF
│   ├── web_search.py    # DuckDuckGo wrapper with offline fallback
│   ├── requirements.txt # Python dependencies
│   └── README.md        # Backend usage information
│
├── frontend/
│   ├── index.html       # Simple chat interface
│   ├── script.js        # Handles SSE streaming and UI updates
│   ├── style.css        # Basic styling
│   └── README.md        # Frontend information
│
└── README.md            # Project overview (this file)
```

## Getting Started

To run the demo locally:

1. Install Python dependencies and start the server:

   ```bash
   cd agent_project/backend
   pip install -r requirements.txt
   uvicorn agent_project.backend.main:app --reload
   ```

2. Visit `http://localhost:8000/ui` and send messages.  The supervisor will
   route your queries to the appropriate agent and stream back updates.

For example:

* Ask a general knowledge question like *"search what is LangGraph"* to
  trigger the web search agent.
* Manage your calendar with commands such as *"list meetings"* or
  *"create meeting Project Kickoff on 2026-02-01 agenda Plan milestones"*.
* Query the document corpus with *"tell me about meeting management"* to
  see the RAG agent in action.

Feel free to extend the `meetings.py` and `rag.py` modules or swap the
heuristic routing in `agents.py` with a large language model.  The
foundations laid out here should make it straightforward to build more
sophisticated agent ecosystems.
