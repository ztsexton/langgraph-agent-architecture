# LangGraph Multi‑Agent Backend

This backend exposes a simple multi‑agent architecture built using
[LangGraph](https://docs.langchain.com/oss/python/langgraph/).  It defines
three worker agents (`web_agent`, `meetings_agent` and `rag_agent`) and a
supervisor that routes user requests to the appropriate agent based on
simple keyword matching.  Responses are streamed to the client via
Server‑Sent Events (SSE).

## Features

* **Supervisor pattern:** A single supervisor node examines the incoming
  message and chooses the appropriate worker to handle the task.  This
  follows the supervisor architecture described in the LangChain docs,
  where a central agent coordinates specialists and is useful when
  multiple domains require different toolsets【806125048713531†L91-L103】.

* **Web search agent:** Uses the DuckDuckGo search API to perform real
  web queries.  If the dependency is missing or network access is
  unavailable the agent returns a mock response.

* **Meetings agent:** Provides simple CRUD operations against an
  in‑memory meeting database.  You can list meetings, create new
  meetings and edit agendas or notes.

* **RAG agent:** Implements a lightweight retrieval augmented search over
  a small document corpus using TF‑IDF vectors.  Answers include a
  citation identifying the source document.

* **Streaming:** All graph updates are streamed to the client using the
  `stream` method of the compiled LangGraph.  Streaming improves
  responsiveness by sending partial results as they become available【998679072524577†L90-L103】.

## Running the server

1. Install dependencies (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server with Uvicorn:

   ```bash
   uvicorn agent_project.backend.main:app --reload
   ```

3. Open your browser to `http://localhost:8000/ui` to use the built‑in
   frontend.  Alternatively you can connect to the SSE endpoint directly
   at `http://localhost:8000/stream?message=Your%20question`.

## Endpoints

* **GET /stream?message=...** – runs the graph for the provided message
  and streams state updates as SSE events.  Each event payload is a JSON
  object containing the name of the node that emitted the update and
  the updated values.
* **GET /** – returns a short description of the API.

Static frontend files are served under the `/ui` prefix.  The
index page contains a minimal chat interface implemented in vanilla
JavaScript.

## Limitations and future work

This example is intentionally small and focuses on demonstrating the
supervisor pattern and streaming with LangGraph.  It uses keyword
matching for routing and rudimentary parsing for the meetings agent.
In a production system you would likely:

* Integrate a large language model for richer intent detection and
  natural language understanding.
* Replace the TF‑IDF–based retrieval with a vector database and
  embeddings model for high‑quality RAG.
* Persist meetings to a database and implement authentication.
* Add proper error handling and logging throughout the system.
* Extend the frontend or use the official Agent Chat UI for
  visualising tool calls and state changes【413813318927496†L80-L86】.

Pull requests are welcome!
