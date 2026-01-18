# LangGraph Multi-Agent Backend

This backend exposes a multi-agent architecture built using LangGraph. It defines three worker agents (`web_agent`, `meetings_agent` and `rag_agent`) and a supervisor that routes user requests to the appropriate agent. When a language model is configured (e.g. Azure OpenAI), the supervisor prefers LLM-based routing; otherwise it falls back to keyword routing. Responses are streamed to the client via Server-Sent Events (SSE).

## Features

- **Supervisor pattern:** A single supervisor node examines the incoming message and chooses the appropriate worker to handle the task. This follows the supervisor architecture described in the LangChain docs, where a central agent coordinates specialists and is useful when multiple domains require different toolsets【806125048713531†L91-L103】.
- **Web search agent:** Uses the DuckDuckGo search API to perform real web queries. If web search fails, it returns no results and the agent will respond based on configuration.
- **Meetings agent:** Provides simple CRUD operations against an in‑memory meeting database. You can list meetings, create new meetings and edit agendas or notes.
- **RAG agent:** Implements a lightweight retrieval augmented search over a small document corpus using TF‑IDF vectors. Answers include a citation identifying the source document.
- **Streaming:** All graph updates are streamed to the client using the `stream` method of the compiled LangGraph. Streaming improves responsiveness by sending partial results as they become available【998679072524577†L90-L103】.
- **Logging:** Incoming chat requests and graph updates are logged by the FastAPI server. The language model helper logs both the prompt and response to provide visibility into each exchange.

## Running the server

1. Install dependencies (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server with Uvicorn:

   ```bash
   uvicorn agent_project.backend.main:app --reload
   ```

3. Open your browser to `http://localhost:8000/ui` to use the built‑in frontend. Alternatively you can connect to the SSE endpoint directly at `http://localhost:8000/stream?message=Your%20question`.

## Configuration and environment variables

Several environment variables control optional integrations and logging:

- **LOG_LEVEL** – Sets the Python logging level for the backend (default: `INFO`).
- **OPENAI_API_KEY** – API key for the OpenAI Chat API. If provided, the LLM helper will call the OpenAI ChatCompletion API.
- **AZURE_OPENAI_API_KEY**, **AZURE_OPENAI_API_BASE**, **AZURE_OPENAI_API_VERSION** and **AZURE_OPENAI_DEPLOYMENT_NAME** – When these variables are set, the LLM helper will call an Azure OpenAI deployment instead of the public OpenAI API. You must specify the base URL (e.g., `https://YOUR_RESOURCE_NAME.openai.azure.com`), the API version (e.g., `2023-10-01-preview`) and the deployment name of your chat model.

If no API keys are set, LLM-based routing/summarisation/answer generation will be disabled.

## Endpoints

- **GET /stream?message=…** – runs the graph for the provided message and streams state updates as SSE events. Each event payload is a JSON object containing the name of the node that emitted the update and the updated values.
- **GET /** – returns a short description of the API.

Static frontend files are served under the **/ui** prefix. The root page contains a minimal chat interface implemented in vanilla JavaScript.

## Limitations and future work

This example is intentionally small and focuses on demonstrating the supervisor pattern and streaming with LangGraph. It uses keyword matching for routing and minimal tools for meetings and search. In a production system you would likely:

- Use a more sophisticated router (e.g., an LLM) to determine the correct agent based on the query.
- Replace the in‑memory meetings manager with a persistent database.
- Use an actual vector database and embedding model for retrieval.
- Replace the vanilla UI with the full AG‑UI or a custom frontend to visualise tool invocations and state transitions【413813318927496†L80-L86】.

## License

This project is released under the MIT license.
