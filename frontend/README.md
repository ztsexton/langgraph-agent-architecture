# Frontend for the Multi‑Agent Demo

This directory contains a minimal, dependency‑free web interface for
interacting with the LangGraph multi‑agent backend.  It is served
automatically by the FastAPI application under the `/ui` prefix.

## Usage

1. Ensure the backend is running (see the instructions in
   `../backend/README.md`).
2. Navigate to `http://localhost:8000/ui` in your browser.
3. Type a question or command into the input box and press **Send**.

The client establishes a new `EventSource` connection for every
message.  As the backend executes the graph it emits state updates via
Server‑Sent Events (SSE).  The JavaScript displays the ``output``
property from any node updates in the chat window.

Although this interface is intentionally simple, it conforms to the
AG‑UI philosophy of streaming incremental agent updates to the user【671429342717890†L103-L111】.  For more
feature‑rich experiences, consider integrating with the official Agent
Chat UI or CopilotKit.  The backend exposed here should be
compatible with those clients out of the box.
