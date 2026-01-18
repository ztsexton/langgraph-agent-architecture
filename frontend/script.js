/*
 * Simple client‑side script to interact with the LangGraph multi‑agent backend.
 *
 * When the user submits a message the script opens a Server–Sent Events (SSE)
 * connection to the ``/stream`` endpoint and listens for incremental updates.
 * Each update contains a JSON object keyed by the node name that produced the
 * update.  The script displays outputs from any worker nodes in the chat
 * history.  A more sophisticated implementation could display routing
 * decisions or intermediate state, but this keeps the UI focused on final
 * responses.
 */

const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const chat = document.getElementById("chat");

// Append a message element to the chat history
function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.classList.add("message");
  div.innerHTML = `<strong>${sender}:</strong> ${text}`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

// Listen for form submissions
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  appendMessage("User", message);
  input.value = "";
  // Open an SSE connection for this message.  A new EventSource is used
  // per request so that each query completes independently.
  const evtSource = new EventSource(`/stream?message=${encodeURIComponent(message)}`);
  evtSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      // Each event contains a single key representing the node name
      const nodeName = Object.keys(data)[0];
      const update = data[nodeName];
      // If the update includes an output, display it
      if (update.output) {
        appendMessage(nodeName, update.output);
      }
    } catch (err) {
      console.error("Failed to parse event data", err);
    }
  };
  evtSource.onerror = () => {
    // Close the connection on error
    evtSource.close();
  };
});
