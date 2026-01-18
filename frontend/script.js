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

function a2uiHasRichContent(node) {
  if (!node || typeof node !== "object") return false;
  const type = node.type;
  if (type === "card" || type === "table" || type === "kv") return true;
  if (type === "container") {
    const children = Array.isArray(node.children) ? node.children : [];
    return children.some((c) => a2uiHasRichContent(c));
  }
  return false;
}

function renderNode(node, container) {
  if (!node || typeof node !== "object") {
    const pre = document.createElement("pre");
    pre.textContent = String(node);
    container.appendChild(pre);
    return;
  }

  const type = node.type;
  if (type === "container") {
    const div = document.createElement("div");
    div.className = "a2ui-container";
    const children = Array.isArray(node.children) ? node.children : [];
    children.forEach((child) => renderNode(child, div));
    container.appendChild(div);
    return;
  }

  if (type === "heading") {
    const level = Math.min(6, Math.max(1, Number(node.level || 2)));
    const h = document.createElement(`h${level}`);
    h.className = "a2ui-heading";
    h.textContent = node.text || "";
    container.appendChild(h);
    return;
  }

  if (type === "text") {
    const p = document.createElement("div");
    p.className = "a2ui-text";
    p.textContent = node.text || "";
    container.appendChild(p);
    return;
  }

  if (type === "list") {
    const ul = document.createElement("ul");
    ul.className = "a2ui-list";
    const items = Array.isArray(node.items) ? node.items : [];
    items.forEach((it) => {
      const li = document.createElement("li");
      li.textContent = String(it);
      ul.appendChild(li);
    });
    container.appendChild(ul);
    return;
  }

  if (type === "links") {
    const div = document.createElement("div");
    div.className = "a2ui-links";
    const items = Array.isArray(node.items) ? node.items : [];
    items.forEach((it) => {
      if (!it || typeof it !== "object") return;
      const a = document.createElement("a");
      a.href = it.href || "#";
      a.target = "_blank";
      a.rel = "noreferrer noopener";
      a.textContent = it.text || it.href || "link";
      div.appendChild(a);
      div.appendChild(document.createElement("br"));
    });
    container.appendChild(div);
    return;
  }

  if (type === "card") {
    const card = document.createElement("div");
    card.className = "a2ui-card";

    const header = document.createElement("div");
    header.className = "a2ui-card-header";

    const title = document.createElement("div");
    title.className = "a2ui-card-title";
    title.textContent = node.title || "";

    const subtitle = document.createElement("div");
    subtitle.className = "a2ui-card-subtitle";
    subtitle.textContent = node.subtitle || "";

    header.appendChild(title);
    if (node.subtitle) header.appendChild(subtitle);

    const body = document.createElement("div");
    body.className = "a2ui-card-body";
    const children = Array.isArray(node.children) ? node.children : [];
    children.forEach((child) => renderNode(child, body));

    card.appendChild(header);
    card.appendChild(body);
    container.appendChild(card);
    return;
  }

  if (type === "kv") {
    const wrap = document.createElement("div");
    wrap.className = "a2ui-kv";
    const items = Array.isArray(node.items) ? node.items : [];
    items.forEach((it) => {
      if (!it || typeof it !== "object") return;
      const row = document.createElement("div");
      row.className = "a2ui-kv-row";
      const label = document.createElement("div");
      label.className = "a2ui-kv-label";
      label.textContent = it.label ?? "";
      const value = document.createElement("div");
      value.className = "a2ui-kv-value";
      value.textContent = it.value === null || it.value === undefined ? "" : String(it.value);
      row.appendChild(label);
      row.appendChild(value);
      wrap.appendChild(row);
    });
    container.appendChild(wrap);
    return;
  }

  if (type === "table") {
    const section = document.createElement("div");
    section.className = "a2ui-table";

    if (node.title) {
      const h = document.createElement("div");
      h.className = "a2ui-table-title";
      h.textContent = node.title;
      section.appendChild(h);
    }

    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const trh = document.createElement("tr");
    const cols = Array.isArray(node.columns) ? node.columns : [];
    cols.forEach((c) => {
      const th = document.createElement("th");
      th.textContent = String(c);
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    const rows = Array.isArray(node.rows) ? node.rows : [];
    rows.forEach((r) => {
      if (!Array.isArray(r)) return;
      const tr = document.createElement("tr");
      r.forEach((cell) => {
        const td = document.createElement("td");
        td.textContent = cell === null || cell === undefined ? "" : String(cell);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    section.appendChild(table);
    container.appendChild(section);
    return;
  }

  // Unknown node type: render JSON for debugging
  const pre = document.createElement("pre");
  pre.className = "a2ui-unknown";
  pre.textContent = JSON.stringify(node, null, 2);
  container.appendChild(pre);
}

function appendA2UIMessage(sender, a2ui, text) {
  if (!a2ui || typeof a2ui !== "object") return;
  if (a2ui.schema !== "a2ui") return;

  const div = document.createElement("div");
  div.classList.add("message");
  div.classList.add("message-rich");

  const header = document.createElement("div");
  header.className = "message-header";
  header.innerHTML = `<strong>${sender}:</strong>`;
  div.appendChild(header);

  const body = document.createElement("div");
  body.className = "message-body";

  if (text) {
    const p = document.createElement("div");
    p.className = "a2ui-text";
    p.textContent = text;
    body.appendChild(p);
  }

  renderNode(a2ui.render, body);
  div.appendChild(body);

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

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
      // If we have a rich UI payload (e.g. weather card), show chat text AND the UI.
      // If the A2UI payload is text-only, render only A2UI to avoid duplicates.
      if (update && update.a2ui) {
        const rich = update.a2ui && update.a2ui.schema === "a2ui" && a2uiHasRichContent(update.a2ui.render);
        appendA2UIMessage(nodeName, update.a2ui, rich ? update.output : undefined);
      } else if (update && update.output) {
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
