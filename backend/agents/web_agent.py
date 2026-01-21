from __future__ import annotations

from ..tools.agent_config import get_agent_settings
from ..tools.langfuse_tracing import end_span, start_span
from ..tools.llm import ask_llm, get_llm
from ..tools.web import search_web

from .types import AgentState


def web_agent(state: AgentState) -> AgentState:
    """Perform a web search and summarise using an LLM when configured."""
    _span = start_span(name="agent:web_agent", input={"state": state}, metadata={"kind": "agent"})
    query = state.get("input", "")
    results = search_web(query, max_results=3)
    web_settings = get_agent_settings("web_agent")

    lines = []
    for idx, res in enumerate(results, start=1):
        title = res.get("title") or "Untitled"
        body = res.get("body") or ""
        href = res.get("href") or ""
        line = f"{idx}. {title} – {body[:120]}"
        if href:
            line += f" ({href})"
        lines.append(line)
    summary = "\n".join(lines)

    llm = get_llm(web_settings.model_name)
    if llm:
        if results:
            prompt = (
                "A user asked the following question:\n\n"
                f"{query}\n\n"
                "Here are the top web search results:\n\n"
                f"{summary}\n\n"
                "Answer the user concisely based only on these results. If the results are insufficient, say so."
            )
        else:
            prompt = (
                "A user asked the following question, but web search returned no results. "
                "Answer as best you can from general knowledge, and clearly say that live search returned no results.\n\n"
                f"User question: {query}"
            )
        try:
            answer = ask_llm(
                prompt,
                model_name=web_settings.model_name,
                system_prompt=web_settings.system_prompt,
            )
            out = {"output": answer}
            end_span(_span, output=out)
            return out
        except Exception:
            pass

    if not results:
        out = {"output": "No search results found."}
        end_span(_span, output=out)
        return out

    out = {"output": summary}
    end_span(_span, output=out)
    return out
