"""
Retrieval‑augmented generation (RAG) tool.

This tool uses an in‑memory vector store to retrieve relevant documents and can
optionally invoke a language model to produce an answer conditioned on those
documents.  It exposes a single function that accepts a query string and
returns a formatted answer with a citation.
"""

from typing import Dict, Any

from ..rag import RAGSearch
from .llm import ask_llm
from .agent_config import get_agent_settings

# Shared search index across calls
_rag_search = RAGSearch()


def answer_question(query: str) -> Dict[str, Any]:
    """Answer a question using retrieval and optional LLM generation.

    Args:
        query: The user’s question.

    Returns:
        A dictionary with ``content`` (the answer) and ``citation`` (reference
        identifier).  If an LLM is configured via environment variables the
        answer will be generated using the retrieved document.  Otherwise the
        raw document content is returned.
    """
    result = _rag_search.search(query)
    doc_content = result["content"]
    citation = result["citation"]
    # Attempt to call LLM to formulate an answer using the document
    prompt = (
        f"You are a helpful assistant.  Use the following information to answer "
        f"the question.  Document:\n\n{doc_content}\n\n"
        f"Question: {query}\n\nAnswer in a concise and direct manner."
    )
    try:
        rag_settings = get_agent_settings("rag_agent")
        answer = ask_llm(
            prompt,
            model_name=rag_settings.model_name,
            system_prompt=rag_settings.system_prompt,
        )
        return {"content": answer, "citation": citation}
    except Exception:
        # If the LLM is not configured or fails, return the raw document
        return {"content": doc_content, "citation": citation}
