"""
Simple RAG (Retrieval‑Augmented Generation) support for the multi‑agent system.

This module defines a lightweight retrieval component using TF‑IDF vectors from
``scikit‑learn`` to find the most relevant document given a natural language
query.  It does not depend on any external embedding services or APIs, making it
quick to set up for demonstration purposes.  A small set of documents is
provided by default but more can be supplied when instantiating the class.

The result returned includes the document title, content and a citation string.
The citation is a simple index into the internal document list.  In a real
application you might replace this with a URL or other identifier.
"""

from __future__ import annotations

from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGSearch:
    """Retrieval component performing simple TF‑IDF similarity search."""

    def __init__(self, documents: Optional[List[Dict[str, str]]] = None) -> None:
        """Initialise the RAG search instance.

        Args:
            documents: A list of dicts with keys ``title`` and ``content``.  If
                not provided, a small default corpus is used.
        """
        if documents is None:
            documents = [
                {
                    "title": "LangGraph Overview",
                    "content": (
                        "LangGraph is a framework that lets developers orchestrate large "
                        "language models using a graph abstraction. It supports multi-agent "
                        "coordination, conditional routing and streaming outputs to build complex "
                        "applications."
                    ),
                },
                {
                    "title": "Meeting Management Tips",
                    "content": (
                        "Organising a productive meeting requires preparing an agenda, inviting "
                        "the right stakeholders, and capturing notes and actions. Following a "
                        "clear structure keeps participants engaged and ensures the meeting stays on track."
                    ),
                },
                {
                    "title": "Web Searching Basics",
                    "content": (
                        "Web search engines crawl and index billions of pages. When you perform a "
                        "search, they return the most relevant documents using ranking algorithms. "
                        "Choosing the right keywords helps to narrow results to the information you need."
                    ),
                },
            ]
        # store documents and build TF‑IDF vectors
        self.documents = documents
        self.vectorizer = TfidfVectorizer().fit([doc["content"] for doc in documents])
        self.doc_vectors = self.vectorizer.transform([doc["content"] for doc in documents])

    def search(self, query: str) -> Dict[str, str]:
        """Return the most relevant document for a given query.

        Args:
            query: Natural language question or statement.

        Returns:
            Dictionary containing ``title``, ``content`` and ``citation`` keys.
        """
        if not query:
            return {
                "title": "",
                "content": "No query provided.",
                "citation": "",
            }
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_idx = sims.argmax()
        best_doc = self.documents[top_idx]
        return {
            "title": best_doc["title"],
            "content": best_doc["content"],
            "citation": f"[doc{top_idx + 1}]",
        }


__all__ = ["RAGSearch"]
