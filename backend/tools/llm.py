"""
Utilities for interacting with a language model via LangChain.

This module exposes helper functions to lazily construct an LLM backed by
Azure OpenAI if the appropriate environment variables are set.  It also
provides a convenience wrapper to send prompts and capture responses.  The
logger defined here integrates with the application‑wide logging so that all
requests and replies are recorded.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema.messages import HumanMessage

logger = logging.getLogger("agent_backend.llm")


_cached_llm: Optional[Any] = None


def get_llm() -> Any:
    """Return a lazily constructed ChatModel instance.

    This function inspects environment variables to determine whether to
    instantiate an Azure OpenAI chat model or the standard OpenAI model.  The
    supported environment variables are:

    - ``AZURE_OPENAI_API_KEY``: API key for Azure.
    - ``AZURE_OPENAI_API_BASE``: Base endpoint, e.g. ``https://YOUR_RESOURCE_NAME.openai.azure.com``.
    - ``AZURE_OPENAI_API_VERSION``: API version, e.g. ``2023-07-01-preview``.
    - ``AZURE_OPENAI_DEPLOYMENT_NAME``: Name of the deployment (model).
    - ``OPENAI_API_KEY``: API key for the standard OpenAI API (fallback).

    Returns:
        An instance of ChatOpenAI or AzureChatOpenAI.
    """
    global _cached_llm
    if _cached_llm is not None:
        return _cached_llm
    # Prefer Azure configuration if all required variables are present
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_base = os.getenv("AZURE_OPENAI_API_BASE")
    azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if azure_key and azure_base and azure_version and azure_deployment:
        logger.info("Initialising Azure OpenAI model")
        _cached_llm = AzureChatOpenAI(
            azure_endpoint=azure_base,
            azure_deployment_name=azure_deployment,
            openai_api_version=azure_version,
            openai_api_key=azure_key,
            temperature=0,
        )
        return _cached_llm
    # Fallback to standard OpenAI API
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("Initialising standard OpenAI model")
        _cached_llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)
        return _cached_llm
    # If no keys are configured return None
    logger.warning(
        "No OpenAI API keys found.  Language model features will be disabled."
    )
    return None


def ask_llm(prompt: str) -> str:
    """Send a prompt to the configured language model and return the response.

    The request and response are logged to ensure visibility into model usage.
    If no model is configured or an error occurs, this function will raise an
    exception.

    Args:
        prompt: The string prompt to send to the chat model.

    Returns:
        The text response from the model.
    """
    llm = get_llm()
    if llm is None:
        raise RuntimeError(
            "No language model configured.  Set the appropriate environment variables."
        )
    logger.info(f"LLM request: {prompt}")
    try:
        response = llm([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"LLM response: {answer}")
        return answer
    except Exception:
        logger.exception("Error during LLM request")
        raise
