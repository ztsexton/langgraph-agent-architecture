"""
Utilities for interacting with a language model via LangChain.

This module exposes helper functions to lazily construct ChatModel instances
based on environment variables and optional caller‑supplied model names.  When
the Azure OpenAI variables are provided the function will instantiate
``AzureChatOpenAI`` using the specified deployment name; otherwise it will
fall back to the standard OpenAI chat model.  To support per‑agent
customisation, callers may pass ``model_name`` and a ``system_prompt`` when
sending a prompt.  All requests and responses are logged for observability.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional, Dict, Iterable

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

logger = logging.getLogger("agent_backend.llm")

# Cache multiple LLM instances keyed by model name to avoid reinitialising
_cached_llms: Dict[str, Any] = {}


def get_llm(model_name: Optional[str] = None) -> Any:
    """Return a lazily constructed chat model instance.

    Args:
        model_name: Optional name of the Azure deployment or OpenAI model to
            initialise.  If ``None``, the default deployment name is taken from
            ``AZURE_OPENAI_DEPLOYMENT_NAME`` or the standard OpenAI model is
            used.

    Returns:
        An instance of ``AzureChatOpenAI`` or ``ChatOpenAI``, or ``None`` if
        no API key configuration is available.
    """
    key = model_name or "__default__"
    if key in _cached_llms:
        return _cached_llms[key]

    # Check for Azure OpenAI configuration
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_base = os.getenv("AZURE_OPENAI_API_BASE")
    azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if azure_key and azure_base and azure_version and azure_deployment:
        logger.info(f"Initialising Azure OpenAI model (deployment={azure_deployment})")
        llm = AzureChatOpenAI(
            azure_endpoint=azure_base,
            azure_deployment_name=azure_deployment,
            openai_api_version=azure_version,
            openai_api_key=azure_key,
            temperature=0,
        )
        _cached_llms[key] = llm
        return llm

    # Fallback to standard OpenAI configuration
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        kwargs = {"openai_api_key": openai_key, "temperature": 0}
        if model_name:
            kwargs["model_name"] = model_name
        logger.info("Initialising standard OpenAI model")
        llm = ChatOpenAI(**kwargs)
        _cached_llms[key] = llm
        return llm

    logger.warning(
        "No OpenAI API keys found.  Language model features will be disabled."
    )
    return None


def ask_llm(
    prompt: str,
    *,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Send a prompt to the configured language model and return the response.

    The caller may specify a ``model_name`` (Azure deployment name or OpenAI
    model) and a ``system_prompt`` which will be prepended as a system message.
    If no model is configured or an error occurs during the request this
    function will raise an exception.

    Args:
        prompt: The user message to send to the LLM.
        model_name: Optional model or deployment name to override the default.
        system_prompt: Optional system prompt to include as context.

    Returns:
        The text response from the LLM.
    """
    llm = get_llm(model_name)
    if llm is None:
        raise RuntimeError(
            "No language model configured.  Set the appropriate environment variables."
        )
    # Build message sequence for the chat model
    messages: list = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    logger.info(f"LLM request (model={model_name}): {prompt}")
    try:
        response = llm(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"LLM response: {answer}")
        return answer
    except Exception:
        logger.exception("Error during LLM request")
        raise
