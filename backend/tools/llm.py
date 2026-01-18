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
from typing import Any, Optional, Dict

try:
    # Optional dependency used only when authenticating to Azure OpenAI via
    # Azure Entra ID (AAD) client credentials.
    from azure.identity import ClientSecretCredential  # type: ignore
except Exception:  # pragma: no cover
    ClientSecretCredential = None  # type: ignore

# Import chat model classes. LangChain moved OpenAI integrations into the
# separate `langchain_openai` package; support both layouts.
try:
    from langchain_openai import ChatOpenAI, AzureChatOpenAI  # type: ignore
except ImportError:  # pragma: no cover
    from langchain.chat_models import ChatOpenAI, AzureChatOpenAI  # type: ignore

# Message classes live in langchain-core in recent versions.
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger("agent_backend.llm")

# Cache multiple LLM instances keyed by model name to avoid reinitialising
_cached_llms: Dict[str, Any] = {}


def _init_azure_chat_openai(
    *,
    azure_endpoint: str,
    azure_deployment: str,
    api_version: str,
    temperature: float,
    api_key: Optional[str] = None,
    azure_ad_token_provider: Optional[Any] = None,
) -> Any:
    """Create an AzureChatOpenAI with best-effort compatibility across versions.

    The `langchain_openai` package has changed parameter names over time.
    We try the newer names first, then fall back.
    """

    # Newer langchain_openai (common): azure_deployment + api_version
    try:
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            temperature=temperature,
            openai_api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
        )
    except TypeError:
        pass

    # Older: azure_deployment_name + openai_api_version
    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment_name=azure_deployment,
        openai_api_version=api_version,
        temperature=temperature,
        openai_api_key=api_key,
        azure_ad_token_provider=azure_ad_token_provider,
    )


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

    # Check for Azure OpenAI configuration.
    # Supports either:
    #  1) API key auth: AZURE_OPENAI_API_KEY
    #  2) Azure Entra ID (AAD) client credentials: AZURE_TENANT_ID/CLIENT_ID/CLIENT_SECRET
    azure_base = os.getenv("AZURE_OPENAI_API_BASE")
    azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    if tenant_id and client_id and client_secret and azure_base and azure_version and azure_deployment:
        if ClientSecretCredential is None:
            raise RuntimeError(
                "Azure Entra ID auth requested but azure-identity isn't installed. "
                "Install `azure-identity` or use AZURE_OPENAI_API_KEY instead."
            )

        logger.info(
            "Initialising Azure OpenAI model using Entra ID client credentials "
            f"(deployment={azure_deployment})"
        )
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

        # Azure OpenAI token scope.
        scope = os.getenv("AZURE_OPENAI_SCOPE", "https://cognitiveservices.azure.com/.default")

        # LangChain expects a callable token provider (string bearer token).
        def token_provider() -> str:
            token = credential.get_token(scope)
            return token.token

        llm = _init_azure_chat_openai(
            azure_endpoint=azure_base,
            azure_deployment=azure_deployment,
            api_version=azure_version,
            azure_ad_token_provider=token_provider,
            api_key=None,
            temperature=0,
        )
        _cached_llms[key] = llm
        return llm

    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_key and azure_base and azure_version and azure_deployment:
        logger.info(f"Initialising Azure OpenAI model (API key auth, deployment={azure_deployment})")
        llm = _init_azure_chat_openai(
            azure_endpoint=azure_base,
            azure_deployment=azure_deployment,
            api_version=azure_version,
            api_key=azure_key,
            azure_ad_token_provider=None,
            temperature=0,
        )
        _cached_llms[key] = llm
        return llm

    # Fallback to standard OpenAI configuration
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        kwargs = {"openai_api_key": openai_key, "temperature": 0}
        if model_name:
            # `langchain_openai` uses `model`, older uses `model_name`.
            kwargs["model"] = model_name
        logger.info("Initialising standard OpenAI model")
        try:
            llm = ChatOpenAI(**kwargs)
        except TypeError:
            # Compatibility fallback
            if "model" in kwargs:
                kwargs["model_name"] = kwargs.pop("model")
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
    history: Optional[list[dict[str, str]]] = None,
) -> str:
    """Send a prompt to the configured language model and return the response.

    The caller may specify a ``model_name`` (Azure deployment name or OpenAI
    model) and a ``system_prompt`` which will be prepended as a system message.
    A ``history`` of prior conversation messages may also be provided to
    influence the model's response.  History should be a list of dicts with
    ``role`` (``"user"`` or ``"assistant"``) and ``content`` keys.
    If no model is configured or an error occurs during the request this
    function will raise an exception.

    Args:
        prompt: The user message to send to the LLM.
        model_name: Optional model or deployment name to override the default.
        system_prompt: Optional system prompt to include as context.
        history: Optional list of prior conversation turns to include.

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
    # Append history messages if provided
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
    # Append the current user prompt as the last message
    messages.append(HumanMessage(content=prompt))
    logger.info(f"LLM request (model={model_name}): {prompt}")
    try:
        # LangChain v1 chat models use the Runnable interface.
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"LLM response: {answer}")
        return answer
    except Exception:
        logger.exception("Error during LLM request")
        raise