"""Agent configuration loader.

This project keeps per-agent model settings and system prompts in
`backend/agent_config.yaml`.

The loader is intentionally small and tolerant:
- If the YAML file is missing or invalid, it falls back to empty defaults.
- Callers can still override model names explicitly at call sites.

The YAML schema:

- default_model: <string | null>
- supervisor/web_agent/meetings_agent/rag_agent:
    system_prompt: <string | null>
    model_name: <string | null>
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Any

import yaml


@dataclass(frozen=True)
class AgentSettings:
    model_name: Optional[str]
    system_prompt: Optional[str]


def _project_root() -> str:
    # backend/tools/agent_config.py -> backend/tools -> backend -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_config_path() -> str:
    return os.path.join(_project_root(), "backend", "agent_config.yaml")


@lru_cache(maxsize=1)
def load_agent_config(path: Optional[str] = None) -> dict[str, Any]:
    config_path = path or os.getenv("AGENT_CONFIG_PATH") or _default_config_path()
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except FileNotFoundError:
        return {}
    except Exception:
        # Keep this loader non-fatal; failures should not crash the server.
        return {}


def get_agent_settings(agent_name: str) -> AgentSettings:
    config = load_agent_config()
    default_model = config.get("default_model")

    agent_block = config.get(agent_name, {}) if isinstance(config, dict) else {}
    if not isinstance(agent_block, dict):
        agent_block = {}

    model_name = agent_block.get("model_name")
    if model_name is None:
        model_name = default_model

    system_prompt = agent_block.get("system_prompt")

    return AgentSettings(
        model_name=model_name if isinstance(model_name, str) else None,
        system_prompt=system_prompt if isinstance(system_prompt, str) else None,
    )
