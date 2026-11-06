"""Dataset-specific adapters for the unified opinion-dynamics JSONL corpus."""

from opinion_dynamics.datasets.adapters.anthropic_persuasion import AnthropicPersuasionAdapter
from opinion_dynamics.datasets.adapters.base import BaseDatasetAdapter
from opinion_dynamics.datasets.adapters.cmv import CMVAdapter
from opinion_dynamics.datasets.adapters.conspiracyai import ConspiracyAIAdapter
from opinion_dynamics.datasets.adapters.debate import DebateAdapter
from opinion_dynamics.datasets.adapters.debategpt import DebateGPTAdapter
from opinion_dynamics.datasets.adapters.persuasiontrace import PersuasionTraceAdapter

ADAPTER_REGISTRY: dict[str, type[BaseDatasetAdapter]] = {
    "DEBATE": DebateAdapter,
    "debate": DebateAdapter,
    "CMV": CMVAdapter,
    "cmv": CMVAdapter,
    "PersuasionTrace": PersuasionTraceAdapter,
    "persuasiontrace": PersuasionTraceAdapter,
    "ConspiracyAI": ConspiracyAIAdapter,
    "conspiracyai": ConspiracyAIAdapter,
    "AnthropicPersuasion": AnthropicPersuasionAdapter,
    "anthropicpersuasion": AnthropicPersuasionAdapter,
    "anthropic_persuasion": AnthropicPersuasionAdapter,
    "DebateGPT": DebateGPTAdapter,
    "debategpt": DebateGPTAdapter,
}


def get_adapter(name: str) -> BaseDatasetAdapter:
    key = name.strip()
    if key not in ADAPTER_REGISTRY:
        raise KeyError(f"Unknown dataset adapter {name!r}. Known: {sorted(set(ADAPTER_REGISTRY))}")
    return ADAPTER_REGISTRY[key]()


__all__ = [
    "ADAPTER_REGISTRY",
    "AnthropicPersuasionAdapter",
    "BaseDatasetAdapter",
    "CMVAdapter",
    "ConspiracyAIAdapter",
    "DebateAdapter",
    "DebateGPTAdapter",
    "PersuasionTraceAdapter",
    "get_adapter",
]
