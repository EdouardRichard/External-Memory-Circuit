"""Provider runtime configuration (006, T002).

Loads the three Provider capability configs (embedding / reranker / llm)
from environment variables following the 001 frozen-Settings pattern
(research §1.4):

- `{EMBEDDING|RERANKER|LLM}_PROVIDER_TYPE` — local_cpu / local_gpu / remote_api
- `{PREFIX}_MODEL` — model identifier
- `{PREFIX}_ENDPOINT` — remote_api base URL (OpenAI-compatible adapter protocol)
- `{PREFIX}_API_KEY_ENV` — credential env var NAME only (constitution V: the
  credential value never enters any config structure)
- `{PREFIX}_CONCURRENCY_LIMIT` (+ `_MAX` ceiling) — per-capability limit,
  clamped to the hard ceiling (clarification Q2: LLM 4/8, Embedding 8/16,
  Reranker 2/4; schema ceiling 64).
- `{PREFIX}_DIMENSION` — optional declared output vector dimension for
  embedding/reranker; unset (or blank) lets the Provider resolve it.
- `{PREFIX}_TIMEOUT_S` — optional per-call timeout in seconds; unset (or
  blank) uses the Provider constructor default.

Default combination (quickstart): local CPU embedding + reranker, remote LLM.
Illegal values (unknown provider_type, non-integer limits) fail loudly here;
cross-field startup validation lives in providers/factory.py (T038).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

VALID_PROVIDER_TYPES = ("local_cpu", "local_gpu", "remote_api")
VALID_CAPABILITIES = ("embedding", "reranker", "llm")

# capability -> (default limit, default hard ceiling), clarification Q2
DEFAULT_CONCURRENCY: dict[str, tuple[int, int]] = {
    "llm": (4, 8),
    "embedding": (8, 16),
    "reranker": (2, 4),
}

# capability -> env prefix
_ENV_PREFIX: dict[str, str] = {
    "embedding": "EMBEDDING",
    "reranker": "RERANKER",
    "llm": "LLM",
}

_DEFAULT_MODELS: dict[str, str] = {
    "embedding": "BAAI/bge-m3",
    "reranker": "BAAI/bge-reranker-v2-m3",
    "llm": "deepseek-v4-flash",
}

_DEFAULT_PROVIDER_TYPES: dict[str, str] = {
    "embedding": "local_cpu",
    "reranker": "local_cpu",
    "llm": "remote_api",
}

# 可选解析项：None 表示未设置（由 Provider 自身默认值或运行时探测决定）
_DEFAULT_DIMENSION: dict[str, int | None] = {"embedding": None, "reranker": None, "llm": None}
_DEFAULT_TIMEOUT_S: dict[str, float | None] = {"embedding": None, "reranker": None, "llm": None}

# Provider-level concurrency ceiling shared with the 006 contract schema
# (common.schema.json ConcurrencyLimit maximum).
SCHEMA_CONCURRENCY_CEILING = 64

_TRUTHY = ("1", "true", "yes", "on")


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


@dataclass(frozen=True)
class ProviderConfig:
    """One capability's resolved Provider configuration (immutable)."""

    capability: str
    provider_type: str
    model: str
    endpoint: str | None
    api_key_env: str | None
    concurrency_limit: int
    concurrency_limit_max: int
    dimension: int | None = None
    timeout_s: float | None = None

    def to_contract_dict(self) -> dict:
        """Serialize in provider-config.schema.json ProviderCapabilityConfig form."""
        return {
            "provider_type": self.provider_type,
            "model": self.model,
            "endpoint": self.endpoint,
            "api_key_env": self.api_key_env,
            "concurrency_limit": self.concurrency_limit,
            "dimension": self.dimension,
            "timeout_s": self.timeout_s,
        }


def load_provider_config(capability: str, env: Mapping[str, str] | None = None) -> ProviderConfig:
    """Load one capability's ProviderConfig from environment variables.

    Raises ValueError on an unknown capability or provider_type, a
    non-integer concurrency value, or a non-positive dimension/timeout
    (fail loudly, never silently fall back — SC-004).
    """
    if capability not in VALID_CAPABILITIES:
        raise ValueError(f"unknown provider capability: {capability!r}")
    lookup = os.getenv if env is None else (lambda name, default=None: env.get(name, default))
    prefix = _ENV_PREFIX[capability]

    provider_type = lookup(f"{prefix}_PROVIDER_TYPE", _DEFAULT_PROVIDER_TYPES[capability])
    if provider_type not in VALID_PROVIDER_TYPES:
        raise ValueError(
            f"invalid provider_type {provider_type!r} for {capability}: "
            f"expected one of {VALID_PROVIDER_TYPES}"
        )

    model = lookup(f"{prefix}_MODEL", _DEFAULT_MODELS[capability])
    if not model:
        raise ValueError(f"{prefix}_MODEL must not be empty for capability {capability}")

    endpoint = lookup(f"{prefix}_ENDPOINT") or None
    api_key_env = lookup(f"{prefix}_API_KEY_ENV") or None

    default_limit, default_ceiling = DEFAULT_CONCURRENCY[capability]
    raw_ceiling = lookup(f"{prefix}_CONCURRENCY_LIMIT_MAX")
    ceiling = default_ceiling
    if raw_ceiling is not None:
        try:
            ceiling = int(raw_ceiling)
        except ValueError as exc:
            raise ValueError(
                f"{prefix}_CONCURRENCY_LIMIT_MAX must be an integer, got {raw_ceiling!r}"
            ) from exc
    # The ceiling itself is bounded by the contract schema maximum.
    ceiling = _clamp(ceiling, 1, SCHEMA_CONCURRENCY_CEILING)

    raw_limit = lookup(f"{prefix}_CONCURRENCY_LIMIT")
    limit = default_limit
    if raw_limit is not None:
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(
                f"{prefix}_CONCURRENCY_LIMIT must be an integer, got {raw_limit!r}"
            ) from exc
    # Clamp the effective limit into [1, ceiling] (FR-009 bounded guardrail).
    limit = _clamp(limit, 1, ceiling)

    # Optional declarations: blank or missing means "unset" (None).
    raw_dimension = lookup(f"{prefix}_DIMENSION")
    dimension = _DEFAULT_DIMENSION[capability]
    if raw_dimension is not None and raw_dimension.strip():
        try:
            dimension = int(raw_dimension)
        except ValueError as exc:
            raise ValueError(
                f"{prefix}_DIMENSION must be a positive integer, got {raw_dimension!r}"
            ) from exc
        if dimension < 1:
            raise ValueError(
                f"{prefix}_DIMENSION must be a positive integer, got {raw_dimension!r}"
            )

    raw_timeout = lookup(f"{prefix}_TIMEOUT_S")
    timeout_s = _DEFAULT_TIMEOUT_S[capability]
    if raw_timeout is not None and raw_timeout.strip():
        try:
            timeout_s = float(raw_timeout)
        except ValueError as exc:
            raise ValueError(
                f"{prefix}_TIMEOUT_S must be positive, got {raw_timeout!r}"
            ) from exc
        if timeout_s <= 0:
            raise ValueError(f"{prefix}_TIMEOUT_S must be positive, got {raw_timeout!r}")

    return ProviderConfig(
        capability=capability,
        provider_type=provider_type,
        model=model,
        endpoint=endpoint,
        api_key_env=api_key_env,
        concurrency_limit=limit,
        concurrency_limit_max=ceiling,
        dimension=dimension,
        timeout_s=timeout_s,
    )


@dataclass(frozen=True)
class ProviderSettings:
    """The three resolved capability configs (embedding/reranker/llm)."""

    embedding: ProviderConfig
    reranker: ProviderConfig
    llm: ProviderConfig

    def by_capability(self, capability: str) -> ProviderConfig:
        return getattr(self, capability)


def load_provider_settings(env: Mapping[str, str] | None = None) -> ProviderSettings:
    """Assemble ProviderSettings from the environment."""
    return ProviderSettings(
        embedding=load_provider_config("embedding", env),
        reranker=load_provider_config("reranker", env),
        llm=load_provider_config("llm", env),
    )
