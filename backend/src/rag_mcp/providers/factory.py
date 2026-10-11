"""Provider factory and startup validation (006, T038).

Assembles the three capability providers (embedding / reranker / llm) from
the unified run configuration and validates the whole configuration at
startup (FR-010/FR-011, SC-004): unknown provider_type, missing remote
endpoint, GPU-without-CUDA, and embedding dimension mismatch all fail
loudly with correctable errors — never silently falling back.

Vendor-neutral (FR-012): OpenAI/Anthropic-compatible endpoints are treated
as adapter protocols only; no vendor SDK is bound. Credentials are
referenced by environment variable NAME (constitution V).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rag_mcp.config.provider_config import (
    ProviderConfig,
    ProviderSettings,
)
from rag_mcp.providers.concurrency import ConcurrencyLimiter, build_limiters

logger = logging.getLogger(__name__)

VALID_PROVIDER_TYPES = ("local_cpu", "local_gpu", "remote_api")


@dataclass(frozen=True)
class ValidationError:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    def to_contract_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": [e.to_dict() for e in self.errors]}


@dataclass
class ProviderBundle:
    """The assembled runtime providers + validation + per-capability limits."""

    embedding: Any
    reranker: Any
    llm: Any
    validation: ValidationResult
    limiters: dict[str, ConcurrencyLimiter]


def _resolve_endpoint(cfg: ProviderConfig, capability: str, llm_base_url: str) -> str | None:
    if cfg.endpoint:
        return cfg.endpoint
    if capability == "llm" and llm_base_url:
        return llm_base_url
    return None


def validate_provider_config(
    providers: ProviderSettings,
    *,
    embedding_dimension: int | None = None,
    active_collection_dimension: int | None = None,
    llm_base_url: str = "",
) -> ValidationResult:
    """Startup validation of the unified Provider configuration (SC-004).

    Every failure is actionable; there is no silent fallback path.
    """
    errors: list[ValidationError] = []

    for capability in ("embedding", "reranker", "llm"):
        cfg = providers.by_capability(capability)
        if cfg.provider_type not in VALID_PROVIDER_TYPES:
            errors.append(
                ValidationError(
                    "INVALID_PROVIDER_TYPE",
                    f"unknown provider_type {cfg.provider_type!r} for {capability}: "
                    f"expected one of {VALID_PROVIDER_TYPES}",
                )
            )
            continue
        if cfg.provider_type == "remote_api":
            endpoint = _resolve_endpoint(cfg, capability, llm_base_url)
            if not endpoint:
                errors.append(
                    ValidationError(
                        "MISSING_ENDPOINT",
                        f"{capability} remote_api requires an endpoint; set "
                        f"{capability.upper()}_ENDPOINT (or LLM_BASE_URL for llm)",
                    )
                )
        if cfg.provider_type == "local_gpu":
            from rag_mcp.providers.local_gpu import is_gpu_available

            if not is_gpu_available():
                errors.append(
                    ValidationError(
                        "GPU_UNAVAILABLE",
                        f"{capability} local_gpu requires CUDA but no GPU is "
                        f"available; refusing to silently fall back to CPU",
                    )
                )

    if embedding_dimension is not None and active_collection_dimension is not None:
        if embedding_dimension != active_collection_dimension:
            errors.append(
                ValidationError(
                    "EMBEDDING_DIMENSION_MISMATCH",
                    f"declared embedding dimension {embedding_dimension} does not "
                    f"match the active collection dimension {active_collection_dimension}; "
                    f"create a new index version and re-vectorize (FR-013, constitution VIII)",
                )
            )

    return ValidationResult(valid=not errors, errors=errors)


def _build_embedding(cfg: ProviderConfig, limiter: ConcurrencyLimiter | None = None):
    if cfg.provider_type == "local_cpu":
        from rag_mcp.providers.local_cpu import LocalCPUEmbeddingProvider

        return LocalCPUEmbeddingProvider(model_name=cfg.model)
    if cfg.provider_type == "local_gpu":
        from rag_mcp.providers.local_gpu import LocalGPUEmbeddingProvider

        return LocalGPUEmbeddingProvider(model_name=cfg.model, device="cuda")
    if cfg.provider_type == "remote_api":
        from rag_mcp.providers.remote_api_embedding import RemoteAPIEmbeddingProvider

        return RemoteAPIEmbeddingProvider(
            endpoint=cfg.endpoint or "",
            model=cfg.model,
            api_key_env=cfg.api_key_env,
            timeout_s=cfg.timeout_s,
            limiter=limiter,
        )
    raise ValueError(f"unsupported embedding provider_type {cfg.provider_type!r}")


def _build_reranker(cfg: ProviderConfig, limiter: ConcurrencyLimiter | None = None):
    if cfg.provider_type in ("local_cpu", "local_gpu"):
        from rag_mcp.providers.local_cpu_reranker import LocalCPUReranker

        # Local providers queue on CPU/GPU; FR-009 limiting is remote-only here.
        return LocalCPUReranker(model_name=cfg.model)
    if cfg.provider_type == "remote_api":
        from rag_mcp.providers.remote_api_reranker import RemoteAPIRerankerProvider

        return RemoteAPIRerankerProvider(
            endpoint=cfg.endpoint or "",
            model=cfg.model,
            api_key_env=cfg.api_key_env,
            timeout_s=cfg.timeout_s,
            limiter=limiter,
        )
    raise ValueError(f"unsupported reranker provider_type {cfg.provider_type!r}")


def _build_llm(cfg: ProviderConfig, settings):
    import os

    if cfg.provider_type == "remote_api":
        from rag_mcp.agents.llm_client import LLMClient

        endpoint = cfg.endpoint or settings.llm_base_url
        api_key = os.getenv(cfg.api_key_env, "") if cfg.api_key_env else settings.llm_api_key
        return LLMClient(
            base_url=endpoint or "",
            api_key=api_key,
            model=cfg.model or settings.llm_model,
        )
    raise ValueError(f"unsupported llm provider_type {cfg.provider_type!r}")


def build_provider_bundle(settings) -> ProviderBundle:
    """Assemble the runtime providers and validate the configuration."""
    providers = settings.providers
    # Build each capability's limiter once, then hand the same objects to the
    # providers that enforce them (FR-009) — bundle.limiters stays the single
    # authoritative set.
    limiters = build_limiters(providers)
    embedding = _build_embedding(providers.embedding, limiter=limiters["embedding"])
    reranker = _build_reranker(providers.reranker, limiter=limiters["reranker"])
    llm = _build_llm(providers.llm, settings)
    validation = validate_provider_config(
        providers,
        llm_base_url=settings.llm_base_url,
    )
    return ProviderBundle(
        embedding=embedding,
        reranker=reranker,
        llm=llm,
        validation=validation,
        limiters=limiters,
    )


def assemble_or_fail(settings) -> ProviderBundle:
    """Startup entry: build providers and refuse to start on invalid config.

    SC-004: illegal configurations fail the instance explicitly; there is
    no silent fallback path.
    """
    bundle = build_provider_bundle(settings)
    if not bundle.validation.valid:
        messages = "; ".join(e.message for e in bundle.validation.errors)
        raise ValueError(f"invalid provider configuration: {messages}")
    return bundle


def check_embedding_dimension(
    embedding_provider: Any, active_collection_dimension: int | None
) -> ValidationResult:
    """Dimension consistency check against the active Dense collection (FR-011/FR-013).

    Returns a mismatch error when the declared embedding dimension differs
    from the active collection dimension; the only legal resolution is a new
    index version + re-vectorization (no in-place mixing).
    """
    if active_collection_dimension is None:
        return ValidationResult(valid=True, errors=[])
    declared = embedding_provider.get_dimension()
    if declared and declared != active_collection_dimension:
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError(
                    "EMBEDDING_DIMENSION_MISMATCH",
                    f"embedding dimension {declared} != active collection "
                    f"dimension {active_collection_dimension}; create a new index "
                    f"version and re-vectorize (FR-013)",
                )
            ],
        )
    return ValidationResult(valid=True, errors=[])
