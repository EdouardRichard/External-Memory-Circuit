"""Standalone MCP server entry point for Streamable HTTP transport.

006 Runtime Hardening: supports `--mode writer|reader` (INSTANCE_MODE env
equivalent, research §1.6). Both forms register the read-only MCP tools
`search_knowledge` / `get_evidence`; neither MCP form runs ingestion, TTL
cleanup or migrations — maintenance belongs to the writer management
process (server.py, FR-004). At startup each form verifies the shared
schema compatibility (FR-007), validates timeout profiles (FR-021) and
registers an instance row with an allocated worker_id (FR-030).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from rag_mcp.config import get_settings
from rag_mcp.config.timeout_profiles import validate_timeout_profiles
from rag_mcp.indexing.qdrant_client import QdrantStore
from rag_mcp.mcp import create_mcp_server

_INSTANCE_MODES = ("writer", "reader")


def _valid_instance_mode(raw: str | None) -> str:
    value = (raw or "writer").strip().lower()
    if value not in _INSTANCE_MODES:
        raise ValueError(f"INSTANCE_MODE must be one of {_INSTANCE_MODES}, got {raw!r}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args; --mode defaults to the INSTANCE_MODE env equivalent."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="RAG MCP server (Streamable HTTP; 006 --mode writer|reader)"
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=list(_INSTANCE_MODES),
        help="Instance mode (default: INSTANCE_MODE env or 'writer')",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="MCP port (default from MCP_PORT, 8080)",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help="Explicit snowflake worker_id (default: WORKER_ID env or auto)",
    )
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = _valid_instance_mode(os.getenv("INSTANCE_MODE", "writer"))
    if args.worker_id is None:
        args.worker_id = settings.worker_id
    return args


def resolve_mode(args: argparse.Namespace) -> str:
    """Resolve the effective instance mode (CLI value already validated)."""
    return _valid_instance_mode(getattr(args, "mode", None))


def background_tasks_for_mode(mode: str) -> list[str]:
    """Background work each MCP form runs (FR-004).

    Neither writer MCP nor reader MCP starts ingestion, TTL cleanup or
    migrations — the maintenance write path belongs to the writer
    management process (server.py). Both forms keep their instance
    registration alive via heartbeat (per-request runtime state, FR-004
    exemption).
    """
    return ["instance_heartbeat"]


def validate_timeout_profiles_at_startup() -> None:
    """T022/FR-021: both MCP forms reject reversed timeout profiles."""
    settings = get_settings()
    errors = validate_timeout_profiles(settings.timeout_profiles)
    if errors:
        raise ValueError(
            "invalid timeout profile configuration: " + "; ".join(errors)
        )


def assemble_runtime_providers(settings):
    """FR-010/FR-011/SC-004: assemble embedding+reranker from run config.

    Replaces the hardcoded local-CPU providers so the reader/writer MCP uses
    the unified Provider configuration; an invalid configuration fails the
    instance loudly instead of silently falling back (no silent downgrade).
    """
    from rag_mcp.providers.factory import assemble_or_fail

    bundle = assemble_or_fail(settings)
    return bundle.embedding, bundle.reranker


def _warmup_provider(label: str, provider) -> None:
    """Eagerly warm a local model if the provider supports warmup.

    Remote providers have no warmup step (they are stateless HTTP adapters);
    GPU providers validate hardware lazily. Both are skipped gracefully.
    """
    warmup = getattr(provider, "warmup", None)
    if callable(warmup):
        print(f"Warming up {label} provider — may take ~30-60s ...")
        warmup()


@dataclass(frozen=True)
class InstanceIdentity:
    """This process's instance identity for the whole lifetime."""

    instance_id: uuid.UUID
    worker_id: int
    mode: str


async def startup_sequence(mode: str, session_factory=None) -> InstanceIdentity:
    """Schema compat check + instance registration (FR-007/FR-030).

    Raises SchemaMismatchError when the shared DB is not at the code's
    migration head (readers never migrate; start the writer first) and
    RuntimeError when worker_id allocation fails.

    When no session_factory is given, a self-contained engine is created and
    disposed within this call (the process-wide engine is loop-bound to the
    serving loop and must not be shared across asyncio.run scopes).
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag_mcp.runtime.instance_registry import InstanceRegistryService
    from rag_mcp.runtime.schema_compat import verify_schema_compat

    settings = get_settings()
    own_engine = None
    if session_factory is None:
        own_engine = create_async_engine(settings.database_url)
        factory = async_sessionmaker(own_engine, expire_on_commit=False)
    else:
        factory = session_factory

    try:
        # Reader/mcp forms only VERIFY the schema version (read-only check).
        async with factory() as session:
            await verify_schema_compat(session)

        instance_id = uuid.uuid4()
        registry = InstanceRegistryService(factory)
        registration = await registry.register(
            instance_id=instance_id,
            instance_mode=mode,
            process_role="mcp",
            worker_id=settings.worker_id,
            expiry_window_s=settings.lease_expiry_window_s,
        )
        if not registration.registered:
            raise RuntimeError(f"instance registration failed: {registration.error}")
        return InstanceIdentity(
            instance_id=instance_id,
            worker_id=registration.worker_id,
            mode=mode,
        )
    finally:
        if own_engine is not None:
            await own_engine.dispose()


async def _instance_heartbeat_loop(
    identity: InstanceIdentity, heartbeat_interval_s: int, expiry_window_s: int
) -> None:
    """Keep this instance's registration alive (data-model §2.4).

    Runs inside the heartbeat thread's own event loop with its own engine
    (never the serving loop's engine: asyncpg connections are loop-bound).
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag_mcp.runtime.activity import PUBLISH_INTERVAL_SECONDS, publish_activity
    from rag_mcp.runtime.instance_registry import InstanceRegistryService

    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        elapsed = heartbeat_interval_s
        while True:
            await asyncio.sleep(min(PUBLISH_INTERVAL_SECONDS, heartbeat_interval_s))
            elapsed += PUBLISH_INTERVAL_SECONDS
            try:
                # 013 T082: this MCP process publishes its own bounded activity
                # snapshot; the writer management process cannot see it in memory.
                async with factory() as session:
                    await publish_activity(session)
                    await session.commit()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.getLogger(__name__).exception("activity publication failed")
            if elapsed < heartbeat_interval_s:
                continue
            elapsed = 0
            try:
                registry = InstanceRegistryService(factory)
                ok = await registry.heartbeat(identity.instance_id, expiry_window_s)
                if not ok:
                    logging.getLogger(__name__).warning(
                        "instance %s heartbeat no longer active (row expired/released)",
                        identity.instance_id,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep the loop alive
                logging.getLogger(__name__).exception("instance heartbeat failed")
    finally:
        await engine.dispose()


def _start_heartbeat_thread(
    identity: InstanceIdentity, settings
) -> "threading.Thread":
    """Run the heartbeat loop in a daemon thread with its own event loop."""
    import threading

    def _run() -> None:
        asyncio.run(
            _instance_heartbeat_loop(
                identity,
                settings.lease_renew_interval_s,
                settings.lease_expiry_window_s,
            )
        )

    thread = threading.Thread(target=_run, name="instance-heartbeat", daemon=True)
    thread.start()
    return thread


def build_mcp_server(
    mode: str,
    embedding_provider=None,
    reranker=None,
    qdrant_store=None,
    session_factory=None,
    identity: InstanceIdentity | None = None,
):
    """Assemble the read-only MCP server for either instance form.

    Both writer and reader forms register search_knowledge + get_evidence
    (FR-001): the MCP plane is read-only; instance_mode only affects
    registration and metrics attribution.
    """
    from rag_mcp.providers.factory import get_provider_bundle

    provider = embedding_provider if embedding_provider is not None else get_provider_bundle().embedding
    if reranker is None:
        reranker = get_provider_bundle().reranker
    qdrant = qdrant_store if qdrant_store is not None else QdrantStore()

    # FR-016/FR-020 (data-model §4.1): install this process's instance
    # identity so retrieval run records carry instance_id / instance_mode
    # (metrics aggregate by instance form).
    if identity is not None:
        from rag_mcp.runtime.instance_context import set_instance

        set_instance(identity.instance_id, identity.mode, identity.worker_id)

    server = create_mcp_server(
        session_factory=session_factory,
        embedding_provider=provider,
        qdrant_store=qdrant,
        reranker=reranker,
        mode=mode,
    )
    return server


def main():
    args = parse_args()
    mode = resolve_mode(args)
    settings = get_settings()
    if args.port is not None:
        os.environ["MCP_PORT"] = str(args.port)

    # Startup validation (FR-007/FR-021/FR-030) before serving.
    validate_timeout_profiles_at_startup()
    identity = asyncio.run(startup_sequence(mode))

    # 013 T082: bind this process's real identity so its published activity
    # snapshot is attributed to the actual MCP instance.
    from rag_mcp.runtime.activity import set_process_identity

    set_process_identity(identity.instance_id, process_role="mcp", instance_mode=mode)

    provider, reranker = assemble_runtime_providers(settings)
    qdrant = QdrantStore()
    server = build_mcp_server(
        mode=mode,
        embedding_provider=provider,
        reranker=reranker,
        qdrant_store=qdrant,
        identity=identity,
    )

    # Warm local models before serving so the first search_knowledge call
    # does not exceed the client's request timeout (bge-m3 lazy load).
    _warmup_provider("embedding", provider)
    _warmup_provider("reranker", reranker)

    print(
        f"MCP server running on 127.0.0.1:{args.port or settings.mcp_port} "
        f"({mode} mode, instance {identity.instance_id}, "
        f"worker_id={identity.worker_id}, Streamable HTTP)"
    )
    # Keep the instance registration alive for the whole process lifetime.
    _start_heartbeat_thread(identity, settings)
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
