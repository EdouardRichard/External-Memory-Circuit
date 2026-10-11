"""REST API routes for knowledge source management (FR-004, FR-010, FR-011, US1, US4)."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from rag_mcp.config import get_settings
from rag_mcp.db import get_session
from rag_mcp.schemas.knowledge_source import (
    KnowledgeSourceListResponse,
    KnowledgeSourceResponse,
)

router = APIRouter(prefix="/api/knowledge-sources", tags=["knowledge-sources"])

# ---------------------------------------------------------------------------
# Background ingestion helpers
# ---------------------------------------------------------------------------

import asyncio
import logging

logger = logging.getLogger(__name__)

_qdrant_store = None


def _get_qdrant_store():
    """Lazily create and cache the QdrantStore singleton."""
    global _qdrant_store
    if _qdrant_store is None:
        from rag_mcp.indexing.qdrant_client import QdrantStore

        _qdrant_store = QdrantStore()
    return _qdrant_store


async def _observe_promotion(session, source_id: int) -> None:
    """Append the actual publication/attempt result for a promoted source (T074)."""
    from rag_mcp.services.memory_service import MemoryService

    try:
        await MemoryService(session).observe_promotion(source_id=source_id)
    except Exception:  # observation must never break ingestion
        logger.exception("Promotion observation failed for source %s", source_id)


async def _run_promotion_observation(source_id: int) -> None:
    from rag_mcp.db import get_session_factory

    factory = get_session_factory()
    try:
        async with factory() as session:
            await _observe_promotion(session, source_id)
    except Exception:  # observation must never break the caller
        logger.exception("Promotion observation task failed for source %s", source_id)


def _schedule_promotion_observation(source_id: int) -> None:
    if not get_settings().ingestion_background:
        return
    try:
        asyncio.get_running_loop().create_task(_run_promotion_observation(source_id))
    except RuntimeError:
        logger.warning("No running event loop; skipping promotion observation")


async def _pending_initial_run(session, source_id: int):
    """The prebuilt pending initial attempt, if the source still has one."""
    from sqlalchemy import select

    from rag_mcp.models.processing_run import ProcessingRun

    return await session.scalar(select(ProcessingRun.run_id).where(
        ProcessingRun.source_id == source_id, ProcessingRun.run_type == "initial",
        ProcessingRun.status == "pending").order_by(ProcessingRun.run_id).limit(1))


async def _run_ingestion(
    source_id: int, graph_ready: bool = False, retry: bool = False, initial_run_id: int | None = None
) -> None:
    """Run the ingestion pipeline for a source in a background task.

    Uses its own DB session (independent of the request-scoped session).
    graph_ready forwards the user's 004 capability declaration (FR-013);
    retry selects reprocess() (user-triggered rebuild) over ingest().
    initial_run_id consumes the prebuilt pending run registered by the shared
    registration step (013 T069/T070) instead of creating a second initial run;
    a crash-recovered promotion task resolves its own pending initial attempt.
    """
    from rag_mcp.db import get_session_factory
    from rag_mcp.providers.factory import get_provider_bundle
    from rag_mcp.services.ingestion_service import IngestionService

    factory = get_session_factory()
    try:
        async with factory() as session:
            service = IngestionService(
                session=session,
                embedding_provider=get_provider_bundle().embedding,
                qdrant_store=_get_qdrant_store(),
            )
            if retry:
                await service.reprocess(source_id, graph_ready=graph_ready)
            else:
                prebuilt = initial_run_id or await _pending_initial_run(session, source_id)
                await service.ingest(source_id, graph_ready=graph_ready, processing_run_id=prebuilt)
            await _observe_promotion(session, source_id)
    except Exception:
        logger.exception("Ingestion failed for source %s", source_id)
        try:
            async with factory() as session:
                await _observe_promotion(session, source_id)
        except Exception:  # observation must never mask the ingestion failure
            logger.exception("Promotion failure observation failed for source %s", source_id)


def _schedule_ingestion(
    source_id: int, graph_ready: bool = False, retry: bool = False, initial_run_id: int | None = None
) -> None:
    """Schedule ingestion through the single activity-visible scheduling path (T082).

    When ``INGESTION_BACKGROUND=false`` (used by tests and any operationally
    deferred-processing mode), no background task is spawned and the source
    remains in ``uploaded`` status awaiting explicit reprocessing.
    """
    from rag_mcp.runtime.scheduling import schedule_ingestion

    schedule_ingestion(source_id, graph_ready=graph_ready, retry=retry, initial_run_id=initial_run_id)


# ---------------------------------------------------------------------------
# Deletion / clear derived-data cleanup (FR-012 / US-4)
# ---------------------------------------------------------------------------


async def _purge_source_derived_data(
    session: AsyncSession,
    source_id: int,
    qdrant_store: "QdrantStore | None" = None,
) -> None:
    """Delete Qdrant points and PG chunks for a deleted source.

    Args:
        session: Async SQLAlchemy session (caller commits the transaction).
        source_id: Source whose derived data is purged.
        qdrant_store: Optional QdrantStore override for testing; defaults to
            the process-wide singleton.

    Idempotent: with no chunk rows there is nothing to remove, and Qdrant
    failures are logged but never block the PG chunk deletion.

    004 (FR-016/AS5.3): graph derived data tied to the source's versions is
    purged as well — graph_expansion_path rows first (chunk FKs), then
    graph_edge/soft_relation for the affected (scope, version_number) pairs.
    """
    from sqlalchemy import delete as sa_delete
    from sqlalchemy import select, text as sa_text

    from rag_mcp.models.chunk import Chunk

    # Collect affected index_versions BEFORE deleting the chunk rows.
    result = await session.execute(
        select(Chunk.index_version).where(Chunk.source_id == source_id).distinct()
    )
    index_versions = list(result.scalars().all())

    store = qdrant_store if qdrant_store is not None else _get_qdrant_store()
    for index_version in index_versions:
        collection = f"chunks_dense_{index_version}"
        try:
            store.delete_points_by_source(collection, source_id)
        except Exception:  # noqa: BLE001 - Qdrant outage must not block PG cleanup
            logger.exception(
                "Failed to purge Qdrant points for source %s (collection %s)",
                source_id,
                collection,
            )

    await _purge_source_graph_relations(session, source_id)

    await session.execute(sa_delete(Chunk).where(Chunk.source_id == source_id))


async def _purge_scope_derived_data(
    session: AsyncSession,
    scope_id: int,
    qdrant_store: "QdrantStore | None" = None,
) -> None:
    """Delete Qdrant points and PG chunks for every source in a cleared scope.

    Args:
        session: Async SQLAlchemy session (caller commits the transaction).
        scope_id: Knowledge scope whose derived data is purged.
        qdrant_store: Optional QdrantStore override for testing.
    """
    from sqlalchemy import delete as sa_delete
    from sqlalchemy import select, text as sa_text

    from rag_mcp.models.chunk import Chunk

    result = await session.execute(
        select(Chunk.index_version)
        .where(Chunk.knowledge_scope_id == scope_id)
        .distinct()
    )
    index_versions = list(result.scalars().all())

    store = qdrant_store if qdrant_store is not None else _get_qdrant_store()
    for index_version in index_versions:
        collection = f"chunks_dense_{index_version}"
        try:
            store.delete_points_by_scope(collection, scope_id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to purge Qdrant points for scope %s (collection %s)",
                scope_id,
                collection,
            )

    # 004 (FR-016/AS5.3, blueprint §5): stop graph retrieval first, then
    # delete graph derived data for the cleared scope.
    await session.execute(sa_text(
        "UPDATE knowledge_versions SET graph_ready = false "
        "WHERE knowledge_scope_id = :sid"
    ), {"sid": scope_id})
    await session.execute(sa_text(
        "DELETE FROM graph_expansion_path WHERE chunk_id IN ("
        "SELECT chunk_id FROM chunks WHERE knowledge_scope_id = :sid)"
    ), {"sid": scope_id})
    await session.execute(sa_text(
        "DELETE FROM soft_relation WHERE knowledge_scope_id = :sid"
    ), {"sid": scope_id})
    await session.execute(sa_text(
        "DELETE FROM graph_edge WHERE knowledge_scope_id = :sid"
    ), {"sid": scope_id})

    await session.execute(
        sa_delete(Chunk).where(Chunk.knowledge_scope_id == scope_id)
    )


async def _purge_source_graph_relations(session, source_id: int) -> None:
    """Purge graph derived data owned by a deleted source's versions (004).

    Graph rows are keyed by (knowledge_scope_id, index_version) —
    knowledge_scope_id is the sole graph isolation key (007 FR-018, 010
    FR-023) — where index_version equals the owning version_number. Deleting a source
    removes the graph rows of its versions; other versions/scopes keep theirs
    (FR-010/FR-016). Expansion-path rows are removed first because they
    reference chunks.
    """
    from sqlalchemy import text as sa_text

    rows = (await session.execute(sa_text(
        "SELECT DISTINCT c.knowledge_scope_id, v.version_number "
        "FROM chunks c JOIN knowledge_versions v ON v.version_id = c.version_id "
        "WHERE c.source_id = :sid"
    ), {"sid": source_id})).fetchall()
    if not rows:
        return

    # Stop graph retrieval for the affected versions before deleting data
    # (blueprint §5 mark-then-delete).
    await session.execute(sa_text(
        "UPDATE knowledge_versions SET graph_ready = false "
        "WHERE version_id IN ("
        "SELECT c.version_id FROM chunks c WHERE c.source_id = :sid)"
    ), {"sid": source_id})
    await session.execute(sa_text(
        "DELETE FROM graph_expansion_path WHERE chunk_id IN ("
        "SELECT chunk_id FROM chunks WHERE source_id = :sid)"
    ), {"sid": source_id})

    for scope_id, version_number in rows:
        await session.execute(sa_text(
            "DELETE FROM soft_relation WHERE knowledge_scope_id = :ksid "
            "AND index_version = :iv"
        ), {"ksid": scope_id, "iv": version_number})
        await session.execute(sa_text(
            "DELETE FROM graph_edge WHERE knowledge_scope_id = :ksid "
            "AND index_version = :iv"
        ), {"ksid": scope_id, "iv": version_number})


async def _run_source_cleanup(source_id: int) -> None:
    """Background task: purge derived data for a deleted source."""
    from rag_mcp.db import get_session_factory

    factory = get_session_factory()
    try:
        async with factory() as session:
            await _purge_source_derived_data(session, source_id)
            await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("Derived-data cleanup failed for source %s", source_id)


async def _run_scope_cleanup(scope_id: int) -> None:
    """Background task: purge derived data for a cleared scope."""
    from rag_mcp.db import get_session_factory

    factory = get_session_factory()
    try:
        async with factory() as session:
            await _purge_scope_derived_data(session, scope_id)
            await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("Derived-data cleanup failed for scope %s", scope_id)


def _schedule_derived_data_cleanup(kind: str, target_id: int) -> None:
    """Schedule derived-data cleanup as a fire-and-forget background task.

    ``kind`` is ``"source"`` or ``"scope"``. Disabled when
    ``INGESTION_BACKGROUND=false`` (tests) so cleanup stays deterministic.
    """
    if not get_settings().ingestion_background:
        logger.info(
            "Background cleanup disabled; %s %s derived data not purged",
            kind,
            target_id,
        )
        return
    try:
        loop = asyncio.get_running_loop()
        if kind == "source":
            loop.create_task(_run_source_cleanup(target_id))
        else:
            loop.create_task(_run_scope_cleanup(target_id))
        logger.info("Scheduled %s derived-data cleanup for %s", kind, target_id)
    except RuntimeError:
        logger.warning("No running event loop; skipping %s cleanup", kind)


def _detect_format(filename: str, content: bytes | None = None) -> str:
    """Detect file format via the FormatHandler registry (008, FR-001/FR-003).

    The registry is the single source of truth for format detection, replacing
    the legacy if/elif chain. Raises RegistryFormatError for unsupported
    formats and extension/content mismatches (.go without a package
    declaration, or a non-OpenAPI .json/.yaml/.yml when no generic entry is
    registered).
    """
    from rag_mcp.services.knowledge_source_registration import detect_format

    return detect_format(filename, content)


@router.post("", response_model=KnowledgeSourceResponse, status_code=201)
async def upload_knowledge_source(
    scope_id: int,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Upload a new knowledge source file (Markdown or Java).

    Args:
        scope_id: Knowledge scope ID to associate the source with.
        file: Uploaded file (multipart/form-data).

    Upload and explicit human promotion share the same registration steps
    (T069): raw object, uploaded KnowledgeSource, exactly one pending initial
    ProcessingRun. Upload keeps its original validation, transaction and
    response behaviour and schedules ingestion only after commit.
    """
    from rag_mcp.services.knowledge_source_registration import RegistrationError, register_uploaded_source

    content = await file.read()
    try:
        registration = await register_uploaded_source(session, scope_id=scope_id, content=content,
                                                      filename=file.filename or "unknown")
    except RegistrationError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from None

    source = registration.source
    await session.commit()

    # Trigger background ingestion pipeline (async, non-blocking); the prebuilt
    # pending initial run registered above is consumed, not duplicated.
    _schedule_ingestion(source.source_id, initial_run_id=registration.initial_run.run_id)

    return KnowledgeSourceResponse(
        source_id=str(source.source_id),
        knowledge_scope_id=str(source.knowledge_scope_id),
        filename=source.filename,
        content_hash=source.content_hash,
        format=source.format,
        size_bytes=source.size_bytes,
        status=source.status,
        processing_error=source.processing_error,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


@router.get("", response_model=KnowledgeSourceListResponse)
async def list_knowledge_sources(
    scope_id: int | None = None,
    session: AsyncSession = Depends(get_session),
):
    """List knowledge sources, optionally filtered by scope."""
    from sqlalchemy import select

    from rag_mcp.models.knowledge_source import KnowledgeSource

    stmt = select(KnowledgeSource).order_by(KnowledgeSource.created_at.desc())
    if scope_id is not None:
        stmt = stmt.where(KnowledgeSource.knowledge_scope_id == scope_id)

    result = await session.execute(stmt)
    sources = list(result.scalars().all())

    items = [
        KnowledgeSourceResponse(
            source_id=str(s.source_id),
            knowledge_scope_id=str(s.knowledge_scope_id),
            filename=s.filename,
            content_hash=s.content_hash,
            format=s.format,
            size_bytes=s.size_bytes,
            status=s.status,
            processing_error=s.processing_error,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sources
    ]
    return KnowledgeSourceListResponse(items=items, total=len(items))


@router.get("/{source_id}", response_model=KnowledgeSourceResponse)
async def get_knowledge_source(
    source_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get a knowledge source by ID."""
    from sqlalchemy import select

    from rag_mcp.models.knowledge_source import KnowledgeSource

    result = await session.execute(
        select(KnowledgeSource).where(KnowledgeSource.source_id == source_id)
    )
    source = result.scalar_one_or_none()
    if source is None:
        raise HTTPException(status_code=404, detail="Knowledge source not found")

    return KnowledgeSourceResponse(
        source_id=str(source.source_id),
        knowledge_scope_id=str(source.knowledge_scope_id),
        filename=source.filename,
        content_hash=source.content_hash,
        format=source.format,
        size_bytes=source.size_bytes,
        status=source.status,
        processing_error=source.processing_error,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


@router.post("/{source_id}/reprocess", status_code=202)
async def reprocess_knowledge_source(
    source_id: int,
    graph_ready: bool = False,
    session: AsyncSession = Depends(get_session),
):
    """Trigger reprocessing of a knowledge source (blueprint §5).

    Creates a new ProcessingRun with run_type='retry'.
    Old version stays published until new version publish succeeds (FR-009).

    004 (FR-013/FR-027): graph_ready=true is the user-triggered declaration
    that the rebuilt version should gain the graph_ready capability; it is
    granted only when graph relations are ready, never by auto migration.
    """
    from sqlalchemy import select

    from rag_mcp.models.knowledge_source import KnowledgeSource

    result = await session.execute(
        select(KnowledgeSource).where(KnowledgeSource.source_id == source_id)
    )
    source = result.scalar_one_or_none()
    if source is None:
        raise HTTPException(status_code=404, detail="Knowledge source not found")

    if source.status not in ("published", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reprocess source in '{source.status}' status",
        )

    # Reset status to trigger reprocessing
    from datetime import datetime, timezone

    source.status = "uploaded"
    source.processing_error = None
    source.updated_at = datetime.now(timezone.utc)
    await session.flush()
    await session.commit()

    # Trigger background ingestion pipeline (same as upload path)
    _schedule_ingestion(source_id, graph_ready=graph_ready, retry=True)
    # Explicit reprocess appends its actual (reset) observation for a promoted
    # source; the ingestion task later appends the real attempt/publication.
    _schedule_promotion_observation(source_id)

    return {
        "message": "Reprocessing triggered",
        "source_id": str(source_id),
        "graph_ready_requested": graph_ready,
    }


@router.delete("/{source_id}", status_code=204)
async def delete_knowledge_source(
    source_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Delete a knowledge source (FR-011, FR-012).

    Marks source as deleted (stops retrieval immediately), then cleans up
    derived data asynchronously. Idempotent - deleting already-deleted source
    returns 204.
    """
    from datetime import datetime, timezone

    from sqlalchemy import select

    from rag_mcp.models.knowledge_source import KnowledgeSource

    result = await session.execute(
        select(KnowledgeSource).where(KnowledgeSource.source_id == source_id)
    )
    source = result.scalar_one_or_none()
    if source is None:
        raise HTTPException(status_code=404, detail="Knowledge source not found")

    if source.status == "deleted":
        # Idempotent: already deleted
        return

    # Step 1: Mark as deleted (immediately stops participation in new retrieval)
    source.status = "deleted"
    source.updated_at = datetime.now(timezone.utc)
    await session.flush()
    await session.commit()

    # Step 2: Asynchronously purge derived data (Qdrant points + PG chunks)
    _schedule_derived_data_cleanup("source", source_id)


@router.post("/scopes/{scope_id}/clear", status_code=202)
async def clear_knowledge_scope(
    scope_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Clear all knowledge sources in a scope (FR-011, FR-012).

    Marks all sources as deleted and the scope as 'deleting'.
    Other scopes are unaffected.
    """
    from datetime import datetime, timezone

    from sqlalchemy import select, update

    from rag_mcp.models.knowledge_scope import KnowledgeScope
    from rag_mcp.models.knowledge_source import KnowledgeSource

    # Verify scope exists
    scope_result = await session.execute(
        select(KnowledgeScope).where(KnowledgeScope.scope_id == scope_id)
    )
    scope = scope_result.scalar_one_or_none()
    if scope is None:
        raise HTTPException(status_code=404, detail="Knowledge scope not found")

    now = datetime.now(timezone.utc)

    # Mark scope as deleting
    scope.status = "deleting"
    scope.updated_at = now

    # Mark all sources in scope as deleted
    await session.execute(
        update(KnowledgeSource)
        .where(KnowledgeSource.knowledge_scope_id == scope_id)
        .values(status="deleted", updated_at=now)
    )

    await session.flush()
    await session.commit()

    # Asynchronously purge derived data (Qdrant points + PG chunks)
    _schedule_derived_data_cleanup("scope", scope_id)

    return {"message": "Scope clearing initiated", "scope_id": str(scope_id)}
