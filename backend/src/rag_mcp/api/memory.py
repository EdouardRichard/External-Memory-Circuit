"""Memory management lives exclusively on the writer management application."""
from datetime import datetime
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import BigInteger, Float, cast, func, select, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from rag_mcp.config import get_settings
from rag_mcp.db import get_session
from rag_mcp.models.knowledge_scope import KnowledgeScope
from rag_mcp.models.domain_profile import DomainProfile
from rag_mcp.models.memory_projection import MemoryEntry
from rag_mcp.models.memory_projection_meta import MemoryProjectionMeta
from rag_mcp.models.memory_event import MemoryEvent
from rag_mcp.models.memory_management_audit import MemoryManagementAudit
from rag_mcp.models.runtime import WriterLease
from rag_mcp.models.scope_binding import ScopeBinding
from rag_mcp.services.memory_reader import public_entry
from rag_mcp.services.memory_service import MemoryService
from rag_mcp.services.scope_resolver import MemoryScopeResolver
from rag_mcp.services.memory_policy import MemoryPolicy

router = APIRouter(prefix="/api/memories", tags=["memories"])


def management_action(action, *, actor):
    return {"authorized": actor == "management", "action": action}


async def require_writer(request: Request, session: AsyncSession = Depends(get_session)):
    lease = getattr(request.app.state, "writer_lease", None)
    if get_settings().instance_mode != "writer" or lease is None:
        raise HTTPException(503, detail={"code": "MEMORY_WRITE_UNAVAILABLE"})
    active = await session.scalar(select(WriterLease.lease_id).where(
        WriterLease.lease_id == lease.lease_id, WriterLease.holder_instance_id == lease.holder_instance_id,
        WriterLease.state == "active", WriterLease.expires_at > func.now()).with_for_update(read=True))
    if active is None:
        raise HTTPException(503, detail={"code": "MEMORY_WRITE_UNAVAILABLE"})


def _service(session):
    from rag_mcp.providers.factory import get_provider_bundle

    return MemoryService(session, embedding_provider=get_provider_bundle().embedding)


def _http_error(exception):
    from rag_mcp.mcp.serialization import memory_error
    body = memory_error(exception).structuredContent["error"]
    return HTTPException(403 if isinstance(exception, PermissionError) else 400, detail=body)


class ScopeCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope_id: int = Field(gt=0)
    reason: str = Field(min_length=1, max_length=4000)


class MemoryCommand(ScopeCommand):
    memory_id: int = Field(gt=0)


class RebuildCommand(ScopeCommand):
    since_event_id: int | None = Field(default=None, gt=0)


class PolicyCommand(ScopeCommand):
    policy: MemoryPolicy


class RollbackCommand(ScopeCommand):
    event_point: int | None = None
    time_point: datetime | None = None


class BindingCommand(ScopeCommand):
    binding_id: int | None = None
    binding_kind: Literal["workdir_prefix", "git_remote", "dir_name"]
    binding_value: str = Field(min_length=1, max_length=1024)
    priority: int = 0
    status: Literal["active", "disabled"] = "active"


class PromoteCommand(ScopeCommand):
    scope_id: int = Field(gt=0, strict=True)
    memory_id: int = Field(gt=0, strict=True)
    candidate_version: str = Field(pattern=r"^[0-9a-f]{64}$")


def _promotion_error(exception):
    """Promotion management errors keep their own REST reasons and status."""
    from rag_mcp.services.memory_governance import PromotionUnavailable

    if isinstance(exception, LookupError):
        return HTTPException(404, detail={"code": "MEMORY_PROMOTION_NOT_FOUND"})
    if isinstance(exception, PromotionUnavailable):
        return HTTPException(503, detail={"code": "MEMORY_PROMOTION_UNAVAILABLE"})
    code = str(exception).split(":", 1)[0].strip()
    if code in {"MEMORY_CANDIDATE_VERSION_CHANGED", "MEMORY_CANDIDATE_NOT_ELIGIBLE"}:
        return HTTPException(409, detail={"code": code, "message": str(exception)})
    if isinstance(exception, PermissionError):
        return HTTPException(403, detail={"code": code or "MEMORY_ROLLBACK_FORBIDDEN"})
    return _http_error(exception)


def _schedule_promotion_ingestion(source_id: int, initial_run_id: int | None = None) -> None:
    from rag_mcp.api.knowledge_sources import _schedule_ingestion

    _schedule_ingestion(source_id, initial_run_id=initial_run_id)


@router.get("/promotion-candidates", dependencies=[Depends(require_writer)])
async def list_promotion_candidates(scope_ref: str = Query(min_length=1), limit: int = Query(default=50, ge=1, le=100),
                                    offset: int = Query(default=0, ge=0),
                                    session: AsyncSession = Depends(get_session)):  # noqa: B008
    """Writer management candidate window; reader never exposes control surfaces."""
    try:
        scope_id = await MemoryScopeResolver(session).resolve(scope_ref)
        report = await _service(session).promotion_candidates(scope_id=scope_id, limit=limit, offset=offset)
    except ValueError as exception:
        raise _http_error(exception) from None
    items = []
    for item in report["items"]:
        items.append({**item, "memory_id": str(item["memory_id"]),
                      "certificate": "inference", "promotion_pointer": item["promotion_pointer"]})
    return {"scope_id": str(scope_id), "items": items, "total": report["total"]}


@router.post("/promote", dependencies=[Depends(require_writer)])
async def promote_candidate(data: PromoteCommand, response: Response,
                            session: AsyncSession = Depends(get_session)):  # noqa: B008
    """Explicit human promotion: registration, pointer and pending run commit together."""
    try:
        result = await _service(session).promote_candidate(scope_id=data.scope_id, memory_id=data.memory_id,
                                                           candidate_version=data.candidate_version,
                                                           actor="management", reason=data.reason)
    except (ValueError, PermissionError, LookupError) as exception:
        raise _promotion_error(exception) from None
    response.status_code = 200 if result["reused"] else 202
    if not result["reused"]:
        _schedule_promotion_ingestion(int(result["source_id"]), int(result["initial_processing_run_id"]))
    return {"schema_version": 1, "scope_id": str(result["scope_id"]), "memory_id": str(result["memory_id"]),
            "candidate_version": result["candidate_version"], "task_id": result["task_id"],
            "event_id": str(result["event_id"]),
            "source_id": str(result["source_id"]),
            "initial_processing_run_id": str(result["initial_processing_run_id"]),
            "status": result["status"], "version_id": None, "request_id": result["request_id"],
            "reused": result["reused"]}


@router.get("/promotions/{task_id}", dependencies=[Depends(require_writer)])
async def promotion_report(task_id: int, scope_ref: str = Query(min_length=1),
                           session: AsyncSession = Depends(get_session)):  # noqa: B008
    """Stable promotion task report; status only from actual source/run/version facts."""
    try:
        scope_id = await MemoryScopeResolver(session).resolve(scope_ref)
        report = await _service(session).promotion_status(task_id=task_id, scope_id=scope_id)
    except LookupError as exception:
        raise _promotion_error(exception) from None
    except ValueError as exception:
        raise _http_error(exception) from None
    return {"schema_version": 1, "scope_id": str(scope_id), "task_id": report["task_id"],
            "memory_id": str(report["memory_id"]), "candidate_version": report["candidate_version"],
            "source_id": str(report["source_id"]),
            "initial_processing_run_id": str(report["initial_processing_run_id"]),
            "attempt_run_ids": [str(item) for item in report["attempt_run_ids"]],
            "status": report["status"],
            "published_version_id": (str(report["published_version_id"])
                                     if report["published_version_id"] is not None else None),
            "result": report["result"],
            "authority_event_ids": [str(item) for item in report["authority_event_ids"]]}


class ConsolidationCommand(BaseModel):
    """The only client-settable consolidation request fields (T077).

    Internal triggers, execution contexts, propagation/proof material, run or
    holder identity, policy and proposals are not client controls at all.
    """
    model_config = ConfigDict(extra="forbid")
    scope_id: int = Field(gt=0, strict=True)
    reason: str = Field(min_length=1, max_length=4000)


def _consolidation_error(exception):
    from rag_mcp.services.consolidation_runtime import ConsolidationRuntimeError

    if not isinstance(exception, ConsolidationRuntimeError):
        return _http_error(exception)
    code = exception.code
    if code == 'CONSOLIDATION_DISABLED':
        return HTTPException(403, detail={"code": code})
    if code == 'CONSOLIDATION_CONFIGURATION_REQUIRED':
        return HTTPException(409, detail={"code": "CONSOLIDATION_CONFIG_REQUIRED"})
    if code == 'CONSOLIDATION_BUSY':
        return HTTPException(409, detail={"code": code,
                                          "run_id": str(exception.run_id) if exception.run_id else None})
    if code == 'CONSOLIDATION_SCOPE_WRITE_BUSY':
        return HTTPException(409, detail={"code": code})
    if code == 'CONSOLIDATION_CAPACITY_EXCEEDED':
        return HTTPException(429, detail={"code": code})
    if code in ('MISSING_KNOWLEDGE_SCOPE', 'CONSOLIDATION_STATUS_INVALID'):
        return HTTPException(400, detail={"code": code})
    return HTTPException(503, detail={"code": "MEMORY_WRITE_UNAVAILABLE"})


@router.post("/consolidation", status_code=202, dependencies=[Depends(require_writer)])
async def start_consolidation(data: ConsolidationCommand, request: Request):
    """Short manual admission: durable eligibility + admitted audit, then 202.

    Window selection and any model work happen in the bounded background worker;
    the HTTP response never waits for them.
    """
    supervisor = getattr(request.app.state, 'consolidation_supervisor', None)
    if supervisor is None:
        raise HTTPException(503, detail={"code": "MEMORY_WRITE_UNAVAILABLE"})
    request_id = str(uuid4())
    try:
        token = await supervisor.submit(data.scope_id, trigger='manual', request_id=request_id,
                                        actor='management')
    except (ValueError, PermissionError) as exception:
        raise _consolidation_error(exception) from None
    return {"schema_version": 1, "run_id": str(token.run_id), "scope_id": str(token.scope_id),
            "request_id": request_id, "trigger": "manual", "execution_context": "distiller_window",
            "status": "admitted", "window": None,
            "report_url": f"/api/memories/consolidation/runs/{token.run_id}?scope_ref={token.scope_id}"}


@router.get("/consolidation/runs", dependencies=[Depends(require_writer)])
async def list_consolidation_runs(scope_ref: str = Query(min_length=1),
                                  limit: int = Query(default=20, ge=1, le=100),
                                  offset: int = Query(default=0, ge=0),
                                  session: AsyncSession = Depends(get_session)):  # noqa: B008
    """Latest cumulative observation per run, stable newest first."""
    from rag_mcp.services import consolidation_report

    try:
        scope_id = await MemoryScopeResolver(session).resolve(scope_ref)
    except ValueError as exception:
        raise _http_error(exception) from None
    return await consolidation_report.list_run_reports(session, scope_id=scope_id, limit=limit, offset=offset)


@router.get("/consolidation/runs/{run_id}", dependencies=[Depends(require_writer)])
async def get_consolidation_run(run_id: str, scope_ref: str = Query(min_length=1),
                                include_history: bool = Query(default=False),
                                session: AsyncSession = Depends(get_session)):  # noqa: B008
    """Same-scope run report; 404 without proof, 410 only with retained identity."""
    from rag_mcp.services import consolidation_report
    from rag_mcp.services.consolidation_report import ConsolidationRunMissing

    try:
        scope_id = await MemoryScopeResolver(session).resolve(scope_ref)
        return await consolidation_report.get_run_report(session, run_id=run_id, scope_id=scope_id,
                                                         include_history=include_history)
    except ConsolidationRunMissing as exception:
        status = 410 if exception.code == consolidation_report.EXPIRED else 404
        raise HTTPException(status, detail={"code": exception.code}) from None
    except ValueError as exception:
        raise _http_error(exception) from None


@router.get("/scopes")
async def list_memory_scopes(session: AsyncSession = Depends(get_session)):
    scopes = (await session.execute(select(KnowledgeScope).where(KnowledgeScope.status == "active")
                                   .order_by(KnowledgeScope.name, KnowledgeScope.scope_id))).scalars().all()
    return {"items": [{"scope_id": str(scope.scope_id), "name": scope.name, "slug": scope.slug,
                       "domain_key": scope.domain_key} for scope in scopes]}


@router.get("")
async def browse_memories(scope_ref: str = Query(min_length=1), limit: int = Query(default=50, ge=1, le=100),
                          offset: int = Query(default=0, ge=0),
                          kind: str | None = Query(default=None, min_length=1),
                          status: str | None = Query(default=None, min_length=1),
                          provenance: str | None = Query(default=None, min_length=1),
                          session_id: str | None = Query(default=None, min_length=1),
                          min_salience: float | None = Query(default=None, ge=0, le=1),
                          session: AsyncSession = Depends(get_session)):
    try:
        sid = await MemoryScopeResolver(session).resolve(scope_ref)
    except ValueError as exception:
        raise _http_error(exception) from None
    entries = func.jsonb_each(MemoryProjectionMeta.payload["state"]["entries"]).table_valued("key", "value").lateral()
    data = cast(entries.c.value, JSONB)
    conditions = [MemoryProjectionMeta.knowledge_scope_id == sid, MemoryProjectionMeta.projection_type == "manifest",
                  MemoryProjectionMeta.status == "complete", MemoryProjectionMeta.payload["verification_version"].as_integer() == 1]
    # FR-036 six-dimension filtering. Every filter is additive and narrows the SAME
    # conditions list the count uses, so `total` is the filtered total and pagination
    # stays correct. Omitting every filter leaves the previous behaviour byte-identical.
    for field, value in (("kind", kind), ("status", status), ("provenance", provenance), ("session_id", session_id)):
        if value is not None:
            conditions.append(data[field].as_string() == value)
    if min_salience is not None:
        # salience is the sibling field projection, not a relation-entry field: read
        # state.salience[<memory_id>].salience through the jsonb_each lateral key.
        conditions.append(
            cast(
                func.jsonb_extract_path_text(MemoryProjectionMeta.payload["state"]["salience"], entries.c.key, "salience"),
                Float,
            ) >= min_salience
        )
    rows = (await session.execute(select(data).select_from(MemoryProjectionMeta).join(entries, true()).where(*conditions)
        .order_by(data["observed_at"].as_string().desc(), cast(data["memory_id"].as_string(), BigInteger).desc())
        .offset(offset).limit(limit))).scalars().all()
    total = await session.scalar(select(func.count()).select_from(MemoryProjectionMeta).join(entries, true()).where(*conditions))
    memories = []
    for row in rows:
        item = public_entry(row)
        for key in ("memory_id", "knowledge_scope_id", "superseded_by"):
            if item.get(key) is not None:
                item[key] = str(item[key])
        item.update(injection_flags=row.get("injection_flags", []), projection_status="complete")
        memories.append(item)
    return {"memories": memories, "total": total, "scope_id": str(sid)}


@router.get("/bindings")
async def browse_bindings(scope_ref: str = Query(min_length=1), session: AsyncSession = Depends(get_session)):
    try:
        sid = await MemoryScopeResolver(session).resolve(scope_ref)
        rows = (await session.execute(select(ScopeBinding).where(ScopeBinding.knowledge_scope_id == sid))).scalars().all()
        return {"items": [{"binding_id": str(row.binding_id), "knowledge_scope_id": str(sid),
            "binding_kind": row.binding_kind, "binding_value": row.binding_value,
            "priority": row.priority, "status": row.status} for row in rows]}
    except ValueError as exception:
        raise _http_error(exception) from None


@router.get("/policy")
async def browse_policy(scope_ref: str = Query(min_length=1), session: AsyncSession = Depends(get_session)):
    try:
        sid = await MemoryScopeResolver(session).resolve(scope_ref)
        scope = await session.get(KnowledgeScope, sid)
        profile = await session.get(DomainProfile, scope.domain_key)
        return {"scope_id": str(sid), "domain_key": scope.domain_key, "policy": profile.memory_policy}
    except ValueError as exception:
        raise _http_error(exception) from None


@router.post("/policy", dependencies=[Depends(require_writer)])
async def update_policy(data: PolicyCommand, session: AsyncSession = Depends(get_session)):
    try:
        return await _service(session).govern("policy", actor="management", scope_id=data.scope_id,
            reason=data.reason, policy=data.policy.model_dump(exclude_unset=True))
    except (ValueError, PermissionError) as exception:
        raise _http_error(exception) from None


async def _command(action, data, session):
    try:
        return await _service(session).govern(action, actor="management", **data.model_dump())
    except (ValueError, PermissionError) as exception:
        raise _http_error(exception) from None


@router.post("/retire", dependencies=[Depends(require_writer)])
async def retire_memory(data: MemoryCommand, session: AsyncSession = Depends(get_session)):
    return await _command("retire", data, session)


@router.post("/purge", dependencies=[Depends(require_writer)])
async def purge_memory(data: MemoryCommand, session: AsyncSession = Depends(get_session)):
    return await _command("purge", data, session)


@router.post("/usage", dependencies=[Depends(require_writer)])
async def record_usage(data: MemoryCommand, session: AsyncSession = Depends(get_session)):
    return await _command("access", data, session)


@router.post("/rollback", dependencies=[Depends(require_writer)])
async def rollback_memory(data: RollbackCommand, session: AsyncSession = Depends(get_session)):
    return await _command("rollback", data, session)


@router.post("/bindings", dependencies=[Depends(require_writer)])
async def grant_binding(data: BindingCommand, session: AsyncSession = Depends(get_session)):
    return await _command("binding", data, session)


@router.post("/rebuild", dependencies=[Depends(require_writer)])
async def rebuild_memory(data: RebuildCommand, session: AsyncSession = Depends(get_session)):
    try:
        request_id = str(uuid4())
        return {"scope_id": data.scope_id, "request_id": request_id, "projections": await _service(session).rebuild(
            data.scope_id, actor="management", reason=data.reason, since_event_id=data.since_event_id,
            request_id=request_id)}
    except (ValueError, PermissionError) as exception:
        raise _http_error(exception) from None


@router.get("/rebuild/audit")
async def rebuild_audit(request_id: str = Query(min_length=1, max_length=128),
                       scope_id: int | None = Query(default=None, gt=0),
                       session: AsyncSession = Depends(get_session)):
    audit = await session.get(MemoryManagementAudit, request_id)
    if audit is None or (scope_id is not None and audit.knowledge_scope_id != scope_id):
        raise HTTPException(404, detail={"code": "MEMORY_AUDIT_NOT_FOUND"})
    return {"request_id": audit.request_id, "operation": audit.operation, "actor": audit.actor,
            "scope_id": audit.knowledge_scope_id, "reason": audit.reason,
            "source_event_id": audit.source_event_id, "since_event_id": audit.since_event_id,
            "result": audit.result, "created_at": audit.created_at.isoformat()}


#: Governance payload fields a management audit reader may see. The stored payload
#: is summarised, never echoed: memory body text has no path here, and the nested
#: promotion `pointer` snapshot stays behind `GET /promotions/{task_id}`.
_AUDIT_PAYLOAD_KEYS = (
    "reason", "payload_version", "grant_type", "purge", "event_point", "retention_stage",
    "domain_key", "binding_id", "binding_kind", "priority", "status", "memory_id",
    "candidate_version", "request_id", "source_id", "impact", "before_fingerprint",
    "after_fingerprint", "policy_before", "policy_after",
)


def audit_payload_summary(payload):
    if not isinstance(payload, dict):
        return {}
    return {key: payload[key] for key in _AUDIT_PAYLOAD_KEYS if key in payload}


@router.get("/audit", dependencies=[Depends(require_writer)])
async def memory_audit(request_id: str = Query(min_length=1, max_length=128),
                       scope_id: int | None = Query(default=None, gt=0),
                       session: AsyncSession = Depends(get_session)):  # noqa: B008
    """T083 (FR-037/FR-039/FR-041): dereference a management pointer into the log.

    Every governance command (retire, purge, rollback, policy, binding and an
    explicit promotion) appends exactly one authority ``MemoryEvent`` carrying its
    own ``request_id``; this is the management-plane read path that turns the
    pointer those commands return into the stored authority row.  A promotion
    reuses its request id for later ``promotion_observed`` grants, so the earliest
    matching event is the authority event for the request itself.  ``GET
    /rebuild/audit`` above keeps its different, keyed-by-request_id contract
    byte-compatible, and this route is deliberately not part of the MCP tool
    surface (``rag_mcp.mcp.memory_tools.tool_names``).
    """
    statement = select(MemoryEvent).where(MemoryEvent.request_id == request_id)
    if scope_id is not None:
        statement = statement.where(MemoryEvent.knowledge_scope_id == scope_id)
    event = (await session.execute(
        statement.order_by(MemoryEvent.event_id).limit(1))).scalars().first()
    if event is None:
        raise HTTPException(404, detail={"code": "MEMORY_AUDIT_NOT_FOUND"})
    return {"schema_version": 1, "request_id": event.request_id, "event_id": str(event.event_id),
            "event_type": event.event_type, "knowledge_scope_id": str(event.knowledge_scope_id),
            "aggregate_id": str(event.aggregate_id), "actor": event.actor,
            "authority": (event.authority or {}).get("source"),
            "actionability": event.actionability, "occurred_at": event.occurred_at.isoformat(),
            "payload_summary": audit_payload_summary(event.payload)}


@router.get("/stats", dependencies=[Depends(require_writer)])
async def memory_statistics(scope_ref: str = Query(min_length=1),
                            session: AsyncSession = Depends(get_session)):
    """T039/FR-044..FR-046: counts, distributions and quantiles for one domain.

    An independent endpoint -- ``GET /runtime/metrics`` (006) is untouched.  The
    response is zero-body by construction: the aggregation selects only counts,
    closed-vocabulary labels and ``memory_salience.salience``, so no free-text key
    has a code path into the payload and ``TRACE_BODY_ENABLED`` cannot change it.
    A read-only instance or an invalid writer lease is refused with 503
    ``MEMORY_WRITE_UNAVAILABLE`` by the shared ``require_writer`` dependency.
    """
    from rag_mcp.services.memory_statistics import domain_statistics

    try:
        scope_id = await MemoryScopeResolver(session).resolve(scope_ref)
    except ValueError as exception:
        raise _http_error(exception) from None
    return await domain_statistics(session, scope_id)
