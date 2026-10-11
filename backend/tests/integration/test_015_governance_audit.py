"""T083 (US6, FR-037/FR-039/FR-041): every management pointer resolves to a row.

The pre-convergence module asserted the same intent for a single action; this
module replaces it with measured evidence for all five governance actions.  For
retire, purge, rollback, policy-edit and explicit promotion it drives the **real
HTTP endpoint** and then dereferences the returned pointer twice:

* against the append-only ``memory_events`` authority log (the returned
  ``event_id`` is a real row whose ``request_id`` is the returned one), and
* through the new management read path ``GET /api/memories/audit``.

Refusals are measured as negative evidence: a refused command (unknown target,
builtin-domain policy edit, illegal policy value, stale candidate version, quota
refusal) leaves the authority log exactly as it found it.

Two deliberate test-infrastructure choices: the deterministic ``StableEmbedding``
fixture from the 013 suite is injected for the management calls (no 2 GB model is
loaded per test; the HTTP surface, the session, the governance transaction and the
append-only log stay real), and the process-wide writer lease 鈥?which this shared
acceptance database hands to one writer at a time 鈥?is waited for instead of
bypassed.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from rag_mcp.db import get_session
from rag_mcp.models.domain_profile import DomainProfile
from rag_mcp.models.knowledge_scope import KnowledgeScope
from rag_mcp.models.memory_event import MemoryEvent
from rag_mcp.server import create_app
from rag_mcp.services.memory_event_store import MemoryEventStore
from rag_mcp.services.memory_service import MemoryService
from rag_mcp.utils.snowflake import generate_id
from tests.integration.consolidation_fixtures import StableEmbedding
from tests.integration.memory_acceptance import writer_owner
from tests.integration.test_012_live_reader import scope_and_payload


def _app_client(app, db_session):
    """Serve every request from the test's session, the established REST pattern.

    The read transaction each request opens is rolled back in the dependency's
    ``finally`` (the repo pattern), so the session is never handed to fixture
    teardown with an open transaction on a pooled connection.
    """
    async def sessions():
        try:
            yield db_session
        finally:
            await db_session.rollback()

    app.dependency_overrides[get_session] = sessions
    return AsyncClient(transport=ASGITransport(app), base_url="http://test")


class _DeterministicEmbedding(StableEmbedding):
    """The 013 stable fixture plus the rest of the provider protocol.

    Deterministic and instant: no 2 GB model is loaded while several suites share
    this host, and the REST/governance/database path under test stays real.
    """

    def warmup(self) -> None:
        return None

    async def embed_query(self, text):
        return (await self.embed_texts([text]))[0]


@asynccontextmanager
async def _writer(engine, *, timeout_s=240.0):
    """Take the writer lease, waiting out another holder of the shared database.

    The lease is the real single-writer invariant (FR-002) and is never bypassed:
    when another worker holds it, this waits for the release or for the bounded
    lease window to expire.  Only the acquisition is retried, never the test body.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        manager = writer_owner(engine)
        try:
            owner = await manager.__aenter__()
        except AssertionError:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(5)
            continue
        try:
            yield owner
        finally:
            await manager.__aexit__(None, None, None)
        return


def _management_service(session):
    """The real service with the deterministic 013 embedding fixture."""
    return MemoryService(session, embedding_provider=_DeterministicEmbedding())


def _inject_embedding(monkeypatch, provider):
    """Inject the deterministic fixture through the provider bundle seam.

    Task 9 removed the ``rag_mcp.api.memory._embedding_provider`` module global, so
    the REST ``_service`` now resolves its provider from ``get_provider_bundle()``;
    patch that seam instead of the deleted global.
    """
    from types import SimpleNamespace

    monkeypatch.setattr("rag_mcp.providers.factory.get_provider_bundle",
                        lambda settings=None: SimpleNamespace(embedding=provider))
    return provider


async def _scope_event_count(session, scope_id):
    return await session.scalar(select(func.count()).select_from(MemoryEvent)
                                .where(MemoryEvent.knowledge_scope_id == scope_id))


async def _pointer(client, session, body, *, event_type, scope_id, payload_keys=()):
    """The returned pointer must be both a stored log row and a readable event."""
    assert body["request_id"], "no request_id in the governance response"
    assert body["event_id"], "no event_id in the governance response"
    event_id, request_id = int(body["event_id"]), body["request_id"]
    # Read the append-only log itself (the repository the authority writes go
    # through), not a cached ORM instance: the pointer must resolve to a row.
    history = await MemoryEventStore(session).replay(scope_id)
    stored = next((item for item in history if item["event_id"] == event_id), None)
    assert stored is not None, f"event_id {event_id} is not a row in the authority log"
    assert stored["request_id"] == request_id
    assert stored["event_type"] == event_type
    assert stored["actor"] == "management" and stored["authority"]["source"] == "management"

    read = await client.get("/api/memories/audit",
                            params={"request_id": request_id, "scope_id": scope_id})
    assert read.status_code == 200, read.text
    record = read.json()
    assert record["event_id"] == str(event_id)
    assert record["request_id"] == request_id
    assert record["event_type"] == event_type
    assert record["knowledge_scope_id"] == str(scope_id)
    assert record["actor"] == "management"
    assert record["occurred_at"] == stored["occurred_at"]
    assert record["schema_version"] == 1
    for key in payload_keys:
        assert key in record["payload_summary"], (key, record["payload_summary"])
    return record


async def _custom_domain(session, prefix):
    """A non-builtin domain: only such a domain accepts a policy edit at all."""
    sid, payload = await scope_and_payload(session)
    key = f"{prefix}-" + uuid4().hex
    seed = await session.get(DomainProfile, "generic")
    session.add(DomainProfile(domain_key=key, name=key, supported_formats=["markdown"],
                              graph_relations={}, default_capabilities={}, is_builtin=False,
                              memory_policy=dict(seed.memory_policy)))
    (await session.get(KnowledgeScope, sid)).domain_key = key
    await session.commit()
    return sid, payload, key


@pytest.mark.asyncio
async def test_retire_purge_rollback_and_policy_pointers_dereference(db_session, engine, monkeypatch):
    _inject_embedding(monkeypatch, _DeterministicEmbedding())
    sid, payload, key = await _custom_domain(db_session, "t083-policy")
    first = await _management_service(db_session).record(payload)
    app = create_app()
    async with _writer(engine) as owner:
        app.state.writer_lease = owner
        async with _app_client(app, db_session) as client:
            retired = await client.post("/api/memories/retire", json={
                "scope_id": sid, "memory_id": first["memory_id"], "reason": "T083 retire audit"})
            assert retired.status_code == 200, retired.text
            record = await _pointer(client, db_session, retired.json(), event_type="retract",
                                    scope_id=sid, payload_keys=("reason", "impact", "before_fingerprint"))
            assert record["payload_summary"]["reason"] == "T083 retire audit"
            assert record["aggregate_id"] == str(first["memory_id"])

            purged = await client.post("/api/memories/purge", json={
                "scope_id": sid, "memory_id": first["memory_id"], "reason": "T083 purge audit"})
            assert purged.status_code == 200, purged.text
            record = await _pointer(client, db_session, purged.json(), event_type="retract",
                                    scope_id=sid, payload_keys=("purge", "impact"))
            assert record["payload_summary"]["purge"] is True

            rolled = await client.post("/api/memories/rollback", json={
                "scope_id": sid, "event_point": first["memory_id"], "reason": "T083 rollback audit"})
            assert rolled.status_code == 200, rolled.text
            record = await _pointer(client, db_session, rolled.json(), event_type="rollback",
                                    scope_id=sid, payload_keys=("event_point", "impact"))
            assert record["payload_summary"]["event_point"] == first["memory_id"]

            edited = await client.post("/api/memories/policy", json={
                "scope_id": sid, "reason": "T083 policy audit", "policy": {"decay_rate": .07}})
            assert edited.status_code == 200, edited.text
            record = await _pointer(client, db_session, edited.json(), event_type="grant",
                                    scope_id=sid, payload_keys=("domain_key", "policy_before", "policy_after"))
            assert record["payload_summary"]["domain_key"] == key
            assert record["payload_summary"]["policy_after"]["decay_rate"] == .07

            # Every action carries its own request identity and its own row.
            pointers = [retired.json(), purged.json(), rolled.json(), edited.json()]
            assert len({item["request_id"] for item in pointers}) == 4
            assert len({item["event_id"] for item in pointers}) == 4

            # The explicit scope query disambiguates instead of disclosing.
            foreign = await client.get("/api/memories/audit", params={
                "request_id": retired.json()["request_id"], "scope_id": sid + 1})
            assert foreign.status_code == 404
            assert foreign.json()["detail"]["code"] == "MEMORY_AUDIT_NOT_FOUND"
    await db_session.rollback()


@pytest.mark.asyncio
async def test_refused_commands_and_quota_refusals_leave_no_authority_row(db_session, engine, monkeypatch):
    _inject_embedding(monkeypatch, _DeterministicEmbedding())
    sid, payload = await scope_and_payload(db_session)
    service = _management_service(db_session)
    first = await service.record(payload)
    app = create_app()
    async with _writer(engine) as owner:
        app.state.writer_lease = owner
        async with _app_client(app, db_session) as client:
            before = await _scope_event_count(db_session, sid)

            unknown = await client.post("/api/memories/retire", json={
                "scope_id": sid, "memory_id": 999999999999999, "reason": "no such entry"})
            assert unknown.status_code == 403, unknown.text
            assert unknown.json()["detail"]["code"] == "MEMORY_ROLLBACK_FORBIDDEN"

            # The selected scope is builtin: the policy command is refused before
            # any write, with the registry's own read-only reason.
            builtin = await client.post("/api/memories/policy", json={
                "scope_id": sid, "reason": "builtin domain", "policy": {"decay_rate": .07}})
            assert builtin.status_code == 400, builtin.text
            assert builtin.json()["detail"]["code"] == "MEMORY_PROVENANCE_INVALID"

            illegal = await client.post("/api/memories/policy", json={
                "scope_id": sid, "reason": "illegal value", "policy": {"decay_rate": -1}})
            assert illegal.status_code == 422, illegal.text

            stale = await client.post("/api/memories/promote", json={
                "scope_id": sid, "memory_id": first["memory_id"],
                "candidate_version": "b" * 64, "reason": "stale candidate"})
            assert stale.status_code == 409, stale.text
            assert stale.json()["detail"]["code"] == "MEMORY_CANDIDATE_VERSION_CHANGED"

            assert await _scope_event_count(db_session, sid) == before

            # A quota refusal is a real refused write too: the policy edit that
            # installs the quota is itself a governance row, the refused record is not.
            key = "t083-quota-" + uuid4().hex
            seed = await db_session.get(DomainProfile, "generic")
            db_session.add(DomainProfile(domain_key=key, name=key, supported_formats=["markdown"],
                                         graph_relations={}, default_capabilities={}, is_builtin=False,
                                         memory_policy=dict(seed.memory_policy)))
            (await db_session.get(KnowledgeScope, sid)).domain_key = key
            await db_session.commit()
            quota = await client.post("/api/memories/policy", json={
                "scope_id": sid, "reason": "quota for T083", "policy": {"per_scope_memory_quota": 1}})
            assert quota.status_code == 200, quota.text
            after_policy = await _scope_event_count(db_session, sid)
            assert after_policy == before + 1
            await _pointer(client, db_session, quota.json(), event_type="grant",
                           scope_id=sid, payload_keys=("policy_after",))

            with pytest.raises(ValueError, match="MEMORY_QUOTA_EXCEEDED"):
                await service.record({**payload, "content": "Over quota for T083"})
            await db_session.rollback()
            assert await _scope_event_count(db_session, sid) == after_policy
    await db_session.rollback()


async def _committed_candidate(session, owner):
    """One real consolidation output marked as a promotion candidate (T066 intent).

    ``tests.integration.consolidation_fixtures.create_scope`` refuses to run
    without ``CONSOLIDATION_ISOLATED_DATABASE``; this module declares the scope it
    owns instead of weakening that guard, and reuses the real anchor, runtime,
    adjudication and commit path unchanged.
    """
    from rag_mcp.orchestration.consolidation_pipeline import ProposalBatch, thaw
    from rag_mcp.services.consolidation_adjudicator import (
        AdjudicationContext,
        adjudicate_batch,
        memory_ref,
    )
    from rag_mcp.services.consolidation_commit import read_evidence
    from rag_mcp.services.consolidation_runtime import ConsolidationRuntime
    from rag_mcp.services.memory_policy import MemoryPolicy
    from tests.integration.consolidation_fixtures import recorded_episode
    from tests.integration.promotion_fixtures import published_anchor
    from tests.integration.test_013_consolidation_dependencies import keep_lease_alive
    from tests.unit.consolidation_cases import facts

    key = "t083-promotion-" + uuid4().hex
    session.add(DomainProfile(domain_key=key, name=key, supported_formats=["markdown"],
                              graph_relations={}, default_capabilities={}, is_builtin=False,
                              memory_policy={"consolidation_enabled": True, "consolidation": {}}))
    scope_id = generate_id()
    session.add(KnowledgeScope(scope_id=scope_id, scope_type="public", slug=key, name=key,
                               domain_key=key))
    await session.commit()

    service = _management_service(session)
    anchor = await published_anchor(session, scope_id)
    source = (await recorded_episode(service, scope_id, "Observed source"))["memory_id"]
    runtime = ConsolidationRuntime(session, owner=owner, memory_service=service)
    await keep_lease_alive(session, owner)
    token = await runtime.admit(scope_id, trigger="manual")
    window = await runtime.select_and_seal(token)
    current = await runtime.read_snapshot(scope_id)
    proposal = {"proposal_id": "p0", "action": "extract_fact",
                "source_refs": [memory_ref(current.entries[source])], "kind": "semantic",
                "content": "Promotable conclusion", "confidence": .97,
                "evidence_refs": [str(anchor["chunk_id"])], "justification": "Observed behavior"}
    batch = ProposalBatch([proposal])
    support = await read_evidence(session, {str(anchor["chunk_id"])})
    context = AdjudicationContext(window=window, inferences={"p0": facts(proposal)},
                                  support_facts=support, support_versions=support)
    policy = MemoryPolicy.model_validate(thaw(window.policy))
    decisions = adjudicate_batch(batch, current, policy, current.vocabulary,
                                 {"count": current.quota_count,
                                  "limit": policy.per_scope_memory_quota},
                                 context, datetime.now(UTC))
    assert decisions.groups and [item.decision for item in decisions.decisions] == ["accept"]
    await session.rollback()
    outcome = await service.commit_approved(decisions, token, runtime=runtime, batch=batch,
                                            context=context)
    assert outcome.status == "completed", outcome.reason_codes
    snapshot = await runtime.read_snapshot(scope_id)
    memory = snapshot.entries[outcome.output_memory_ids[0]]
    await session.rollback()
    assert memory["candidate_version"] and memory.get("candidate_basis")
    return {"scope": scope_id, "memory_id": memory["memory_id"],
            "candidate_version": memory["candidate_version"]}


@pytest.mark.asyncio
async def test_promotion_pointer_is_returned_and_dereferenceable(db_session, engine, monkeypatch):
    _inject_embedding(monkeypatch, _DeterministicEmbedding())
    app = create_app()
    async with _writer(engine) as owner:
        app.state.writer_lease = owner
        fixture = await _committed_candidate(db_session, owner)
        scope, memory_id = fixture["scope"], fixture["memory_id"]
        await db_session.rollback()
        async with _app_client(app, db_session) as client:
            request = {"scope_id": scope, "memory_id": memory_id,
                       "candidate_version": fixture["candidate_version"],
                       "reason": "T083 promotion audit"}
            before = await _scope_event_count(db_session, scope)
            created = await client.post("/api/memories/promote", json=request)
            assert created.status_code == 202, created.text
            body = created.json()
            assert body["event_id"] == str(body["task_id"]), \
                "the pointer still survives only as task_id"
            record = await _pointer(client, db_session, body, event_type="grant", scope_id=scope,
                                    payload_keys=("grant_type", "candidate_version", "source_id"))
            assert record["payload_summary"]["grant_type"] == "promotion_requested"
            assert record["payload_summary"]["candidate_version"] == fixture["candidate_version"]
            assert record["payload_summary"]["request_id"] == body["request_id"]
            assert await _scope_event_count(db_session, scope) == before + 1

            repeated = await client.post("/api/memories/promote", json=request)
            assert repeated.status_code == 200, repeated.text
            assert repeated.json()["reused"] is True
            assert (repeated.json()["event_id"], repeated.json()["request_id"]) == (
                body["event_id"], body["request_id"])
            assert await _scope_event_count(db_session, scope) == before + 1

            refused = await client.post("/api/memories/promote",
                                        json={**request, "candidate_version": "b" * 64})
            assert refused.status_code == 409, refused.text
            assert refused.json()["detail"]["code"] == "MEMORY_CANDIDATE_VERSION_CHANGED"
            assert await _scope_event_count(db_session, scope) == before + 1

            # GET /promotions/{task_id} keeps exposing the same authority event ids.
            report = await client.get(f"/api/memories/promotions/{body['task_id']}",
                                      params={"scope_ref": str(scope)})
            assert report.status_code == 200, report.text
            assert report.json()["authority_event_ids"] == [body["event_id"]]
    await db_session.rollback()


@pytest.mark.asyncio
async def test_audit_read_path_is_writer_only_and_absent_from_the_mcp_surface(db_session, engine, monkeypatch):
    from rag_mcp.mcp import create_mcp_server
    from rag_mcp.mcp.memory_tools import tool_names

    provider = _DeterministicEmbedding()
    _inject_embedding(monkeypatch, provider)

    app = create_app()
    paths = set(app.openapi()["paths"])
    assert "/api/memories/audit" in paths and "/api/memories/rebuild/audit" in paths

    writer_tools = {tool.name for tool in await create_mcp_server(
        embedding_provider=provider, mode="writer").list_tools()}
    reader_tools = {tool.name for tool in await create_mcp_server(
        embedding_provider=provider, mode="reader").list_tools()}
    assert writer_tools == set(tool_names("writer"))
    assert reader_tools == set(tool_names("reader"))
    assert not any("audit" in name for name in writer_tools | reader_tools)

    sid, payload = await scope_and_payload(db_session)
    first = await _management_service(db_session).record(payload)
    async with _app_client(app, db_session) as client:
        refused = await client.get("/api/memories/audit", params={"request_id": "no-lease"})
        assert refused.status_code == 503, refused.text
        assert refused.json()["detail"]["code"] == "MEMORY_WRITE_UNAVAILABLE"

    async with _writer(engine) as owner:
        app.state.writer_lease = owner
        async with _app_client(app, db_session) as client:
            missing = await client.get("/api/memories/audit",
                                       params={"request_id": "t083-no-such-request"})
            assert missing.status_code == 404, missing.text
            assert missing.json()["detail"]["code"] == "MEMORY_AUDIT_NOT_FOUND"

            rebuilt = await client.post("/api/memories/rebuild", json={
                "scope_id": sid, "reason": "T083 rebuild audit compatibility"})
            assert rebuilt.status_code == 200, rebuilt.text
            legacy = await client.get("/api/memories/rebuild/audit",
                                      params={"request_id": rebuilt.json()["request_id"]})
            assert legacy.status_code == 200, legacy.text
            assert set(legacy.json()) == {"request_id", "operation", "actor", "scope_id", "reason",
                                          "source_event_id", "since_event_id", "result", "created_at"}
            # A rebuild writes a management audit row, never an authority event.
            assert (await client.get("/api/memories/audit",
                                     params={"request_id": rebuilt.json()["request_id"]})
                    ).status_code == 404

            retired = await client.post("/api/memories/retire", json={
                "scope_id": sid, "memory_id": first["memory_id"], "reason": "T083 scope check"})
            assert retired.status_code == 200, retired.text
            pointer = retired.json()
            assert (await client.get("/api/memories/audit", params={
                "request_id": pointer["request_id"], "scope_id": sid})).status_code == 200
            assert (await client.get("/api/memories/audit", params={
                "request_id": pointer["request_id"], "scope_id": sid + 1})).status_code == 404
    await db_session.rollback()

