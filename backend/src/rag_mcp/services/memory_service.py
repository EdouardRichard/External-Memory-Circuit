from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import func, select, text

from rag_mcp.errors import MemoryContentConflictError, attachment_degradation_reason
from rag_mcp.indexing.qdrant_client import QdrantStore
from rag_mcp.models.domain_profile import DomainProfile
from rag_mcp.models.knowledge_scope import KnowledgeScope
from rag_mcp.models.knowledge_source import KnowledgeSource
from rag_mcp.models.knowledge_version import KnowledgeVersion
from rag_mcp.models.memory_event import MemoryEvent
from rag_mcp.models.memory_management_audit import MemoryManagementAudit
from rag_mcp.models.memory_projection import MemoryEntry
from rag_mcp.models.memory_projection_meta import MemoryProjectionMeta
from rag_mcp.models.processing_run import ProcessingRun
from rag_mcp.models.session import MemorySession
from rag_mcp.services.memory_event_store import MemoryEventStore
from rag_mcp.services.memory_projection_store import MemoryProjectionStore, ProjectionFailure
from rag_mcp.services.memory_reducer import reduce_events
from rag_mcp.services.memory_validators import (
    MemoryProvenanceValidator,
    check_quota,
    derive_ttl,
    detect_submission,
    redact_submission,
    validate_supersede,
)
from rag_mcp.utils.snowflake import generate_id


# --- 014 frozen contract constants -------------------------------------------
# Protocol limits, NOT policy defaults: a domain policy may only tighten them
# (data-model §3, plan.md "冻结契约常量"). They are module-level so the tool layer
# and the tests assert the same numbers instead of re-typing them.
ATTACH_ITEMS_HARD_LIMIT = 5
ATTACH_CHARACTERS_HARD_LIMIT = 800
ATTACH_EXCERPT_HARD_LIMIT = 200
ATTACH_TIMEOUT_MS_HARD_LIMIT = 800
#: 014 contract: ``memory_context`` is 1-4000 characters, while the direct
#: ``recall_memory`` query channel stays at 2000 (T080).
ATTACH_CONTEXT_MAX_CHARACTERS = 4000


def attachment_budget(policy) -> dict:
    """Effective attachment budget for one resolved scope policy.

    The policy is validated strictly (unknown keys rejected, integers exclude
    bool, floats reject NaN/Infinity) and then clamped by the frozen contract
    limits, so a misconfigured domain can never widen the protocol budget.
    """
    from rag_mcp.services.memory_policy import MemoryPolicy

    validated = MemoryPolicy.model_validate(policy or {})
    return {
        "top_k": min(validated.attach_top_k, ATTACH_ITEMS_HARD_LIMIT),
        "max_chars": min(validated.attach_max_chars, ATTACH_CHARACTERS_HARD_LIMIT),
        "excerpt_chars": min(validated.attach_excerpt_chars, ATTACH_EXCERPT_HARD_LIMIT),
        "timeout_ms": min(validated.attach_timeout_ms, ATTACH_TIMEOUT_MS_HARD_LIMIT),
        # Threshold dual track (research §3): an explicit memory_context uses the
        # permissive existing key, a session-only signal uses the conservative档.
        "min_score_with_context": validated.attach_min_score,
        "min_score_conservative": validated.attach_conservative_min_score,
        "delivered_ttl_seconds": validated.delivered_ttl_seconds,
    }


#: Attachment item key order, frozen to the contract declaration order.
ATTACHMENT_ITEM_KEYS = (
    "memory_id", "knowledge_scope_id", "kind", "provenance", "confidence", "title",
    "content_excerpt", "truncated", "content_length", "evidence_refs", "inference_meta",
    "valid_from", "valid_to", "observed_at", "session_id", "agent_id", "status",
    "superseded_by", "injection_flags", "attach_reason", "match",
)


def _aware(value):
    """Parse a timestamp, returning ``None`` for anything unparsable or naive.

    A6 (research §6): a malformed or timezone-naive timestamp must make the row a
    non-candidate, never raise out of the read path.
    """
    if value is None:
        return None
    from rag_mcp.orchestration.packing import timestamp as _timestamp

    try:
        return _timestamp(value)
    except (ValueError, TypeError, AttributeError):
        return None


def attachment_visible(row, *, now) -> bool:
    """The attachment-layer visibility predicate (FR-007, data-model §7.1).

    Excludes ``quarantined``/``superseded``/``retired`` (status), ``compressed``
    and ``archived`` (retention stage), closed rows (``valid_to``), expired rows
    and writes that never completed. Any unusable timestamp fails closed.
    """
    if not isinstance(row, Mapping):
        return False
    if row.get("status") != "active":
        return False
    if row.get("retention_stage") != "active":
        return False
    if row.get("valid_to") is not None:
        return False
    if row.get("write_status") not in (None, "complete"):
        return False
    observed = _aware(row.get("observed_at"))
    if observed is None:
        return False
    valid_from = row.get("valid_from")
    if valid_from is not None and _aware(valid_from) is None:
        return False
    expires = row.get("expires_at")
    if expires is not None:
        expiry = _aware(expires)
        if expiry is None or expiry <= now:
            return False
    return True


def attachment_hard_anchor_ok(row) -> bool:
    """A ``hard`` item needs a per-item attribution anchor, or it is excluded.

    ``soft``/``distilled`` items carry their provenance in ``inference_meta``
    instead, so the anchor requirement applies to ``hard`` only (FR-002/FR-007).
    """
    if row.get("provenance") != "hard":
        return True
    return bool(row.get("evidence_refs"))


def attachment_min_score(policy, *, has_context: bool, has_session: bool) -> float:
    """The threshold actually applied to ``match.dense_similarity``.

    T106 ruling (spec Edge Cases / FR-006 / FR-015 / Q5): the permissive
    ``attach_min_score`` track applies only when **both** explicit signals are
    present.  A context-only or session-only request uses the conservative
    "宁缺勿滥" track (``attach_conservative_min_score``).
    """
    budget = attachment_budget(policy)
    permissive = bool(has_context and has_session)
    return budget["min_score_with_context"] if permissive else budget["min_score_conservative"]


def _entry_text(row) -> tuple[str, int]:
    """Return ``(available_text, full_content_length)`` for a row or public entry.

    The attachment layer reuses ``MemoryReader.recall`` and therefore normally
    receives ``public_entry`` dicts, which expose a 300-character
    ``content_excerpt`` plus the true ``content_length``. Because the 014 excerpt
    limit (200) is below 300, that is enough to render the attachment excerpt
    without opening a second query channel. Raw projection rows (``content_text``)
    are also accepted so the pure selection stays directly testable.
    """
    if row.get("content_text") is not None:
        text = row["content_text"]
        return text, len(text)
    excerpt = row.get("content_excerpt")
    if excerpt is None:
        return "", int(row.get("content_length") or 0)
    return excerpt, int(row.get("content_length") or len(excerpt))


def attachment_item(row, *, match, attach_reason: str, excerpt_chars: int) -> dict:
    """Assemble one contract-aligned attachment item, in the frozen key order.

    ``valid_to``/``superseded_by`` are pinned to ``null`` and ``status`` to
    ``active`` because non-active or closed rows are excluded before this point;
    the evidence-layer locating fields are never copied (Constitution IV).
    """
    text, content_length = _entry_text(row)
    excerpt = text[:excerpt_chars]
    item = {
        "memory_id": row["memory_id"],
        "knowledge_scope_id": row["knowledge_scope_id"],
        "kind": row.get("kind"),
        "provenance": row.get("provenance"),
        "confidence": row.get("confidence"),
        "title": row.get("title"),
        "content_excerpt": excerpt,
        "truncated": content_length > len(excerpt),
        "content_length": content_length,
        "evidence_refs": list(row.get("evidence_refs") or []),
        "inference_meta": row.get("inference_meta"),
        "valid_from": row.get("valid_from"),
        "valid_to": None,
        "observed_at": row.get("observed_at"),
        "session_id": row.get("session_id"),
        "agent_id": row.get("agent_id"),
        "status": "active",
        "superseded_by": None,
        "injection_flags": dict(row.get("injection_flags") or {}),
        "attach_reason": attach_reason,
        "match": match,
    }
    return {key: item[key] for key in ATTACHMENT_ITEM_KEYS}


def attachment_candidates(
    rows,
    *,
    matches=None,
    policy=None,
    has_context: bool,
    session_id=None,
    now=None,
    delivered: Iterable[int] = (),
    include_delivered: bool = False,
    anchor_verified_ids: Iterable[int] | None = None,
) -> dict:
    """Pure attachment selection: visibility, threshold, dedup, budget, ordering.

    No IO, no clock and no model: ``now`` is explicit, ordering is by
    ``(-dense_similarity, memory_id)`` and the budget comes from the validated
    domain policy clamped by the frozen contract limits.

    Threshold track (T106 ruling): the permissive ``attach_min_score`` applies
    only when ``has_context`` **and** a ``session_id`` are both present; either
    signal alone selects the conservative track.
    """
    from rag_mcp.orchestration.packing import text_characters

    budget = attachment_budget(policy)
    threshold = attachment_min_score(policy, has_context=has_context,
                                     has_session=session_id is not None)
    moment = now or datetime.now(UTC)
    entries = list(rows.values()) if isinstance(rows, Mapping) else list(rows)
    matches = matches or {}
    delivered_ids = set(delivered or ())
    # ``None`` means "presence-only" (the pure-function contract); the attach
    # path passes the live per-item re-verification result (T075/FR-003).
    verified_hard = None if anchor_verified_ids is None else set(anchor_verified_ids)

    visible: list[tuple[int, Mapping, Mapping | None]] = []
    filtered_inactive = 0
    unscored = 0
    for row in entries:
        if not isinstance(row, Mapping):
            continue
        memory_id = row.get("memory_id")
        if memory_id is None:
            continue
        if not attachment_visible(row, now=moment) or not attachment_hard_anchor_ok(row):
            filtered_inactive += 1
            continue
        if (verified_hard is not None and row.get("provenance") == "hard"
                and int(memory_id) not in verified_hard):
            # A hard item whose anchors cannot be re-verified live never enters
            # the attachment (FR-003); it is counted as state-filtered.
            filtered_inactive += 1
            continue
        match = matches.get(memory_id)
        if match is None:
            match = row.get("match")
        if not isinstance(match, Mapping) or match.get("dense_similarity") is None:
            # Without the dense component the documented threshold cannot be
            # applied, so the candidate is not admitted (fail closed) — but it
            # is still a visible candidate and must not vanish from the counts
            # (T091/FR-014).
            unscored += 1
            continue
        visible.append((int(memory_id), row, match))

    candidates = len(visible) + unscored
    failed_paths: list[str] = []
    if visible:
        eligible = [entry for entry in visible if entry[2]["dense_similarity"] >= threshold]
        if not eligible:
            failed_paths = ["below_min_score"]
    else:
        eligible = []
        if unscored or (not matches and entries):
            failed_paths = ["memory_unavailable"]
        elif filtered_inactive:
            # T091: the pool existed but every row was state-invisible.
            failed_paths = ["state_filtered"]

    eligible.sort(key=lambda entry: (-entry[2]["dense_similarity"], entry[0]))

    dropped_delivered = 0
    if delivered_ids and not include_delivered:
        dropped_delivered = sum(1 for mid, _, _ in eligible if mid in delivered_ids)
        eligible = [entry for entry in eligible if entry[0] not in delivered_ids]

    attach_reason = ("session_and_context" if session_id is not None and has_context
                     else "context_match" if has_context else "session_recent")

    truncated_by_budget = 0
    selected: list[dict] = []
    if len(eligible) > budget["top_k"]:
        truncated_by_budget += len(eligible) - budget["top_k"]
        eligible = eligible[: budget["top_k"]]

    characters = 0
    for memory_id, row, match in eligible:
        item = attachment_item(row, match=match, attach_reason=attach_reason,
                               excerpt_chars=budget["excerpt_chars"])
        size = text_characters(item)
        if characters + size > budget["max_chars"]:
            truncated_by_budget += 1
            continue
        characters += size
        selected.append(item)

    if not selected and truncated_by_budget:
        # T091: the pool existed but the whole pool was cropped by the budget.
        failed_paths = sorted(set(failed_paths + ["budget_exhausted"]))

    return {
        "items": selected,
        "counts": {
            "returned": len(selected),
            "candidates": candidates,
            "truncated_by_budget": truncated_by_budget,
            "dropped_delivered": dropped_delivered,
            "filtered_inactive": filtered_inactive,
            "characters": characters,
        },
        "failed_paths": list(failed_paths),
    }


def detect_context_flags(memory_context):
    """Run the existing injection detection on ``memory_context``, first.

    Detection is **先行** (Clarifications Q5): it must complete before any recall,
    scoring, sorting or assembly. It never raises and never relaxes a filter or a
    threshold — a failed detector degrades to ``detection_degraded`` and the
    context is still treated as untrusted data only.
    """
    if memory_context is None:
        return {}, []
    try:
        sanitized = detect_submission({"content": memory_context})
    except Exception:  # noqa: BLE001 - detection failure must not block retrieval
        return {}, ["detection_degraded"]
    return dict(sanitized.injection_flags or {}), []


def _hard_replacement_command(event, target, validation):
    """Pure adapter called only after the existing record authorization checks."""
    from rag_mcp.services.consolidation_adjudicator import _COMMAND_ISSUER_SEAL, _governed_command

    return _governed_command(event, target=target, validation=validation, _issuer=_COMMAND_ISSUER_SEAL)


class MemoryService:
    def __init__(self, session, projection_store=None, *, embedding_provider=None, qdrant_store=None, projection_root=None):
        from rag_mcp.providers.factory import get_provider_bundle

        self.session = session
        self.projections = projection_store or MemoryProjectionStore(session,
            embedding_provider=embedding_provider or get_provider_bundle().embedding,
            qdrant_store=qdrant_store, projection_root=projection_root)

    def _ensure_vector_store(self):
        if self.projections.qdrant is None:
            self.projections.qdrant = QdrantStore()

    async def recall(self, **parameters):
        from rag_mcp.services.memory_reader import MemoryReader
        return await MemoryReader(self.session, self.projections).recall(**parameters)

    async def attach(self, *, scope_ref, query=None, session_id=None, memory_context=None,
                     policy=None, include_delivered=False):
        """014 attachment layer: a parameterised reuse of the existing recall path.

        No new query channel is opened (research §1/§3): the same
        ``MemoryReader.recall`` performs scope resolution, visibility, delivered
        dedup, fusion and auditing. This method only

        * runs the existing injection detection on ``memory_context`` **first**,
          before any recall/scoring/sorting/assembly (Clarifications Q5),
        * applies the 014 budget, threshold dual track, state re-verification and
          ordering over the recall result, and
        * degrades on its own: any failure is reported as a reason and never
          propagates to the primary retrieval (FR-004/SC-003).

        Returns ``{"items", "counts", "failed_paths", "injection_flags"}``.
        """
        from rag_mcp.services.memory_reader import MemoryReader

        flags, detection_failed = detect_context_flags(memory_context)
        empty_counts = {"returned": 0, "candidates": 0, "truncated_by_budget": 0,
                        "dropped_delivered": 0, "filtered_inactive": 0, "characters": 0}
        result = {"items": [], "counts": dict(empty_counts),
                  "failed_paths": list(detection_failed), "injection_flags": flags}

        # An explicit memory_context is the recall query; a session-only signal
        # reuses the primary query with the conservative threshold (research §3).
        recall_query = memory_context if memory_context is not None else query
        has_context = memory_context is not None
        try:
            if policy is None:
                # T079: the tool path must consume the resolved domain policy, not
                # silently fall back to the built-in defaults.
                policy = await self._attachment_policy(scope_ref)
            budget = attachment_budget(policy)
            async with asyncio.timeout(budget["timeout_ms"] / 1000):
                recall = await MemoryReader(self.session, self.projections).recall(
                    scope_ref=scope_ref,
                    query=recall_query,
                    session_id=session_id,
                    limit=max(budget["top_k"], 1) * 4,
                    include_delivered=include_delivered,
                    include_injection_flags=True,
                    # T080: the 014 contract allows a 4000-character
                    # memory_context; only the attachment channel may use it.
                    max_query_characters=ATTACH_CONTEXT_MAX_CHARACTERS,
                    tool="search_knowledge",
                    channel="attached",
                )
        except (TimeoutError, asyncio.TimeoutError):
            result["failed_paths"] = sorted(set(result["failed_paths"] + ["attachment_timeout"]))
            return result
        except Exception as exception:  # noqa: BLE001 - the attachment degrades on its own
            result["failed_paths"] = sorted(
                set(result["failed_paths"] + [attachment_degradation_reason(exception)])
            )
            return result

        notice_paths = list((recall.get("memory_notice") or {}).get("failed_paths") or [])
        if recall.get("completion_status") == "failed":
            result["failed_paths"] = sorted(
                set(result["failed_paths"] + notice_paths + ["memory_unavailable"])
            )
            return result
        # T081: a ``partial`` recall (sub-path unavailability) must not vanish.
        result["failed_paths"] = sorted(set(result["failed_paths"] + notice_paths))

        selection = attachment_candidates(
            list(recall.get("memories") or []),
            policy=policy,
            has_context=has_context,
            session_id=session_id,
            delivered=(),
            include_delivered=include_delivered,
            anchor_verified_ids=await self._verify_hard_anchors(list(recall.get("memories") or [])),
        )
        if flags:
            # T076: the detected memory_context flags ride along with each item.
            for item in selection["items"]:
                merged = dict(item.get("injection_flags") or {})
                merged.update(flags)
                item["injection_flags"] = merged
        result["items"] = selection["items"]
        result["counts"] = {**empty_counts, **selection["counts"]}
        # Delivered dedup happens once, inside the single read channel; surface
        # its count rather than recomputing it here.
        result["counts"]["dropped_delivered"] = int(
            (recall.get("counts") or {}).get("dropped_delivered") or 0
        )
        result["failed_paths"] = sorted(set(result["failed_paths"] + selection["failed_paths"]))
        return result

    async def _verify_hard_anchors(self, entries) -> set[int]:
        """T075/FR-003: live per-item attribution re-verification for ``hard`` items.

        Presence of ``evidence_refs`` is not enough: each anchor is re-read and
        must still be a published, attributed chunk in a published source. An
        unreadable anchor store is treated as "cannot verify", so every hard item
        is excluded (fail closed) rather than attaching an unverifiable claim.
        """
        needed: dict[int, list[str]] = {}
        for row in entries:
            if isinstance(row, Mapping) and row.get("provenance") == "hard" \
                    and row.get("memory_id") is not None:
                needed[int(row["memory_id"])] = [str(ref) for ref in (row.get("evidence_refs") or [])]
        if not needed:
            return set()
        identifiers = sorted({ref for refs in needed.values() for ref in refs})
        if not identifiers:
            return set()
        from rag_mcp.services.consolidation_commit import read_evidence

        try:
            facts = await read_evidence(self.session, identifiers)
        except Exception:  # noqa: BLE001 - unreadable anchors never authorize a hard item
            return set()
        verified: set[int] = set()
        for memory_id, refs in needed.items():
            for ref in refs:
                fact = facts.get(ref) or {}
                if (fact.get("status") == "published" and fact.get("source_status") == "published"
                        and fact.get("attributed")):
                    verified.add(memory_id)
                    break
        return verified

    async def _attachment_policy(self, scope_ref):
        """T079: the domain policy the attachment budget/threshold must obey.

        Every resolved scope's ``memory_policy`` is merged conservatively —
        smallest counts, character budgets, excerpt, timeout and delivery window,
        highest score thresholds — so a permissive domain can never widen the
        attachment budget for a stricter one.
        """
        from rag_mcp.models.domain_profile import DomainProfile
        from rag_mcp.models.knowledge_scope import KnowledgeScope
        from rag_mcp.services.memory_policy import MemoryPolicy
        from rag_mcp.services.scope_resolver import MemoryScopeResolver

        scope_ids = await MemoryScopeResolver(self.session).resolve_many(scope_ref)
        policies = []
        for scope_id in scope_ids:
            scope = await self.session.get(KnowledgeScope, scope_id)
            profile = await self.session.get(DomainProfile, scope.domain_key) if scope is not None else None
            policies.append(MemoryPolicy.model_validate(
                (profile.memory_policy if profile is not None else None) or {}))
        if not policies:
            return MemoryPolicy()
        if len(policies) == 1:
            return policies[0]
        merged = {field: min(int(getattr(item, field)) for item in policies)
                  for field in ("attach_top_k", "attach_max_chars", "attach_excerpt_chars",
                                "attach_timeout_ms", "delivered_ttl_seconds")}
        merged["attach_conservative_min_score"] = max(
            float(item.attach_conservative_min_score) for item in policies)
        # The conservative track may never drop below the permissive minimum.
        merged["attach_min_score"] = min(
            max(float(item.attach_min_score) for item in policies),
            merged["attach_conservative_min_score"],
        )
        return MemoryPolicy.model_validate(merged)

    async def start_work(self, **parameters):
        from rag_mcp.services.memory_reader import MemoryReader
        return await MemoryReader(self.session, self.projections).start_work(**parameters)

    async def govern(self, action, **parameters):
        from rag_mcp.services.memory_governance import MemoryGovernance
        return await MemoryGovernance(self).execute(action, **parameters)

    async def apply_event(self, event_data):
        # Raw caller events are not an authorized memory-write surface.
        raise PermissionError("MEMORY_WRITE_UNAVAILABLE: use validated memory commands")

    async def commit_approved(self, decisions, token, *, runtime, batch, context):
        from rag_mcp.services.consolidation_commit import commit_approved
        return await commit_approved(self, decisions, token, runtime=runtime, batch=batch, context=context)

    async def promotion_candidates(self, *, scope_id, limit=50, offset=0):
        """Derived candidate read model with current promotability (T073).

        Candidates come from authoritative approved markers; every item is
        re-verified against the current corpus anchors so an already marked but
        no longer eligible candidate stays visible for audit without claiming it
        is still promotable.
        """
        from rag_mcp.services.consolidation_adjudicator import candidate_eligibility
        from rag_mcp.services.consolidation_commit import read_evidence
        from rag_mcp.services.memory_policy import MemoryPolicy

        scope = await self.session.get(KnowledgeScope, scope_id)
        if scope is None or scope.status != "active":
            raise ValueError("MISSING_KNOWLEDGE_SCOPE")
        profile = await self.session.get(DomainProfile, scope.domain_key)
        policy = MemoryPolicy.model_validate((profile.memory_policy if profile else None) or {})
        threshold = policy.consolidation.candidate_min_confidence if policy.consolidation else 1.0
        state = reduce_events(await MemoryEventStore(self.session).replay(scope_id))
        marked = [row for row in state["entries"].values() if row.get("candidate_version")]
        identifiers = {str(reference) for row in marked for reference in row.get("evidence_refs") or ()}
        facts = await read_evidence(self.session, identifiers) if identifiers else {}
        items = []
        for row in sorted(marked, key=lambda item: (item.get("observed_at") or "", item["memory_id"]), reverse=True):
            eligible, reasons = candidate_eligibility(row, facts, scope_id=scope_id, threshold=threshold)
            items.append({"memory_id": row["memory_id"], "kind": row.get("kind"),
                          "provenance": row.get("provenance"), "confidence": row.get("confidence"),
                          "promote_candidate_at": row.get("promote_candidate_at"),
                          "candidate_version": row.get("candidate_version"),
                          "evidence_attributions": [dict(item) for item in
                                                    (row.get("candidate_basis") or {}).get("evidence_attributions", ())],
                          "promotable": eligible, "ineligibility_reasons": list(reasons),
                          "promotion_pointer": row.get("promotion_pointer")})
        return {"scope_id": scope_id, "items": items[offset:offset + limit], "total": len(items)}

    async def promote_candidate(self, *, scope_id, memory_id, candidate_version, actor, reason, request_id=None):
        """Explicit human promotion short transaction (T072)."""
        from rag_mcp.services.memory_governance import MemoryGovernance

        return await MemoryGovernance(self).promote(scope_id=scope_id, memory_id=memory_id,
                                                    candidate_version=candidate_version, actor=actor,
                                                    reason=reason, request_id=request_id)

    async def promotion_status(self, *, task_id, scope_id):
        """Stable promotion task report derived from actual facts (T073)."""
        from rag_mcp.models.chunk import Chunk

        if isinstance(task_id, str) and task_id.isdecimal():
            task_id = int(task_id)
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise LookupError("MEMORY_PROMOTION_NOT_FOUND")
        history = await MemoryEventStore(self.session).replay(scope_id)
        request = next((event for event in history if event["event_id"] == task_id
                        and event["event_type"] == "grant"
                        and event["payload"].get("grant_type") in ("promotion_requested", "promotion_observed")), None)
        if request is None:
            raise LookupError("MEMORY_PROMOTION_NOT_FOUND")
        state = reduce_events(history)
        entry = state["entries"].get(request["payload"]["memory_id"]) or {}
        pointer = entry.get("promotion_pointer") or request["payload"]["pointer"]
        source = await self.session.get(KnowledgeSource, pointer["source_id"], populate_existing=True)
        runs = (await self.session.execute(select(ProcessingRun).where(
            ProcessingRun.source_id == pointer["source_id"]).order_by(ProcessingRun.run_id))).scalars().all()
        attempt_run_ids = [run.run_id for run in runs] or list(pointer["attempt_run_ids"])
        published_version_id = None
        if source is not None and source.status == "published":
            published_version_id = await self.session.scalar(
                select(KnowledgeVersion.version_id)
                .join(Chunk, Chunk.version_id == KnowledgeVersion.version_id)
                .where(Chunk.source_id == pointer["source_id"], KnowledgeVersion.status == "published")
                .order_by(KnowledgeVersion.version_number.desc()).limit(1))
        if source is None:
            status = "failed" if pointer["result"] else "accepted"
        elif published_version_id is not None:
            status = "published"
        elif source.status in ("failed", "deleted"):
            status = "failed"
        elif source.status == "processing" or any(run.status == "running" for run in runs):
            status = "processing"
        elif pointer["status"] == "failed" and pointer["result"]:
            status = "failed"
        else:
            status = "uploaded"
        return {"task_id": str(task_id), "memory_id": pointer["memory_id"],
                "candidate_version": pointer["candidate_version"], "source_id": pointer["source_id"],
                "initial_processing_run_id": pointer["initial_processing_run_id"],
                "attempt_run_ids": attempt_run_ids, "status": status,
                "published_version_id": published_version_id, "result": pointer["result"],
                "authority_event_ids": list(pointer["authority_event_ids"])}

    async def observe_promotion(self, *, source_id, status=None, result=None):
        """Append a permanent promotion_observed grant for real attempts (T074).

        Uploaded, processing and failed never report published: the recorded
        status comes from the actual source/run/version facts unless an explicit
        pre-dispatch failure is being recorded. Regular uploads have no
        promotion request and are left untouched. The original request pointer
        is never mutated; a new permanent observation extends its history.
        """
        from rag_mcp.services.memory_governance import MemoryGovernance

        return await MemoryGovernance(self).observe_promotion(source_id=source_id, status=status, result=result)

    async def recover_consolidation(self, token, *, runtime):
        from rag_mcp.services.consolidation_commit import recover_consolidation
        return await recover_consolidation(self, token, runtime=runtime)

    async def record(self, payload):
        scope_id = payload.get("scope_id")
        scope = await self.session.get(KnowledgeScope, scope_id) if isinstance(scope_id, int) else None
        if not scope or scope.status != "active":
            # FR-061/FR-018: the rejection must carry the candidate domains, exactly
            # like the resolver-level rejection (ScopeBindingError is a ValueError,
            # so existing callers and their `match=` assertions are unaffected).
            from rag_mcp.services.scope_resolver import missing_scope_error

            raise await missing_scope_error(self.session)
        clean = redact_submission(payload)
        validation = await MemoryProvenanceValidator(self.session).validate(clean)
        sanitized = detect_submission(clean)
        await self.session.execute(text("SELECT pg_advisory_xact_lock(:scope)"), {"scope": scope_id})
        profile = await self.session.get(DomainProfile, scope.domain_key)
        policy = profile.memory_policy or {}
        metadata = {key: clean.get(key) for key in (
            "kind", "provenance", "inference_meta", "confidence", "title", "session_id", "agent_id",
            "task_context", "supersedes_memory_id")}
        metadata["evidence_refs"] = sorted(set(clean.get("evidence_refs") or []))
        metadata["tags"] = sorted(set(clean.get("tags") or []))
        digest = hashlib.sha256(sanitized.content.encode()).hexdigest()
        matches = (await self.session.scalars(select(MemoryEntry).where(
            MemoryEntry.knowledge_scope_id == scope_id, MemoryEntry.content_hash == digest
        ).order_by(MemoryEntry.source_event_id, MemoryEntry.memory_id))).all()
        request_id = str(uuid4())
        if matches:
            if any(row.write_status != "complete" for row in matches):
                raise ValueError("MEMORY_WRITE_UNAVAILABLE")
            conflict = next((row for row in matches if row.submission_meta != metadata), None)
            if conflict:
                raise MemoryContentConflictError(conflict.memory_id)
            existing = matches[0]
            return {"memory_id": existing.memory_id, "status": existing.status,
                    "provenance_validation": validation, "injection_flags": existing.injection_flags,
                    "request_id": request_id}
        history = await MemoryEventStore(self.session).replay(scope_id)
        current = await self.projections.current(scope_id)
        if history and (not current or current.source_event_id != history[-1]["event_id"]):
            raise ValueError("MEMORY_WRITE_UNAVAILABLE")
        if clean.get("supersedes_memory_id"):
            target = await self.session.get(MemoryEntry, clean["supersedes_memory_id"])
            validate_supersede({**clean, "target": {"scope_id": target.knowledge_scope_id,
                "status": target.status, "provenance": target.provenance} if target else {}})
        count = await self.session.scalar(select(func.count()).select_from(MemoryEntry).where(
            MemoryEntry.knowledge_scope_id == scope_id, MemoryEntry.status == "active", MemoryEntry.write_status == "complete"
        ))
        check_quota(count, policy.get("per_scope_memory_quota", 5000))
        now = datetime.now(UTC)
        ttl = derive_ttl(clean["kind"], policy)
        identifier = generate_id()
        event_payload = {**metadata, "content_text": sanitized.content, "content_hash": digest,
            "submission_meta": metadata, "status": sanitized.status, "injection_flags": sanitized.injection_flags,
            "provenance_validation": validation, "decay_rate": policy.get("decay_rate", .05),
            "created_at": now.isoformat(), "updated_at": now.isoformat(),
            "expires_at": (now + timedelta(days=ttl)).isoformat() if ttl is not None else None}
        event = MemoryEvent(event_id=identifier, aggregate_id=identifier,
            event_type="revise" if clean.get("supersedes_memory_id") else "assert",
            knowledge_scope_id=scope_id, payload=event_payload, actor="memory_tool", request_id=request_id,
            session_id=clean.get("session_id"), occurred_at=now, valid_from=now,
            authority={"source": "validated_evidence" if clean["provenance"] == "hard" else "inference"},
            scope_meta={"knowledge_scope_id": scope_id}, mutability={"correction": "supersede"},
            provenance_meta=validation, recoverability={"source": "event_log"}, actionability="evidence")
        event_fields = {column.name: getattr(event, column.name) for column in MemoryEvent.__table__.columns if column.name != "created_at"}
        if clean.get('supersedes_memory_id') and target.provenance == 'hard':
            from rag_mcp.services.consolidation_adjudicator import guard_effect

            target_fact = {key: getattr(target, key) for key in (
                'memory_id', 'knowledge_scope_id', 'source_event_id', 'state_event_id', 'content_hash', 'provenance')}
            target_fact['state_event_id'] = target_fact['state_event_id'] or target_fact['source_event_id']
            command = _hard_replacement_command(event_fields, target_fact, validation)
            checked = guard_effect({'operation': 'replace', 'aggregate_id': target.memory_id,
                                    'replacement_id': identifier}, target_fact, command=command)
            if checked.decision != 'accept':
                raise PermissionError('HARD_MEMORY_PROTECTED')
        try:
            async with self.session.begin_nested():
                await MemoryEventStore(self.session).append(event)
                state = reduce_events(await MemoryEventStore(self.session).replay(scope_id))
                self._ensure_vector_store()
                try:
                    await self.projections.materialize(state, scope_id, identifier)
                    integrity = await self.projections.inspect(state, scope_id)
                except ProjectionFailure:
                    raise
                except Exception as error:
                    raise ProjectionFailure("integrity") from error
                if not all(row["matches_replay"] for row in integrity.values()):
                    raise ProjectionFailure("integrity")
                if clean.get("session_id"):
                    session = await self.session.get(MemorySession, clean["session_id"])
                    if session is None:
                        self.session.add(MemorySession(session_id=clean["session_id"], agent_id=clean.get("agent_id") or "unknown",
                            primary_scope_id=scope_id, started_at=now, last_active_at=now, status="active", expires_at=now + timedelta(days=7)))
                    else:
                        session.last_active_at = now
                        session.expires_at = now + timedelta(days=7)
            await self.session.commit()
            # 013 T083: a successful ordinary eligible write emits only an O(1)
            # offline volume hint. The foreground never selects a window, waits
            # for or calls the Distiller, or starts an awaited consolidation job.
            from rag_mcp.runtime.activity import mark_volume_hint

            mark_volume_hint(scope_id)
            # 014 T047: the same O(1) discipline for the read-only consumption
            # layer. This only marks the scope dirty and schedules a bounded
            # per-scope worker; it never awaits, never raises into the write and
            # skips entirely while the projection switch is off, so the write
            # critical path and its failure semantics are unchanged.
            from rag_mcp.runtime.memory_projection import mark_memory_projection_dirty

            mark_memory_projection_dirty(scope_id)
        except ProjectionFailure as failure:
            if failure.path == "relation":
                await self.session.rollback()
                raise
            # The savepoint restored the previous PG view and left our scoped
            # transaction lock held. Retain the failed command without moving
            # the completed manifest consumed by readers.
            try:
                await MemoryEventStore(self.session).append(MemoryEvent(**event_fields))
                state = reduce_events(await MemoryEventStore(self.session).replay(scope_id))
                await self.projections.retain_failure(state, scope_id, identifier, failure.path)
                await self.session.commit()
            except Exception:
                await self.session.rollback()
                raise
            raise ValueError(f"MEMORY_WRITE_UNAVAILABLE:{failure.path}") from None
        except Exception:
            await self.session.rollback()
            raise
        return {"memory_id": identifier, "status": sanitized.status, "provenance_validation": validation,
                "injection_flags": sanitized.injection_flags, "request_id": request_id}

    async def inspect_projections(self, scope_id):
        self._ensure_vector_store()
        state = reduce_events(await MemoryEventStore(self.session).replay(scope_id))
        return await self.projections.inspect(state, scope_id)

    async def rebuild(self, scope_id, *, actor, reason="management rebuild", since_event_id=None, request_id=None):
        from rag_mcp.runtime.activity import get_runtime_activity

        # 013 T082: wrap the actual rebuild lifetime (including exceptions and
        # cancellation) rather than only the REST handler.
        with get_runtime_activity().track('rebuild'):
            return await self._rebuild(scope_id, actor=actor, reason=reason,
                                       since_event_id=since_event_id, request_id=request_id)

    async def _rebuild(self, scope_id, *, actor, reason="management rebuild", since_event_id=None,
                       request_id=None):
        if actor != "management" or not isinstance(scope_id, int) or isinstance(scope_id, bool):
            raise PermissionError("MEMORY_ROLLBACK_FORBIDDEN")
        await self.session.execute(text("SELECT pg_advisory_xact_lock(:scope)"), {"scope": scope_id})
        if since_event_id is not None:
            if not isinstance(since_event_id, int) or isinstance(since_event_id, bool) or since_event_id <= 0:
                raise ValueError("MEMORY_WRITE_UNAVAILABLE: invalid rebuild event point")
            point = await self.session.get(MemoryEvent, since_event_id)
            if point is None:
                raise ValueError("MEMORY_WRITE_UNAVAILABLE: invalid rebuild event point")
            if point.knowledge_scope_id != scope_id:
                raise ValueError("MEMORY_EVIDENCE_SCOPE_MISMATCH")
        history = await MemoryEventStore(self.session).replay(scope_id)
        if not history:
            raise ValueError("MEMORY_WRITE_UNAVAILABLE")
        from rag_mcp.runtime.projection_rebuild import MemoryHistory
        recovered = await MemoryHistory(self).load(scope_id, since_event_id=since_event_id)
        state = recovered.state
        current = await self.projections.current(scope_id)
        completed_event_id = current.source_event_id if current else 0
        pending_event_ids = set((await self.session.execute(select(MemoryProjectionMeta.source_event_id).where(
            MemoryProjectionMeta.knowledge_scope_id == scope_id, MemoryProjectionMeta.projection_type == "pending",
            MemoryProjectionMeta.status == "failed", MemoryProjectionMeta.source_event_id > completed_event_id))).scalars().all())
        self._ensure_vector_store()
        try:
            async with self.session.begin_nested():
                for event in history:
                    payload = event["payload"]
                    if event["event_id"] not in pending_event_ids or event["event_type"] != "grant" or "policy_after" not in payload:
                        continue
                    profile = await self.session.scalar(select(DomainProfile).where(
                        DomainProfile.domain_key == payload["domain_key"]).with_for_update())
                    if profile is None or profile.is_builtin:
                        raise ValueError("MEMORY_WRITE_UNAVAILABLE: pending policy domain unavailable")
                    if profile.memory_policy == payload["policy_after"]:
                        continue
                    if (profile.memory_policy or {}) != payload["policy_before"]:
                        raise ValueError("MEMORY_WRITE_UNAVAILABLE: pending policy conflicts with current profile")
                    profile.memory_policy = payload["policy_after"]
                    await self.session.flush()
                await self.projections.materialize(state, scope_id, history[-1]["event_id"])
                report = await self.projections.inspect(state, scope_id)
                if not all(row["matches_replay"] for row in report.values()):
                    raise ProjectionFailure("integrity")
                versions = await self.projections.versions(scope_id)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        for name, row in report.items():
            row.update(recovery_source=recovered.source, source_event_id=history[-1]["event_id"], knowledge_scope_id=scope_id,
                       scope_id=scope_id, since_event_id=since_event_id,
                       range={"from_event_id": history[0]["event_id"], "through_event_id": history[-1]["event_id"],
                              "since_event_id": since_event_id, "event_count": len(history)},
                       cross_scope_check={"passed": True, "foreign_scope_count": 0, "scope_ids": [scope_id]},
                       schema_version=versions["schema_version"], projection_version=versions["projection_versions"][name])
            if name == "dense":
                row["index_version"] = versions["index_version"]
        audit_request_id = request_id or str(uuid4())
        self.session.add(MemoryManagementAudit(
            request_id=audit_request_id, operation="rebuild", actor=actor,
            knowledge_scope_id=scope_id, reason=reason,
            source_event_id=history[-1]["event_id"], since_event_id=since_event_id,
            result={"scope_id": scope_id, "projections": report},
        ))
        await self.session.commit()
        for row in report.values():
            row["request_id"] = audit_request_id
        return report
