"""RetrievalService: orchestrates semantic search across the RAG knowledge base.

Resolves project references, embeds queries, searches Qdrant with scope and
version filters, enforces per-source guards, records audit trails, and returns
structured responses conforming to mcp-search-output.schema.json.

Blueprint §12 (retrieval guardrails), §14 (four-state completion status).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_mcp.config import get_settings
from rag_mcp.fusion.rrf import rrf_fuse
from rag_mcp.indexing.qdrant_client import QdrantStore
from rag_mcp.indexing.sparse_encoder import BM25SparseEncoder
from rag_mcp.models.chunk import Chunk
from rag_mcp.models.knowledge_scope import KnowledgeScope
from rag_mcp.models.knowledge_version import KnowledgeVersion
from rag_mcp.models.project import Project
from rag_mcp.models.retrieval_run import RetrievalRun
from rag_mcp.providers.active_model import active_embedding_model, derive_index_version
from rag_mcp.providers.base import EmbeddingProvider, RerankerProvider
from rag_mcp.utils.snowflake import generate_id

logger = logging.getLogger(__name__)

# Module-level cache for BM25SparseEncoder instances (T036, FR-015/SC-005).
# Key: sorted tuple of scope_ids. Value: fitted BM25SparseEncoder.
# Hash-based term IDs ensure cached encoder produces compatible IDs with
# stored sparse vectors. Cache is process-scoped; restart rebuilds it.
_sparse_encoder_cache: dict[tuple[int, ...], "BM25SparseEncoder"] = {}


def _non_empty_entries(refs: list[str] | None) -> list[str]:
    """Return the non-empty / non-whitespace entries of a scope reference list."""
    if not refs:
        return []
    out: list[str] = []
    for r in refs:
        if isinstance(r, str) and r.strip():
            out.append(r)
    return out



def _non_empty_entries(refs: list[str] | None) -> list[str]:
    """Return the non-empty / non-whitespace entries of a scope reference list."""
    if not refs:
        return []
    out: list[str] = []
    for r in refs:
        if isinstance(r, str) and r.strip():
            out.append(r)
    return out


def invalidate_sparse_encoder_cache() -> None:
    """Clear the sparse encoder cache (call after ingestion publishes new data)."""
    global _sparse_encoder_cache
    _sparse_encoder_cache.clear()
    logger.info("Sparse encoder cache invalidated")


async def persist_retrieval_run(
    session: AsyncSession,
    *,
    query: str | None,
    project_scopes: list,
    completion_status: str,
    evidence_count: int,
    duration_ms: int,
    retrieval_mode: str = "dense",
    subpath_timings: dict | None = None,
    evidence_ref_ids: list | None = None,
    format: str | None = None,
    run_id: int | None = None,
    tool: str = "search_knowledge",
    error_summary: dict | None = None,
    provider_usage: dict | None = None,
    trace_enabled: bool = True,
    ttl_days: int = 7,
    instance_id=None,
    instance_mode=None,
) -> int | None:
    """Write one append-only RetrievalRun row (FR-016/FR-018/FR-020).

    Shared by the deterministic path, the agentic path and get_evidence so
    every retrieval activity lands in retrieval_runs for the by-Tool /
    by-instance-mode runtime metrics. Audit writes must never break the
    caller: any failure is logged and None returned.
    """
    try:
        now = datetime.now(timezone.utc)
        run = RetrievalRun(
            run_id=run_id if run_id is not None else generate_id(),
            query_text=query if trace_enabled else None,
            project_scopes=project_scopes,
            completion_status=completion_status,
            evidence_count=evidence_count,
            duration_ms=duration_ms,
            retrieval_mode=retrieval_mode,
            subpath_timings=subpath_timings,
            evidence_ref_ids=evidence_ref_ids or [],
            format=format,
            tool=tool,
            instance_id=instance_id,
            instance_mode=instance_mode,
            trace_body_recorded=trace_enabled,
            error_summary=error_summary,
            provider_usage=provider_usage,
            created_at=now,
            expires_at=now + timedelta(days=int(ttl_days)),
        )
        session.add(run)
        await session.flush()
        return run.run_id
    except Exception:
        logger.error("Failed to record RetrievalRun", exc_info=True)
        return None


class RetrievalService:
    """Orchestrates semantic retrieval with project-scope isolation.

    Responsibilities:
    - Resolve human-friendly project references to internal scope IDs
    - Embed queries and search Qdrant with scope + version filters
    - Enforce max_evidence_per_source guard
    - Determine four-state completion status
    - Record RetrievalRun audit trail
    - Return structured response matching mcp-search-output.schema.json
    """

    def __init__(
        self,
        session: AsyncSession,
        qdrant_store: QdrantStore,
        embedding_provider: EmbeddingProvider,
        reranker: RerankerProvider | None = None,
    ) -> None:
        self._session = session
        self._qdrant_store = qdrant_store
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        project_scopes: list[str],
        top_k: int = 5,
        task_context: dict | None = None,
        domain_scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute a scoped semantic search and return structured results.

        Args:
            query: Natural language query string.
            project_scopes: List of project references (ID, alias, or repo_path).
            top_k: Maximum number of evidence items to return (1-20).
            task_context: Optional task context for future relevance boosting.

        Returns:
            Dict conforming to mcp-search-output.schema.json.

        The deterministic whole-call guardrail (RetrievalConfig.total_timeout_ms,
        SC-005 / blueprint section 19) is enforced here: a search exceeding the
        total budget degrades to a failed SEARCH_TIMEOUT response rather than
        hanging (T040).
        """
        retrieval_cfg = self._settings.retrieval
        try:
            return await asyncio.wait_for(
                self._search_impl(query, project_scopes, top_k, task_context, domain_scopes),
                timeout=retrieval_cfg.total_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Search exceeded the %dms total-timeout guardrail",
                retrieval_cfg.total_timeout_ms,
            )
            return {
                "completion_status": "failed",
                "evidence": [],
                "error": {
                    "code": "SEARCH_TIMEOUT",
                    "message": (
                        f"Search exceeded the {retrieval_cfg.total_timeout_ms}ms "
                        "total-timeout guardrail."
                    ),
                },
                "request_id": str(uuid.uuid4()),
            }

    async def _search_impl(
        self,
        query: str,
        project_scopes: list[str],
        top_k: int = 5,
        task_context: dict | None = None,
        domain_scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Body of search(); see the public method for the contract."""
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        from rag_mcp.services.provider_usage import ProviderUsageAccumulator

        usage = ProviderUsageAccumulator()

        # Clamp top_k to configured bounds
        retrieval_cfg = self._settings.retrieval
        top_k = max(1, min(top_k, retrieval_cfg.top_k_max))

        try:
            # 1. Resolve project references to knowledge_scope_ids
            if domain_scopes and _non_empty_entries(domain_scopes):
                resolved_ids, error_info = await self.resolve_knowledge_scopes(project_scopes, domain_scopes)
            else:
                resolved_ids, error_info = await self.resolve_project_refs(project_scopes)

            if error_info is not None:
                # Resolution failed — return error response
                duration_ms = int((time.monotonic() - start_time) * 1000)
                await self._record_retrieval_run(
                    query=query,
                    project_scopes=project_scopes,
                    completion_status="failed",
                    evidence_count=0,
                    duration_ms=duration_ms,
                    retrieval_mode="dense",
                    subpath_timings=None,
                    evidence_ref_ids=[],
                    provider_usage=usage.to_dict(),
                    error_summary={
                        "code": error_info.get("code", "RESOLUTION_FAILED"),
                        "message": error_info.get("message", ""),
                        "failed_paths": [],
                    },
                )
                return {
                    "completion_status": "failed",
                    "evidence": [],
                    "error": error_info,
                    "request_id": request_id,
                }

            if not resolved_ids:
                # Should not happen after error check, but defensive
                duration_ms = int((time.monotonic() - start_time) * 1000)
                await self._record_retrieval_run(
                    query=query,
                    project_scopes=project_scopes,
                    completion_status="failed",
                    evidence_count=0,
                    duration_ms=duration_ms,
                    retrieval_mode="dense",
                    subpath_timings=None,
                    evidence_ref_ids=[],
                    provider_usage=usage.to_dict(),
                    error_summary={
                        "code": "MISSING_PROJECT_SCOPE",
                        "message": "No valid project scopes could be resolved.",
                        "failed_paths": [],
                    },
                )
                return {
                    "completion_status": "failed",
                    "evidence": [],
                    "error": {
                        "code": "MISSING_PROJECT_SCOPE",
                        "message": "No valid project scopes could be resolved.",
                    },
                    "request_id": request_id,
                }

            # 2. Embed the query
            query_vector = await self._embedding_provider.embed_query(query)
            usage.record_embedding(1)

            # 3. Hybrid or Dense recall based on version capabilities (FR-013)
            #    Only lexical_ready versions participate in the Sparse path.
            subpath_timings: dict | None = None
            retrieval_mode = "dense"

            hybrid_result = await self._try_hybrid_recall(
                query, query_vector, resolved_ids, top_k * 2,
            )
            failed_paths: list[str] = []
            graph_trace_data: dict | None = None
            if hybrid_result is not None:
                (raw_results, subpath_timings, failed_paths,
                 graph_used, graph_trace_data, _rich_candidates) = hybrid_result
                retrieval_mode = "graph_enhanced" if graph_used else "hybrid"
            else:
                # Dense-only path (001 behavior, backward compat)
                index_version = self._derive_index_version()
                collection_name = f"chunks_dense_{index_version}"
                raw_results = self._qdrant_store.search(
                    collection=collection_name,
                    vector=query_vector,
                    scope_ids=resolved_ids,
                    limit=top_k * 2,  # Over-fetch for per-source guard
                )

            if subpath_timings and "rerank_ms" in subpath_timings:
                usage.record_rerank(1)

            # 5. Filter to published versions only
            filtered_results = await self._filter_published_versions(raw_results)

            # 6. Enforce max_evidence_per_source guard
            guarded_results = self._apply_per_source_guard(
                filtered_results,
                max_per_source=retrieval_cfg.max_evidence_per_source,
            )

            # 7. Trim to requested top_k
            final_results = guarded_results[:top_k]

            # 8. Build evidence items
            evidence_items = await self._build_evidence_items(final_results)

            # 8b. Annotate graph-recalled evidence with hard/soft relation
            # metadata (004, FR-004/SC-009). Additive per the 004 annotation
            # contract extension; only present when the graph path ran.
            if graph_trace_data is not None and graph_trace_data["graph_candidates"]:
                await self._annotate_graph_evidence(
                    evidence_items, graph_trace_data["graph_candidates"]
                )

            # 9. Determine completion status
            completion_status = self._determine_completion_status(
                evidence_items=evidence_items,
                requested_top_k=top_k,
                total_available=len(filtered_results),
            )

            # 9b. If hybrid sub-paths failed, status is partial (FR-016)
            if failed_paths:
                completion_status = "partial"

            # 10. Build gaps for partial status
            gaps: list[dict[str, str]] = []
            if completion_status == "partial":
                gaps = self._infer_gaps(evidence_items, top_k)
                for fp in failed_paths:
                    gaps.append({
                        "description": f"Retrieval sub-path failed: {fp}",
                        "suggested_action": "Check service health and retry.",
                    })

            # 11. Record RetrievalRun audit trail
            duration_ms = int((time.monotonic() - start_time) * 1000)
            evidence_ref_ids = [e["evidence_id"] for e in evidence_items]
            run_id = generate_id()
            await self._record_retrieval_run(
                query=query,
                project_scopes=project_scopes,
                completion_status=completion_status,
                evidence_count=len(evidence_items),
                duration_ms=duration_ms,
                retrieval_mode=retrieval_mode,
                subpath_timings=subpath_timings,
                evidence_ref_ids=evidence_ref_ids,
                run_id=run_id,
                provider_usage=usage.to_dict(),
                error_summary=(
                    {
                        "code": "PARTIAL_PATHS_FAILED",
                        "message": "One or more retrieval sub-paths failed",
                        "failed_paths": failed_paths,
                    }
                    if failed_paths
                    else None
                ),
            )

            # 11b. Runtime graph-expansion trace ledger (004, T041/T045,
            # FR-026, DM-1): recorded only when the graph path was attempted.
            # Internal ledger — never part of the external MCP response
            # (FR-011, Constitution VII).
            if graph_trace_data is not None:
                await self._record_graph_trace(
                    request_id=request_id,
                    run_id=run_id,
                    scope_ids=resolved_ids,
                    subpath_timings=subpath_timings,
                    failed_paths=failed_paths,
                    completion_status=completion_status,
                    evidence_items=evidence_items,
                    graph_trace_data=graph_trace_data,
                )

            # 12. Assemble response
            response: dict[str, Any] = {
                "completion_status": completion_status,
                "evidence": evidence_items,
                "request_id": request_id,
            }
            if completion_status == "partial":
                response["gaps"] = gaps

            logger.info(
                "Search completed: status=%s, evidence=%d, duration=%dms, request_id=%s",
                completion_status, len(evidence_items), duration_ms, request_id,
            )
            return response

        except Exception as exc:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "Search failed: %s, request_id=%s", exc, request_id, exc_info=True,
            )
            # Best-effort audit record
            try:
                await self._record_retrieval_run(
                    query=query,
                    project_scopes=project_scopes,
                    completion_status="failed",
                    evidence_count=0,
                    duration_ms=duration_ms,
                    retrieval_mode="dense",
                    subpath_timings=None,
                    evidence_ref_ids=[],
                    provider_usage=usage.to_dict(),
                    error_summary={
                        "code": "SYSTEM_ERROR",
                        "message": f"Internal retrieval error: {type(exc).__name__}",
                        "failed_paths": [],
                    },
                )
            except Exception:
                logger.error("Failed to record retrieval run for failed search", exc_info=True)

            return {
                "completion_status": "failed",
                "evidence": [],
                "error": {
                    "code": "SYSTEM_ERROR",
                    "message": f"Internal retrieval error: {type(exc).__name__}",
                },
                "request_id": request_id,
            }

    async def resolve_project_refs(
        self, refs: list[str],
    ) -> tuple[list[int], dict | None]:
        """Resolve project references to knowledge_scope_ids.

        Each ref can be:
        - A numeric project_id (string representation of Snowflake ID)
        - An alias string
        - A repo_path string
        - A numeric public knowledge scope ID (001 Phase 9, FR-016:
          public knowledge MUST use a distinct public scope and MUST NOT
          masquerade as a project; public scope IDs resolve directly to
          themselves without a Project row)

        Args:
            refs: List of project reference strings.

        Returns:
            Tuple of (resolved_scope_ids, error_info_or_none).
            If any ref is ambiguous, returns error with candidates.
            If no refs resolve, returns error with MISSING_PROJECT_SCOPE.
        """
        resolved_scope_ids: list[int] = []
        all_candidates: list[dict[str, Any]] = []

        for ref in refs:
            ref_stripped = ref.strip()
            if not ref_stripped:
                continue

            # Public knowledge scope: a numeric ref matching an active
            # public scope resolves directly (Constitution I; public scope
            # is never managed through projects). Project refs fall
            # through unchanged.
            try:
                numeric_ref = int(ref_stripped)
            except ValueError:
                numeric_ref = None
            if numeric_ref is not None:
                pub_result = await self._session.execute(
                    select(KnowledgeScope.scope_id).where(
                        KnowledgeScope.scope_id == numeric_ref,
                        KnowledgeScope.scope_type == "public",
                        KnowledgeScope.status == "active",
                    )
                )
                pub_scope_id = pub_result.scalar_one_or_none()
                if pub_scope_id is not None:
                    if pub_scope_id not in resolved_scope_ids:
                        resolved_scope_ids.append(pub_scope_id)
                    continue

            projects = await self._find_projects_by_ref(ref_stripped)

            if len(projects) == 0:
                # Ref didn't match anything — skip silently per spec
                # (only fail if NO refs resolve at all)
                logger.warning("Project ref '%s' did not match any project", ref_stripped)
                continue

            if len(projects) > 1:
                # Ambiguous reference — collect candidates for error
                candidates = [
                    {
                        "project_id": str(p.project_id),
                        "name": p.name,
                        "alias": p.alias,
                        "repo_path": p.repo_path,
                    }
                    for p in projects
                ]
                all_candidates.extend(candidates)

                return [], {
                    "code": "AMBIGUOUS_PROJECT_REF",
                    "message": (
                        f"Project reference '{ref_stripped}' matches multiple projects. "
                        f"Please specify using a unique project_id, alias, or repo_path."
                    ),
                    "candidates": candidates,
                }

            # Exactly one match
            project = projects[0]
            scope_id = project.knowledge_scope_id
            if scope_id not in resolved_scope_ids:
                resolved_scope_ids.append(scope_id)

        if not resolved_scope_ids:
            return [], {
                "code": "MISSING_PROJECT_SCOPE",
                "message": (
                    "None of the provided project references could be resolved. "
                    "Please provide valid project IDs, aliases, or repository paths."
                ),
            }

        return resolved_scope_ids, None


    async def resolve_knowledge_scopes(
        self,
        project_scope: list[str] | None = None,
        domain_scope: list[str] | None = None,
    ) -> tuple[list[int], dict | None]:
        """Resolve the union of project_scope and domain_scope references (007, T017)."""
        project_scope = project_scope or []
        domain_scope = domain_scope or []
        domain_involved = bool(_non_empty_entries(domain_scope))

        resolved_ids: list[int] = []
        all_candidates: list[dict[str, Any]] = []
        ambiguous = False

        # project_scope: legacy semantics (identical to resolve_project_refs).
        for ref in project_scope:
            ref_stripped = ref.strip() if isinstance(ref, str) else ""
            if not ref_stripped:
                continue
            try:
                numeric_ref = int(ref_stripped)
            except ValueError:
                numeric_ref = None
            if numeric_ref is not None:
                pub_result = await self._session.execute(
                    select(KnowledgeScope.scope_id).where(
                        KnowledgeScope.scope_id == numeric_ref,
                        KnowledgeScope.scope_type == "public",
                        KnowledgeScope.status == "active",
                    )
                )
                pub_scope_id = pub_result.scalar_one_or_none()
                if pub_scope_id is not None:
                    if pub_scope_id not in resolved_ids:
                        resolved_ids.append(pub_scope_id)
                    continue
            projects = await self._find_projects_by_ref(ref_stripped)
            if len(projects) == 0:
                continue
            if len(projects) > 1:
                ambiguous = True
                all_candidates.extend([
                    {
                        "project_id": str(p.project_id),
                        "name": p.name,
                        "alias": p.alias,
                        "repo_path": p.repo_path,
                    }
                    for p in projects
                ])
                continue
            scope_id = projects[0].knowledge_scope_id
            if scope_id not in resolved_ids:
                resolved_ids.append(scope_id)

        # domain_scope: new resolution chain.
        for ref in domain_scope:
            ref_stripped = ref.strip() if isinstance(ref, str) else ""
            if not ref_stripped:
                continue
            entry = await self._resolve_domain_scope_entry(ref_stripped)
            if entry is None:
                continue
            if isinstance(entry, list):
                ambiguous = True
                all_candidates.extend(entry)
                continue
            if entry not in resolved_ids:
                resolved_ids.append(entry)

        if ambiguous:
            if domain_involved:
                return [], {
                    "code": "AMBIGUOUS_DOMAIN_REF",
                    "message": (
                        "Domain reference is ambiguous. Please specify using a "
                        "unique scope slug or numeric scope ID."
                    ),
                    "candidates": all_candidates,
                }
            return [], {
                "code": "AMBIGUOUS_PROJECT_REF",
                "message": (
                    "Project reference matches multiple projects. "
                    "Please specify using a unique project_id, alias, or repo_path."
                ),
                "candidates": all_candidates,
            }

        if not resolved_ids:
            if domain_involved:
                return [], {
                    "code": "MISSING_KNOWLEDGE_SCOPE",
                    "message": (
                        "None of the provided knowledge domain references could be "
                        "resolved. Please provide valid scope IDs, slugs, or type:name references."
                    ),
                }
            return [], {
                "code": "MISSING_PROJECT_SCOPE",
                "message": (
                    "None of the provided project references could be resolved. "
                    "Please provide valid project IDs, aliases, or repository paths."
                ),
            }

        return resolved_ids, None

    async def _resolve_domain_scope_entry(self, ref: str) -> int | list[dict] | None:
        """Resolve one domain_scope entry: numeric → slug → type:name (FR-008)."""
        try:
            numeric = int(ref)
            result = await self._session.execute(
                select(KnowledgeScope).where(
                    KnowledgeScope.scope_id == numeric,
                    KnowledgeScope.status == "active",
                )
            )
            scope = result.scalar_one_or_none()
            return scope.scope_id if scope is not None else None
        except ValueError:
            pass

        result = await self._session.execute(
            select(KnowledgeScope).where(
                KnowledgeScope.slug == ref,
                KnowledgeScope.status == "active",
            )
        )
        scope = result.scalar_one_or_none()
        if scope is not None:
            return scope.scope_id

        if ":" in ref:
            scope_type, _, name = ref.partition(":")
            if scope_type in ("project", "public") and name:
                result = await self._session.execute(
                    select(KnowledgeScope).where(
                        KnowledgeScope.scope_type == scope_type,
                        KnowledgeScope.name == name,
                        KnowledgeScope.status == "active",
                    )
                )
                matches = list(result.scalars().all())
                if len(matches) == 1:
                    return matches[0].scope_id
                if len(matches) > 1:
                    return [self._domain_candidate(s) for s in matches]
        return None

    @staticmethod
    def _domain_candidate(scope: KnowledgeScope) -> dict[str, Any]:
        """Build a domain-scope candidate (scope_type/domain_key/slug)."""
        return {
            "scope_id": str(scope.scope_id),
            "name": scope.name,
            "scope_type": scope.scope_type,
            "domain_key": scope.domain_key,
            "slug": scope.slug,
        }

    async def recall_candidates(

        self,
        query: str,
        scope_ids: list[int],
        limit: int = 20,
        use_graph: bool = False,
        graph_relation_types: list[str] | None = None,
        graph_hop: int | None = None,
    ) -> dict[str, Any]:
        """Hybrid recall for the agentic pipeline (005, T058).

        Runs the SAME retrieval components as search() — QdrantStore
        dense+sparse recall, RRF fusion, Rerank, 004 graph expansion with its
        guardrails — but returns raw recall candidates with per-retriever
        metadata instead of an MCP response. The deterministic search() path
        is untouched.

        Args:
            query: The sub-problem query text.
            scope_ids: Resolved knowledge scope IDs (already isolated).
            limit: Over-fetch limit for the recall sub-paths.
            use_graph: Run the 004 graph sub-path for this query (planner
                signal), independent of the global graph switch (FR-033).
            graph_relation_types: Planner-selected relation directions.
            graph_hop: Planner-selected hop cap, clamped to 004 hop_max.

        Returns:
            Dict with 'candidates' (per-retriever metadata + payload),
            'subpath_timings', 'failed_paths', 'graph_used'.
        """
        from rag_mcp.orchestration.retrieval_pipeline import (
            clamp01,
            final_candidate_score,
        )

        query_vector = await self._embedding_provider.embed_query(query)

        hybrid_result = await self._try_hybrid_recall(
            query,
            query_vector,
            scope_ids,
            limit,
            graph_mode="on" if use_graph else "off",
            graph_relation_types=graph_relation_types,
            graph_hop=graph_hop,
        )

        if hybrid_result is not None:
            (_raw, subpath_timings, failed_paths,
             graph_used, _graph_trace, rich_candidates) = hybrid_result
        else:
            # Dense-only fallback (001 behaviour): no lexical_ready versions,
            # hybrid collection missing, or query without sparse vocab terms.
            subpath_timings = {}
            failed_paths = []
            graph_used = False
            index_version = self._derive_index_version()
            collection_name = f"chunks_dense_{index_version}"
            hybrid_name = f"chunks_hybrid_{index_version}"
            t_dense = time.monotonic()
            dense_results: list[dict[str, Any]] = []
            if self._qdrant_store.collection_exists(collection_name):
                dense_results = self._qdrant_store.search(
                    collection=collection_name,
                    vector=query_vector,
                    scope_ids=scope_ids,
                    limit=limit,
                )
            if not dense_results and self._qdrant_store.collection_exists(hybrid_name):
                # Dense-only recall against the hybrid collection (named
                # vectors): keeps the agentic path available when the data
                # lives in the 002 hybrid collection (T065).
                dense_results = self._qdrant_store.search_dense_named(
                    collection=hybrid_name,
                    vector=query_vector,
                    scope_ids=scope_ids,
                    limit=limit,
                )
            subpath_timings["dense_recall_ms"] = round(
                (time.monotonic() - t_dense) * 1000, 2
            )
            rich_candidates = [
                {
                    "chunk_id": str((r.get("payload") or {}).get("chunk_id", r.get("id", ""))),
                    "retrievers": ["dense"],
                    "dense_score": float(r.get("score", 0.0)),
                    "sparse_score": None,
                    "dense_rank": i + 1,
                    "sparse_rank": None,
                    "graph_rank": None,
                    "graph_structure_weight": None,
                    "fused_score": 0.0,
                    "final_rank": i + 1,
                    "rerank_score": None,
                    "payload": r.get("payload", {}) or {},
                }
                for i, r in enumerate(dense_results)
            ]

        # Published-version isolation before candidates enter the ledger input
        version_ids: set[int] = set()
        for c in rich_candidates:
            vid = (c.get("payload") or {}).get("version_id")
            if vid is not None:
                try:
                    version_ids.add(int(vid))
                except (ValueError, TypeError):
                    pass
        published: set[int] = set()
        if version_ids:
            result = await self._session.execute(
                select(KnowledgeVersion.version_id).where(
                    KnowledgeVersion.version_id.in_(version_ids),
                    KnowledgeVersion.status == "published",
                )
            )
            published = set(result.scalars().all())

        candidates: list[dict[str, Any]] = []
        for c in rich_candidates:
            payload = c.get("payload") or {}
            try:
                vid = int(payload.get("version_id", 0))
            except (ValueError, TypeError):
                vid = 0
            if vid not in published:
                continue
            score = final_candidate_score(
                dense_score=c.get("dense_score"),
                sparse_score=c.get("sparse_score"),
                rerank_score=c.get("rerank_score"),
                fused_score=c.get("fused_score") or 0.0,
                structure_weight=c.get("graph_structure_weight"),
            )
            candidates.append({
                "chunk_id": c["chunk_id"],
                "retrievers": c["retrievers"],
                "dense_score": c.get("dense_score"),
                "sparse_score": c.get("sparse_score"),
                "dense_rank": c.get("dense_rank"),
                "sparse_rank": c.get("sparse_rank"),
                "graph_rank": c.get("graph_rank"),
                "graph_structure_weight": c.get("graph_structure_weight"),
                "fused_score": c.get("fused_score") or 0.0,
                "final_rank": c.get("final_rank"),
                "rerank_score": c.get("rerank_score"),
                "score": clamp01(score),
                "payload": payload,
            })

        return {
            "candidates": candidates,
            "subpath_timings": subpath_timings,
            "failed_paths": failed_paths,
            "graph_used": graph_used,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _find_projects_by_ref(self, ref: str) -> list[Project]:
        """Find projects matching a reference string.

        Tries matching by:
        1. Numeric project_id
        2. Exact alias match
        3. Exact repo_path match
        4. Partial repo_path match (endswith)
        5. Numeric knowledge_scope_id (005 T061: eval datasets and callers
           may address a project through its knowledge scope stable ID;
           additive — refs that previously failed to resolve now resolve)

        Returns all matching projects (may be 0, 1, or more).
        """
        conditions = []

        # Try numeric ID match (project_id or knowledge_scope_id)
        try:
            numeric_id = int(ref)
            conditions.append(Project.project_id == numeric_id)
            conditions.append(Project.knowledge_scope_id == numeric_id)
        except ValueError:
            pass

        # Alias match (case-insensitive)
        conditions.append(Project.alias.ilike(ref))

        # Exact repo_path match
        conditions.append(Project.repo_path == ref)

        # Partial repo_path match (for paths like 'org/repo')
        conditions.append(Project.repo_path.ilike(f"%{ref}%"))

        result = await self._session.execute(
            select(Project).where(or_(*conditions))
        )
        return list(result.scalars().all())

    async def _filter_published_versions(
        self, raw_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter Qdrant results to only include chunks from published versions.

        Queries the database to verify each result's version_id has status='published'.
        """
        if not raw_results:
            return []

        # Collect unique version_ids from payloads
        version_ids: set[int] = set()
        for r in raw_results:
            vid = r.get("payload", {}).get("version_id")
            if vid is not None:
                try:
                    version_ids.add(int(vid))
                except (ValueError, TypeError):
                    pass

        if not version_ids:
            return []

        # Query published version IDs
        result = await self._session.execute(
            select(KnowledgeVersion.version_id).where(
                KnowledgeVersion.version_id.in_(version_ids),
                KnowledgeVersion.status == "published",
            )
        )
        published_version_ids: set[int] = set(result.scalars().all())

        # Filter results
        return [
            r for r in raw_results
            if int(r.get("payload", {}).get("version_id", 0)) in published_version_ids
        ]

    def _apply_per_source_guard(
        self,
        results: list[dict[str, Any]],
        max_per_source: int,
    ) -> list[dict[str, Any]]:
        """Enforce max_evidence_per_source guard.

        Limits the number of evidence items from any single source_id to
        prevent dominance by one document in results.
        """
        source_counts: dict[str, int] = defaultdict(int)
        guarded: list[dict[str, Any]] = []

        for r in results:
            source_id = str(r.get("payload", {}).get("source_id", ""))
            if source_counts[source_id] < max_per_source:
                guarded.append(r)
                source_counts[source_id] += 1

        return guarded

    async def _build_evidence_items(
        self, results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build EvidenceItem dicts from Qdrant search results.

        Fetches chunk metadata from PostgreSQL to enrich payload data
        with source_version and knowledge_scope_type.
        """
        if not results:
            return []

        # Collect chunk_ids for batch lookup
        chunk_ids: list[int] = []
        for r in results:
            cid = r.get("payload", {}).get("chunk_id")
            if cid is not None:
                try:
                    chunk_ids.append(int(cid))
                except (ValueError, TypeError):
                    pass

        # Batch fetch chunks for enrichment
        chunk_map: dict[int, Chunk] = {}
        if chunk_ids:
            result = await self._session.execute(
                select(Chunk).where(Chunk.chunk_id.in_(chunk_ids))
            )
            for chunk in result.scalars().all():
                chunk_map[chunk.chunk_id] = chunk

        # Also fetch version numbers for each version_id
        version_ids: set[int] = set()
        for r in results:
            vid = r.get("payload", {}).get("version_id")
            if vid is not None:
                try:
                    version_ids.add(int(vid))
                except (ValueError, TypeError):
                    pass

        # Fetch scope types so evidence carries its true knowledge-domain
        # identity (001 Phase 9, FR-016 / Constitution I & II: public
        # evidence MUST retain knowledge_scope_type='public'; previously
        # hardcoded to 'project').
        scope_ids: set[int] = set()
        for r in results:
            sid = r.get("payload", {}).get("knowledge_scope_id")
            if sid is not None:
                try:
                    scope_ids.add(int(sid))
                except (ValueError, TypeError):
                    pass
        scope_type_map: dict[int, str] = {}
        if scope_ids:
            sresult = await self._session.execute(
                select(
                    KnowledgeScope.scope_id, KnowledgeScope.scope_type
                ).where(KnowledgeScope.scope_id.in_(scope_ids))
            )
            for sid, stype in sresult.all():
                scope_type_map[sid] = stype

        version_number_map: dict[int, int] = {}
        if version_ids:
            vresult = await self._session.execute(
                select(KnowledgeVersion.version_id, KnowledgeVersion.version_number).where(
                    KnowledgeVersion.version_id.in_(version_ids),
                )
            )
            for vid, vnum in vresult.all():
                version_number_map[vid] = vnum

        # Build evidence items preserving result order
        evidence_items: list[dict[str, Any]] = []
        for r in results:
            payload = r.get("payload", {})
            chunk_id_str = str(payload.get("chunk_id", ""))
            chunk_id_int = int(chunk_id_str) if chunk_id_str else 0
            chunk = chunk_map.get(chunk_id_int)

            version_id = int(payload.get("version_id", 0))
            source_version = version_number_map.get(version_id, 0)

            # Determine scope type from the knowledge_scopes row; an
            # unavailable row falls back to the empty string (domain-neutral
            # fallback, 010 T058) — never a hardcoded 'project' default
            # (FR-019 / Constitution XI).
            try:
                scope_sid = int(payload.get("knowledge_scope_id", 0))
            except (ValueError, TypeError):
                scope_sid = 0
            scope_type = scope_type_map.get(scope_sid, "")

            content_excerpt = ""
            if chunk:
                content_excerpt = chunk.content_text[:500]
            else:
                # Fallback: use position_path from payload
                content_excerpt = payload.get("position_path", "")[:500]

            evidence_item = {
                "evidence_id": chunk_id_str,
                "content_excerpt": content_excerpt,
                "source_version": source_version,
                "source_position": (
                    chunk.position_path if chunk
                    else payload.get("position_path", "")
                ),
                "knowledge_scope_id": str(payload.get("knowledge_scope_id", "")),
                "knowledge_scope_type": scope_type,
                "relevance_score": round(float(r.get("score", 0.0)), 4),
            }
            evidence_items.append(evidence_item)

        return evidence_items

    @staticmethod
    def _determine_completion_status(
        evidence_items: list[dict[str, Any]],
        requested_top_k: int,
        total_available: int,
    ) -> str:
        """Determine the four-state completion status.

        - complete: evidence found and coverage appears sufficient
        - partial: some evidence found but clear gaps exist
        - no_evidence: system executed normally but found nothing
        - failed: handled separately via exception path
        """
        if not evidence_items:
            return "no_evidence"

        # If we got fewer results than requested AND fewer than available,
        # it suggests partial coverage
        if len(evidence_items) < requested_top_k and total_available <= requested_top_k:
            # We returned everything available but it's less than requested
            # This is still "complete" if we have evidence — the knowledge base
            # simply has limited content for this scope
            if len(evidence_items) >= 1:
                return "complete"

        # If we have evidence items, consider it complete for 001
        # Future versions may use reranking scores to determine partial
        return "complete"

    @staticmethod
    def _infer_gaps(
        evidence_items: list[dict[str, Any]],
        requested_top_k: int,
    ) -> list[dict[str, str]]:
        """Infer evidence gaps for partial completion status."""
        gaps: list[dict[str, str]] = []

        if len(evidence_items) < requested_top_k:
            gaps.append({
                "description": (
                    f"Only {len(evidence_items)} evidence item(s) found "
                    f"out of {requested_top_k} requested."
                ),
                "suggested_action": (
                    "Consider broadening the knowledge domain scope or adding more "
                    "knowledge sources to improve coverage."
                ),
            })

        return gaps

    async def _has_graph_ready_versions(self, scope_ids: list[int]) -> bool:
        """Check if any published version in the given scopes has graph_ready.

        FR-013/FR-014: only graph_ready versions participate in graph expansion.
        """
        result = await self._session.execute(
            select(KnowledgeVersion.version_id).where(
                KnowledgeVersion.knowledge_scope_id.in_(scope_ids),
                KnowledgeVersion.status == "published",
                KnowledgeVersion.graph_ready == True,  # noqa: E712
            ).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def _has_graph_edges(self, scope_ids: list[int]) -> bool:
        """Check if graph relations exist for the given scopes (FR-013).

        Hard edges (graph_edge) or ACTIVE soft relations (soft_relation,
        FR-005) both make the graph path worthwhile for a scope.
        """
        from sqlalchemy import text as sa_text
        result = await self._session.execute(
            sa_text(
                "SELECT 1 WHERE EXISTS ("
                "SELECT 1 FROM graph_edge WHERE knowledge_scope_id = ANY(:sids)"
                ") OR EXISTS ("
                "SELECT 1 FROM soft_relation "
                "WHERE lifecycle_state = 'active' AND knowledge_scope_id = ANY(:sids)"
                ") LIMIT 1"
            ),
            {"sids": list(scope_ids)},
        )
        return result.scalar_one_or_none() is not None

    async def _annotate_graph_evidence(
        self,
        evidence_items: list[dict[str, Any]],
        graph_candidates: list[dict[str, Any]],
    ) -> None:
        """Attach hard/soft relation annotations to graph-recalled evidence.

        Each graph-recalled evidence item gains an additive 'relation' field
        (004 annotation contract): hard relations are verifiable evidence with
        deterministic parse_evidence; soft relations are marked inferred with
        confidence/model/lifecycle metadata and never masquerade as hard
        (FR-004/SC-009, Constitution III). Annotation failures never break the
        response — items simply stay unannotated.
        """
        try:
            from rag_mcp.graph.models import GraphEdge, SoftRelation
            from rag_mcp.services.evidence_service import EvidenceService

            cand_by_chunk = {
                str(c.get("chunk_id", "")): c for c in graph_candidates
            }
            evidence_svc = EvidenceService(self._session)
            for item in evidence_items:
                cand = cand_by_chunk.get(str(item.get("evidence_id", "")))
                if cand is None or not cand.get("edge_path"):
                    continue
                first_hop = cand["edge_path"][0]
                try:
                    edge_id = int(first_hop.get("edge_id"))
                except (TypeError, ValueError):
                    continue
                if first_hop.get("is_hard"):
                    edge = await self._session.get(GraphEdge, edge_id)
                    if edge is not None:
                        annotated = evidence_svc.annotate_evidence(
                            item, relation_edge=edge
                        )
                        item["relation"] = annotated["relation"]
                else:
                    relation = await self._session.get(SoftRelation, edge_id)
                    if relation is not None:
                        annotated = evidence_svc.annotate_evidence(
                            item, soft_relation=relation
                        )
                        item["relation"] = annotated["relation"]
        except Exception:  # noqa: BLE001 - annotation must never break search
            logger.error("Failed to annotate graph evidence", exc_info=True)

    async def _record_graph_trace(
        self,
        request_id: str,
        run_id: int,
        scope_ids: list[int],
        subpath_timings: dict | None,
        failed_paths: list[str],
        completion_status: str,
        evidence_items: list[dict[str, Any]],
        graph_trace_data: dict[str, Any],
    ) -> None:
        """Finalise and persist the runtime graph-expansion trace (004, FR-026).

        Builds the schema-conformant trace dict via GraphTraceRecorder,
        backfills evidence_id on surviving candidates and bridges them to
        graph_expansion_path rows keyed by the RetrievalRun run_id (DM-1).
        Trace failures never break the search response.
        """
        try:
            from rag_mcp.graph.trace_recorder import GraphTraceRecorder

            gcfg = self._settings.graph
            recorder = GraphTraceRecorder(
                request_id,
                [str(s) for s in scope_ids],
                {
                    "hop_default": gcfg.hop_default,
                    "hop_max": gcfg.hop_max,
                    "candidate_budget": gcfg.candidate_budget,
                    "graph_sub_timeout_ms": gcfg.graph_sub_timeout_ms,
                    "total_timeout_ms": gcfg.total_timeout_ms,
                    "direction_default": gcfg.direction_default,
                },
            )
            if subpath_timings:
                recorder.record_timings(subpath_timings)
            recorder.record_graph_candidates(graph_trace_data["graph_candidates"])
            recorder.record_fused_candidates(graph_trace_data["fused_candidates"])
            for fp in failed_paths:
                recorder.record_failed_path(fp)
            recorder.set_completion_status(completion_status)
            evidence_ids = [e["evidence_id"] for e in evidence_items]
            recorder.set_evidence_ref_ids(evidence_ids)
            # DM-1 backfill: evidence_id equals the surviving chunk_id.
            recorder.backfill_evidence_ids({eid: eid for eid in evidence_ids})

            trace = recorder.to_trace_dict()
            self._last_graph_trace = trace
            logger.debug("Graph expansion trace: %s", trace)

            await recorder.persist_paths(self._session, run_id, None)
        except Exception:  # noqa: BLE001 - tracing must never break search
            logger.error("Failed to record graph expansion trace", exc_info=True)

    async def _graph_scope_version(self, scope_id: int):
        """Resolve a scope's graph-eligible version_number (FR-018).

        Returns the version_number of the newest published version that
        satisfies the capability gate (graph_ready=true AND the FR-015
        dense+lexical implication, checked via graph.capabilities), or None
        when the scope has no graph-eligible published version (FR-014).
        knowledge_scope_id is the sole graph isolation key — no Project row
        is required (Constitution I v1.3.0, cross-domain isolation).
        """
        from rag_mcp.graph.capabilities import is_graph_ready_version

        result = await self._session.execute(
            select(KnowledgeVersion).where(
                KnowledgeVersion.knowledge_scope_id == scope_id,
                KnowledgeVersion.status == "published",
                KnowledgeVersion.graph_ready == True,  # noqa: E712
            ).order_by(KnowledgeVersion.version_number.desc())
        )
        eligible = next(
            (v for v in result.scalars().all() if is_graph_ready_version(v)), None
        )
        if eligible is None:
            return None
        return eligible.version_number

    async def _graph_recall(
        self,
        scope_ids: list[int],
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        relation_types: list[str] | None = None,
        hop: int | None = None,
        direction: str | None = None,
    ) -> list[dict[str, Any]]:
        """Graph-expansion recall as the 3rd retriever input (004, FR-006).

        Seeds expansion from the Dense/Sparse hit chunks and expands within
        each scope's graph-eligible published version (FR-013/FR-014 gating,
        FR-010 isolation). Guardrails (hop/budget/direction) come from
        GraphConfig via GraphExpansionEngine (FR-017). Returns candidates as
        fusion-ready dicts carrying graph_rank/structure_weight/edge_path.

        005 (T058): optional planner-directed parameters — relation_types /
        hop / direction override the config defaults while the 004 guardrails
        (hop_max clamp, budget, sub-timeout) stay enforced (FR-033).
        """
        from rag_mcp.graph.expansion import GraphExpansionEngine
        from rag_mcp.graph.store.base import GraphScope

        # Capability pre-gates (FR-013/FR-014): skip unless a graph_ready
        # published version with actual edges exists in the requested scopes.
        if not await self._has_graph_ready_versions(scope_ids):
            return []
        if not await self._has_graph_edges(scope_ids):
            return []

        # Seed chunks: union of Dense/Sparse hits (deterministic order).
        seed_ids: list[int] = []
        seen: set[int] = set()
        for r in dense_results + sparse_results:
            payload = r.get("payload") or {}
            cid = payload.get("chunk_id", r.get("id"))
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            if cid_int not in seen:
                seen.add(cid_int)
                seed_ids.append(cid_int)
        if not seed_ids:
            return []

        engine = GraphExpansionEngine(self._session)
        graph_cfg = self._settings.graph
        # Planner hop choice is clamped into the 004 guardrail band (FR-033)
        if hop is not None:
            hop = max(1, min(int(hop), graph_cfg.hop_max))
        graph_results: list[dict[str, Any]] = []
        for scope_id in scope_ids:
            version_number = await self._graph_scope_version(scope_id)
            if version_number is None:
                continue
            scope = GraphScope(
                knowledge_scope_id=scope_id,
                index_version=version_number,
            )
            candidates = await engine.expand(
                seed_ids,
                scope,
                hop=hop,
                direction=direction,
                relation_types=relation_types,
            )
            for cand in candidates:
                graph_results.append({
                    "chunk_id": str(cand.chunk_id),
                    "knowledge_scope_id": str(cand.knowledge_scope_id),
                    "graph_rank": cand.graph_rank,
                    "structure_weight": cand.structure_weight,
                    "hop_count": cand.hop_count,
                    "edge_path": cand.edge_path,
                    "start_chunk_id": str(cand.start_chunk_id),
                    "relation_is_hard": cand.relation_is_hard,
                })

        logger.info("Graph recall: %d candidates across %d scope(s)",
                    len(graph_results), len(scope_ids))
        return graph_results

    async def _synthesize_graph_payloads(
        self, chunk_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Build retrieval-result payloads for graph-only candidates from PG.

        Graph-recalled chunks absent from Dense/Sparse have no Qdrant payload;
        their metadata is fetched from the chunks table so they can flow
        through the published-version filter and evidence building unchanged.
        """
        int_ids: list[int] = []
        for cid in chunk_ids:
            try:
                int_ids.append(int(cid))
            except (TypeError, ValueError):
                continue
        if not int_ids:
            return {}

        result = await self._session.execute(
            select(Chunk).where(Chunk.chunk_id.in_(int_ids))
        )
        payloads: dict[str, dict[str, Any]] = {}
        for chunk in result.scalars().all():
            payloads[str(chunk.chunk_id)] = {
                "id": chunk.chunk_id,
                "score": 0.0,
                "payload": {
                    "chunk_id": str(chunk.chunk_id),
                    "version_id": str(chunk.version_id),
                    "knowledge_scope_id": str(chunk.knowledge_scope_id),
                    "source_id": str(chunk.source_id),
                    "position_path": chunk.position_path or "",
                    "chunk_type": chunk.chunk_type,
                    "index_version": chunk.index_version,
                },
            }
        return payloads

    async def _has_lexical_ready_versions(self, scope_ids: list[int]) -> bool:
        """Check if any published version in the given scopes has lexical_ready.

        FR-013: only lexical_ready versions participate in the Sparse path.
        """
        result = await self._session.execute(
            select(KnowledgeVersion.version_id).where(
                KnowledgeVersion.knowledge_scope_id.in_(scope_ids),
                KnowledgeVersion.status == "published",
                KnowledgeVersion.capabilities["lexical_ready"].astext == "true",
            ).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def _try_hybrid_recall(
        self,
        query: str,
        query_vector: list[float],
        scope_ids: list[int],
        limit: int,
        graph_mode: str | None = None,
        graph_relation_types: list[str] | None = None,
        graph_hop: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict, list[str]] | None:
        """Attempt hybrid Dense+Sparse recall + RRF fusion + Rerank.

        Returns (raw_results, subpath_timings, failed_paths, graph_used,
        graph_trace_data, rich_candidates) if hybrid is available, or None if
        no lexical_ready versions or hybrid collection missing (caller falls
        back to dense-only path).

        005 (T058): graph_mode overrides the global graph switch for the
        agentic pipeline — None = honour GRAPH_ENHANCED_RETRIEVAL_ENABLED,
        'on' = run the graph sub-path, 'off' = skip it. rich_candidates carry
        per-retriever scores for the evidence ledger input.

        FR-013: only lexical_ready versions participate in the Sparse path.
        FR-008: both Dense and Sparse enforce scope+version filter.
        FR-016: partial degradation on sparse/rerank failure.
        FR-005: rerank only processes <= rerank_budget candidates.
        """
        # Check for lexical_ready published versions (FR-013)
        if not await self._has_lexical_ready_versions(scope_ids):
            return None

        # Determine hybrid collection name
        index_version = self._derive_index_version()
        collection_name = f"chunks_hybrid_{index_version}"

        # Check hybrid collection exists
        if not self._qdrant_store.collection_exists(collection_name):
            logger.debug(
                "Hybrid collection %s does not exist, falling back to dense",
                collection_name,
            )
            return None

        # Build sparse encoder from published chunk texts in the resolved scopes
        encoder = await self._build_sparse_encoder(scope_ids)
        if encoder is None:
            return None

        # Encode query for sparse retrieval (IDF-weighted, FR-025)
        sparse_query = encoder.encode(query)
        if not sparse_query["indices"]:
            # Query has no in-vocab terms -> sparse path adds nothing
            logger.debug("Query has no in-vocab sparse terms, falling back to dense")
            return None

        # Dense + Sparse recall (FR-002, both enforce scope filter FR-008).
        # Dense is the always-on deterministic recall; Sparse is an independent
        # sub-path that may fail/timeout and degrades to dense-only with a
        # recorded failed path (partial, FR-016/SC-009) without dropping Dense
        # evidence (T040/T041).
        hybrid_cfg = self._settings.hybrid_retrieval
        t_recall_start = time.monotonic()
        dense_results = self._qdrant_store.search_dense_named(
            collection=collection_name,
            vector=query_vector,
            scope_ids=scope_ids,
            version_id=None,
            limit=limit,
        )
        failed_paths: list[str] = []
        try:
            sparse_results = await asyncio.wait_for(
                asyncio.to_thread(
                    self._qdrant_store.search_sparse,
                    collection_name,
                    sparse_query,
                    scope_ids,
                    None,
                    limit,
                ),
                timeout=hybrid_cfg.sparse_query_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Sparse recall timed out after %dms; degrading to dense-only",
                hybrid_cfg.sparse_query_timeout_ms,
            )
            failed_paths.append("sparse_timeout")
            sparse_results = []
        except Exception as sparse_exc:  # noqa: BLE001 - degrade, keep dense
            logger.warning(
                "Sparse recall failed: %s; degrading to dense-only", sparse_exc,
            )
            failed_paths.append("sparse_failed")
            sparse_results = []
        recall_elapsed = (time.monotonic() - t_recall_start) * 1000

        # Graph recall (004, FR-006/FR-007): the 3rd retriever input, behind
        # the GRAPH_ENHANCED_RETRIEVAL_ENABLED switch (FR-024). Guarded by the
        # graph sub-timeout; failures degrade to hybrid-only with a recorded
        # failed path (FR-018 partial), never blocking the response.
        graph_cfg = self._settings.graph
        graph_results: list[dict[str, Any]] = []
        graph_elapsed = 0.0
        graph_used = False
        # 005 (T058): the agentic pipeline overrides the global graph switch
        # with the planner's per-sub-problem signal choice (FR-033); the
        # deterministic default path keeps honouring only the global switch.
        graph_enabled = (
            graph_cfg.enabled if graph_mode is None else graph_mode == "on"
        )
        # Trace payload assembled whenever the graph path is attempted (T045);
        # stays None when the switch is off so the default path is untouched.
        graph_trace_data: dict[str, Any] | None = (
            {"graph_candidates": [], "fused_candidates": []}
            if graph_enabled else None
        )
        if graph_enabled:
            t_graph_start = time.monotonic()
            try:
                graph_results = await asyncio.wait_for(
                    self._graph_recall(
                        scope_ids,
                        dense_results,
                        sparse_results,
                        relation_types=graph_relation_types,
                        hop=graph_hop,
                    ),
                    timeout=graph_cfg.graph_sub_timeout_ms / 1000.0,
                )
                graph_used = bool(graph_results)
            except asyncio.TimeoutError:
                logger.warning("Graph expansion timed out after %dms",
                               graph_cfg.graph_sub_timeout_ms)
                failed_paths.append("graph_expansion_timeout")
            except Exception as graph_exc:  # noqa: BLE001 - degrade, keep hybrid
                logger.warning("Graph expansion failed: %s", graph_exc)
                failed_paths.append("graph_expansion_failed")
            graph_elapsed = (time.monotonic() - t_graph_start) * 1000

        # RRF fusion (FR-003, deterministic tie-breaker FR-017); graph joins
        # the same fusion pool as the 3rd ranked list (FR-006).
        # rrf_k and fusion_algorithm come from HybridRetrievalConfig (T040).
        # Only 'rrf' is implemented (research.md §1.2 reserves DBSF); an
        # unsupported value is recorded and falls back deterministically.
        fusion_algorithm = hybrid_cfg.fusion_algorithm
        if fusion_algorithm != "rrf":
            logger.warning(
                "fusion_algorithm=%r is not implemented (only 'rrf'); "
                "falling back to RRF",
                fusion_algorithm,
            )
        t_fusion_start = time.monotonic()
        fused = rrf_fuse(dense_results, sparse_results, k=hybrid_cfg.rrf_k,
                         graph_results=graph_results or None)
        fusion_elapsed = (time.monotonic() - t_fusion_start) * 1000

        # Build chunk_id -> result map from both dense and sparse results
        result_map: dict[str, dict[str, Any]] = {}
        for r in dense_results + sparse_results:
            payload = r.get("payload") or {}
            chunk_id = str(payload.get("chunk_id", r.get("id", "")))
            if chunk_id and chunk_id not in result_map:
                result_map[chunk_id] = r

        if graph_trace_data is not None:
            graph_trace_data["graph_candidates"] = graph_results
            graph_trace_data["fused_candidates"] = [
                {
                    "chunk_id": str(c.chunk_id),
                    "knowledge_scope_id": str(c.knowledge_scope_id),
                    "source_retrievers": list(c.source_retrievers),
                    "dense_score": c.dense_score,
                    "sparse_score": c.sparse_score,
                    "graph_structure_weight": c.graph_structure_weight,
                    "graph_rank": c.graph_rank,
                    "fused_score": c.fused_score,
                    "rerank_score": c.rerank_score,
                    "final_rank": c.final_rank,
                }
                for c in fused
            ]

        # Enrich graph-only candidates (recalled by edges, absent from
        # Dense/Sparse) with PG metadata so they can become evidence (FR-006).
        if graph_results:
            graph_only = [
                str(g.get("chunk_id", ""))
                for g in graph_results
                if str(g.get("chunk_id", "")) not in result_map
            ]
            if graph_only:
                enriched = await self._synthesize_graph_payloads(graph_only)
                result_map.update(enriched)

        # Convert fused candidates to raw_results format (sorted by fused score)
        raw_results: list[dict[str, Any]] = []
        for cand in fused:
            r = result_map.get(cand.chunk_id)
            if r is not None:
                raw_results.append({
                    "id": r.get("id"),
                    "score": cand.fused_score,
                    "payload": r.get("payload", {}),
                })

        # Rerank (FR-004/FR-005): only if reranker is configured and results exist
        rerank_elapsed = 0.0
        rerank_scores: dict[str, float] = {}
        if self._reranker is not None and raw_results:
            rerank_budget = self._settings.hybrid_retrieval.rerank_budget
            # Trim to rerank budget (FR-005, blueprint §18.5)
            rerank_input = raw_results[:rerank_budget]

            # Fetch chunk content_text for rerank (query+passage pairs)
            chunk_ids: list[int] = []
            for r in rerank_input:
                cid = r.get("payload", {}).get("chunk_id")
                if cid is not None:
                    try:
                        chunk_ids.append(int(cid))
                    except (ValueError, TypeError):
                        pass

            content_map = await self._fetch_chunk_content(chunk_ids)

            candidates = [
                {
                    "chunk_id": str(r.get("payload", {}).get("chunk_id", "")),
                    "content_text": content_map.get(
                        int(r.get("payload", {}).get("chunk_id", 0)), ""
                    ),
                    "fused_score": r.get("score", 0.0),
                }
                for r in rerank_input
            ]

            try:
                t_rerank_start = time.monotonic()
                reranked = await self._reranker.rerank(
                    query, candidates, top_k=limit,
                )
                rerank_elapsed = (time.monotonic() - t_rerank_start) * 1000

                # Rebuild raw_results from reranked order
                reranked_map: dict[str, dict[str, Any]] = {}
                for r in reranked:
                    cid = str(r.get("chunk_id", ""))
                    if cid:
                        reranked_map[cid] = r
                        if r.get("rerank_score") is not None:
                            rerank_scores[cid] = float(r["rerank_score"])

                new_raw: list[dict[str, Any]] = []
                for r in reranked:
                    cid = str(r.get("chunk_id", ""))
                    orig = result_map.get(cid, {})
                    new_raw.append({
                        "id": orig.get("id"),
                        "score": r.get("rerank_score", r.get("fused_score", 0.0)),
                        "payload": orig.get("payload", {}),
                    })
                raw_results = new_raw

            except Exception as rerank_exc:
                logger.warning("Rerank failed, using RRF results: %s", rerank_exc)
                failed_paths.append("rerank_failed")

        subpath_timings = {
            "dense_recall_ms": round(recall_elapsed, 2),
            "sparse_recall_ms": round(recall_elapsed, 2),  # parallel
            "fusion_ms": round(fusion_elapsed, 4),
            "rerank_ms": round(rerank_elapsed, 2),
            "total_ms": round(
                recall_elapsed + graph_elapsed + fusion_elapsed + rerank_elapsed, 2
            ),
        }
        if graph_enabled:
            subpath_timings["graph_recall_ms"] = round(graph_elapsed, 2)

        # 005 (T058): rich candidates with per-retriever metadata — the
        # evidence ledger input for the agentic pipeline. Order matches
        # raw_results (post-rerank). Deterministic path ignores this field.
        fused_by_cid = {str(c.chunk_id): c for c in fused}
        rich_candidates: list[dict[str, Any]] = []
        for r in raw_results:
            payload = r.get("payload", {}) or {}
            cid = str(payload.get("chunk_id", r.get("id", "")))
            fc = fused_by_cid.get(cid)
            if fc is None:
                continue
            rich_candidates.append({
                "chunk_id": cid,
                "retrievers": list(fc.source_retrievers),
                "dense_score": fc.dense_score,
                "sparse_score": fc.sparse_score,
                "dense_rank": fc.dense_rank,
                "sparse_rank": fc.sparse_rank,
                "graph_rank": fc.graph_rank,
                "graph_structure_weight": fc.graph_structure_weight,
                "fused_score": fc.fused_score,
                "final_rank": fc.final_rank,
                "rerank_score": rerank_scores.get(cid),
                "payload": payload,
            })

        logger.info(
            "Hybrid recall: dense=%d sparse=%d graph=%d fused=%d rerank=%dms, "
            "failed=%s, timings=%s",
            len(dense_results), len(sparse_results), len(graph_results),
            len(raw_results), rerank_elapsed, failed_paths, subpath_timings,
        )

        return (
            raw_results, subpath_timings, failed_paths,
            graph_used, graph_trace_data, rich_candidates,
        )

    async def _fetch_chunk_content(self, chunk_ids: list[int]) -> dict[int, str]:
        """Fetch content_text for chunks by ID (for rerank query+passage pairs)."""
        if not chunk_ids:
            return {}
        result = await self._session.execute(
            select(Chunk.chunk_id, Chunk.content_text).where(
                Chunk.chunk_id.in_(chunk_ids)
            )
        )
        return {row[0]: row[1] for row in result.all()}

    async def _build_sparse_encoder(self, scope_ids: list[int]) -> BM25SparseEncoder | None:
        """Build or retrieve a cached BM25SparseEncoder for the given scopes.

        Uses a module-level cache keyed by sorted scope_ids to avoid a full DB
        scan + fit() on every hybrid search call (T036, FR-015/SC-005).

        Hash-based term IDs ensure the cached encoder produces compatible IDs
        with stored sparse vectors regardless of when it was built.
        """
        cache_key = tuple(sorted(scope_ids))
        cached = _sparse_encoder_cache.get(cache_key)
        if cached is not None:
            logger.debug("Sparse encoder cache hit for scopes %s", cache_key)
            return cached

        logger.debug("Sparse encoder cache miss for scopes %s — building", cache_key)
        result = await self._session.execute(
            select(Chunk.content_text)
            .join(KnowledgeVersion, Chunk.version_id == KnowledgeVersion.version_id)
            .where(
                Chunk.knowledge_scope_id.in_(scope_ids),
                KnowledgeVersion.status == "published",
            )
            .order_by(Chunk.chunk_id)
        )
        texts = [row[0] for row in result.all()]
        if not texts:
            return None

        encoder = BM25SparseEncoder()
        encoder.fit(texts)
        _sparse_encoder_cache[cache_key] = encoder
        logger.info("Sparse encoder cached for scopes %s (%d terms)", cache_key, encoder.vocab_size)
        return encoder

    async def _record_retrieval_run(
        self,
        query: str,
        project_scopes: list[str],
        completion_status: str,
        evidence_count: int,
        duration_ms: int,
        retrieval_mode: str = "dense",
        subpath_timings: dict | None = None,
        evidence_ref_ids: list[str] | None = None,
        format: str | None = None,
        run_id: int | None = None,
        tool: str = "search_knowledge",
        error_summary: dict | None = None,
        provider_usage: dict | None = None,
    ) -> None:
        """Record an append-only RetrievalRun audit entry (003 adds format, FR-027).

        A caller-supplied run_id lets the graph trace ledger bridge
        graph_expansion_path rows to this run (004, DM-1).

        006 (FR-016/FR-018/FR-019): honors the unified trace-body switch
        (query_text=NULL + trace_body_recorded=FALSE when disabled), records
        tool / instance attribution / error_summary / provider_usage, and
        computes expires_at from the configurable RETRIEVAL_TTL_DAYS.
        """
        from rag_mcp.runtime.instance_context import get_instance

        instance_id, instance_mode, _worker_id = get_instance()
        await persist_retrieval_run(
            self._session,
            query=query,
            project_scopes=project_scopes,
            completion_status=completion_status,
            evidence_count=evidence_count,
            duration_ms=duration_ms,
            retrieval_mode=retrieval_mode,
            subpath_timings=subpath_timings,
            evidence_ref_ids=evidence_ref_ids,
            format=format,
            run_id=run_id,
            tool=tool,
            error_summary=error_summary,
            provider_usage=provider_usage,
            trace_enabled=bool(self._settings.trace_body_enabled),
            ttl_days=int(self._settings.retrieval_ttl_days),
            instance_id=instance_id,
            instance_mode=instance_mode,
        )

    def _derive_index_version(self) -> str:
        """Derive the Qdrant collection index_version from the active provider model."""
        return derive_index_version(active_embedding_model(self._settings))
