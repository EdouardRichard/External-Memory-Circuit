"""Read-only gate proof loading for candidate link expansion (013 T061).

This module owns the shared pure report/binding validators and the bounded,
non-blocking registry/report loader. It never imports eval, never holds a
writer handle, and never caches an authorization: every call re-reads and
re-hashes the deployed bytes. Only the parse/validation of identical bytes may
be cached, keyed by content hash.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from rag_mcp.providers.active_model import active_embedding_model

REGISTRY_LIMIT_BYTES = 256 * 1024
REPORT_LIMIT_BYTES = 8 * 1024 * 1024
IO_BUDGET_MS = 100
GATE_REASONS = ('GATE_NOT_CONFIGURED', 'GATE_MISSING', 'GATE_INVALID', 'GATE_HASH_MISMATCH',
                'GATE_SCOPE_MISMATCH', 'GATE_BINDING_STALE', 'GATE_EXPIRED',
                'GATE_VARIANT_NOT_AUTHORIZED', 'GATE_IO_BUDGET_EXCEEDED')
SCHEMA_FILES = ('distiller-output.schema.json', 'consolidate-event.schema.json',
                'benefit-report.schema.json', 'gate-registry.schema.json')
IMPLEMENTATION_FILES = ('agents/memory_distiller.py', 'orchestration/consolidation_pipeline.py',
                        'services/consolidation_adjudicator.py', 'services/consolidation_commit.py',
                        'services/consolidation_gate.py', 'services/consolidation_runtime.py',
                        'services/memory_projection_store.py', 'services/memory_reader.py',
                        'services/memory_reducer.py', 'services/memory_validators.py')
RECALL_WEIGHTS = {'dense': 1., 'recency': .5, 'kind': .3, 'salience': .2}
GATE_CHECKS = {'quality': {'six_query_coverage', 'aggregate_recomputed', 'relative_gain_thresholds'},
               'safety': {'hard_metrics_zero', 'scope_isolation'},
               'regression': {'reproducibility_passed', 'replay_zero_network', 'legacy_contract_unchanged'}}
_RATE_KEYS = ('mrr', 'ndcg', 'hit_rate', 'recall_at_k', 'precision_at_k')
GATE_VARIANTS = ('baseline', 'consolidated_direct', 'consolidated_candidate_expansion')
# Frozen physical-rank trace identity (evaluation-contract.md "Relevance and
# metrics"): every arm records the alias observed at each retrieved position so
# the report's per-query metrics are recomputable instead of self-reported.
TRACE_VERSION = '013.trace.1'

_IO_SLOTS = threading.BoundedSemaphore(2)
_IO_THREADS = ThreadPoolExecutor(max_workers=2, thread_name_prefix='consolidation-gate')
_PARSE_CACHE: dict[str, object] = {}
_PARSE_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class GateProof:
    available: bool
    reason_code: str
    scope_id: str | None = None
    report_sha256: str | None = None
    gate_variant: str | None = None
    binding: Mapping | None = None
    expires_at: str | None = None


def canonical_json(value) -> bytes:
    """Shared JSON-value normalization for binding materials (ensure_ascii)."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _contracts_dir() -> Path:
    return Path(__file__).resolve().parents[4] / 'specs/013-memory-consolidation-loop/contracts'


def _schema_validator(name):
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource

    schemas = [json.loads((_contracts_dir() / schema).read_text(encoding='utf-8')) for schema in SCHEMA_FILES]
    registry = Registry().with_resources((schema['$id'], Resource.from_contents(schema)) for schema in schemas)
    formats = FormatChecker()

    @formats.checks('date-time', raises=ValueError)
    def timestamp(value):
        return not isinstance(value, str) or datetime.fromisoformat(value).tzinfo is not None

    return Draft202012Validator(json.loads((_contracts_dir() / name).read_text(encoding='utf-8')),
                                registry=registry, format_checker=formats)


def _validate_schema(name, document):
    """Bundled local Schema validation; every failure is a deterministic ValueError."""
    from jsonschema import ValidationError

    try:
        _schema_validator(name).validate(document)
    except ValidationError as error:
        raise ValueError('document does not satisfy the packaged schema') from error
    except (RecursionError, TypeError) as error:
        raise ValueError('document cannot be validated') from error


def _strict_loads(raw: bytes):
    def pairs(items):
        keys = [key for key, _ in items]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate JSON object key')
        return dict(items)

    def constant(value):
        raise ValueError('non-finite JSON number')

    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise ValueError('invalid JSON document') from error


def validate_gate_binding(binding) -> Mapping:
    if not isinstance(binding, Mapping):
        raise TypeError('gate binding must be an object')
    fields = ('scope_id', 'data_hash', 'policy_hash', 'vocabulary_hash', 'prompt_hash',
              'schema_hash', 'model_version', 'implementation_hash', 'recall_config_hash')
    if set(binding) != set(fields):
        raise ValueError('gate binding fields mismatch')
    if not isinstance(binding['scope_id'], str) or not binding['scope_id'].isdecimal() \
            or binding['scope_id'].startswith('0'):
        raise ValueError('gate binding scope must be a decimal string')
    for key in fields:
        if key in ('scope_id', 'model_version'):
            continue
        value = binding[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'gate binding {key} must be a sha256 hex digest')
    if not isinstance(binding['model_version'], str) or not binding['model_version']:
        raise ValueError('gate binding model_version required')
    return MappingProxyType(dict(binding))


def _mean(values):
    return sum(values) / len(values)


def _close(left, right, tolerance=1e-9):
    return left is not None and right is not None and abs(left - right) <= tolerance


def binary_metrics(ranked_aliases, relevance_units, k):
    """Binary-gain retrieval metrics at preserved physical rank.

    ``ranked_aliases`` is the ordered alias observed at each physical position
    (``None`` for a retrieved item that is not a frozen relevance unit). Gains
    are strictly binary: the first occurrence of a relevance unit gains 1 and
    every later duplicate gains 0, so a repeated alias can never inflate
    recall, precision or nDCG. Only the first ``k`` positions are scored and a
    missing rank contributes zero. Shared by the eval comparison runner and the
    production report validator, which recomputes rather than trusts them.
    """
    units = list(relevance_units)
    known = set(units)
    gains = []
    seen = set()
    for position, alias in enumerate(list(ranked_aliases)[:k], start=1):
        if alias is not None and alias in known and alias not in seen:
            seen.add(alias)
            gains.append((position, 1))
        else:
            gains.append((position, 0))
    dcg = sum(gain / math.log2(position + 1) for position, gain in gains)
    ideal_hits = min(len(units), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return {
        'mrr': next((1.0 / position for position, gain in gains if gain), 0.0),
        'ndcg': (dcg / idcg) if idcg else 0.0,
        'hit_rate': 1.0 if any(gain for _, gain in gains) else 0.0,
        'recall_at_k': (len(seen) / len(units)) if units else 0.0,
        'precision_at_k': sum(gain for _, gain in gains) / k,
    }


def macro_average(rows):
    """One-query-one-weight macro average; an unobserved arm is never zero-filled."""
    rows = list(rows)
    if not rows:
        return {metric: None for metric in _RATE_KEYS}
    return {metric: sum(row[metric] for row in rows) / len(rows) for metric in _RATE_KEYS}


def _frozen_relevance_units(relevance_units):
    units = []
    for unit in relevance_units if isinstance(relevance_units, (list, tuple)) else ():
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError('report relevance unit must be a non-empty label')
        if unit in units:
            raise ValueError('report relevance unit repeated')
        units.append(unit)
    if not units:
        raise ValueError('report query has no frozen relevance unit')
    return units


def recompute_query_metrics(query, k) -> dict:
    """Recompute all three arms from the frozen physical-rank trace."""
    units = _frozen_relevance_units(query['relevance_units'])
    trace = query['result_trace']
    if not isinstance(trace, Mapping) or trace.get('trace_version') != TRACE_VERSION:
        raise ValueError('report result trace version invalid')
    if trace.get('k') != k:
        raise ValueError('report result trace k mismatch')
    variants = trace.get('variants')
    if not isinstance(variants, Mapping) or set(variants) != set(GATE_VARIANTS):
        raise ValueError('report result trace variants incomplete')
    recomputed = {}
    for variant in GATE_VARIANTS:
        ranked = variants[variant]
        if not isinstance(ranked, list) or len(ranked) != k:
            raise ValueError('report result trace must record every physical rank')
        if any(alias is not None and alias not in units for alias in ranked):
            raise ValueError('report result trace alias is not a frozen unit')
        recomputed[variant] = binary_metrics(ranked, units, k)
    return recomputed


def _observed_metrics(container, context):
    if not isinstance(container, Mapping):
        raise TypeError(f'report {context} observation missing')
    for metric in _RATE_KEYS:
        if container.get(metric) is None:
            raise ValueError(f'report {context} observation missing')
    return container


def _validate_passed_semantics(report):
    binding = validate_gate_binding(report['gate_binding'])
    environment = report['environment']
    for key in ('policy_hash', 'vocabulary_hash', 'prompt_hash', 'schema_hash'):
        if environment.get(key) != binding[key]:
            raise ValueError('report environment diverges from gate binding')
    if environment.get('model_version') != binding['model_version']:
        raise ValueError('report environment diverges from gate binding')
    queries = report['queries']
    if len(queries) < 6 or len({q['query_id'] for q in queries}) != len(queries):
        raise ValueError('report query coverage invalid')
    if any(q['scope_id'] != binding['scope_id'] for q in queries):
        raise ValueError('report covers more than one scope')
    categories = [q['primary_category'] for q in queries]
    if not all(categories.count(kind) >= 2 for kind in ('extraction', 'correction', 'merge')):
        raise ValueError('report category coverage invalid')
    kinds = {q['extraction_kind'] for q in queries if q['primary_category'] == 'extraction'}
    if kinds != {'semantic', 'procedural'}:
        raise ValueError('report extraction coverage invalid')
    variants = GATE_VARIANTS
    for query in queries:
        recomputed = recompute_query_metrics(query, report['k'])
        for variant in variants:
            observed = _observed_metrics(query[variant], f'query {variant}')
            for metric in _RATE_KEYS:
                if not _close(observed[metric], recomputed[variant][metric]):
                    raise ValueError('report per-query metrics are not the frozen physical-rank result')
    for variant in variants:
        aggregate = _observed_metrics(report['aggregates'][variant], f'aggregate {variant}')
        for metric in _RATE_KEYS:
            observed = [q[variant][metric] for q in queries]
            if not _close(_mean(observed), aggregate[metric]):
                raise ValueError('report aggregates are not the recomputed macro average')
    baseline = report['aggregates']['baseline']
    zero = baseline['mrr'] == 0 or baseline['ndcg'] == 0
    gains = report['relative_gains']
    if gains['baseline_zero'] != zero:
        raise ValueError('report baseline_zero inconsistent')
    expansion = report['aggregates']['consolidated_candidate_expansion']
    for metric in ('mrr', 'ndcg'):
        expected = None if zero else (expansion[metric] - baseline[metric]) / baseline[metric]
        if expected is None:
            if gains[metric] is not None:
                raise ValueError('report gains must be null on a zero baseline')
        elif not _close(gains[metric], expected) or gains[metric] < .03:
            raise ValueError('report relative gains invalid')
    for variant in ('consolidated_direct', 'consolidated_candidate_expansion'):
        for metric in ('hit_rate', 'recall_at_k', 'precision_at_k'):
            if report['aggregates'][variant][metric] < baseline[metric] - 1e-12:
                raise ValueError('report hides a direct-path regression')
    for name, required in GATE_CHECKS.items():
        gate = report['gates'][name]
        if gate['status'] != 'passed' or not required <= set(gate['checks']):
            raise ValueError('report gate checks incomplete')
        if not all(gate['checks'][check] is True for check in required) or not gate['evidence']:
            raise ValueError('report gate evidence incomplete')
    hard = report['hard_metrics']
    if any(hard[key] != 0 for key in ('cross_scope_leaks', 'quarantined_inputs', 'soft_overturns_hard',
                                      'automatic_promotions', 'invalid_outputs_applied', 'stale_holder_commits',
                                      'incomplete_outputs_consumed', 'rebuild_llm_calls')):
        raise ValueError('report hard metrics nonzero')
    if any(hard[key] != 1 for key in ('source_chain_complete_rate', 'schema_validity_rate',
                                      'projection_integrity_rate')):
        raise ValueError('report integrity rates incomplete')
    cache = report['cache']
    if (cache['replay_real_network_calls'] != 0 or not cache['evidence_complete']
        or cache['response_match_rate'] != 1 or cache['missing'] or cache['corrupt']
        or cache['version_mismatch'] or cache['replay_usage']['source'] != 'replay_zero'):
        raise ValueError('report cache evidence incomplete')
    reproducibility = report['reproducibility']
    if (reproducibility['status'] != 'passed' or not reproducibility['zero_baseline_exact_match']
        or not reproducibility['safety_exact_match'] or reproducibility['max_non_latency_relative_drift'] is None
        or reproducibility['max_non_latency_relative_drift'] > .01):
        raise ValueError('report reproducibility invalid')
    defaults = report['default_configuration']
    if (defaults['consolidation_enabled'] or defaults['link_expansion_enabled']
            or defaults['policy_published']):
        raise ValueError('report must not publish policy')
    if (not report['run_ids'] or report['failed_paths'] or not environment.get('frozen_clock')
            or not environment.get('model_version') or not environment.get('projection_fingerprints')):
        raise ValueError('report environment incomplete')
    if not report['default_enable_eligible']:
        raise ValueError('report is not default-enable eligible')


def validate_report(report) -> Mapping:
    """Pure shared 013.2 report validation; raises ValueError on any violation."""
    if not isinstance(report, Mapping):
        raise TypeError('report must be an object')
    _validate_schema('benefit-report.schema.json', report)
    if any(not isinstance(query, Mapping) or 'query_id' not in query for query in report['queries']):
        raise ValueError('report query entry invalid')
    if len({q['query_id'] for q in report['queries']}) != len(report['queries']):
        raise ValueError('duplicate report query id')
    if report['gate_binding'] is not None:
        binding = validate_gate_binding(report['gate_binding'])
        if any(q['scope_id'] != binding['scope_id'] for q in report['queries']):
            raise ValueError('report covers more than one scope')
    # A zero baseline is not a quality gain and is never computed with epsilon,
    # infinity or a fabricated number: the relative gains stay null. This holds
    # for every status so an incomplete report cannot smuggle a computed gain.
    baseline = report['aggregates']['baseline']
    gains = report['relative_gains']
    if isinstance(baseline, Mapping):
        zero = baseline.get('mrr') == 0 or baseline.get('ndcg') == 0
        if gains.get('baseline_zero') is not zero:
            raise ValueError('report baseline_zero inconsistent')
        if zero and (gains.get('mrr') is not None or gains.get('ndcg') is not None):
            raise ValueError('report gains must be null on a zero baseline')
    if report['status'] == 'passed':
        _validate_passed_semantics(report)
    return MappingProxyType(dict(report))


def ordinary_source_events(events) -> list:
    """Verified ordinary authority events; 013 internal effects/controls excluded."""
    selected = []
    for event in events:
        payload = event.get('payload') or {}
        if event.get('event_type') == 'consolidate' and payload.get('payload_version') == 2:
            continue
        if event.get('event_type') == 'grant' and payload.get('grant_type') in (
                'consolidation_window', 'consolidation_propagation'):
            continue
        occurred = event.get('occurred_at')
        selected.append({'event_id': str(event['event_id']), 'event_type': event['event_type'],
                         'aggregate_id': str(event['aggregate_id']),
                         'observed_at': occurred.isoformat() if hasattr(occurred, 'isoformat') else occurred,
                         'payload': payload})
    selected.sort(key=lambda row: int(row['event_id']))
    return selected


def data_material_hash(scope_id, source_events, published_evidence) -> str:
    material = {'binding_version': 1, 'scope_id': str(scope_id),
                'source_events': list(source_events), 'published_evidence': list(published_evidence)}
    return sha256(canonical_json(material)).hexdigest()


def schema_hash() -> str:
    root = _contracts_dir()
    return sha256(canonical_json({name: sha256((root / name).read_bytes()).hexdigest()
                                  for name in SCHEMA_FILES})).hexdigest()


def implementation_hash() -> str:
    package = Path(__file__).resolve().parents[1]
    materials = {f'src/rag_mcp/{name}': sha256((package / name).read_bytes()).hexdigest()
                 for name in IMPLEMENTATION_FILES}
    versions = Path(__file__).resolve().parents[3] / 'alembic' / 'versions'
    for path in sorted(versions.glob('0*_*.py')):
        prefix = path.name.split('_', 1)[0]
        # Every applied 013 consolidation migration from 0095 onward participates
        # in the implementation fingerprint (a later guard change must invalidate
        # an existing gate binding); 0094 and earlier are the 012 foundation.
        if len(prefix) == 4 and prefix.isdigit() and prefix >= '0095':
            materials[f'alembic/{path.name}'] = sha256(path.read_bytes()).hexdigest()
    return sha256(canonical_json(materials)).hexdigest()


def prompt_hash(profile) -> str:
    from rag_mcp.config.domain_profiles import DISTILLER_PROMPT_VERSION, DISTILLER_SYSTEM_PROMPT

    material = {'role': 'memory_distiller', 'prompt_version': DISTILLER_PROMPT_VERSION,
                'system': DISTILLER_SYSTEM_PROMPT,
                'domain': {'domain_key': profile.domain_key, 'name': profile.name,
                           'description': profile.description or ''}}
    return sha256(canonical_json(material)).hexdigest()


def recall_config_hash(settings, weights=None) -> str:
    material = {'embedding_model': active_embedding_model(settings), 'weights': dict(weights or RECALL_WEIGHTS),
                'limit_max': 50, 'timeout_ms': 3000, 'content_budget_chars': 6000,
                'dense_candidate_min': 40, 'dense_candidate_factor': 4}
    return sha256(canonical_json(material)).hexdigest()


async def gather_current_binding(session, scope_id, *, settings=None, weights=None) -> dict:
    """Read-only current binding for one scope; failure means no authorization."""
    from sqlalchemy import select

    from rag_mcp.config import get_settings
    from rag_mcp.config.domain_profiles import memory_vocabulary_version, validate_memory_link_vocabulary
    from rag_mcp.models.chunk import Chunk
    from rag_mcp.models.domain_profile import DomainProfile
    from rag_mcp.models.knowledge_scope import KnowledgeScope
    from rag_mcp.models.knowledge_source import KnowledgeSource
    from rag_mcp.models.knowledge_version import KnowledgeVersion
    from rag_mcp.models.memory_projection_meta import MemoryProjectionMeta
    from rag_mcp.services.memory_event_store import MemoryEventStore
    from rag_mcp.services.memory_policy import MemoryPolicy

    settings = settings or get_settings()
    scope = await session.get(KnowledgeScope, scope_id, populate_existing=True)
    if scope is None or scope.status != 'active':
        raise ValueError('MISSING_KNOWLEDGE_SCOPE')
    profile = await session.get(DomainProfile, scope.domain_key, populate_existing=True)
    policy = MemoryPolicy.model_validate(profile.memory_policy or {})
    manifest = await session.get(MemoryProjectionMeta, f'current:{scope_id}', populate_existing=True)
    high_water = manifest.source_event_id if manifest and manifest.status == 'complete' else 0
    events = await MemoryEventStore(session).replay(scope_id, through_event_id=high_water) if high_water else []
    rows = (await session.execute(select(Chunk, KnowledgeVersion, KnowledgeSource)
        .join(KnowledgeVersion, Chunk.version_id == KnowledgeVersion.version_id)
        .join(KnowledgeSource, Chunk.source_id == KnowledgeSource.source_id)
        .where(Chunk.knowledge_scope_id == scope_id, KnowledgeVersion.status == 'published'))).all()
    evidence = [{'chunk_id': str(chunk.chunk_id), 'source_id': str(source.source_id),
                 'version_id': str(version.version_id), 'version_number': version.version_number,
                 'position_path': chunk.position_path,
                 'content_hash': sha256(chunk.content_text.encode()).hexdigest(),
                 'chunk_type': chunk.chunk_type, 'embedding_model': chunk.embedding_model,
                 'index_version': chunk.index_version,
                 'source_available': source.status in ('uploaded', 'processing', 'published')}
                for chunk, version, source in rows]
    evidence.sort(key=lambda row: int(row['chunk_id']))
    return {'scope_id': str(scope_id),
            'data_hash': data_material_hash(scope_id, ordinary_source_events(events), evidence),
            'policy_hash': sha256(canonical_json(policy.model_dump())).hexdigest(),
            'vocabulary_hash': memory_vocabulary_version(
                validate_memory_link_vocabulary(profile.memory_link_vocabulary or [])),
            'prompt_hash': prompt_hash(profile), 'schema_hash': schema_hash(),
            'model_version': settings.llm_model, 'implementation_hash': implementation_hash(),
            'recall_config_hash': recall_config_hash(settings, weights)}


def _parse_timestamp(value):
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError('invalid timestamp') from error
    if stamp.tzinfo is None:
        raise ValueError('timestamp requires timezone')
    return stamp


def _load_sync(path, scope_id, binding, now) -> GateProof:
    from rag_mcp.config import get_settings

    root = Path(path)
    try:
        if not root.is_absolute():
            return GateProof(False, 'GATE_INVALID')
        resolved = root.resolve()
        data_root = Path(get_settings().data_root).resolve()
        if resolved == data_root or data_root in resolved.parents:
            return GateProof(False, 'GATE_INVALID')
        if not resolved.is_file():
            return GateProof(False, 'GATE_MISSING')
        raw = resolved.read_bytes()
        if len(raw) > REGISTRY_LIMIT_BYTES:
            return GateProof(False, 'GATE_INVALID')
        registry = _strict_loads(raw)
        _validate_schema('gate-registry.schema.json', registry)
        entries = registry['entries']
        if len({entry['gate_binding']['scope_id'] for entry in entries}) != len(entries):
            return GateProof(False, 'GATE_INVALID')
        entry = next((item for item in entries if item['gate_binding']['scope_id'] == scope_id), None)
        if entry is None:
            return GateProof(False, 'GATE_MISSING')
        expires = _parse_timestamp(entry['expires_at'])
        if expires <= now:
            return GateProof(False, 'GATE_EXPIRED')
        report_path = (resolved.parent / entry['report_path']).resolve()
        if resolved.parent not in report_path.parents:
            return GateProof(False, 'GATE_INVALID')
        if not report_path.is_file():
            return GateProof(False, 'GATE_MISSING')
        report_bytes = report_path.read_bytes()
        if len(report_bytes) > REPORT_LIMIT_BYTES:
            return GateProof(False, 'GATE_INVALID')
        digest = sha256(report_bytes).hexdigest()
        if digest != entry['report_sha256']:
            return GateProof(False, 'GATE_HASH_MISMATCH')
        with _PARSE_CACHE_LOCK:
            report = _PARSE_CACHE.get(digest)
        if report is None:
            report = validate_report(_strict_loads(report_bytes))
            with _PARSE_CACHE_LOCK:
                _PARSE_CACHE[digest] = report
        generated = _parse_timestamp(report['generated_at'])
        if report['status'] != 'passed' or not report['default_enable_eligible']:
            return GateProof(False, 'GATE_VARIANT_NOT_AUTHORIZED')
        if report['gate_variant'] != 'consolidated_candidate_expansion':
            return GateProof(False, 'GATE_VARIANT_NOT_AUTHORIZED')
        if expires <= generated:
            return GateProof(False, 'GATE_EXPIRED')
        if dict(report['gate_binding'] or {}) != dict(entry['gate_binding']):
            return GateProof(False, 'GATE_BINDING_STALE')
        if dict(binding).get('scope_id') != scope_id:
            return GateProof(False, 'GATE_SCOPE_MISMATCH')
        if dict(entry['gate_binding']) != dict(binding):
            return GateProof(False, 'GATE_BINDING_STALE')
        return GateProof(True, 'GATE_AUTHORIZED', scope_id, digest, report['gate_variant'],
                         MappingProxyType(dict(entry['gate_binding'])), entry['expires_at'])
    except (ValueError, TypeError):
        return GateProof(False, 'GATE_INVALID')
    except OSError:
        return GateProof(False, 'GATE_MISSING')


def _load_sync_guarded(path, scope_id, binding, now) -> GateProof:
    try:
        return _load_sync(path, scope_id, binding, now)
    finally:
        # The slot belongs to the actual read; a timed-out waiter never frees it.
        _IO_SLOTS.release()


async def load_gate_proof(scope_id, current_binding, now, *, registry_path=None, remaining_budget_ms=IO_BUDGET_MS):
    """Bounded, non-blocking proof load; a late or slow proof never authorizes."""
    from rag_mcp.config import get_settings

    path = registry_path or get_settings().consolidation_gate_registry_path
    if not path:
        return GateProof(False, 'GATE_NOT_CONFIGURED')
    budget = min(IO_BUDGET_MS, remaining_budget_ms) / 1000
    if budget <= 0:
        return GateProof(False, 'GATE_IO_BUDGET_EXCEEDED')
    if not _IO_SLOTS.acquire(blocking=False):
        return GateProof(False, 'GATE_IO_BUDGET_EXCEEDED')
    try:
        work = _IO_THREADS.submit(_load_sync_guarded, str(path), str(scope_id),
                                  dict(current_binding), now)
    except BaseException:
        _IO_SLOTS.release()
        raise
    future = asyncio.wrap_future(work)
    future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=budget)
    except TimeoutError:
        return GateProof(False, 'GATE_IO_BUDGET_EXCEEDED')
