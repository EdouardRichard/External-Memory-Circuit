"""T097: independent six-way authority restoration from one sealed capsule.

Proves the requirement that every ``(round, arm)`` of a consolidation
comparison starts from its own restoration of the same frozen original
authority:

1. a fresh source database is published through the ordinary services and
   sealed into one immutable capsule (native PostgreSQL copy, byte-exact
   private data-root subtrees, sealed Qdrant points);
2. six ``(round, arm)`` identities are allocated with distinct databases,
   distinct Qdrant store processes and distinct private data roots;
3. two identities are really restored, and each is verified equal to the sealed
   capsule (authority digest, migration version, cutoff, projection manifest,
   byte-level data-root manifest, Qdrant point digest);
4. an ordinary write through ``MemoryService`` into one identity changes only
   that identity: the other identity still verifies equal to the capsule.

Nothing here writes to the sealed capsule or into another arm, and no resource
is reused across arms.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / ".superpowers" / "sdd" / "013-tasks" / "isolation_runner.py"
RESTORE_CLI = ROOT / "eval" / "restore_consolidation_arm.py"
RESTORE_BASE = Path(os.environ.get("CONSOLIDATION_RESTORE_BASE", "C:/t097"))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

from consolidation_restore_support import (
    Identity,
    RestoreError,
    allocate_identities,
    assert_independent,
    drop_database,
    load_run,
    restore_identity,
    stop_qdrant,
)


def _runner(*arguments: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RUNNER), *arguments], cwd=ROOT, check=False,
                          capture_output=True, text=True, timeout=timeout, env=dict(os.environ))


def _restore(*arguments: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return _runner("eval", str(RESTORE_CLI), *arguments, timeout=timeout)


def _payload(result: subprocess.CompletedProcess) -> dict:
    assert "{" in result.stdout, result.stdout + result.stderr
    return json.loads(result.stdout[result.stdout.index("{"):])


def _async_url(database: str) -> str:
    return make_url(os.environ["DATABASE_URL"]).set(database=database).render_as_string(hide_password=False)


def _memory_collection() -> str:
    from rag_mcp.providers.active_model import active_embedding_model, derive_memory_collection

    return derive_memory_collection(active_embedding_model())


def _scope_of(database: str) -> int:
    from sqlalchemy import create_engine, text

    engine = create_engine(make_url(os.environ["DATABASE_URL_SYNC"]).set(database=database)
                           .render_as_string(hide_password=False))
    try:
        with engine.connect() as connection:
            return int(connection.scalar(text(
                "SELECT scope_id FROM knowledge_scopes WHERE domain_key LIKE 'c013-restore-%' LIMIT 1")))
    finally:
        engine.dispose()


async def _seed_source(engine, *, episodes: int = 3) -> int:
    """Publish one real scope with real memory events through the ordinary path."""
    from rag_mcp.models.domain_profile import DomainProfile
    from rag_mcp.models.knowledge_scope import KnowledgeScope
    from rag_mcp.services.memory_service import MemoryService
    from rag_mcp.utils.snowflake import generate_id
    from tests.integration.consolidation_fixtures import StableEmbedding, recorded_episode
    from tests.integration.memory_acceptance import writer_owner

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        key = "c013-restore-" + uuid4().hex
        session.add(DomainProfile(domain_key=key, name=key, supported_formats=["markdown"],
                                  graph_relations={}, default_capabilities={}, is_builtin=False,
                                  memory_policy={}))
        await session.flush()
        scope = KnowledgeScope(scope_id=generate_id(), scope_type="public", slug=key, name=key,
                               domain_key=key)
        session.add(scope)
        await session.commit()
        scope_id = int(scope.scope_id)
    async with writer_owner(engine), factory() as session:
        service = MemoryService(session, embedding_provider=StableEmbedding())
        for index in range(episodes):
            await recorded_episode(service, scope_id, f"Observed restoration source {index}")
        await session.commit()
    return scope_id


async def _append_episode(engine, scope_id: int, qdrant_url: str, projection_root: Path) -> None:
    """One ordinary post-restore write inside a single identity (its own store and root)."""
    from rag_mcp.indexing.qdrant_client import QdrantStore
    from rag_mcp.services.memory_service import MemoryService
    from tests.integration.consolidation_fixtures import StableEmbedding, recorded_episode
    from tests.integration.memory_acceptance import writer_owner

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with writer_owner(engine), factory() as session:
        service = MemoryService(session, embedding_provider=StableEmbedding(),
                                qdrant_store=QdrantStore(url=qdrant_url),
                                projection_root=projection_root)
        await recorded_episode(service, scope_id, "Post-restore arm-specific observation")
        await session.commit()


def _teardown(databases: list[str], run_file: Path, workdir: Path) -> None:
    if run_file.exists():
        try:
            run = load_run(run_file)
        except Exception:  # noqa: BLE001 - a broken run file must not block database cleanup
            run = None
        if run is not None:
            for identity in run.identities:
                stop_qdrant(identity)
    for database in databases:
        with contextlib.suppress(Exception):
            # Cleanup is best effort: the real assertions already ran.
            drop_database(database)
    shutil.rmtree(workdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_two_restorations_are_independent_and_match_the_frozen_capsule():
    token = uuid4().hex[:10]
    source_database = f"memory_consolidation_013_t097src_{token}"
    workdir = RESTORE_BASE / token
    capsule_dir = workdir / "capsule"
    run_file = workdir / "run.json"
    databases = [source_database]
    try:
        workdir.mkdir(parents=True)
        provision = _restore("provision", "--database", source_database)
        assert provision.returncode == 0, provision.stdout + provision.stderr
        migrated = _runner("--database", source_database, "alembic", "upgrade", "head")
        assert migrated.returncode == 0, migrated.stderr[-2000:]

        source_engine = create_async_engine(_async_url(source_database), echo=False)
        try:
            scope_id = await _seed_source(source_engine)
        finally:
            # The native template copy needs the source database quiescent.
            await source_engine.dispose()

        sealed = _restore("seal", "--source-database", source_database,
                          "--source-data-root", str(os.environ["DATA_ROOT"]),
                          "--capsule-dir", str(capsule_dir), "--scopes", str(scope_id),
                          "--token", token, "--collections", _memory_collection())
        assert sealed.returncode == 0, sealed.stdout + sealed.stderr
        capsule = json.loads((capsule_dir / "capsule.json").read_text(encoding="utf-8"))
        databases.append(capsule["capsule_database"])
        assert _payload(sealed)["authority_digest"] == capsule["authority"]["authority_digest"]
        assert capsule["authority"]["scopes"] == [scope_id]

        allocated = _restore("allocate", "--run-id", token, "--base", str(RESTORE_BASE),
                             "--output", str(run_file),
                             "--qdrant-port-base", str(17400 + int(token[:3], 16)))
        assert allocated.returncode == 0, allocated.stdout + allocated.stderr
        allocation = _payload(allocated)
        identities = allocation["identities"]
        assert len(identities) == 6
        assert allocation["proof"] == {"identities": 6, "distinct_databases": 6,
                                       "distinct_data_roots": 6, "distinct_qdrant_stores": 6}

        arm_a = _restore("restore", "--capsule-dir", str(capsule_dir), "--run", str(run_file),
                         "--round", "record", "--arm", "baseline")
        assert arm_a.returncode == 0, arm_a.stdout + arm_a.stderr
        arm_b = _restore("restore", "--capsule-dir", str(capsule_dir), "--run", str(run_file),
                         "--round", "record", "--arm", "consolidated_direct")
        assert arm_b.returncode == 0, arm_b.stdout + arm_b.stderr
        report_a, report_b = _payload(arm_a), _payload(arm_b)
        databases.extend([report_a["database"], report_b["database"]])

        for report in (report_a, report_b):
            assert report["authority_digest"] == capsule["authority"]["authority_digest"]
            assert report["manifest_digest"] == capsule["project_manifest"]["manifest_digest"]
            assert report["data_root_digest"] == capsule["data_root_digest"]
            assert report["alembic_version"] == capsule["alembic_version"]
        assert report_a["database"] != report_b["database"]
        assert report_a["data_root"] != report_b["data_root"]
        assert report_a["qdrant_url"] != report_b["qdrant_url"]

        arm_a_engine = create_async_engine(_async_url(report_a["database"]), echo=False)
        try:
            await _append_episode(arm_a_engine, _scope_of(report_a["database"]), report_a["qdrant_url"],
                                  Path(report_a["data_root"]) / "memory_projection")
        finally:
            await arm_a_engine.dispose()

        changed = _restore("verify", "--capsule-dir", str(capsule_dir), "--run", str(run_file),
                           "--round", "record", "--arm", "baseline")
        assert changed.returncode == 2, changed.stdout + changed.stderr
        assert "differs from the sealed capsule" in changed.stdout

        untouched = _restore("verify", "--capsule-dir", str(capsule_dir), "--run", str(run_file),
                             "--round", "record", "--arm", "consolidated_direct")
        assert untouched.returncode == 0, untouched.stdout + untouched.stderr
        verified = _payload(untouched)
        assert verified["authority_digest"] == capsule["authority"]["authority_digest"]
        assert verified["data_root_digest"] == capsule["data_root_digest"]
    finally:
        _teardown(databases, run_file, workdir)


def test_six_arm_identities_never_share_a_resource():
    """Allocation itself must already refuse a shared database, root or store."""
    run = allocate_identities("alloctest", base=RESTORE_BASE, qdrant_port_base=17800)
    assert assert_independent(run) == {"identities": 6, "distinct_databases": 6,
                                       "distinct_data_roots": 6, "distinct_qdrant_stores": 6}
    assert {(identity.round, identity.arm) for identity in run.identities} == {
        (round_name, arm) for round_name in ("record", "replay")
        for arm in ("baseline", "consolidated_direct", "consolidated_candidate_expansion")}
    run.identities[1] = run.identities[0]
    with pytest.raises(RestoreError):
        assert_independent(run)


def test_restore_refuses_to_reuse_an_existing_identity(tmp_path):
    """An existing target root is refused before anything is materialised."""
    identity = Identity(round="record", arm="baseline",
                        database="memory_consolidation_013_t097probe_missing",
                        data_root=tmp_path / "root", qdrant_url="http://127.0.0.1:17999",
                        qdrant_storage=tmp_path / "qdrant", qdrant_http_port=17999,
                        qdrant_grpc_port=18000)
    (tmp_path / "root").mkdir()
    with pytest.raises(RestoreError, match="data root already exists"):
        restore_identity(tmp_path / "no-capsule", identity)
