"""Shared pytest fixtures for RAG MCP tests."""

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_mcp.config import get_settings

# Tests exercise the upload endpoint but do not depend on the asynchronous
# ingestion pipeline (which loads the real bge-m3 model and runs against live
# Qdrant). Disable background ingestion so uploads stay in 'uploaded' status,
# keeping tests deterministic and free of un-awaited background tasks.
os.environ.setdefault("INGESTION_BACKGROUND", "false")


def _sanitize_proxy_environment() -> None:
    """Drop malformed bracketed IPv6 entries from NO_PROXY (environment defect).

    This host exports ``NO_PROXY='localhost,127.0.0.1,::1,[::1]'``. The bracketed
    entry makes httpx build an invalid URLPattern for EVERY client, so any test that
    constructs one fails with ``httpx.InvalidURL: Invalid port: ':1]'`` before it
    reaches the code under test (measured: ``http://[::1]:P``, ``http://127.0.0.1:P``
    and ``http://localhost:P`` all fail as-is and all succeed once the bracketed entry
    is removed). Only this process's environment is normalised; nothing global changes
    and the services the tests talk to are real.
    """
    for name in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(name)
        if not value:
            continue
        entries = [entry.strip() for entry in value.split(",") if entry.strip()]
        cleaned = [entry for entry in entries if not (entry.startswith("[") and entry.endswith("]"))]
        if cleaned != entries:
            os.environ[name] = ",".join(cleaned)


_sanitize_proxy_environment()


@pytest.fixture(scope="session", autouse=True)
def consolidation_evidence_bundle(request):
    """T094: real 013 acceptance-evidence collection, gated on the environment.

    Created for every run so a whole-suite 013 acceptance run exports evidence,
    but it only writes files when ``CONSOLIDATION_EVIDENCE_DIR`` is configured
    (the 012 observer's ``--memory-evidence`` caliber). The per-test
    ``consolidation_evidence`` fixture always works in memory, so a test can
    assert on real observations with or without an export directory.
    """
    target = os.environ.get("CONSOLIDATION_EVIDENCE_DIR")
    if not target:
        yield None
        return
    from tests.integration.consolidation_evidence import EvidenceBundle

    bundle = EvidenceBundle(target)
    request.config._consolidation_evidence_bundle = bundle
    yield bundle


@pytest.fixture(scope="session", autouse=True)
def memory_eval_evidence_bundle(request):
    """T009: real 015 per-case evidence collection, gated on the environment.

    Strictly isomorphic with ``consolidation_evidence_bundle`` above: created for
    every run so a whole-suite 015 run exports evidence, but it only writes files
    when ``MEMORY_EVAL_EVIDENCE_DIR`` is configured. Unset means a lazy no-op.
    """
    target = os.environ.get("MEMORY_EVAL_EVIDENCE_DIR")
    if not target:
        yield None
        return
    from tests.integration.memory_eval_evidence import MemoryEvalEvidenceBundle

    bundle = MemoryEvalEvidenceBundle(target)
    request.config._memory_eval_evidence_bundle = bundle
    yield bundle


@pytest.fixture
def memory_eval_evidence(request):
    """Per-test accessor for the 015 evidence bundle (None when not configured)."""
    return request.getfixturevalue("memory_eval_evidence_bundle")


@pytest.fixture
def consolidation_evidence(request, monkeypatch):
    """Per-test recorder for real 013 observations and observed hard counts.

    When an export directory is configured the real memory services are also
    observed with the 012 observer caliber; the wrapped production call is
    returned unchanged, so no test outcome depends on the observation.
    """
    from tests.integration.consolidation_evidence import EvidenceObserver, instrument

    bundle = request.getfixturevalue("consolidation_evidence_bundle")
    observer = EvidenceObserver(bundle, request.node.nodeid)
    if bundle is not None:
        instrument(monkeypatch, observer)
    return observer


def pytest_sessionfinish(session, exitstatus):
    bundle = getattr(session.config, "_consolidation_evidence_bundle", None)
    if bundle is not None:
        bundle.finalize(int(exitstatus))
    # T009: additive 015 export. The 013 branch above is untouched, and an unset
    # MEMORY_EVAL_EVIDENCE_DIR means no bundle was ever created.
    memory_bundle = getattr(session.config, "_memory_eval_evidence_bundle", None)
    if memory_bundle is not None:
        memory_bundle.finalize(int(exitstatus))


@pytest_asyncio.fixture(autouse=True)
async def global_database_runtime_lifetime():
    from rag_mcp.db import dispose_engine

    yield
    # The process-wide pool must be closed before this test's loop is closed.
    await dispose_engine()


@pytest.fixture(autouse=True)
def _reset_provider_bundle_between_tests():
    """Review Focus #5：缓存不得在用例间泄漏。"""
    from rag_mcp.providers.factory import reset_provider_bundle

    reset_provider_bundle()
    yield
    reset_provider_bundle()


@pytest_asyncio.fixture
async def engine():
    """Create async engine connected to database."""
    settings = get_settings()
    eng = create_async_engine(settings.database_url, echo=False)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test session."""
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def memory_writer_owner(engine):
    from tests.integration.memory_acceptance import writer_owner
    async with writer_owner(engine) as owner:
        yield owner


@pytest_asyncio.fixture
async def test_client(engine):
    """Create an async test client with DB session dependency override."""
    from httpx import ASGITransport, AsyncClient

    from rag_mcp.api.knowledge_sources import get_session as ks_get_session
    from rag_mcp.api.projects import get_session as projects_get_session
    from rag_mcp.api.runtime_metrics import get_session as metrics_get_session
    from rag_mcp.server import app

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[projects_get_session] = _get_session
    app.dependency_overrides[ks_get_session] = _get_session
    app.dependency_overrides[metrics_get_session] = _get_session

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()


@pytest_asyncio.fixture
def sample_markdown():
    """Sample Markdown content for parser tests."""
    return """# 项目概述

这是一个示例项目的文档。

## 安装指南

### 环境要求

- Python 3.12+
- PostgreSQL 16

### 安装步骤

1. 克隆仓库
2. 安装依赖：`pip install -r requirements.txt`
3. 配置数据库连接

## API 配置

数据库密码: password=MySecret123
API密钥: api_key=sk-abc123def456ghi789jkl012mno345
Token: bearer_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U

## 架构说明

系统采用前后端分离架构。
"""


@pytest_asyncio.fixture
def sample_java():
    """Sample Java source code for parser tests."""
    return '''package com.example.service;

import java.util.List;
import java.util.Optional;

/**
 * 用户服务类，处理用户相关业务逻辑。
 */
public class UserService {

    private static final String DB_PASSWORD = "SuperSecret456";
    private final UserRepository repository;

    public UserService(UserRepository repository) {
        this.repository = repository;
    }

    /**
     * 根据ID查找用户。
     * @param id 用户ID
     * @return 用户实体
     */
    public Optional<User> findById(Long id) {
        return repository.findById(id);
    }

    /**
     * 获取所有活跃用户。
     */
    public List<User> getActiveUsers() {
        return repository.findByStatus("active");
    }

    private void validateToken(String token) {
        // Token validation logic
        if (token == null || token.isEmpty()) {
            throw new IllegalArgumentException("Token required");
        }
    }
}
'''
