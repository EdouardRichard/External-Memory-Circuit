"""生产代码不得再直接实例化本地 Provider（在线适配，SC-2/SC-3）."""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "rag_mcp"

# Provider 定义文件与 local_gpu 子类例外：它们本来就要引用本地实现
_ALLOWED = {
    "providers/local_cpu.py",
    "providers/local_cpu_reranker.py",
    "providers/local_gpu.py",
    "providers/factory.py",
}

_SUSPECT = {"LocalCPUEmbeddingProvider", "LocalCPUReranker", "LocalGPUEmbeddingProvider"}


def _violations() -> list[str]:
    found = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in _SUSPECT:
                # 仅 Name 引用（实例化/导入）算违规
                found.append(f"{rel}:{node.lineno}:{node.id}")
    return found


def test_no_local_provider_instantiation_in_production_code() -> None:
    violations = _violations()
    assert violations == [], f"生产代码仍引用本地 Provider: {violations}"


def test_recall_config_hash_uses_active_model(monkeypatch) -> None:
    """RAG 门闸指纹必须取生效模型，而不是 settings.embedding_model（Review Focus #1）."""
    from types import SimpleNamespace

    from rag_mcp.config import get_settings
    from rag_mcp.services.consolidation_gate import recall_config_hash

    settings = get_settings()
    legacy = "legacy-settings-embedding-model"      # 故意与 provider 值不同

    def fake(provider_model: str):
        return SimpleNamespace(
            embedding_model=legacy,                  # settings.embedding_model 保持不变
            providers=SimpleNamespace(embedding=SimpleNamespace(model=provider_model)),
        )

    # embedding_model 相同、provider 模型不同 → 指纹必须不同（证明取的是 provider 值）
    assert recall_config_hash(fake("model-a")) != recall_config_hash(fake("model-b"))
    # provider 模型相同、embedding_model 相同 → 指纹相同（确定性）
    assert recall_config_hash(fake("model-a")) == recall_config_hash(fake("model-a"))
