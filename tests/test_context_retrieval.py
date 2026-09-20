"""Focused tests for the context retrieval registry (core seam).

These tests verify the byte-identical invariant: with no retrievers
registered, retrieve_context returns an empty list (no side effects).
"""
from __future__ import annotations

import pytest

from ouroboros.context_retrieval import (
    list_retrievers,
    register_retriever,
    retrieve_context,
    unregister_retriever,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a clean registry before and after each test."""
    for name in list_retrievers():
        unregister_retriever(name)
    yield
    for name in list_retrievers():
        unregister_retriever(name)


def test_empty_registry_returns_empty_list():
    """Byte-identical path: no retrievers -> empty list, no side effects."""
    result = retrieve_context("hello world")
    assert result == []
    assert list_retrievers() == []


def test_register_and_retrieve():
    """A registered retriever's blocks are returned."""
    calls = []

    def handler(query: str) -> list:
        calls.append(query)
        return [f"block1 for {query}", "block2"]

    register_retriever("test_skill", handler, description="test")
    result = retrieve_context("my query")
    assert calls == ["my query"]
    assert result == ["block1 for my query", "block2"]


def test_unregister_removes():
    """After unregister, the retriever is gone."""
    register_retriever("tmp", lambda q: ["x"])
    assert "tmp" in list_retrievers()
    unregister_retriever("tmp")
    assert "tmp" not in list_retrievers()
    assert retrieve_context("q") == []


def test_unregister_nonexistent_is_noop():
    """Unregistering a name that was never registered does not raise."""
    unregister_retriever("never_existed")  # must not raise


def test_failing_retriever_is_skipped():
    """A retriever that raises is caught; other retrievers still work."""
    def bad_handler(q: str) -> list:
        raise RuntimeError("simulated Qdrant down")

    def good_handler(q: str) -> list:
        return ["survivor"]

    register_retriever("bad", bad_handler)
    register_retriever("good", good_handler)
    result = retrieve_context("q")
    assert result == ["survivor"]


def test_multiple_retrievers_combined():
    """Blocks from multiple retrievers are concatenated in registration order."""
    register_retriever("a", lambda q: ["from_a"])
    register_retriever("b", lambda q: ["from_b1", "from_b2"])
    result = retrieve_context("q")
    assert result == ["from_a", "from_b1", "from_b2"]


def test_empty_blocks_filtered():
    """Empty-string blocks from a retriever are filtered out."""
    register_retriever("sparse", lambda q: ["real", "", None, "  "])
    result = retrieve_context("q")
    assert result == ["real", "  "]  # "  " is truthy; "" and None are falsy


def test_reregister_replaces():
    """Re-registering the same name replaces the handler."""
    register_retriever("x", lambda q: ["v1"])
    register_retriever("x", lambda q: ["v2"])
    assert retrieve_context("q") == ["v2"]


def test_idempotent_register():
    """Registering the same name twice does not duplicate."""
    register_retriever("idempotent", lambda q: ["one"])
    register_retriever("idempotent", lambda q: ["one"])
    assert list_retrievers().count("idempotent") == 1
    assert retrieve_context("q") == ["one"]


def test_thread_safety_smoke():
    """Concurrent register/retrieve/unregister does not crash (smoke test)."""
    import threading as th
    errors = []

    def worker(i):
        try:
            register_retriever(f"w{i}", lambda q, _i=i: [f"from_{_i}"])
            retrieve_context("q")
            unregister_retriever(f"w{i}")
        except Exception as e:
            errors.append(e)

    threads = [th.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert retrieve_context("q") == []
