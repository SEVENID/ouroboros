"""Context retrieval registry: optional pre-LLM context augmentation.

This module provides a generic, off-by-default mechanism for registered
providers (typically skills) to augment the LLM context before dispatch.
When no provider is registered (the default state), all functions are
no-ops that return empty results -- the core path is byte-identical.

Design invariants:
- Thread-safe (RLock).
- No external dependencies (stdlib only).
- Any provider failure is caught, logged, and skipped -- never crashes the loop.
- Registration/unregistration are idempotent.
- The registry is process-local: out-of-process skill extensions that register
  in a child process are invisible to the parent loop (graceful no-op).

Usage (in a skill's plugin.py, in-process only):
    from ouroboros.context_retrieval import register_retriever, unregister_retriever

    def register(api):
        api.register_tool("vector_search", handler=..., ...)
        register_retriever("vector_memory", handler=my_retrieve, description="Qdrant RAG")
        api.on_unload(lambda: unregister_retriever("vector_memory"))
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List

__all__ = ["register_retriever", "unregister_retriever", "list_retrievers", "retrieve_context", "last_user_query"]

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_retrievers: Dict[str, Dict[str, Any]] = {}


def register_retriever(
    name: str,
    handler: Callable[[str], List[str]],
    *,
    description: str = "",
    timeout_sec: int = 10,
) -> None:
    """Register a context retrieval handler under the given name.

    The handler receives a query string and returns a list of context blocks
    (each a non-empty string). An empty list means "no relevant context".
    Re-registering with the same name replaces the previous handler.
    """
    with _lock:
        _retrievers[name] = {
            "handler": handler,
            "timeout_sec": max(1, int(timeout_sec)),
            "description": str(description or ""),
        }
    logger.debug("context_retrieval: registered retriever '%s'", name)


def unregister_retriever(name: str) -> None:
    """Remove a previously registered retriever. No-op if not present."""
    with _lock:
        _retrievers.pop(name, None)
    logger.debug("context_retrieval: unregistered retriever '%s'", name)


def list_retrievers() -> List[str]:
    """Return the names of all currently registered retrievers."""
    with _lock:
        return list(_retrievers.keys())


def retrieve_context(query: str, *, task_id: str = "") -> List[str]:
    """Call all registered retrievers with the given query.

    Returns a combined list of context blocks from all successful retrievers.
    On any retriever failure, logs a warning and skips that retriever.
    Returns an empty list when no retrievers are registered (byte-identical path).
    """
    with _lock:
        snapshot = list(_retrievers.items())
    if not snapshot:
        return []
    blocks: List[str] = []
    for name, entry in snapshot:
        try:
            result = entry["handler"](query)
            if result:
                blocks.extend(b for b in result if b)
        except Exception:
            logger.warning(
                "context_retrieval: retriever '%s' failed (task=%s); skipping",
                name, task_id or "?", exc_info=True,
            )
    return blocks


def last_user_query(messages: list, *, max_chars: int = 2000) -> str:
    """Extract the text of the last user-role message, truncated to max_chars.

    Handles both string content and multimodal (list-of-parts) content.
    Returns an empty string when no user message is present.
    """
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text") or "")
            text = " ".join(parts)
        else:
            text = ""
        text = (text or "").strip()
        if text:
            return text[:max_chars]
    return ""
