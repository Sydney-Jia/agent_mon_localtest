"""Explicit synchronous observation context. No frame/stack inspection."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, asdict
import uuid

_context = ContextVar("agentmon_kv_context", default=None)
_deletion = ContextVar("agentmon_kv_deletion", default="UNKNOWN")


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    decision_id: str
    attempt_id: str | None = None
    model_call_id: str | None = None
    http_attempt: int | None = None
    context_version: str | None = None
    batch_id: str | None = None

    def message(self):
        if not self.request_id or not self.decision_id:
            raise ValueError("EXPLICIT_REQUEST_DECISION_REQUIRED")
        return asdict(self)


@contextmanager
def request_scope(context):
    if isinstance(context, dict):
        context = RequestContext(**context)
    context.message()
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def current():
    return _context.get()


def for_request(request, **links):
    return RequestContext(request.request_id, uuid.uuid4().hex, **links)


@contextmanager
def deletion_scope(reason):
    if reason not in ("CAPACITY_REASSIGNMENT", "EXPLICIT_INVALIDATION", "RESET",
                      "FAULT_INVALIDATION", "ALIAS_REPLACEMENT", "UNKNOWN"):
        raise ValueError("UNREVIEWED_DELETION_REASON")
    token = _deletion.set(reason)
    try:
        yield
    finally:
        _deletion.reset(token)


def deletion_reason():
    return _deletion.get()
