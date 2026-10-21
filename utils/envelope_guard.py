"""Worker side of the versioned task envelope (C-1009, EliteaAI/elitea_issues#6940).

A governed task states which capabilities its worker must have. The node-level approver
makes a worker that lacks them stay silent, so another worker can take the task; the
entry-point guard re-validates because an approver's failure is silent and a task may also
arrive by a path that never queried approvers.
"""

import threading
import time

from .credential_slots_support import (
    CAPABILITY_CREDENTIAL_SLOTS_BATCH_V1, CAPABILITY_CREDENTIAL_SLOTS_V1, slots_supported,
)
from .task_envelope import (
    CAPABILITY_ENVELOPE_V1, EnvelopeRejected, read_envelope,
)

SUPPORTED_CAPABILITIES = frozenset({CAPABILITY_ENVELOPE_V1})

CAPABILITIES_EVENT = "task_node_capabilities"
CAPABILITIES_REQUEST_EVENT = "task_node_capabilities_request"
DEFAULT_HEARTBEAT_S = 30


class _LabelledCounter:
    """prometheus_client Counter when installed, always an in-process count as well."""

    def __init__(self, name, documentation, label):
        self._counts = {}
        self._prom = None
        try:
            from prometheus_client import Counter  # pylint: disable=C0415,E0401
            self._prom = Counter(name, documentation, [label])
        except Exception:  # pylint: disable=W0718
            pass

    def inc(self, value):
        self._counts[value] = self._counts.get(value, 0) + 1
        if self._prom is not None:
            self._prom.labels(value).inc()

    def total(self, value=None):
        if value is None:
            return sum(self._counts.values())
        return self._counts.get(value, 0)


def supported_capabilities():
    """The envelope capabilities this worker honours: the base set, plus governed credential slots
    (and their one-call batch form) only when the SDK ships them and the sealed transport self-test passes (#6946)."""
    if slots_supported():
        return SUPPORTED_CAPABILITIES | {CAPABILITY_CREDENTIAL_SLOTS_V1, CAPABILITY_CREDENTIAL_SLOTS_BATCH_V1}
    return SUPPORTED_CAPABILITIES


WORKER_CAPABILITY_REJECT = _LabelledCounter(
    "worker_capability_reject_total",
    "Tasks refused by this worker because its envelope could not be honoured",
    "capability",
)


def envelope_task_approver(extra=()):
    """Arbiter node-level approver: False when the envelope is not one this worker honours.

    Non-blocking and exception-safe by construction (it is called while Main holds its
    start lock); a payload without an envelope is approved, as before. Arbiter's
    ``task_start_query`` carries no kwargs, so the decision happens at ``task_start_request``;
    a declined request is not acknowledged and the requestor tries the next candidate.
    ``extra`` adds capabilities that only this node has (the ``governed_ops`` pool).
    """
    def _approver(event_name, event_payload):
        _ = event_name
        kwargs = (event_payload or {}).get("kwargs") or {}
        try:
            read_envelope(kwargs, supported_capabilities() | frozenset(extra))
        except EnvelopeRejected as exc:
            try:
                from pylon.core.tools import log  # pylint: disable=C0415,E0401
                log.warning("task declined: envelope rejected (%s)", exc.reason_code)
            except Exception:  # pylint: disable=W0718
                pass
            return False
        return True
    #
    return _approver


def enforce_envelope(kwargs, extra=()):
    """Entry-point guard: raise ``EnvelopeRejected`` (and count it) for a refused envelope."""
    try:
        read_envelope(kwargs or {}, supported_capabilities() | frozenset(extra))
    except EnvelopeRejected as exc:
        for capability in exc.missing or ("-",):
            WORKER_CAPABILITY_REJECT.inc(capability)
        raise


def rejection_result(exc):
    """The error shape of tasks that report failures as a result dict."""
    missing = f" ({', '.join(exc.missing)})" if exc.missing else ""
    return {
        "success": False,
        "error": f"Task rejected by worker: {exc.reason_code}{missing}",
        "reason_code": exc.reason_code,
    }


def capabilities_payload(ident, pool, now=None, extra=()):
    return {
        "ident": ident,
        "pool": pool,
        "capabilities": sorted(supported_capabilities() | frozenset(extra)),
        "ts": time.time() if now is None else now,
    }


def start_capability_heartbeat(emit, interval_s=DEFAULT_HEARTBEAT_S):
    """Call ``emit()`` every ``interval_s`` on a daemon thread; return a stop function.

    Main drops capability entries it has not heard from within its TTL, so a worker has
    to keep re-announcing (arbiter's own announce is not periodic).
    """
    stopped = threading.Event()

    def _run():
        while not stopped.wait(interval_s):
            try:
                emit()
            except Exception:  # pylint: disable=W0718
                pass
    #
    emit()
    threading.Thread(target=_run, name="task-capabilities-heartbeat", daemon=True).start()
    return stopped.set
