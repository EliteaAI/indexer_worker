"""Worker side of the versioned task envelope (C-1009, EliteaAI/elitea_issues#6940).

A governed task states which capabilities its worker must have. The node-level approver
makes a worker that lacks them stay silent, so another worker can take the task; the
entry-point guard re-validates because an approver's failure is silent and a task may also
arrive by a path that never queried approvers.
"""

import threading
import time

from .security_metrics_client import SecurityCounter
from .task_envelope import (
    CAPABILITY_ENVELOPE_V1, EnvelopeRejected, read_envelope,
)

SUPPORTED_CAPABILITIES = frozenset({CAPABILITY_ENVELOPE_V1})

CAPABILITIES_EVENT = "task_node_capabilities"
CAPABILITIES_REQUEST_EVENT = "task_node_capabilities_request"
DEFAULT_HEARTBEAT_S = 30


WORKER_CAPABILITY_REJECT = SecurityCounter(
    "worker_capability_reject_total",
    "Tasks refused by this worker because its envelope could not be honoured",
    ("capability",),
)


def envelope_task_approver():
    """Arbiter node-level approver: False when the envelope is not one this worker honours.

    Non-blocking and exception-safe by construction (it is called while Main holds its
    start lock); a payload without an envelope is approved, as before. Arbiter's
    ``task_start_query`` carries no kwargs, so the decision happens at ``task_start_request``;
    a declined request is not acknowledged and the requestor tries the next candidate.
    """
    def _approver(event_name, event_payload):
        _ = event_name
        kwargs = (event_payload or {}).get("kwargs") or {}
        try:
            read_envelope(kwargs, SUPPORTED_CAPABILITIES)
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


def enforce_envelope(kwargs):
    """Entry-point guard: raise ``EnvelopeRejected`` (and count it) for a refused envelope."""
    try:
        read_envelope(kwargs or {}, SUPPORTED_CAPABILITIES)
    except EnvelopeRejected as exc:
        for capability in exc.missing or ("-",):
            WORKER_CAPABILITY_REJECT.inc(capability=capability)
        raise


def rejection_result(exc):
    """The error shape of tasks that report failures as a result dict."""
    missing = f" ({', '.join(exc.missing)})" if exc.missing else ""
    return {
        "success": False,
        "error": f"Task rejected by worker: {exc.reason_code}{missing}",
        "reason_code": exc.reason_code,
    }


def capabilities_payload(ident, pool, now=None):
    return {
        "ident": ident,
        "pool": pool,
        "capabilities": sorted(SUPPORTED_CAPABILITIES),
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
