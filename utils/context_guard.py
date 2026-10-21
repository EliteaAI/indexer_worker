"""Execution-context guard (C-1009, EliteaAI/elitea_issues#6756).

A governed task carries ``envelope.context_ref`` stamped by Main's admission. Before any LLM or
tool work the worker validates that reference with Main over the worker bus and refuses the run
on any failure. A legacy envelope makes no call at all.
"""

import threading

from elitea_sdk.runtime.utils.utils import PREDICT_RUN_ID_KWARGS_KEY  # pylint: disable=E0401

from .task_envelope import PROFILE_GOVERNED, EnvelopeRejected, read_envelope

STALE_CONTEXT = "STALE_CONTEXT"
DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"

# Fixed, value-free text shown to the user for each refusal code.
REJECTION_MESSAGES = {
    STALE_CONTEXT: "This run is no longer valid. Please start it again.",
    "EPOCH_MISMATCH": "Access for this run was revoked before it started. Please start it again.",
    "BINDING_DIGEST_MISMATCH": "The agent changed after this run was approved. Please start it again.",
    "DENIED_BINDING": "This agent is not approved to run in this context.",
    DEPENDENCY_UNAVAILABLE: "The run could not be authorized right now. Please try again shortly.",
    "UNKNOWN_SCHEMA_VERSION": "This run uses a security context version this worker does not support.",
    # Governed credential release (#6946). Fixed per code: no titles, ids or field names.
    "CREDENTIAL_REFERENCE_REQUIRED": "A governed credential reference is required; the run was not started.",
    "CREDENTIAL_REFERENCE_INACTIVE": "The credential reference is not active; the run was not started.",
    "CREDENTIAL_GENERATION_MISMATCH": "The credential changed since the run was approved; the run was not started.",
    "DENIED_GRANT": "The credential grant is no longer in force; the run was not started.",
    "DENIED_RBAC": "Permission to use this credential was denied; the run was not started.",
    "MANDATE_INACTIVE": "The mandate for this run is not active; the run was not started.",
    "MANDATE_EXPIRED": "The mandate for this run has expired; the run was not started.",
    # Autonomous runs (#6947): the mandate's runtime limit, enforced by Main's deadline sweep.
    "RUNTIME_LIMIT_EXCEEDED": "The run exceeded the time its mandate allows and was stopped.",
    "RATE_LIMITED": "This run asked for its credentials too often and was stopped.",
    "AUDIT_UNAVAILABLE": "Auditing is unavailable; the run was not started.",
    "WORKER_CAPABILITY_UNAVAILABLE": "This worker cannot apply governed credentials; the run was not started.",
    "UNSUPPORTED_AUTH_MODE": "This toolkit cannot use a governed credential yet; the run was not started.",
    **dict.fromkeys(
        ("STORE_DENIED", "STORE_SEALED", "STORE_UNAVAILABLE", "STORE_TLS_INVALID", "STORE_NOT_FOUND",
         "STORE_VERSION_CONFLICT"),
        "The credential store is unavailable; the run was not started. Try again later.",
    ),
}
DEFAULT_REJECTION_MESSAGE = "This run was not authorized."

DEFAULT_TIMEOUT_S = 5
REF_LOG_CHARS = 8


class ContextRejected(EnvelopeRejected):
    """Main refused the execution context, or could not be asked."""


def rejection_message(reason_code):
    return REJECTION_MESSAGES.get(reason_code, DEFAULT_REJECTION_MESSAGE)


class _Counter:
    """prometheus_client Counter when installed, always an in-process count as well."""

    def __init__(self, name, documentation, labels):
        self._counts = {}
        self._prom = None
        try:
            from prometheus_client import Counter  # pylint: disable=C0415,E0401
            self._prom = Counter(name, documentation, list(labels))
        except Exception:  # pylint: disable=W0718
            pass

    def inc(self, *values):
        self._counts[values] = self._counts.get(values, 0) + 1
        if self._prom is not None:
            self._prom.labels(*values).inc()

    def total(self, *values):
        if not values:
            return sum(self._counts.values())
        return self._counts.get(values, 0)


CONTEXT_VALIDATION = _Counter(
    "context_validation_total",
    "Execution-context validations by this worker, by result and reason",
    ("result", "reason_code"),
)


def _log_info(message):
    try:
        from pylon.core.tools import log  # pylint: disable=C0415,E0401
        log.info(message)
    except Exception:  # pylint: disable=W0718
        import logging  # pylint: disable=C0415
        logging.getLogger(__name__).info(message)


def _record(result, reason_code):
    CONTEXT_VALIDATION.inc(result, reason_code)
    _log_info(f"security_metric name=context_validation_total result={result} reason_code={reason_code}")


def _reject(reason_code, ref=None):
    _record("deny", reason_code)
    if ref:
        _log_info(f"execution context refused: {reason_code} ref={str(ref)[:REF_LOG_CHARS]}")
    raise ContextRejected(reason_code)


RUN_RPC_TIMEOUT_S = 15  # transport default of a run's shared node; every call passes its own timeout


class RunRpcNode:
    """The one ``RpcNode`` of a governed run, shared by context validation and credential release (S9).

    Created and started on first use, so a legacy run (which never calls Main) creates none. The
    owner of the run calls ``close()`` once, in the task's ``finally``. A forked task has no running
    RPC response listener of its own: the node starts one over the already-started Worker/Main
    transport, as ``agent_execution_common`` does for signing.
    """

    def __init__(self, local_event_node, *, id_prefix="indexer_run_", proxy_timeout=RUN_RPC_TIMEOUT_S):
        self._event_node = local_event_node
        self._id_prefix = id_prefix
        self._proxy_timeout = proxy_timeout
        self._node = None
        self._lock = threading.Lock()

    def proxy(self, timeout_s):
        """RPC proxy of the shared node with this call's own timeout."""
        with self._lock:
            if self._node is None:
                from arbiter.rpcnode import RpcNode  # pylint: disable=C0415,E0401
                node = RpcNode(self._event_node, id_prefix=self._id_prefix, proxy_timeout=self._proxy_timeout)
                node.start()
                self._node = node
            node = self._node
        return node.timeout(timeout_s)

    def close(self):
        with self._lock:
            node, self._node = self._node, None
        if node is not None:
            try:
                node.stop()
            except Exception:  # pylint: disable=W0718
                pass


def _validate_with_main(local_event_node, timeout_s, rpc=None, **payload):
    if rpc is not None:
        return rpc.proxy(timeout_s).restricted_validate_execution_context(**payload)
    # No shared node: one for this call only.
    from arbiter.rpcnode import RpcNode  # pylint: disable=C0415,E0401
    node = RpcNode(local_event_node, id_prefix="indexer_ctx_", proxy_timeout=timeout_s)
    node.start()
    try:
        return node.proxy.restricted_validate_execution_context(**payload)
    finally:
        node.stop()


def enforce_execution_context(kwargs, local_event_node, *, task_name, timeout_s=DEFAULT_TIMEOUT_S, project_id=None,
                              rpc=None):
    """Return Main's context view for a governed run, ``None`` for a legacy one; raise ``ContextRejected``.

    ``project_id`` is the run project Main stamped in the task meta; the payload's own is the fallback.
    ``rpc`` is the run's shared ``RunRpcNode`` (the caller closes it); without one the call uses a node of its own.
    """
    envelope = read_envelope(kwargs)
    if envelope.security_profile != PROFILE_GOVERNED:
        return None
    ref = envelope.context_ref
    project_id = project_id if project_id is not None else kwargs.get("project_id")
    if not ref or project_id is None:
        _reject(STALE_CONTEXT)
    try:
        response = _validate_with_main(
            local_event_node, timeout_s, rpc,
            project_id=project_id,
            context_ref=ref,
            caller={
                "kind": "indexer_worker",
                "task_name": task_name,
                "run_id": kwargs.get(PREDICT_RUN_ID_KWARGS_KEY),
            },
        )
    except Exception:  # pylint: disable=W0718  # includes queue.Empty (proxy timeout)
        _reject(DEPENDENCY_UNAVAILABLE, ref)
    if isinstance(response, dict) and response.get("ok") is True and isinstance(response.get("context"), dict):
        _record("ok", "OK")
        return response["context"]
    code = response.get("reason_code") if isinstance(response, dict) else None
    _reject(code if isinstance(code, str) and code else STALE_CONTEXT, ref)

