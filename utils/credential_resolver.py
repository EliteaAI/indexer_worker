"""Trusted credential resolver of a governed run (C-1009 WP-1D, EliteaAI/elitea_issues#6946).

A governed payload carries credential *slots*, never material. The SDK calls the resolver
attached to the run's Elitea client (``client.credential_resolver``) while it builds a toolkit;
the resolver asks Main for a release over the worker bus and opens the reply, which Main
sealed to a key generated for that one call (``credential_transport``).

Nothing here keeps material: one X25519 keypair per call whose private half is dropped as
soon as the reply is opened, no cache, no logging of values, no ids or reasons in user-facing
text. A refusal or failure is a ``GovernedCredentialError`` carrying a reason code only.
"""

import json
import threading
from datetime import datetime, timezone

from .context_guard import PREDICT_RUN_ID_KWARGS_KEY, _Counter, _log_info
from .task_envelope import read_envelope

RELEASE_RPC = "restricted_release_credential"
AAD_VERSION = "v1"
DEFAULT_TIMEOUT_S = 15

DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
STALE_CONTEXT = "STALE_CONTEXT"

CREDENTIAL_SLOT_RESOLUTION = _Counter(
    "credential_slot_resolution_total",
    "Governed credential slot resolutions by this worker, by result and reason",
    ("result", "reason_code"),
)


def _record(result, reason_code):
    CREDENTIAL_SLOT_RESOLUTION.inc(result, reason_code)
    _log_info(f"security_metric name=credential_slot_resolution_total result={result} reason_code={reason_code}")


def governed_credential_code(exc):
    """Reason code of a ``GovernedCredentialError`` (matched by shape: class identity can change on SDK reload)."""
    code = getattr(exc, "reason_code", None)
    if type(exc).__name__ == "GovernedCredentialError" and isinstance(code, str) and code:
        return code
    return None


def _sdk_types():
    from elitea_sdk.runtime.credential_slots import CredentialEnvelope, GovernedCredentialError  # pylint: disable=C0415,E0401
    return CredentialEnvelope, GovernedCredentialError


def _parse_expires_at(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expires_at")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class TrustedCredentialResolver:
    """Callable ``resolver(*, toolkit_instance_id, field, slot) -> CredentialEnvelope`` for one task.

    ``context_ref`` set: a governed agent run. Unset: a probe, whose ticket is the ``probe_ref``
    carried by the slot. One ``RpcNode`` serves every call of the task; ``close()`` stops it.
    """

    def __init__(self, local_event_node, *, project_id, task_name, context_ref=None, run_id=None,
                 timeout_s=DEFAULT_TIMEOUT_S):
        self._event_node = local_event_node
        self._project_id = project_id
        self._task_name = task_name
        self._context_ref = context_ref
        self._run_id = run_id
        self._timeout_s = timeout_s
        self._node = None
        self._lock = threading.Lock()

    # A resolver is shared by reference with whatever copies the client; it never pickles.
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        raise TypeError("TrustedCredentialResolver must not be pickled")

    def __repr__(self):
        return f"TrustedCredentialResolver(task={self._task_name})"

    def close(self):
        with self._lock:
            node, self._node = self._node, None
        if node is not None:
            try:
                node.stop()
            except Exception:  # pylint: disable=W0718
                pass

    def _rpc(self):
        with self._lock:
            if self._node is None:
                # A forked task has no running RPC response listener of its own; one per run.
                from arbiter.rpcnode import RpcNode  # pylint: disable=C0415,E0401
                node = RpcNode(self._event_node, id_prefix="indexer_rel_", proxy_timeout=self._timeout_s)
                node.start()
                self._node = node
            return self._node

    def _refusal(self, error_type, code):
        _record("deny", code)
        return error_type(code)

    def __call__(self, *, toolkit_instance_id, field, slot):
        envelope_type, error_type = _sdk_types()
        context_ref = self._context_ref
        probe_ref = None
        if context_ref is None:
            probe_ref = slot.get("probe_ref") if isinstance(slot, dict) else None
            if not isinstance(probe_ref, str) or not probe_ref:
                raise self._refusal(error_type, STALE_CONTEXT)
        from .credential_transport import generate_recipient_keypair  # pylint: disable=C0415
        private_key, recipient_key = generate_recipient_keypair()
        try:
            try:
                reply = self._rpc().proxy.restricted_release_credential(
                    project_id=self._project_id,
                    context_ref=context_ref,
                    probe_ref=probe_ref,
                    caller={"kind": "indexer_worker", "task_name": self._task_name, "run_id": self._run_id},
                    toolkit_instance_id=toolkit_instance_id,
                    field=field,
                    slot=slot,
                    recipient_key=recipient_key,
                )
            except Exception:  # pylint: disable=W0718  # includes queue.Empty (proxy timeout)
                raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE) from None
            if not isinstance(reply, dict):
                raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE)
            if reply.get("ok") is not True:
                code = reply.get("reason_code")
                raise self._refusal(error_type, code if isinstance(code, str) and code else DEPENDENCY_UNAVAILABLE)
            result = self._open(reply, private_key, envelope_type, error_type, context_ref or probe_ref,
                                toolkit_instance_id, field, slot)
        finally:
            del private_key
        _record("ok", "OK")
        return result

    def _open(self, reply, private_key, envelope_type, error_type, ref, toolkit_instance_id, field, slot):
        release_id = reply.get("release_id")
        if not isinstance(release_id, str) or not release_id:
            raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE)
        from .credential_transport import TransportError, open_sealed  # pylint: disable=C0415
        decision_id = slot.get("decision_id") if isinstance(slot, dict) else None
        aad = "|".join((
            AAD_VERSION, str(self._project_id), ref, toolkit_instance_id, field,
            decision_id if isinstance(decision_id, str) else "", release_id,
        )).encode("utf-8")
        try:
            plaintext = open_sealed(reply.get("sealed"), private_key, aad)
            body = json.loads(plaintext)
            del plaintext
            fields, generation = body["fields"], body["generation"]
            expires_at = _parse_expires_at(body.get("expires_at"))
            if (
                    not isinstance(fields, dict) or isinstance(generation, bool) or not isinstance(generation, int)
                    or not all(isinstance(k, str) and isinstance(v, str) for k, v in fields.items())
                    or (isinstance(reply.get("fields"), list) and set(reply["fields"]) != set(fields))
            ):
                raise ValueError("body")
            envelope = envelope_type(fields, generation, expires_at)
        except (TransportError, ValueError, KeyError, TypeError):
            # Never chain: the original text could echo the opened body.
            raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE) from None
        finally:
            body = fields = None  # noqa: F841  # pylint: disable=W0642
        if expires_at is not None and expires_at <= datetime.now(timezone.utc):
            raise self._refusal(error_type, STALE_CONTEXT)
        return envelope


def attach_credential_resolver(client, context_view, kwargs, local_event_node, *, task_name, project_id):
    """Attach the resolver to a governed run's client; return it (the caller closes it), or ``None`` for a legacy run.

    ``context_view`` is what ``enforce_execution_context`` returned: ``None`` means legacy, and a
    legacy client is left exactly as it was (no attribute).
    """
    if context_view is None:
        return None
    resolver = TrustedCredentialResolver(
        local_event_node,
        project_id=project_id if project_id is not None else kwargs.get("project_id"),
        task_name=task_name,
        context_ref=read_envelope(kwargs).context_ref,
        run_id=kwargs.get(PREDICT_RUN_ID_KWARGS_KEY),
    )
    client.credential_resolver = resolver
    return resolver
