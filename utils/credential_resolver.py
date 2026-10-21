"""Trusted credential resolver of a governed run (C-1009 WP-1D, EliteaAI/elitea_issues#6946).

A governed payload carries credential *slots*, never material. The SDK calls the resolver
attached to the run's Elitea client (``client.credential_resolver``) while it builds a toolkit;
the resolver asks Main for a release over the worker bus and opens the reply, which Main
sealed to a key generated for that one call (``credential_transport``).

``resolve_many`` releases every slot of a toolkit build in one bus call (one keypair, one reply,
each item sealed under its own AAD); ``__call__`` is the one-slot form for SDKs that resolve slot
by slot. Main decides: the batch call is used only when its validate reply said ``batch_release``
(``ContextView.batch_release``); otherwise, and for probes, every slot is its own call. A bus failure
of either call is ``DEPENDENCY_UNAVAILABLE``, never a reason to try the other form.

Nothing here keeps material: one X25519 keypair per call whose private half is dropped as
soon as the reply is opened, no cache, no logging of values, no ids or reasons in user-facing
text. A refusal or failure is a ``GovernedCredentialError`` carrying a reason code only.
"""

import json
from datetime import datetime, timezone

from .context_guard import PREDICT_RUN_ID_KWARGS_KEY, RunRpcNode, _Counter, _log_info
from .task_envelope import read_envelope

AAD_VERSION = "v1"
DEFAULT_TIMEOUT_S = 15
MAX_BATCH_SLOTS = 64  # Main's MAX_GOVERNED_CREDENTIALS_PER_RUN
BATCH_TIMEOUT_PER_SLOT_S = 0.5
MAX_BATCH_TIMEOUT_S = 60
RELEASE_REPLY_KEYS = frozenset({"ok", "releases"})
RELEASE_ITEM_KEYS = frozenset({"toolkit_instance_id", "field", "release_id", "fields", "expires_at", "sealed"})

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


def batch_timeout(base_s, slots):
    """Bus timeout of a release batch: the single-call timeout plus half a second per slot, at most a minute."""
    return min(base_s + BATCH_TIMEOUT_PER_SLOT_S * slots, MAX_BATCH_TIMEOUT_S)


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
    """``resolver.resolve_many(*, toolkit_instance_id, items) -> {field: CredentialEnvelope}`` for one task, and
    the one-slot form ``resolver(*, toolkit_instance_id, field, slot) -> CredentialEnvelope``.

    ``context_ref`` set: a governed agent run. Unset: a probe, whose ticket is the ``probe_ref``
    carried by the slot. One ``RpcNode`` serves every call of the task: the run's shared ``rpc``
    (its owner closes it), or one of its own that ``close()`` stops.
    """

    def __init__(self, local_event_node, *, project_id, task_name, context_ref=None, run_id=None,
                 timeout_s=DEFAULT_TIMEOUT_S, rpc=None, batch_release=False):
        self._project_id = project_id
        self._task_name = task_name
        self._context_ref = context_ref
        self._run_id = run_id
        self._timeout_s = timeout_s
        self._owns_rpc = rpc is None
        self._batch_release = batch_release is True  # Main said so in its validate reply
        self._shared_rpc = RunRpcNode(local_event_node, id_prefix="indexer_rel_") if rpc is None else rpc

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
        if self._owns_rpc:
            self._shared_rpc.close()

    def _rpc(self, timeout_s=None):
        return self._shared_rpc.proxy(self._timeout_s if timeout_s is None else timeout_s)

    def _refusal(self, error_type, code):
        _record("deny", code)
        return error_type(code)

    def __call__(self, *, toolkit_instance_id, field, slot):
        """One slot (SDKs that resolve slot by slot): a one-item batch when Main serves batches."""
        return self.resolve_many(toolkit_instance_id=toolkit_instance_id, items=[{"field": field, "slot": slot}])[field]

    def resolve_many(self, *, toolkit_instance_id, items):
        """Release every ``{field, slot}`` item of one toolkit build with one bus call; ``{field: envelope}``.

        All or nothing: the first refusal or unopenable item raises its typed ``GovernedCredentialError``.
        """
        envelope_type, error_type = _sdk_types()
        entries = self._entries(items, error_type)
        if not entries:
            return {}
        if self._batch_release:
            return self._release_batch(toolkit_instance_id, entries, envelope_type, error_type)
        return {
            field: self._release_one(toolkit_instance_id, field, slot, envelope_type, error_type)
            for field, slot in entries
        }

    def _entries(self, items, error_type):
        if not isinstance(items, (list, tuple)):
            raise self._refusal(error_type, STALE_CONTEXT)
        if len(items) > MAX_BATCH_SLOTS:  # a malformed request, not a missing reference
            raise self._refusal(error_type, STALE_CONTEXT)
        entries, seen = [], set()
        for item in items:
            field = item.get("field") if isinstance(item, dict) else None
            slot = item.get("slot") if isinstance(item, dict) else None
            if not isinstance(field, str) or not field or field in seen or not isinstance(slot, dict):
                raise self._refusal(error_type, STALE_CONTEXT)
            seen.add(field)
            entries.append((field, slot))
        return entries

    def _probe_ref(self, slot, error_type):
        probe_ref = slot.get("probe_ref") if isinstance(slot, dict) else None
        if not isinstance(probe_ref, str) or not probe_ref:
            raise self._refusal(error_type, STALE_CONTEXT)
        return probe_ref

    def _release_batch(self, toolkit_instance_id, entries, envelope_type, error_type):
        context_ref = self._context_ref
        probe_ref = None
        if context_ref is None:
            refs = {self._probe_ref(slot, error_type) for _field, slot in entries}
            if len(refs) != 1:  # one ticket per batch
                raise self._refusal(error_type, STALE_CONTEXT)
            (probe_ref,) = refs
        from .credential_transport import generate_recipient_keypair  # pylint: disable=C0415
        private_key, recipient_key = generate_recipient_keypair()  # one keypair for the batch
        try:
            try:
                reply = self._rpc(batch_timeout(self._timeout_s, len(entries))).restricted_release_credentials(
                    project_id=self._project_id,
                    context_ref=context_ref,
                    probe_ref=probe_ref,
                    caller={"kind": "indexer_worker", "task_name": self._task_name, "run_id": self._run_id},
                    recipient_key=recipient_key,
                    slots=[{"toolkit_instance_id": toolkit_instance_id, "field": field, "slot": slot}
                           for field, slot in entries],
                )
            except Exception:  # pylint: disable=W0718  # includes queue.Empty (proxy timeout): never a fallback
                raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE) from None
            if not isinstance(reply, dict):
                raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE)
            if reply.get("ok") is not True:
                code = reply.get("reason_code")
                raise self._refusal(error_type, code if isinstance(code, str) and code else DEPENDENCY_UNAVAILABLE)
            if not set(reply) <= RELEASE_REPLY_KEYS:
                raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE)
            releases = self._releases_by_field(reply.get("releases"), toolkit_instance_id, entries, error_type)
            ref = context_ref or probe_ref
            result = {
                field: self._open(releases[field], private_key, envelope_type, error_type, ref,
                                  toolkit_instance_id, field, slot)
                for field, slot in entries
            }
        finally:
            del private_key
        for _ in result:
            _record("ok", "OK")
        return result

    def _releases_by_field(self, releases, toolkit_instance_id, entries, error_type):
        """The reply's releases keyed by field: exactly the requested set, each for this toolkit, known keys only,
        no repeated field or release id."""
        if not isinstance(releases, list) or len(releases) != len(entries):
            raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE)
        by_field, release_ids = {}, set()
        for release in releases:
            if not isinstance(release, dict) or not set(release) <= RELEASE_ITEM_KEYS:
                raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE)
            field, release_id = release.get("field"), release.get("release_id")
            if (not isinstance(field, str) or field in by_field or release.get("toolkit_instance_id") != toolkit_instance_id
                    or not isinstance(release_id, str) or release_id in release_ids):
                raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE)
            by_field[field] = release
            release_ids.add(release_id)
        if set(by_field) != {field for field, _slot in entries}:
            raise self._refusal(error_type, DEPENDENCY_UNAVAILABLE)
        return by_field

    def _release_one(self, toolkit_instance_id, field, slot, envelope_type, error_type):
        """Single-slot release (``restricted_release_credential``): a Main without the batch handler."""
        context_ref = self._context_ref
        probe_ref = None
        if context_ref is None:
            probe_ref = self._probe_ref(slot, error_type)
        from .credential_transport import generate_recipient_keypair  # pylint: disable=C0415
        private_key, recipient_key = generate_recipient_keypair()
        try:
            try:
                reply = self._rpc().restricted_release_credential(
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


def attach_credential_resolver(client, context_view, kwargs, local_event_node, *, task_name, project_id, rpc=None):
    """Attach the resolver to a governed run's client; return it (the caller closes it), or ``None`` for a legacy run.

    ``rpc`` is the run's shared ``RunRpcNode`` (also used to validate the context); its owner closes it.

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
        rpc=rpc,
        batch_release=getattr(context_view, "batch_release", False),
    )
    client.credential_resolver = resolver
    return resolver
