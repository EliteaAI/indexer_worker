"""C-1009 WP-1D (#6946): the worker-side trusted resolver, capabilities and readable refusals.

The resolver is exercised against a fake worker bus whose "Main" seals its replies with the
shared transport, so the request/reply contract is checked end to end. Tests that need the
SDK's ``credential_slots`` module (``real_sdk_slots``) use the installed one and are skipped without it; set
ELITEA_CREDENTIAL_SLOTS_PATH to a checkout of ``credential_slots.py`` to run them before the SDK release carrying
it is pinned.
"""

import copy
import datetime
import gc
import importlib.util
import json
import os
import pathlib
import pickle
import queue
import sys
import types
import unittest.mock

import pytest

from utils import context_guard, credential_resolver, credential_slots_support, envelope_guard, governed_ops
from utils import credential_transport as transport
from utils.task_envelope import attach_envelope

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
SLOTS_MODULE = "elitea_sdk.runtime.credential_slots"
RUN_KEY = context_guard.PREDICT_RUN_ID_KWARGS_KEY
CONTEXT_REF = "0123456789abcdef-context"
PROBE_REF = "fedcba9876543210-probe"
VALUE = "canary-" + os.urandom(8).hex()  # runtime-assembled: nothing secret-shaped in the source


# --- fixtures -------------------------------------------------------------------------------

def _stand_in_slots_module():
    """Only the two types the resolver builds, as the SDK documents them (used when the SDK does not ship the module yet)."""
    from pydantic import SecretStr
    module = types.ModuleType(SLOTS_MODULE)

    class GovernedCredentialError(Exception):
        def __init__(self, reason_code):
            super().__init__(reason_code)
            self.reason_code = reason_code

    class CredentialEnvelope:
        def __init__(self, fields, generation, expires_at=None):
            self.fields = {k: v if isinstance(v, SecretStr) else SecretStr(v) for k, v in fields.items()}
            self.generation, self.expires_at = generation, expires_at

        def __repr__(self):
            return f"CredentialEnvelope(fields={sorted(self.fields)}, generation={self.generation})"

    module.GovernedCredentialError, module.CredentialEnvelope = GovernedCredentialError, CredentialEnvelope
    return module


@pytest.fixture
def sdk_slots(monkeypatch):
    """The SDK's credential_slots: installed, else loaded from ELITEA_CREDENTIAL_SLOTS_PATH, else a stand-in of the
    two types the resolver needs (tests that need the real module use ``real_sdk_slots``)."""
    try:
        return importlib.import_module(SLOTS_MODULE)
    except ImportError:
        pass
    path = os.environ.get("ELITEA_CREDENTIAL_SLOTS_PATH")
    if path:
        spec = importlib.util.spec_from_file_location(SLOTS_MODULE, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = _stand_in_slots_module()
    monkeypatch.setitem(sys.modules, SLOTS_MODULE, module)
    return module


@pytest.fixture
def real_sdk_slots(sdk_slots):
    if not hasattr(sdk_slots, "apply_credential_slots"):
        pytest.skip("installed SDK has no runtime.credential_slots (set ELITEA_CREDENTIAL_SLOTS_PATH)")
    return sdk_slots


@pytest.fixture(autouse=True)
def fresh_support_cache():
    credential_slots_support.slots_supported.cache_clear()
    yield
    credential_slots_support.slots_supported.cache_clear()


NO_OVERRIDE = object()


class FakeBus:
    """Stands in for the worker bus with Main's release handler behind it."""

    def __init__(self):
        self.requests = []        # single-slot releases (restricted_release_credential)
        self.batch_requests = []  # batch releases (restricted_release_credentials)
        self.batch_release = None    # what Main's validate reply says: True / False / None (key absent = older Main)
        self.validate_requests = []  # restricted_validate_execution_context
        self.nodes = []
        self.fields = {"access_token": VALUE}
        self.generation = 1
        self.expires_in = 300
        self.override = NO_OVERRIDE   # any reply or exception to return/raise instead of a real sealed reply
        self.seal_decision_id = None  # seal with a different decision id (a Main that disagrees)

    def _entry(self, request, release_id):
        """One sealed release for ``request`` (project_id, context_ref/probe_ref, toolkit_instance_id, field, slot, recipient_key)."""
        decision = self.seal_decision_id if self.seal_decision_id is not None else request["slot"].get("decision_id") or ""
        aad = "|".join((
            "v1", str(request["project_id"]), request["context_ref"] or request["probe_ref"],
            request["toolkit_instance_id"], request["field"], decision, release_id,
        )).encode()
        expires = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=self.expires_in))
        expires_at = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
        body = json.dumps({"fields": self.fields, "generation": self.generation, "expires_at": expires_at})
        return {
            "release_id": release_id, "fields": list(self.fields), "expires_at": expires_at,
            "sealed": transport.seal(body.encode(), request["recipient_key"], aad),
        }

    def validate(self, **request):
        self.validate_requests.append(request)
        reply = {"ok": True, "context": {"run_id": "run-1", "attempt": 1}}
        if self.batch_release is not None:
            reply["batch_release"] = self.batch_release
        return reply

    def release(self, **request):
        self.requests.append(request)
        if isinstance(self.override, BaseException):
            raise self.override
        if self.override is not NO_OVERRIDE:
            return self.override
        return {"ok": True, **self._entry(request, f"rel-{len(self.requests)}")}

    def release_many(self, **request):
        self.batch_requests.append(request)
        if isinstance(self.override, BaseException):
            raise self.override
        if self.override is not NO_OVERRIDE:
            return self.override
        releases = []
        for i, item in enumerate(request["slots"]):
            single = {**item, **{k: request[k] for k in ("project_id", "context_ref", "probe_ref", "recipient_key")}}
            releases.append({"toolkit_instance_id": item["toolkit_instance_id"], "field": item["field"],
                             **self._entry(single, f"rel-{len(self.batch_requests)}-{i}")})
        return {"ok": True, "releases": releases}


def _install_bus(monkeypatch, bus):
    class FakeRpc:
        def __init__(self, event_node, id_prefix=None, proxy_timeout=None):
            self.args = (event_node, id_prefix, proxy_timeout)
            self.started = self.stopped = False
            self.timeouts = []
            self.proxy = types.SimpleNamespace(
                restricted_release_credential=lambda **r: bus.release(**r),
                restricted_release_credentials=lambda **r: bus.release_many(**r),
                restricted_validate_execution_context=lambda **r: bus.validate(**r),
            )
            bus.nodes.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def timeout(self, seconds):
            self.timeouts.append(seconds)
            return self.proxy

    arbiter = types.ModuleType("arbiter")
    rpcnode = types.ModuleType("arbiter.rpcnode")
    rpcnode.RpcNode = FakeRpc
    arbiter.rpcnode = rpcnode
    monkeypatch.setitem(sys.modules, "arbiter", arbiter)
    monkeypatch.setitem(sys.modules, "arbiter.rpcnode", rpcnode)
    return bus


@pytest.fixture
def bus(monkeypatch):
    """A Main without the batch handler (its validate reply carries no ``batch_release``): these tests pin the
    single-slot request/reply contract. ``batch_bus`` is the Main that serves batches and says so."""
    return _install_bus(monkeypatch, FakeBus())


@pytest.fixture
def batch_bus(monkeypatch):
    bus = FakeBus()
    bus.batch_release = True
    return _install_bus(monkeypatch, bus)


@pytest.fixture
def logs(monkeypatch):
    lines = []
    monkeypatch.setattr(credential_resolver, "_log_info", lines.append)
    return lines


def _slot(**extra):
    return {"v": 1, "credential_ref": "ref-1", "credential_generation": 1,
            "decision_id": "decision-1", "fields": ["access_token"], **extra}


def _resolver(node="node", **overrides):
    params = {"project_id": 7, "task_name": "indexer_agent", "context_ref": CONTEXT_REF, "run_id": "run-1"}
    params.update(overrides)
    return credential_resolver.TrustedCredentialResolver(node, **params)


def _bresolver(node="node", **overrides):
    """A resolver whose run's validate reply said Main serves batches."""
    return _resolver(node, batch_release=True, **overrides)


def _call(resolver, slot=None, toolkit="12", field="github_configuration"):
    return resolver(toolkit_instance_id=toolkit, field=field, slot=slot or _slot())


def _code(excinfo):
    return excinfo.value.reason_code


# --- the request Main receives ---------------------------------------------------------------

def test_run_request_shape(bus, sdk_slots):
    _call(_resolver("the-node"))
    (request,) = bus.requests
    assert set(request) == {
        "project_id", "context_ref", "probe_ref", "caller", "toolkit_instance_id", "field", "slot", "recipient_key",
    }
    assert request["project_id"] == 7
    assert request["context_ref"] == CONTEXT_REF and request["probe_ref"] is None
    assert request["caller"] == {"kind": "indexer_worker", "task_name": "indexer_agent", "run_id": "run-1"}
    assert (request["toolkit_instance_id"], request["field"]) == ("12", "github_configuration")
    assert request["slot"] == _slot()
    assert len(__import__("base64").b64decode(request["recipient_key"])) == 32


def test_probe_request_takes_its_ticket_from_the_slot(bus, sdk_slots):
    resolver = _resolver(context_ref=None, run_id=None, task_name="indexer_governed_check_connection")
    envelope = _call(resolver, _slot(probe_ref=PROBE_REF, decision_id=None), toolkit="probe", field="configuration")
    (request,) = bus.requests
    assert request["context_ref"] is None and request["probe_ref"] == PROBE_REF
    assert request["caller"] == {"kind": "indexer_worker", "task_name": "indexer_governed_check_connection",
                                 "run_id": None}
    assert envelope.fields["access_token"].get_secret_value() == VALUE


def test_probe_without_a_ticket_is_refused_before_any_call(bus, sdk_slots):
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _call(_resolver(context_ref=None), _slot())
    assert _code(err) == "STALE_CONTEXT" and bus.requests == []


# --- one keypair per call, one RpcNode per run ------------------------------------------------

def test_one_keypair_per_call_and_one_rpcnode_per_run(bus, sdk_slots):
    resolver = _resolver("the-node")
    for _ in range(3):
        _call(resolver)
    assert len({r["recipient_key"] for r in bus.requests}) == 3
    assert len(bus.nodes) == 1
    node = bus.nodes[0]
    assert node.started and not node.stopped
    assert node.args[0] == "the-node" and node.timeouts == [credential_resolver.DEFAULT_TIMEOUT_S] * 3
    resolver.close()
    resolver.close()  # idempotent
    assert node.stopped


def test_no_rpcnode_before_the_first_call(bus, sdk_slots):
    resolver = _resolver()
    assert bus.nodes == []
    resolver.close()
    assert bus.nodes == []


def test_private_key_is_dropped_after_the_call(bus, sdk_slots, monkeypatch):
    keys = []
    real = transport.generate_recipient_keypair

    def spy():
        private, public = real()
        keys.append(private)
        return private, public

    monkeypatch.setattr(transport, "generate_recipient_keypair", spy)
    resolver = _resolver()
    _call(resolver)
    gc.collect()
    # only the test's own list holds it: the same count as a key nothing else ever saw
    untouched = [real()[0]]
    assert len(keys) == 1 and sys.getrefcount(keys[0]) == sys.getrefcount(untouched[0])
    assert not any(isinstance(v, transport.X25519PrivateKey) for v in vars(resolver).values())


# --- the envelope ------------------------------------------------------------------------------

def test_reply_is_opened_into_an_envelope(bus, sdk_slots):
    envelope = _call(_resolver())
    assert isinstance(envelope, sdk_slots.CredentialEnvelope)
    assert envelope.fields["access_token"].get_secret_value() == VALUE
    assert envelope.generation == 1
    assert envelope.expires_at > datetime.datetime.now(datetime.timezone.utc)
    assert VALUE not in repr(envelope) and VALUE not in str(envelope)


def test_nested_paths_are_passed_through(bus, sdk_slots):
    bus.fields = {"auth.token": VALUE, "api_key": VALUE + "-2"}
    envelope = _call(_resolver(), _slot(fields=["auth.token", "api_key"]))
    assert set(envelope.fields) == {"auth.token", "api_key"}


# --- refusals and failures ---------------------------------------------------------------------

@pytest.mark.parametrize("code", ["DENIED_GRANT", "RATE_LIMITED", "STORE_UNAVAILABLE", "CREDENTIAL_REFERENCE_INACTIVE"])
def test_a_refusal_reply_raises_its_reason_code(bus, sdk_slots, code):
    bus.override = {"ok": False, "reason_code": code, "message": "text that must not matter"}
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _call(_resolver())
    assert _code(err) == code
    assert "text that must not matter" not in str(err.value)


@pytest.mark.parametrize("reply", [None, "nope", {"ok": False}, {"ok": "yes"}, {"ok": True},
                                   {"ok": True, "release_id": "r", "sealed": None}])
def test_malformed_replies_are_a_dependency_failure(bus, sdk_slots, reply):
    bus.override = reply
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _call(_resolver())
    assert _code(err) == "DEPENDENCY_UNAVAILABLE"


@pytest.mark.parametrize("failure", [queue.Empty(), ConnectionError(VALUE), RuntimeError("boom")])
def test_bus_failures_are_a_dependency_failure_without_the_original_text(bus, sdk_slots, failure):
    bus.override = failure
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _call(_resolver())
    assert _code(err) == "DEPENDENCY_UNAVAILABLE"
    assert err.value.__cause__ is None and err.value.__suppress_context__ is True
    assert VALUE not in str(err.value)


def test_a_reply_sealed_for_another_context_does_not_open(bus, sdk_slots):
    bus.seal_decision_id = "some-other-decision"
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _call(_resolver())
    assert _code(err) == "DEPENDENCY_UNAVAILABLE"


def test_a_reply_sealed_to_another_key_does_not_open(bus, sdk_slots):
    other_public = transport.generate_recipient_keypair()[1]
    real = bus.release

    def release(**request):
        return real(**{**request, "recipient_key": other_public})

    bus.nodes  # noqa: B018  (fixture side effects only)
    bus.release = release
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _call(_resolver())
    assert _code(err) == "DEPENDENCY_UNAVAILABLE"


def test_an_expired_envelope_is_refused(bus, sdk_slots):
    bus.expires_in = -5
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _call(_resolver())
    assert _code(err) == "STALE_CONTEXT"


def test_outer_field_list_must_match_the_sealed_body(bus, sdk_slots):
    real = bus.release

    def release(**request):
        reply = real(**request)
        reply["fields"] = ["something_else"]
        return reply

    bus.release = release
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _call(_resolver())
    assert _code(err) == "DEPENDENCY_UNAVAILABLE"


def test_nothing_secret_reaches_logs_metrics_or_repr(bus, sdk_slots, logs):
    resolver = _resolver()
    _call(resolver)
    bus.override = {"ok": False, "reason_code": "DENIED_GRANT"}
    with pytest.raises(sdk_slots.GovernedCredentialError):
        _call(resolver)
    assert logs and all(VALUE not in line for line in logs)
    assert any("result=ok" in line for line in logs) and any("reason_code=DENIED_GRANT" in line for line in logs)
    assert VALUE not in repr(resolver)
    assert credential_resolver.CREDENTIAL_SLOT_RESOLUTION.total("ok", "OK") >= 1


def test_resolver_is_shared_on_copy_and_never_pickled(bus, sdk_slots):
    resolver = _resolver()
    assert copy.copy(resolver) is resolver and copy.deepcopy(resolver) is resolver
    with pytest.raises(TypeError):
        pickle.dumps(resolver)


# --- attaching to the client -------------------------------------------------------------------

def _kwargs(context_ref=CONTEXT_REF):
    kwargs = attach_envelope({}, "governed")
    kwargs["envelope"]["context_ref"] = context_ref
    kwargs.update({"project_id": 7, RUN_KEY: "run-1"})
    return kwargs


def test_governed_run_attaches_the_resolver_and_legacy_does_not(bus, sdk_slots):
    legacy, governed = types.SimpleNamespace(), types.SimpleNamespace()
    assert credential_resolver.attach_credential_resolver(
        legacy, None, attach_envelope({"project_id": 7}, "legacy"), "node",
        task_name="indexer_agent", project_id=7) is None
    assert not hasattr(legacy, "credential_resolver")

    resolver = credential_resolver.attach_credential_resolver(
        governed, {"run_id": "run-1"}, _kwargs(), "node", task_name="indexer_agent", project_id=7)
    assert governed.credential_resolver is resolver
    _call(resolver)
    (request,) = bus.requests
    assert request["context_ref"] == CONTEXT_REF and request["caller"]["run_id"] == "run-1"


def test_attach_falls_back_to_the_payload_project(bus, sdk_slots):
    client = types.SimpleNamespace()
    resolver = credential_resolver.attach_credential_resolver(
        client, {}, _kwargs(), "node", task_name="indexer_agent", project_id=None)
    _call(resolver)
    assert bus.requests[0]["project_id"] == 7


@pytest.mark.parametrize("name", ["indexer_agent", "indexer_predict_agent"])
def test_agent_entry_points_are_wired(name):
    source = (PLUGIN_ROOT / "methods" / f"{name}.py").read_text()
    assert "context_view = enforce_execution_context(" in source
    assert f'task_name="{name}", project_id=tasknode_task.meta.get("project_id")' in source
    assert "credential_resolver = attach_credential_resolver(" in source
    # skipped only for a governed run, closed with the task, readable on a refused slot
    assert "if credential_resolver is None" in source and "unsecret_mcp_tools(" in source
    assert "credential_resolver.close()" in source
    assert "governed_code = governed_credential_code(e)" in source
    assert "human_readable=rejection_message(governed_code)" in source


def test_unsecret_mcp_tools_is_guarded_by_the_legacy_check_in_both_entry_points():
    agent = (PLUGIN_ROOT / "methods" / "indexer_agent.py").read_text()
    predict = (PLUGIN_ROOT / "methods" / "indexer_predict_agent.py").read_text()
    assert "if credential_resolver is None and version_details and isinstance(version_details.get(\"tools\"), list):" in agent
    assert ("if credential_resolver is None:  # governed: nothing to resolve here, refused upstream (#6946)\n"
            "            adhoc_tools = unsecret_mcp_tools(adhoc_tools, client)") in predict


def test_governed_credential_code_matches_by_shape(sdk_slots):
    assert credential_resolver.governed_credential_code(sdk_slots.GovernedCredentialError("DENIED_GRANT")) == "DENIED_GRANT"
    assert credential_resolver.governed_credential_code(RuntimeError("DENIED_GRANT")) is None

    class GovernedCredentialError(Exception):  # a reloaded SDK class: same shape, different identity
        reason_code = "RATE_LIMITED"

    assert credential_resolver.governed_credential_code(GovernedCredentialError()) == "RATE_LIMITED"


# --- readable messages ---------------------------------------------------------------------------

CREDENTIAL_CODES = [
    "CREDENTIAL_REFERENCE_REQUIRED", "CREDENTIAL_REFERENCE_INACTIVE", "CREDENTIAL_GENERATION_MISMATCH",
    "DENIED_GRANT", "DENIED_RBAC", "DENIED_BINDING", "EPOCH_MISMATCH", "STALE_CONTEXT",
    "MANDATE_INACTIVE", "MANDATE_EXPIRED", "RATE_LIMITED", "AUDIT_UNAVAILABLE", "DEPENDENCY_UNAVAILABLE",
    "WORKER_CAPABILITY_UNAVAILABLE", "UNSUPPORTED_AUTH_MODE",
    "STORE_DENIED", "STORE_SEALED", "STORE_UNAVAILABLE", "STORE_TLS_INVALID", "STORE_NOT_FOUND",
    "STORE_VERSION_CONFLICT",
]


@pytest.mark.parametrize("code", CREDENTIAL_CODES)
def test_every_credential_code_has_its_own_readable_message(code):
    message = context_guard.rejection_message(code)
    assert message != context_guard.DEFAULT_REJECTION_MESSAGE
    assert code not in message and "_" not in message  # a sentence, never a code or an id


def test_specified_texts():
    assert context_guard.rejection_message("RATE_LIMITED") == "This run asked for its credentials too often and was stopped."
    assert context_guard.rejection_message("UNSUPPORTED_AUTH_MODE") == (
        "This toolkit cannot use a governed credential yet; the run was not started.")
    store = {context_guard.rejection_message(c) for c in CREDENTIAL_CODES if c.startswith("STORE_")}
    assert store == {"The credential store is unavailable; the run was not started. Try again later."}


def test_messages_agree_with_the_sdk_for_the_codes_it_names(real_sdk_slots):
    sdk_slots = real_sdk_slots
    for code in CREDENTIAL_CODES:
        assert context_guard.rejection_message(code), code
    for code in ("WORKER_CAPABILITY_UNAVAILABLE", "UNSUPPORTED_AUTH_MODE", "RATE_LIMITED"):
        assert context_guard.rejection_message(code) == sdk_slots.GovernedCredentialError(code).message


# --- capabilities ----------------------------------------------------------------------------------

def test_slots_supported_needs_the_sdk_module_and_the_transport_self_test(sdk_slots, monkeypatch):
    assert credential_slots_support.slots_supported() is True
    credential_slots_support.slots_supported.cache_clear()
    monkeypatch.setattr(credential_slots_support, "_transport_self_test", lambda: False)
    assert credential_slots_support.slots_supported() is False
    credential_slots_support.slots_supported.cache_clear()
    monkeypatch.setattr(credential_slots_support, "_transport_self_test", lambda: True)
    monkeypatch.setitem(sys.modules, SLOTS_MODULE, None)  # the SDK has no such module
    assert credential_slots_support.slots_supported() is False


def test_self_test_really_round_trips_and_rejects_a_foreign_aad():
    assert credential_slots_support._transport_self_test() is True


def test_self_test_failure_inside_transport_means_unsupported(sdk_slots, monkeypatch):
    def broken(*_a, **_k):
        raise transport.TransportError()

    monkeypatch.setattr(transport, "open_sealed", broken)
    assert credential_slots_support.slots_supported() is False


def test_credential_slots_capability_is_announced_only_when_supported(monkeypatch):
    monkeypatch.setattr(credential_slots_support, "slots_supported", lambda: False)
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: False)
    assert envelope_guard.capabilities_payload("a", "agents", now=1)["capabilities"] == ["task_envelope.v1"]
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: True)
    assert envelope_guard.capabilities_payload("a", "agents", now=1)["capabilities"] == [
        "credential_slots.v1", "credential_slots_batch.v1", "credential_slots_run.v1", "task_envelope.v1"]


def test_governed_ops_capability_rides_only_on_the_pool_announce(monkeypatch):
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: True)
    agents = envelope_guard.capabilities_payload("agents_1", "agents", now=1)
    pool = envelope_guard.capabilities_payload(
        "governed_ops_1", "governed_ops", now=1, extra=governed_ops.governed_ops_capabilities())
    assert "governed_ops_pool.v1" not in agents["capabilities"]
    assert pool["capabilities"] == [
        "credential_slots.v1", "credential_slots_batch.v1", "credential_slots_run.v1", "governed_ops_pool.v1",
        "task_envelope.v1"]


def _requires(*capabilities):
    kwargs = attach_envelope({}, "governed")
    kwargs["envelope"]["required_capabilities"] = ["task_envelope.v1", *capabilities]
    return {"name": "x", "kwargs": kwargs}


def test_approver_and_guard_honour_slots_only_when_supported(monkeypatch):
    approve = envelope_guard.envelope_task_approver()
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: False)
    assert approve("q", _requires("credential_slots.v1")) is False
    with pytest.raises(Exception) as err:
        envelope_guard.enforce_envelope(_requires("credential_slots.v1")["kwargs"])
    assert "credential_slots.v1" in err.value.missing
    assert approve("q", _requires("credential_slots_batch.v1")) is False
    assert approve("q", _requires("credential_slots_run.v1")) is False
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: True)
    assert approve("q", _requires("credential_slots.v1")) is True
    assert approve("q", _requires("credential_slots_batch.v1")) is True
    assert approve("q", _requires("credential_slots_run.v1")) is True
    envelope_guard.enforce_envelope(_requires("credential_slots.v1", "credential_slots_batch.v1")["kwargs"])


def test_pool_capability_is_honoured_only_by_the_pool_approver(monkeypatch):
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: True)
    task = _requires("credential_slots.v1", "governed_ops_pool.v1")
    assert envelope_guard.envelope_task_approver()("q", task) is False
    pool_approver = envelope_guard.envelope_task_approver(extra=governed_ops.governed_ops_capabilities())
    assert pool_approver("q", task) is True


# --- the governed_ops pool ---------------------------------------------------------------------------

def test_pool_is_off_unless_explicitly_enabled_and_supported(monkeypatch):
    monkeypatch.setattr(governed_ops, "slots_supported", lambda: True)
    assert governed_ops.governed_ops_enabled({}) is False
    assert governed_ops.governed_ops_enabled({"governed_ops_enabled": False}) is False
    assert governed_ops.governed_ops_enabled({"governed_ops_enabled": "true"}) is False
    assert governed_ops.governed_ops_enabled({"governed_ops_enabled": True}) is True
    monkeypatch.setattr(governed_ops, "slots_supported", lambda: False)
    assert governed_ops.governed_ops_enabled({"governed_ops_enabled": True}) is False


@pytest.mark.parametrize("value,expected", [
    (None, 4), (0, 4), (-1, 4), ("8", 4), (True, 4), (2.5, 4), (1, 1), (16, 16),
])
def test_task_limit_is_always_finite(value, expected):
    assert governed_ops.governed_ops_task_limit({"governed_ops_task_limit": value}) == expected
    assert governed_ops.governed_ops_task_limit({}) == governed_ops.DEFAULT_TASK_LIMIT == 4


def test_pool_node_parameters():
    captured = {}
    fake_arbiter = types.SimpleNamespace(TaskNode=lambda event_node, **kw: captured.update(kw, event_node=event_node))
    approver = object()
    governed_ops.create_governed_task_node(fake_arbiter, "evt", {"governed_ops_task_limit": 3}, "/tmp/t", approver)
    assert captured["event_node"] == "evt"
    assert captured["pool"] == "governed_ops" and captured["task_limit"] == 3
    assert captured["multiprocessing_context"] == "fork" and captured["kill_on_stop"] is True
    assert captured["ident_prefix"] == "governed_ops_" and captured["task_approver"] is approver
    assert captured["start_attempts"] == 1 and captured["task_retention_period"] == 300
    assert "result_transport" not in captured  # default transport, not files


def test_default_config_ships_the_pool_off():
    text = (PLUGIN_ROOT / "config.yml").read_text()
    assert "\ngoverned_ops_enabled: false\n" in text and "\ngoverned_ops_task_limit: 4\n" in text
    assert "governed_ops" not in (PLUGIN_ROOT / "admin_schema.json").read_text()


# --- module wiring (module.py loaded as part of a package, with a fake arbiter) ------------------------

class FakeNode:
    def __init__(self, event_node=None, **kwargs):
        self.kwargs = kwargs
        self.ident = kwargs.get("ident_prefix", "evt_") + "1"
        self.pool = kwargs.get("pool")
        self.task_limit = kwargs.get("task_limit")
        self.started = self.stopped = False
        self.tasks = {}
        self.emitted = []

    def clone(self):
        return FakeNode(ident_prefix="evt_")

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def register_task(self, func, name, *approvers):
        self.tasks[name] = (func, approvers)

    def emit(self, name, payload):
        self.emitted.append((name, payload))


ALIASED = ("context_guard", "credential_resolver", "credential_slots_support", "credential_transport",
           "envelope_guard", "governed_ops", "task_envelope")


@pytest.fixture
def package(monkeypatch):
    """The plugin as a package (``iw6946``) with pylon, arbiter and worker_core stubbed.

    The plugin's own utils are the already-imported ``utils.*`` modules, so nothing here depends on
    which SDK stubs earlier test files left in ``sys.modules``.
    """
    calls = types.SimpleNamespace(log=[])
    log = types.SimpleNamespace(**{n: (lambda *a, _n=n, **k: calls.log.append((_n, a))) for n in
                                   ("debug", "info", "warning", "error", "exception")})
    pylon_tools = types.ModuleType("pylon.core.tools")
    pylon_tools.log = log
    pylon_tools.module = types.SimpleNamespace(ModuleModel=object)
    pylon_tools.web = types.SimpleNamespace(method=lambda *_a, **_k: (lambda fn: fn))
    arbiter = types.ModuleType("arbiter")
    arbiter.TaskNode = FakeNode
    tools = types.ModuleType("tools")
    tools.worker_core = types.SimpleNamespace(event_node=FakeNode(ident_prefix="core_"), tasknode_tmp="/tmp/tasknode")
    for name, mod in (("pylon", types.ModuleType("pylon")), ("pylon.core", types.ModuleType("pylon.core")),
                      ("pylon.core.tools", pylon_tools), ("arbiter", arbiter), ("tools", tools)):
        monkeypatch.setitem(sys.modules, name, mod)
    root = types.ModuleType("iw6946")
    root.__path__ = [str(PLUGIN_ROOT)]
    monkeypatch.setitem(sys.modules, "iw6946", root)
    for name in ALIASED:
        monkeypatch.setitem(sys.modules, f"iw6946.utils.{name}", sys.modules[f"utils.{name}"])
    yield types.SimpleNamespace(calls=calls)
    for key in [k for k in sys.modules if k.startswith("iw6946.") and k[len("iw6946."):].split(".")[0] != "utils"
                or k.startswith("iw6946.utils.") and k.rsplit(".", 1)[1] not in ALIASED]:
        del sys.modules[key]


def _module(package, config):
    spec = importlib.util.spec_from_file_location("iw6946.module", PLUGIN_ROOT / "module.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["iw6946.module"] = loaded
    spec.loader.exec_module(loaded)
    return loaded.Module(None, types.SimpleNamespace(config=config))


def _slots_supported(monkeypatch, value):
    monkeypatch.setattr(governed_ops, "slots_supported", lambda: value)
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: value)


def test_pool_is_absent_when_disabled(package, monkeypatch):
    instance = _module(package, {})
    _slots_supported(monkeypatch, True)
    instance._init_governed_ops()
    assert instance.governed_task_node is None and instance.governed_event_node is None
    assert not [w for w in package.calls.log if w[0] == "warning"]


def test_pool_starts_when_enabled_and_supported(package, monkeypatch):
    instance = _module(package, {"governed_ops_enabled": True, "governed_ops_task_limit": 2})
    _slots_supported(monkeypatch, True)
    instance._init_governed_ops()
    node = instance.governed_task_node
    assert node.started and node.pool == "governed_ops" and node.task_limit == 2
    assert instance.governed_event_node.started


def test_pool_stays_off_and_warns_when_slots_are_unsupported(package, monkeypatch):
    instance = _module(package, {"governed_ops_enabled": True})
    _slots_supported(monkeypatch, False)
    instance._init_governed_ops()
    assert instance.governed_task_node is None
    assert [w for w in package.calls.log if w[0] == "warning"]


def test_module_registers_and_tears_down_the_probe_task_only_with_the_pool():
    source = (PLUGIN_ROOT / "module.py").read_text()
    assert source.count("indexer_governed_check_connection, GOVERNED_CHECK_CONNECTION_TASK") == 2
    assert source.count("if self.governed_task_node is not None:") == 3
    # started before the first capability announce, and not an agent-pool task
    assert source.index("self._init_governed_ops()") < source.index("start_capability_heartbeat(")
    assert "self.agent_task_node.register_task(\n                self.indexer_governed_check_connection" not in source


def test_capabilities_request_announces_the_pool_only_when_it_exists(package, monkeypatch):
    _slots_supported(monkeypatch, True)
    method = importlib.import_module("iw6946.methods.indexer_task_capabilities").Method
    agents = FakeNode(ident_prefix="agents_", pool="agents")
    agents.pool = "agents"
    me = types.SimpleNamespace(agent_event_node=FakeNode(), agent_task_node=agents, governed_task_node=None)
    method.task_node_capabilities_request(me, None, None)
    assert [p["pool"] for _, p in me.agent_event_node.emitted] == ["agents"]
    me.agent_event_node.emitted.clear()
    me.governed_task_node = FakeNode(pool="governed_ops", ident_prefix="governed_ops_")
    method.task_node_capabilities_request(me, None, None)
    emitted = {p["pool"]: p for _, p in me.agent_event_node.emitted}
    assert set(emitted) == {"agents", "governed_ops"}
    assert "governed_ops_pool.v1" in emitted["governed_ops"]["capabilities"]
    assert "governed_ops_pool.v1" not in emitted["agents"]["capabilities"]


# --- flag off / legacy run is untouched ------------------------------------------------------------------------

def test_legacy_run_makes_no_call_and_leaves_payload_and_client_alone(bus, sdk_slots, monkeypatch):
    kwargs = attach_envelope({"project_id": 7, "tools": [{"type": "github", "settings": {"x": {"password": "p"}}}]}, "legacy")
    before = copy.deepcopy(kwargs)
    client = types.SimpleNamespace()
    view = context_guard.enforce_execution_context(kwargs, "node", task_name="indexer_agent")
    assert view is None
    assert credential_resolver.attach_credential_resolver(
        client, view, kwargs, "node", task_name="indexer_agent", project_id=7) is None
    assert kwargs == before and vars(client) == {}
    assert bus.requests == [] and bus.nodes == []


def test_governed_run_leaves_the_payload_byte_identical(bus, sdk_slots):
    kwargs = _kwargs()
    before = copy.deepcopy(kwargs)
    client = types.SimpleNamespace()
    resolver = credential_resolver.attach_credential_resolver(
        client, {}, kwargs, "node", task_name="indexer_agent", project_id=7)
    _call(resolver)
    assert kwargs == before and pickle.loads(pickle.dumps(kwargs)) == before


# --- the governed check-connection task -------------------------------------------------------------------------

class FakeMcpAuthorizationRequired(Exception):
    def to_dict(self):
        return {"message": str(self), "server_url": "https://mcp.example.test/mcp/"}


class FakeConfig:
    seen = None
    action = staticmethod(lambda settings: {"success": True})

    @staticmethod
    def check_connection(settings):
        FakeConfig.seen = copy.deepcopy(settings)
        return FakeConfig.action(settings)


@pytest.fixture
def probe(package, bus, real_sdk_slots, monkeypatch):
    """The governed task loaded from the package, with the SDK configuration registry and event node faked."""
    FakeConfig.seen = None
    FakeConfig.action = staticmethod(lambda settings: {"success": True})
    local_node = FakeNode()
    agent_common = types.ModuleType("iw6946.utils.agent_execution_common")
    agent_common.setup_event_node = lambda _context: local_node
    monkeypatch.setitem(sys.modules, "iw6946.utils.agent_execution_common", agent_common)
    configurations = types.ModuleType("elitea_sdk.configurations")
    configurations.get_class_configurations = lambda: {"github": FakeConfig}
    monkeypatch.setitem(sys.modules, "elitea_sdk.configurations", configurations)
    funcs = types.ModuleType("iw6946.utils.funcs")
    funcs.normalize_mcp_auth_metadata_urls = lambda metadata: {**metadata, "normalized": True}
    funcs.dev_reload_sdk = lambda _name: None
    monkeypatch.setitem(sys.modules, "iw6946.utils.funcs", funcs)
    mcp_oauth = types.ModuleType("elitea_sdk.runtime.utils.mcp_oauth")
    mcp_oauth.McpAuthorizationRequired = FakeMcpAuthorizationRequired
    monkeypatch.setitem(sys.modules, "elitea_sdk.runtime.utils.mcp_oauth", mcp_oauth)
    task = types.SimpleNamespace(meta={"project_id": 7}, multiprocessing_context="fork")
    monkeypatch.setitem(sys.modules, "tasknode_task", task)
    method = importlib.import_module("iw6946.methods.indexer_governed_check_connection").Method
    _slots_supported(monkeypatch, True)

    def run(settings=None, configuration_type="github", **kwargs):
        kwargs = {**attach_envelope({}, "governed"), **kwargs}
        if settings is None:
            settings = {"base_url": "https://example.test", "credential_slot": _slot(probe_ref=PROBE_REF, decision_id=None)}
        return method.indexer_governed_check_connection(
            types.SimpleNamespace(), configuration_type=configuration_type, settings=settings, **kwargs)

    return types.SimpleNamespace(run=run, node=local_node, task=task, package=package)


def test_probe_resolves_the_slot_and_hands_check_connection_plain_values(probe, bus):
    result = probe.run()
    assert result == {"success": True}
    assert FakeConfig.seen == {"base_url": "https://example.test", "access_token": VALUE}
    (request,) = bus.requests
    assert request["probe_ref"] == PROBE_REF and request["context_ref"] is None
    assert request["toolkit_instance_id"] == "probe" and request["field"] == "configuration"
    assert request["caller"] == {"kind": "indexer_worker", "task_name": "indexer_governed_check_connection",
                                 "run_id": None}
    assert request["project_id"] == 7


def test_probe_closes_its_bus_node_and_event_node(probe, bus):
    probe.run()
    assert bus.nodes[0].stopped and probe.node.stopped


def test_probe_scrubs_a_result_that_echoes_the_value(probe):
    FakeConfig.action = staticmethod(lambda s: {"success": False, "error": f"bad token {VALUE}", "n": [f"x{VALUE}"]})
    result = probe.run()
    assert VALUE not in json.dumps(result) and result["success"] is False


def test_probe_scrubs_a_raised_error_and_its_log_line(probe):
    def boom(settings):
        raise RuntimeError(f"401 for {settings['access_token']}")

    FakeConfig.action = staticmethod(boom)
    result = probe.run()
    assert isinstance(result, str) and VALUE not in result and "401" in result
    assert all(VALUE not in repr(entry) for entry in probe.package.calls.log)


def test_probe_scrubs_a_validation_error(probe):
    from pydantic import BaseModel, ValidationError

    class Model(BaseModel):
        count: int

    def invalid(settings):
        try:
            Model(count=settings["access_token"])
        except ValidationError as exc:
            assert VALUE in str(exc.errors())  # the unscrubbed text really carries it
            raise

    FakeConfig.action = staticmethod(invalid)
    result = probe.run()
    assert isinstance(result, str) and VALUE not in result
    assert all(VALUE not in repr(entry) for entry in probe.package.calls.log)


def test_probe_returns_requires_authorization_unchanged(probe):
    ex = FakeMcpAuthorizationRequired("authorize please")

    def needs_auth(settings):
        raise ex

    FakeConfig.action = staticmethod(needs_auth)
    assert probe.run() == {
        "success": False, "requires_authorization": True, "error": "authorize please",
        "auth_metadata": {"message": "authorize please", "server_url": "https://mcp.example.test/mcp/",
                          "normalized": True},
    }


@pytest.mark.parametrize("code", ["DENIED_RBAC", "STORE_UNAVAILABLE", "CREDENTIAL_REFERENCE_INACTIVE", "STALE_CONTEXT"])
def test_probe_refusal_is_a_readable_typed_result_and_check_is_not_called(probe, bus, code):
    bus.override = {"ok": False, "reason_code": code, "message": "ignored"}
    result = probe.run()
    assert result == {"success": False, "reason_code": code, "error": context_guard.rejection_message(code)}
    assert FakeConfig.seen is None and bus.nodes[0].stopped


def test_probe_needs_a_slot(probe, bus):
    result = probe.run(settings={"base_url": "https://example.test"})
    assert result["reason_code"] == "CREDENTIAL_REFERENCE_REQUIRED" and bus.requests == []


def test_probe_refuses_a_legacy_envelope(probe, bus):
    result = probe.run(envelope={"schema_version": 1, "security_profile": "legacy", "required_capabilities": []})
    assert result["reason_code"] == "STALE_CONTEXT" and bus.requests == [] and FakeConfig.seen is None


def test_probe_refuses_an_envelope_it_cannot_honour(probe, bus):
    envelope = attach_envelope({}, "governed")["envelope"]
    envelope["required_capabilities"] = ["task_envelope.v1", "future.v9"]
    result = probe.run(envelope=envelope)
    assert result["success"] is False and result["reason_code"] == "CAPABILITY_MISSING" and bus.requests == []


def test_probe_accepts_the_pool_capability(probe):
    envelope = attach_envelope({}, "governed")["envelope"]
    envelope["required_capabilities"] = ["task_envelope.v1", "credential_slots.v1", "governed_ops_pool.v1"]
    assert probe.run(envelope=envelope) == {"success": True}


def test_probe_without_a_project_is_stale(probe, bus):
    probe.task.meta = {}
    assert probe.run()["reason_code"] == "STALE_CONTEXT" and bus.requests == []
    assert probe.run(project_id=7) == {"success": True}  # the payload's own project is the fallback


def test_probe_unknown_type_is_reported_as_the_ungoverned_task_does(probe):
    assert probe.run(configuration_type="nope") == "Not supported toolkit type: nope"


# --- batch release: one bus call per toolkit build (perf PR 4: M1 worker side, S9, L7) ---------------------------------

def _items(n, **extra):
    return [{"field": f"configuration_{i}", "slot": _slot(decision_id=f"decision-{i}", **extra)} for i in range(n)]


def _many(resolver, items, toolkit="12"):
    return resolver.resolve_many(toolkit_instance_id=toolkit, items=items)


@pytest.fixture
def keypairs(monkeypatch):
    made = []
    real = transport.generate_recipient_keypair

    def spy():
        pair = real()
        made.append(pair[1])
        return pair

    monkeypatch.setattr(transport, "generate_recipient_keypair", spy)
    return made


def test_a_governed_run_with_six_slots_makes_two_bus_calls_on_one_node(batch_bus, sdk_slots, keypairs):
    """validate (1) + one release batch (1) == 2 bus calls, one RpcNode, one keypair; the single-slot handler is idle."""
    kwargs = _kwargs()
    run_rpc = context_guard.RunRpcNode("node")
    view = context_guard.enforce_execution_context(kwargs, "node", task_name="indexer_agent", project_id=7, rpc=run_rpc)
    resolver = credential_resolver.attach_credential_resolver(
        types.SimpleNamespace(), view, kwargs, "node", task_name="indexer_agent", project_id=7, rpc=run_rpc)
    result = _many(resolver, _items(6))
    assert set(result) == {f"configuration_{i}" for i in range(6)}
    assert all(e.fields["access_token"].get_secret_value() == VALUE and e.generation == 1 for e in result.values())
    assert len(batch_bus.validate_requests) + len(batch_bus.batch_requests) == 2
    assert len(batch_bus.validate_requests) == 1 and len(batch_bus.batch_requests) == 1
    assert batch_bus.requests == []
    assert len(batch_bus.nodes) == 1 and len(keypairs) == 1
    node = batch_bus.nodes[0]
    # validate: its own 5 s; the batch: 15 s + 0.5 s per slot
    assert node.started and node.timeouts == [context_guard.DEFAULT_TIMEOUT_S, 18.0]
    assert view.batch_release is True
    resolver.close()
    assert not node.stopped  # the run owns the shared node
    run_rpc.close()
    run_rpc.close()  # idempotent
    assert node.stopped


def test_batch_request_shape_and_one_slot_per_item(batch_bus, sdk_slots):
    items = _items(3)
    _many(_bresolver("the-node"), items, toolkit="12")
    (request,) = batch_bus.batch_requests
    assert set(request) == {"project_id", "context_ref", "probe_ref", "caller", "recipient_key", "slots"}
    assert request["project_id"] == 7 and request["context_ref"] == CONTEXT_REF and request["probe_ref"] is None
    assert request["caller"] == {"kind": "indexer_worker", "task_name": "indexer_agent", "run_id": "run-1"}
    assert len(__import__("base64").b64decode(request["recipient_key"])) == 32
    assert request["slots"] == [{"toolkit_instance_id": "12", "field": i["field"], "slot": i["slot"]} for i in items]


def test_every_item_is_opened_with_its_own_aad(batch_bus, sdk_slots):
    """A reply item sealed for another decision id does not open, even though the batch shares one key."""
    result = _many(_bresolver(), _items(3))
    assert len(result) == 3
    real = batch_bus.release_many

    def swapped(**request):
        reply = real(**request)
        first, second = reply["releases"][:2]
        first["sealed"], second["sealed"] = second["sealed"], first["sealed"]
        return reply

    batch_bus.release_many = swapped
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _many(_bresolver(), _items(3))
    assert _code(err) == "DEPENDENCY_UNAVAILABLE"


def test_the_one_slot_call_is_a_one_item_batch(batch_bus, sdk_slots, keypairs):
    envelope = _call(_bresolver())
    assert envelope.fields["access_token"].get_secret_value() == VALUE
    (request,) = batch_bus.batch_requests
    assert [s["field"] for s in request["slots"]] == ["github_configuration"] and batch_bus.requests == []
    assert len(keypairs) == 1


def test_private_key_of_the_batch_is_dropped_after_all_items_open(batch_bus, sdk_slots, monkeypatch):
    keys = []
    real = transport.generate_recipient_keypair

    def spy():
        private, public = real()
        keys.append(private)
        return private, public

    monkeypatch.setattr(transport, "generate_recipient_keypair", spy)
    resolver = _bresolver()
    _many(resolver, _items(4))
    gc.collect()
    untouched = [real()[0]]
    assert len(keys) == 1 and sys.getrefcount(keys[0]) == sys.getrefcount(untouched[0])
    assert not any(isinstance(v, transport.X25519PrivateKey) for v in vars(resolver).values())


@pytest.mark.parametrize("code", ["DENIED_GRANT", "RATE_LIMITED", "STORE_UNAVAILABLE", "CREDENTIAL_REFERENCE_INACTIVE"])
def test_a_batch_refusal_maps_to_its_reason_code(batch_bus, sdk_slots, code):
    batch_bus.override = {"ok": False, "reason_code": code, "message": "must not matter",
                          "toolkit_instance_id": "12", "field": "configuration_1"}
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _many(_bresolver(), _items(3))
    assert _code(err) == code and "must not matter" not in str(err.value)
    assert batch_bus.requests == []


@pytest.mark.parametrize("reply", [None, "nope", {"ok": False}, {"ok": True}, {"ok": True, "releases": None},
                                   {"ok": True, "releases": "x"}, {"ok": True, "releases": [None]},
                                   {"ok": True, "releases": [], "extra": 1}])
def test_malformed_batch_replies_are_a_dependency_failure(batch_bus, sdk_slots, reply):
    batch_bus.override = reply
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _many(_bresolver(), _items(2))
    assert _code(err) == "DEPENDENCY_UNAVAILABLE"


@pytest.mark.parametrize("tamper", ["missing", "extra", "duplicate", "foreign_toolkit", "unknown_field",
                                    "duplicate_release_id", "unknown_item_key", "unknown_reply_key"])
def test_the_released_set_must_be_exactly_the_requested_set(batch_bus, sdk_slots, tamper):
    real = batch_bus.release_many

    def release_many(**request):
        reply = real(**request)
        releases = reply["releases"]
        if tamper == "missing":
            releases.pop()
        elif tamper == "extra":
            releases.append({**releases[0], "field": "another"})
        elif tamper == "duplicate":
            releases[1] = dict(releases[0])
        elif tamper == "foreign_toolkit":
            releases[0]["toolkit_instance_id"] = "99"
        elif tamper == "duplicate_release_id":
            releases[1]["release_id"] = releases[0]["release_id"]
        elif tamper == "unknown_item_key":
            releases[0]["note"] = "x"
        elif tamper == "unknown_reply_key":
            reply["note"] = "x"
        else:
            releases[0]["field"] = "not_requested"
        return reply

    batch_bus.release_many = release_many
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _many(_bresolver(), _items(3))
    assert _code(err) == "DEPENDENCY_UNAVAILABLE"


@pytest.mark.parametrize("failure", [queue.Empty(), ConnectionError(VALUE), RuntimeError("boom"), NotImplementedError("x")])
def test_batch_bus_failures_are_a_dependency_failure_and_never_a_fallback(batch_bus, sdk_slots, failure):
    batch_bus.override = failure
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _many(_bresolver(), _items(2))
    assert _code(err) == "DEPENDENCY_UNAVAILABLE" and VALUE not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__ is True
    assert batch_bus.requests == []  # no single-slot retry


def test_an_expired_item_refuses_the_batch(batch_bus, sdk_slots):
    batch_bus.expires_in = -5
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _many(_bresolver(), _items(2))
    assert _code(err) == "STALE_CONTEXT"


def test_batch_logs_and_metrics_carry_no_material(batch_bus, sdk_slots, logs):
    resolver = _bresolver()
    before = credential_resolver.CREDENTIAL_SLOT_RESOLUTION.total("ok", "OK")
    _many(resolver, _items(3))
    assert credential_resolver.CREDENTIAL_SLOT_RESOLUTION.total("ok", "OK") == before + 3  # one per released slot
    assert logs and all(VALUE not in line for line in logs) and VALUE not in repr(resolver)


# --- Main decides: the validate reply's batch_release flag ------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [(True, True), (False, False), (None, False)])
def test_the_view_carries_the_flag_from_the_validate_reply(batch_bus, sdk_slots, value, expected):
    batch_bus.batch_release = value  # None: the key is absent (an older Main)
    view = context_guard.enforce_execution_context(_kwargs(), "node", task_name="indexer_agent")
    assert view.batch_release is expected and view == {"run_id": "run-1", "attempt": 1}


@pytest.mark.parametrize("junk", ["true", 1, "yes", [True], {}])
def test_only_an_explicit_true_counts(batch_bus, sdk_slots, junk):
    batch_bus.batch_release = junk
    assert context_guard.enforce_execution_context(_kwargs(), "node", task_name="indexer_agent").batch_release is False


def test_attach_passes_the_flag_on_and_a_plain_view_means_no_batch(bus, sdk_slots):
    for view, expected in ((context_guard.ContextView({}, batch_release=True), True),
                           (context_guard.ContextView({}), False), ({"run_id": "r"}, False)):
        resolver = credential_resolver.attach_credential_resolver(
            types.SimpleNamespace(), view, _kwargs(), "node", task_name="indexer_agent", project_id=7)
        assert resolver._batch_release is expected


def test_flag_true_releases_a_toolkit_build_in_one_call(batch_bus, sdk_slots):
    _many(_resolver(batch_release=True), _items(6))
    assert len(batch_bus.batch_requests) == 1 and batch_bus.requests == []


def test_flag_false_releases_slot_by_slot_without_ever_calling_the_batch_handler(bus, sdk_slots, keypairs):
    result = _many(_resolver(batch_release=False), _items(6))
    assert len(result) == 6 and bus.batch_requests == [] and len(bus.requests) == 6
    assert len({r["recipient_key"] for r in bus.requests}) == 6 == len(keypairs)  # a keypair per single call
    assert [r["field"] for r in bus.requests] == [i["field"] for i in _items(6)]
    assert len(bus.nodes) == 1


def test_the_flag_also_decides_the_one_slot_call(batch_bus, sdk_slots):
    _call(_resolver(batch_release=False))
    assert len(batch_bus.requests) == 1 and batch_bus.batch_requests == []
    _call(_resolver(batch_release=True))
    assert len(batch_bus.batch_requests) == 1 and len(batch_bus.requests) == 1


def test_a_refusal_slot_by_slot_keeps_its_code_and_stops_at_the_first(bus, sdk_slots):
    bus.override = {"ok": False, "reason_code": "DENIED_RBAC"}
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _many(_resolver(batch_release=False), _items(3))
    assert _code(err) == "DENIED_RBAC" and len(bus.requests) == 1


def test_the_batch_timeout_scales_with_the_slots(batch_bus, sdk_slots):
    for n, expected in ((1, 15.5), (2, 16.0), (6, 18.0), (64, 47.0)):
        batch_bus.nodes.clear()
        _many(_resolver(batch_release=True), _items(n))
        assert batch_bus.nodes[0].timeouts == [expected]
    assert credential_resolver.batch_timeout(15, 90) == 60 == credential_resolver.batch_timeout(15, 500)
    assert credential_resolver.batch_timeout(15, 89) == 59.5


# --- the real bus (arbiter RpcNode over the mock event node) -------------------------------------------------------------------

def _arbiter_root():
    candidates = [os.environ.get("ELITEA_ARBITER_PATH")]
    candidates += [str(parent / "arbiter" / "arbiter") for parent in pathlib.Path(__file__).resolve().parents]
    for candidate in candidates:
        if candidate and (pathlib.Path(candidate) / "rpcnode" / "rpcnode.py").is_file():
            return candidate
    return None


@pytest.fixture
def real_bus(monkeypatch):
    """arbiter's own RpcNode and MockEventNode, loaded without the package's heavy ``__init__`` (broker clients)."""
    root = _arbiter_root()
    if root is None:
        pytest.skip("arbiter sources not found (set ELITEA_ARBITER_PATH to the directory holding rpcnode/)")
    import logging
    saved = {k: v for k, v in sys.modules.items() if k == "arbiter" or k.startswith("arbiter.")}
    for key in saved:
        del sys.modules[key]
    package, eventnode = types.ModuleType("arbiter"), types.ModuleType("arbiter.eventnode")
    package.__path__, eventnode.__path__ = [root], [os.path.join(root, "eventnode")]
    package.log = logging.getLogger("arbiter-test")
    sys.modules["arbiter"], sys.modules["arbiter.eventnode"] = package, eventnode
    from arbiter.eventnode.mock import MockEventNode  # pylint: disable=E0401,C0415
    from arbiter.rpcnode import RpcNode  # pylint: disable=E0401,C0415
    event_node = MockEventNode()
    event_node.start()
    try:
        yield types.SimpleNamespace(event_node=event_node, RpcNode=RpcNode)
    finally:
        event_node.stop()
        for key in [k for k in sys.modules if k == "arbiter" or k.startswith("arbiter.")]:
            del sys.modules[key]
        sys.modules.update(saved)


def _subscriptions(event_node):
    return sum(len(callbacks) for callbacks in event_node.event_callbacks.values())


def _provider(real_bus, **handlers):
    node = real_bus.RpcNode(real_bus.event_node, id_prefix="main_")
    node.start()
    for name, handler in handlers.items():
        node.register(handler, name)
    return node


def test_no_provider_times_out_to_dependency_unavailable_with_no_single_slot_retry(real_bus, sdk_slots, monkeypatch):
    singles = []
    provider = _provider(real_bus, restricted_release_credential=lambda **r: singles.append(r))  # an old Main
    monkeypatch.setattr(credential_resolver, "BATCH_TIMEOUT_PER_SLOT_S", 0)
    run_rpc = context_guard.RunRpcNode(real_bus.event_node)
    try:
        resolver = credential_resolver.TrustedCredentialResolver(
            real_bus.event_node, project_id=7, task_name="indexer_agent", context_ref=CONTEXT_REF, run_id="run-1",
            timeout_s=0.2, rpc=run_rpc, batch_release=True)  # a Main that said batch, but has no handler registered
        with pytest.raises(sdk_slots.GovernedCredentialError) as err:
            _many(resolver, _items(3))
        assert _code(err) == "DEPENDENCY_UNAVAILABLE" and err.value.__cause__ is None
        assert singles == []
    finally:
        run_rpc.close()
        provider.stop()


def test_the_real_bus_serves_a_batch_end_to_end_and_close_leaves_no_subscriptions(real_bus, sdk_slots, keypairs):
    fake = FakeBus()
    provider = _provider(real_bus, restricted_release_credentials=fake.release_many,
                         restricted_release_credential=fake.release,
                         restricted_validate_execution_context=lambda **r: {**fake.validate(**r), "batch_release": True})
    baseline = _subscriptions(real_bus.event_node)  # the provider's own four
    run_rpc = context_guard.RunRpcNode(real_bus.event_node)
    try:
        kwargs = _kwargs()
        view = context_guard.enforce_execution_context(kwargs, real_bus.event_node, task_name="indexer_agent",
                                                       project_id=7, rpc=run_rpc)
        resolver = credential_resolver.attach_credential_resolver(
            types.SimpleNamespace(), view, kwargs, real_bus.event_node, task_name="indexer_agent", project_id=7,
            rpc=run_rpc)
        result = _many(resolver, _items(6))
        assert len(result) == 6 and len(fake.batch_requests) == 1 and fake.requests == [] and len(keypairs) == 1
        assert len(fake.validate_requests) == 1
        assert _subscriptions(real_bus.event_node) == baseline + 4  # one shared node, validate and release
    finally:
        run_rpc.close()
    assert _subscriptions(real_bus.event_node) == baseline
    with pytest.raises(RuntimeError):
        run_rpc.proxy(1)
    provider.stop()


def test_a_start_failure_leaves_nothing_subscribed(real_bus, sdk_slots, monkeypatch):
    baseline = _subscriptions(real_bus.event_node)
    real_subscribe = real_bus.event_node.subscribe

    def failing(event_name, callback):
        if event_name == "service_request":
            raise RuntimeError("subscribe failed")
        return real_subscribe(event_name, callback)

    monkeypatch.setattr(real_bus.event_node, "subscribe", failing)
    run_rpc = context_guard.RunRpcNode(real_bus.event_node)
    with pytest.raises(RuntimeError):
        run_rpc.proxy(1)
    assert _subscriptions(real_bus.event_node) == baseline
    run_rpc.close()


# --- the items themselves ------------------------------------------------------------------------------------------------

def test_no_items_means_no_call_and_no_node(batch_bus, sdk_slots):
    assert _many(_bresolver(), []) == {}
    assert batch_bus.batch_requests == [] and batch_bus.requests == [] and batch_bus.nodes == []


@pytest.mark.parametrize("items", [
    "x", [None], [{"field": "a"}], [{"slot": _slot()}], [{"field": "", "slot": _slot()}],
    [{"field": "a", "slot": "nope"}], [{"field": "a", "slot": _slot()}, {"field": "a", "slot": _slot()}],
])
def test_malformed_items_are_refused_before_any_call(batch_bus, sdk_slots, items):
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _many(_bresolver(), items)
    assert _code(err) == "STALE_CONTEXT" and batch_bus.batch_requests == [] and batch_bus.nodes == []


def test_the_batch_is_bounded_at_the_run_limit(batch_bus, sdk_slots):
    assert len(_many(_bresolver(), _items(credential_resolver.MAX_BATCH_SLOTS))) == credential_resolver.MAX_BATCH_SLOTS
    batch_bus.batch_requests.clear()
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _many(_bresolver(), _items(credential_resolver.MAX_BATCH_SLOTS + 1))
    assert _code(err) == "STALE_CONTEXT" and batch_bus.batch_requests == []


def test_a_probe_batch_carries_its_one_ticket(batch_bus, sdk_slots):
    resolver = _bresolver(context_ref=None, run_id=None, task_name="indexer_governed_check_connection")
    _many(resolver, _items(2, probe_ref=PROBE_REF), toolkit="probe")
    (request,) = batch_bus.batch_requests
    assert request["context_ref"] is None and request["probe_ref"] == PROBE_REF
    for items in (_items(2), [_items(1, probe_ref=PROBE_REF)[0], {**_items(2, probe_ref="other-ticket")[1]}]):
        with pytest.raises(sdk_slots.GovernedCredentialError) as err:
            _many(resolver, items, toolkit="probe")
        assert _code(err) == "STALE_CONTEXT"
    assert len(batch_bus.batch_requests) == 1


# --- run-level release: every toolkit of a run in one call (perf PR 5: resolve_all) -------------------------------------------

def _run_items(n, per_toolkit=1, **extra):
    """n toolkits x per_toolkit slots, each as ``{toolkit_instance_id, field, slot}``."""
    return [{"toolkit_instance_id": f"tk-{t}", "field": f"configuration_{f}",
             "slot": _slot(decision_id=f"decision-{t}-{f}", **extra)}
            for t in range(n) for f in range(per_toolkit)]


def _all(resolver, items):
    return resolver.resolve_all(items=items)


def test_a_run_of_six_toolkits_is_one_release_call_and_two_bus_calls(batch_bus, sdk_slots, keypairs):
    kwargs = _kwargs()
    run_rpc = context_guard.RunRpcNode("node")
    view = context_guard.enforce_execution_context(kwargs, "node", task_name="indexer_agent", project_id=7, rpc=run_rpc)
    resolver = credential_resolver.attach_credential_resolver(
        types.SimpleNamespace(), view, kwargs, "node", task_name="indexer_agent", project_id=7, rpc=run_rpc)
    items = _run_items(6)
    result = _all(resolver, items)
    assert list(result) == [(i["toolkit_instance_id"], i["field"]) for i in items]
    assert all(e.fields["access_token"].get_secret_value() == VALUE for e in result.values())
    assert len(batch_bus.validate_requests) + len(batch_bus.batch_requests) == 2 and batch_bus.requests == []
    (request,) = batch_bus.batch_requests
    assert request["slots"] == [{k: i[k] for k in ("toolkit_instance_id", "field", "slot")} for i in items]
    assert len(batch_bus.nodes) == 1 and len(keypairs) == 1
    assert batch_bus.nodes[0].timeouts == [context_guard.DEFAULT_TIMEOUT_S, 18.0]
    run_rpc.close()


def test_the_same_field_name_in_two_toolkits_is_two_slots(batch_bus, sdk_slots):
    items = [{"toolkit_instance_id": tk, "field": "github_configuration", "slot": _slot(decision_id=f"d-{tk}")}
             for tk in ("a", "b")]
    assert list(_all(_bresolver(), items)) == [("a", "github_configuration"), ("b", "github_configuration")]
    assert len(batch_bus.batch_requests) == 1


def test_a_run_over_the_batch_limit_is_chunked_in_order(batch_bus, sdk_slots, keypairs):
    limit = credential_resolver.MAX_BATCH_SLOTS
    items = _run_items(limit + 1)
    result = _all(_bresolver(), items)
    assert list(result) == [(i["toolkit_instance_id"], i["field"]) for i in items]
    first, second = batch_bus.batch_requests
    assert [s["toolkit_instance_id"] for s in first["slots"]] == [i["toolkit_instance_id"] for i in items[:limit]]
    assert [s["toolkit_instance_id"] for s in second["slots"]] == [items[limit]["toolkit_instance_id"]]
    assert batch_bus.requests == [] and len(batch_bus.nodes) == 1
    assert len({r["recipient_key"] for r in batch_bus.batch_requests}) == 2 == len(keypairs)  # a keypair per call
    assert batch_bus.nodes[0].timeouts == [credential_resolver.batch_timeout(15, limit),
                                           credential_resolver.batch_timeout(15, 1)]


def test_a_refused_slot_is_named_in_the_log_only_and_stops_the_run(batch_bus, sdk_slots, logs):
    batch_bus.override = {"ok": False, "reason_code": "DENIED_GRANT", "message": "must not matter",
                          "toolkit_instance_id": "tk-3", "field": "configuration_0"}
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _all(_bresolver(), _run_items(6))
    assert _code(err) == "DENIED_GRANT"
    assert "tk-3" not in str(err.value) and "configuration_0" not in str(err.value) and "must not matter" not in str(err.value)
    assert any("DENIED_GRANT" in line and "tk-3" in line and "configuration_0" in line for line in logs)
    assert len(batch_bus.batch_requests) == 1 and batch_bus.requests == []


def test_a_refusal_without_a_named_slot_logs_no_names_and_keeps_its_code(batch_bus, sdk_slots, logs):
    batch_bus.override = {"ok": False, "reason_code": "RATE_LIMITED", "toolkit_instance_id": {"x": 1}, "field": ""}
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _all(_bresolver(), _run_items(2))
    assert _code(err) == "RATE_LIMITED" and not any("credential_slot_refusal" in line for line in logs)


def test_a_later_chunk_refusal_aborts_the_run_with_its_code(batch_bus, sdk_slots):
    real = batch_bus.release_many

    def second_fails(**request):
        if not batch_bus.batch_requests:  # the first chunk is served
            return real(**request)
        batch_bus.batch_requests.append(request)
        return {"ok": False, "reason_code": "STORE_UNAVAILABLE", "toolkit_instance_id": "tk-64", "field": "configuration_0"}

    batch_bus.release_many = second_fails
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _all(_bresolver(), _run_items(credential_resolver.MAX_BATCH_SLOTS + 1))
    assert _code(err) == "STORE_UNAVAILABLE" and len(batch_bus.batch_requests) == 2


@pytest.mark.parametrize("tamper", ["swapped_toolkit", "missing", "duplicate_key"])
def test_the_run_reply_must_match_the_requested_toolkit_and_field_pairs(batch_bus, sdk_slots, tamper):
    real = batch_bus.release_many

    def release_many(**request):
        reply = real(**request)
        releases = reply["releases"]
        if tamper == "swapped_toolkit":
            releases[0]["toolkit_instance_id"] = releases[1]["toolkit_instance_id"]
        elif tamper == "missing":
            releases.pop()
        else:
            releases[1] = {**releases[0], "release_id": "rel-other"}
        return reply

    batch_bus.release_many = release_many
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _all(_bresolver(), _run_items(3))
    assert _code(err) == "DEPENDENCY_UNAVAILABLE"


@pytest.mark.parametrize("items", [
    "x", None, [None], [{}], [{"field": "a", "slot": _slot()}],
    [{"toolkit_instance_id": "", "field": "a", "slot": _slot()}],
    [{"toolkit_instance_id": 12, "field": "a", "slot": _slot()}],
    [{"toolkit_instance_id": "t", "slot": _slot()}],
    [{"toolkit_instance_id": "t", "field": "", "slot": _slot()}],
    [{"toolkit_instance_id": "t", "field": "a", "slot": "nope"}],
    [{"toolkit_instance_id": "t", "field": "a", "slot": _slot()}] * 2,  # (toolkit, field) at most once
])
def test_run_items_with_a_bad_shape_or_a_repeated_pair_are_refused_before_any_call(batch_bus, sdk_slots, items):
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _all(_bresolver(), items)
    assert _code(err) == "STALE_CONTEXT"
    assert batch_bus.batch_requests == [] and batch_bus.requests == [] and batch_bus.nodes == []


def test_an_empty_run_makes_no_call_and_no_node(batch_bus, sdk_slots):
    assert _all(_bresolver(), []) == {} and batch_bus.nodes == [] and batch_bus.batch_requests == []


def test_without_the_batch_flag_the_run_falls_back_to_one_resolve_many_per_toolkit(bus, sdk_slots, keypairs):
    items = _run_items(3, per_toolkit=2)
    result = _all(_resolver(batch_release=False), items)
    assert list(result) == [(i["toolkit_instance_id"], i["field"]) for i in items]
    assert bus.batch_requests == [] and len(bus.requests) == 6 == len(keypairs)
    assert [(r["toolkit_instance_id"], r["field"]) for r in bus.requests] == list(result)
    assert len(bus.nodes) == 1


def test_the_fallback_groups_interleaved_items_by_toolkit(bus, sdk_slots, monkeypatch):
    calls = []
    resolver = _resolver()
    real = resolver.resolve_many

    def spy(*, toolkit_instance_id, items):
        calls.append((toolkit_instance_id, [i["field"] for i in items]))
        return real(toolkit_instance_id=toolkit_instance_id, items=items)

    monkeypatch.setattr(resolver, "resolve_many", spy)
    slot = _slot()
    items = [{"toolkit_instance_id": tk, "field": f, "slot": slot} for tk, f in (("a", "x"), ("b", "x"), ("a", "y"))]
    result = _all(resolver, items)
    assert calls == [("a", ["x", "y"]), ("b", ["x"])] and set(result) == {("a", "x"), ("a", "y"), ("b", "x")}


def test_a_refusal_in_the_fallback_keeps_its_code_and_stops_at_the_first(bus, sdk_slots):
    bus.override = {"ok": False, "reason_code": "DENIED_GRANT"}
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _all(_resolver(), _run_items(3))
    assert _code(err) == "DENIED_GRANT" and len(bus.requests) == 1


def test_a_bus_failure_of_the_run_call_is_a_dependency_failure_and_never_a_fallback(batch_bus, sdk_slots):
    batch_bus.override = queue.Empty()
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _all(_bresolver(), _run_items(3))
    assert _code(err) == "DEPENDENCY_UNAVAILABLE" and batch_bus.requests == []


def test_a_probe_run_needs_one_ticket_across_all_chunks_before_any_call(batch_bus, sdk_slots):
    resolver = _bresolver(context_ref=None, run_id=None, task_name="indexer_governed_check_connection")
    items = _run_items(credential_resolver.MAX_BATCH_SLOTS + 1, probe_ref=PROBE_REF)
    items[-1]["slot"]["probe_ref"] = "other-ticket"
    with pytest.raises(sdk_slots.GovernedCredentialError) as err:
        _all(resolver, items)
    assert _code(err) == "STALE_CONTEXT" and batch_bus.batch_requests == [] and batch_bus.nodes == []
    assert len(_all(resolver, _run_items(2, probe_ref=PROBE_REF))) == 2


def test_run_release_logs_and_metrics_carry_no_material(batch_bus, sdk_slots, logs):
    before = credential_resolver.CREDENTIAL_SLOT_RESOLUTION.total("ok", "OK")
    _all(_bresolver(), _run_items(4))
    assert credential_resolver.CREDENTIAL_SLOT_RESOLUTION.total("ok", "OK") == before + 4
    assert logs and all(VALUE not in line for line in logs)


# --- S9 / L7: one node per governed run; a legacy run has none -------------------------------------------------------------

def test_validate_and_release_share_one_rpcnode(batch_bus, sdk_slots):
    run_rpc = context_guard.RunRpcNode("node")
    context_guard.enforce_execution_context(_kwargs(), "node", task_name="indexer_agent", rpc=run_rpc)
    assert len(batch_bus.nodes) == 1 and not batch_bus.nodes[0].stopped  # no longer stopped after validate
    _many(_bresolver(rpc=run_rpc), _items(2))
    assert len(batch_bus.nodes) == 1
    run_rpc.close()
    assert batch_bus.nodes[0].stopped


def test_a_governed_run_without_slots_creates_one_node_for_validation_only(batch_bus, sdk_slots):
    run_rpc = context_guard.RunRpcNode("node")
    kwargs = _kwargs()
    view = context_guard.enforce_execution_context(kwargs, "node", task_name="indexer_agent", rpc=run_rpc)
    credential_resolver.attach_credential_resolver(
        types.SimpleNamespace(), view, kwargs, "node", task_name="indexer_agent", project_id=7, rpc=run_rpc)
    assert len(batch_bus.nodes) == 1 and len(batch_bus.validate_requests) == 1
    assert batch_bus.batch_requests == [] and batch_bus.requests == []
    run_rpc.close()


def test_legacy_run_makes_no_bus_call_and_creates_no_rpcnode(batch_bus, sdk_slots):
    kwargs = attach_envelope({"project_id": 7}, "legacy")
    run_rpc = context_guard.RunRpcNode("node")
    client = types.SimpleNamespace()
    view = context_guard.enforce_execution_context(kwargs, "node", task_name="indexer_agent", rpc=run_rpc)
    assert credential_resolver.attach_credential_resolver(
        client, view, kwargs, "node", task_name="indexer_agent", project_id=7, rpc=run_rpc) is None
    run_rpc.close()
    assert vars(client) == {}
    assert batch_bus.validate_requests == [] and batch_bus.batch_requests == [] and batch_bus.requests == []
    assert batch_bus.nodes == []


def test_a_resolver_without_a_shared_node_owns_and_closes_its_own(batch_bus, sdk_slots):
    resolver = _bresolver()
    _many(resolver, _items(2))
    (node,) = batch_bus.nodes
    assert node.started and not node.stopped
    resolver.close()
    assert node.stopped


class _StubModule(types.ModuleType):
    """A module whose every attribute is a mock: stands in for the heavy imports of an entry-point module."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = unittest.mock.MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


ENTRY_STUBS = ("methods.agent_common", "utils.checkpoint_utils", "utils.agent_execution_common",
               "utils.langfuse_callback", "utils.image_helpers", "utils.funcs", "utils.parallel_dispatch_contract",
               "utils.usage_tool_events", "utils.parallel_hitl_router", "utils.predict_router", "utils.mcp_auth_tools",
               "utils.exceptions", "utils.node_interface")
ENTRY_REAL = ("context_guard", "credential_resolver", "envelope_guard", "task_envelope", "credential_slots_support",
              "credential_transport")


@pytest.fixture
def entry(real_bus, sdk_slots, monkeypatch):
    """The real agent entry points (``indexer_agent``, ``indexer_predict_agent``) over the real bus, with their heavy
    collaborators mocked; the event node is the shared one (a non-fork task) and ``create_elitea_client`` fails."""
    pylon_tools = types.ModuleType("pylon.core.tools")
    pylon_tools.log = unittest.mock.MagicMock()
    pylon_tools.web = types.SimpleNamespace(method=lambda *_a, **_k: (lambda fn: fn))
    for name, mod in (("pylon", types.ModuleType("pylon")), ("pylon.core", types.ModuleType("pylon.core")),
                      ("pylon.core.tools", pylon_tools)):
        monkeypatch.setitem(sys.modules, name, mod)
    mcp_oauth = types.ModuleType("elitea_sdk.runtime.utils.mcp_oauth")
    mcp_oauth.McpAuthorizationRequired = type("McpAuthorizationRequired", (Exception,), {})
    monkeypatch.setitem(sys.modules, "elitea_sdk.runtime.utils.mcp_oauth", mcp_oauth)
    sdk_utils = types.ModuleType("elitea_sdk.runtime.utils.utils")  # other test files leave this package stubbed
    sdk_utils.PREDICT_RUN_ID_KWARGS_KEY, sdk_utils.PREDICT_RUN_ID_HEADER = RUN_KEY, "X-Predict-Run-Id"
    monkeypatch.setitem(sys.modules, "elitea_sdk.runtime.utils.utils", sdk_utils)
    root = types.ModuleType("iw6946")
    root.__path__ = [str(PLUGIN_ROOT)]
    monkeypatch.setitem(sys.modules, "iw6946", root)
    for name in ENTRY_REAL:
        monkeypatch.setitem(sys.modules, f"iw6946.utils.{name}", sys.modules[f"utils.{name}"])
    stubs = {}
    for name in ENTRY_STUBS:
        stubs[name] = _StubModule(f"iw6946.{name}")
        monkeypatch.setitem(sys.modules, f"iw6946.{name}", stubs[name])
    common = stubs["utils.agent_execution_common"]
    common.setup_event_node = lambda _context: real_bus.event_node
    common.with_tracing_span = lambda _tp, _name, _sid, _mid, _pid, func, *a, **k: func(*a, **k)

    def failing_client(*_a, **_k):
        raise RuntimeError("client creation failed")

    common.create_elitea_client = failing_client
    task = types.SimpleNamespace(meta={"project_id": 7}, multiprocessing_context="thread")
    monkeypatch.setitem(sys.modules, "tasknode_task", task)
    methods = {}
    for name in ("indexer_agent", "indexer_predict_agent"):
        module = importlib.import_module(f"iw6946.methods.{name}")
        methods[name] = type("Host", (module.Method,), {})()
    yield types.SimpleNamespace(real_bus=real_bus, methods=methods, task=task, common=common,
                                provider=lambda **handlers: _provider(real_bus, **handlers))
    for key in [k for k in sys.modules if k.startswith("iw6946.methods")]:
        del sys.modules[key]


def _run_entry(entry, name, kwargs):
    host = entry.methods[name]
    if name == "indexer_agent":
        return host._indexer_agent_task(None, {}, {}, "tok", {}, stream_id="s", message_id="m", **kwargs)
    return host._indexer_predict_agent_task(None, {}, "tok", {}, stream_id="s", message_id="m", **kwargs)


ENTRY_NAMES = ["indexer_agent", "indexer_predict_agent"]


@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_the_run_node_is_closed_when_the_body_fails_between_validation_and_its_big_try(entry, name):
    """Validation started the shared node; creating the client then fails: nothing stays subscribed on the shared event node."""
    fake = FakeBus()
    fake.batch_release = True
    provider = entry.provider(restricted_validate_execution_context=fake.validate)
    baseline = _subscriptions(entry.real_bus.event_node)
    try:
        with pytest.raises(RuntimeError, match="client creation failed"):
            _run_entry(entry, name, _kwargs())
        assert len(fake.validate_requests) == 1
        assert _subscriptions(entry.real_bus.event_node) == baseline
    finally:
        provider.stop()


@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_the_run_node_is_closed_on_the_refusal_path(entry, name):
    provider = entry.provider(restricted_validate_execution_context=lambda **r: {"ok": False, "reason_code": "EPOCH_MISMATCH"})
    baseline = _subscriptions(entry.real_bus.event_node)
    try:
        _run_entry(entry, name, _kwargs())  # the readable refusal is a result, not an exception
        assert _subscriptions(entry.real_bus.event_node) == baseline
        assert entry.common.execution_error.called is False  # it lives in agent_common
    finally:
        provider.stop()


@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_a_legacy_run_starts_no_node_at_all(entry, name):
    baseline = _subscriptions(entry.real_bus.event_node)
    kwargs = attach_envelope({"project_id": 7}, "legacy")
    with pytest.raises(RuntimeError, match="client creation failed"):
        _run_entry(entry, name, kwargs)
    assert _subscriptions(entry.real_bus.event_node) == baseline


@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_entry_points_hand_the_shared_node_to_validation_and_release(entry, name):
    source = (PLUGIN_ROOT / "methods" / f"{name}.py").read_text()
    assert "rpc=run_rpc" in source[source.index("enforce_execution_context("):source.index("except ContextRejected")]
    assert "rpc=run_rpc" in source[source.index("attach_credential_resolver("):source.index("should_continue = ")]
    assert source.index("credential_resolver.close()") < source.rindex("run_rpc.close()")  # before the event node stops


def test_old_sdk_slot_by_slot_calls_still_resolve_through_the_real_apply(batch_bus, real_sdk_slots):
    settings = {"base_url": "x", "github_configuration": {"credential_slot": _slot()}}
    applied, released = real_sdk_slots.apply_credential_slots(settings, "12", _bresolver())
    assert applied["github_configuration"]["access_token"] == VALUE and released == frozenset({VALUE})
    assert len(batch_bus.batch_requests) == 1 and batch_bus.requests == []
    batch_bus.batch_requests.clear()
    real_sdk_slots.apply_credential_slots(settings, "12", _resolver())  # a run whose Main did not say batch
    assert batch_bus.batch_requests == [] and len(batch_bus.requests) == 1


def test_the_real_sdk_run_level_helpers_release_six_toolkits_in_one_bus_call(batch_bus, real_sdk_slots):
    if not hasattr(real_sdk_slots, "resolve_run_slots"):
        pytest.skip("installed SDK has no run-level slot release")
    resolver = _bresolver()
    settings = {f"tk-{n}": {"github_configuration": {"credential_slot": _slot(decision_id=f"d-{n}")}} for n in range(6)}
    items = [i for tk, s in settings.items() for i in real_sdk_slots.collect_slot_items(s, tk)]
    released = real_sdk_slots.resolve_run_slots(resolver, items)
    assert len(batch_bus.batch_requests) == 1 and batch_bus.requests == []
    for tk, s in settings.items():
        applied, values = real_sdk_slots.apply_credential_slots(s, tk, None, released=released[tk])
        assert applied["github_configuration"]["access_token"] == VALUE and values == frozenset({VALUE})
