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
        self.requests = []
        self.nodes = []
        self.fields = {"access_token": VALUE}
        self.generation = 1
        self.expires_in = 300
        self.override = NO_OVERRIDE   # any reply or exception to return/raise instead of a real sealed reply
        self.seal_decision_id = None  # seal with a different decision id (a Main that disagrees)

    def release(self, **request):
        self.requests.append(request)
        if isinstance(self.override, BaseException):
            raise self.override
        if self.override is not NO_OVERRIDE:
            return self.override
        release_id = f"rel-{len(self.requests)}"
        decision = self.seal_decision_id if self.seal_decision_id is not None else request["slot"].get("decision_id") or ""
        aad = "|".join((
            "v1", str(request["project_id"]), request["context_ref"] or request["probe_ref"],
            request["toolkit_instance_id"], request["field"], decision, release_id,
        )).encode()
        expires = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=self.expires_in))
        expires_at = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
        body = json.dumps({"fields": self.fields, "generation": self.generation, "expires_at": expires_at})
        return {
            "ok": True, "release_id": release_id, "fields": list(self.fields), "expires_at": expires_at,
            "sealed": transport.seal(body.encode(), request["recipient_key"], aad),
        }


@pytest.fixture
def bus(monkeypatch):
    bus = FakeBus()

    class FakeRpc:
        def __init__(self, event_node, id_prefix=None, proxy_timeout=None):
            self.args = (event_node, id_prefix, proxy_timeout)
            self.started = self.stopped = False
            self.proxy = types.SimpleNamespace(restricted_release_credential=bus.release)
            bus.nodes.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    arbiter = types.ModuleType("arbiter")
    rpcnode = types.ModuleType("arbiter.rpcnode")
    rpcnode.RpcNode = FakeRpc
    arbiter.rpcnode = rpcnode
    monkeypatch.setitem(sys.modules, "arbiter", arbiter)
    monkeypatch.setitem(sys.modules, "arbiter.rpcnode", rpcnode)
    return bus


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
    assert node.args[0] == "the-node" and node.args[2] == credential_resolver.DEFAULT_TIMEOUT_S
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
        "credential_slots.v1", "task_envelope.v1"]


def test_governed_ops_capability_rides_only_on_the_pool_announce(monkeypatch):
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: True)
    agents = envelope_guard.capabilities_payload("agents_1", "agents", now=1)
    pool = envelope_guard.capabilities_payload(
        "governed_ops_1", "governed_ops", now=1, extra=governed_ops.governed_ops_capabilities())
    assert "governed_ops_pool.v1" not in agents["capabilities"]
    assert pool["capabilities"] == ["credential_slots.v1", "governed_ops_pool.v1", "task_envelope.v1"]


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
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: True)
    assert approve("q", _requires("credential_slots.v1")) is True
    envelope_guard.enforce_envelope(_requires("credential_slots.v1")["kwargs"])


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
