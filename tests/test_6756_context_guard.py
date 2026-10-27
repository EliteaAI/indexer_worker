"""C-1009 (#6756): a governed run is validated with Main over the worker bus before any work."""

import pathlib
import queue
import sys
import types

import pytest

from utils import context_guard
from utils.context_guard import ContextRejected, enforce_execution_context
from utils.task_envelope import EnvelopeRejected, attach_envelope

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = "0123456789abcdef-secret-tail"
RUN_KEY = context_guard.PREDICT_RUN_ID_KWARGS_KEY


class FakeRpc:
    """Stands in for arbiter's RpcNode; records construction, calls and lifecycle."""
    instances = []
    response = {"ok": True, "context": {"run_id": "run-1", "attempt": 1}}

    def __init__(self, event_node, id_prefix=None, proxy_timeout=None):
        self.args = (event_node, id_prefix, proxy_timeout)
        self.calls = []
        self.started = self.stopped = False
        self.proxy = types.SimpleNamespace(restricted_validate_execution_context=self._call)
        FakeRpc.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def _call(self, **payload):
        self.calls.append(payload)
        if isinstance(FakeRpc.response, BaseException):
            raise FakeRpc.response
        return FakeRpc.response


@pytest.fixture(autouse=True)
def fake_bus(monkeypatch):
    FakeRpc.instances = []
    FakeRpc.response = {"ok": True, "context": {"run_id": "run-1", "attempt": 1}}
    arbiter = types.ModuleType("arbiter")
    rpcnode = types.ModuleType("arbiter.rpcnode")
    rpcnode.RpcNode = FakeRpc
    arbiter.rpcnode = rpcnode
    monkeypatch.setitem(sys.modules, "arbiter", arbiter)
    monkeypatch.setitem(sys.modules, "arbiter.rpcnode", rpcnode)


@pytest.fixture
def logs(monkeypatch):
    lines = []
    monkeypatch.setattr(context_guard, "_log_info", lines.append)
    return lines


def _governed(ref=REF, **extra):
    kwargs = attach_envelope({}, "governed")
    if ref is not None:
        kwargs["envelope"]["context_ref"] = ref
    kwargs.update({"project_id": 7, RUN_KEY: "run-1", "api_token": "tok-secret"}, **extra)
    return kwargs


def _count(result, reason):
    return context_guard.CONTEXT_VALIDATION.total(result, reason)


def test_legacy_and_missing_envelope_make_no_rpc_and_no_count():
    before = context_guard.CONTEXT_VALIDATION.total()
    assert enforce_execution_context({}, object(), task_name="indexer_agent") is None
    assert enforce_execution_context(
        attach_envelope({"project_id": 7}, "legacy"), object(), task_name="indexer_agent") is None
    assert FakeRpc.instances == []
    assert context_guard.CONTEXT_VALIDATION.total() == before


def test_governed_without_context_ref_is_stale_and_makes_no_rpc():
    before = _count("deny", "STALE_CONTEXT")
    with pytest.raises(ContextRejected) as err:
        enforce_execution_context(_governed(ref=None), object(), task_name="indexer_agent")
    assert err.value.reason_code == "STALE_CONTEXT"
    assert isinstance(err.value, EnvelopeRejected)
    assert FakeRpc.instances == []
    assert _count("deny", "STALE_CONTEXT") == before + 1


def test_ok_response_returns_context_counts_and_cleans_up(logs):
    before = _count("ok", "OK")
    node = object()
    ctx = enforce_execution_context(_governed(), node, task_name="indexer_agent", timeout_s=3)
    assert ctx == {"run_id": "run-1", "attempt": 1}
    assert _count("ok", "OK") == before + 1
    (rpc,) = FakeRpc.instances
    assert rpc.args == (node, "indexer_ctx_", 3)
    assert rpc.started and rpc.stopped
    assert "security_metric name=context_validation_total result=ok reason_code=OK" in logs


def test_default_timeout_is_five_seconds():
    enforce_execution_context(_governed(), object(), task_name="indexer_agent")
    assert FakeRpc.instances[0].args[2] == 5


def test_caller_payload_is_exactly_the_three_keys_and_the_run_id():
    enforce_execution_context(_governed(), object(), task_name="indexer_predict_agent")
    (call,) = FakeRpc.instances[0].calls
    assert call == {
        "project_id": 7,
        "context_ref": REF,
        "caller": {"kind": "indexer_worker", "task_name": "indexer_predict_agent", "run_id": "run-1"},
    }


def test_meta_project_id_wins_over_payload():
    enforce_execution_context(_governed(), object(), task_name="indexer_agent", project_id=42)
    assert FakeRpc.instances[0].calls[0]["project_id"] == 42


def test_deny_response_raises_with_main_reason_code(logs):
    FakeRpc.response = {"ok": False, "reason_code": "EPOCH_MISMATCH", "message": "revoked"}
    before = _count("deny", "EPOCH_MISMATCH")
    with pytest.raises(ContextRejected) as err:
        enforce_execution_context(_governed(), object(), task_name="indexer_agent")
    assert err.value.reason_code == "EPOCH_MISMATCH"
    assert _count("deny", "EPOCH_MISMATCH") == before + 1
    assert FakeRpc.instances[0].stopped


@pytest.mark.parametrize("response", [None, {}, {"ok": True}, {"ok": False}, "yes"])
def test_malformed_response_is_refused(response):
    FakeRpc.response = response
    with pytest.raises(ContextRejected) as err:
        enforce_execution_context(_governed(), object(), task_name="indexer_agent")
    assert err.value.reason_code == "STALE_CONTEXT"


@pytest.mark.parametrize("failure", [queue.Empty(), TimeoutError("slow"), RuntimeError("bus down")])
def test_timeout_or_error_is_dependency_unavailable_and_creates_nothing(failure):
    FakeRpc.response = failure
    before = _count("deny", "DEPENDENCY_UNAVAILABLE")
    with pytest.raises(ContextRejected) as err:
        enforce_execution_context(_governed(), object(), task_name="indexer_agent")
    assert err.value.reason_code == "DEPENDENCY_UNAVAILABLE"
    assert _count("deny", "DEPENDENCY_UNAVAILABLE") == before + 1
    assert FakeRpc.instances[0].stopped


def test_logs_never_carry_more_than_eight_ref_chars_or_kwargs(logs):
    FakeRpc.response = queue.Empty()
    with pytest.raises(ContextRejected):
        enforce_execution_context(_governed(), object(), task_name="indexer_agent")
    FakeRpc.response = {"ok": True, "context": {}}
    enforce_execution_context(_governed(), object(), task_name="indexer_agent")
    text = "\n".join(logs)
    assert REF[:8] in text
    assert REF[:9] not in text
    assert "tok-secret" not in text
    assert "api_token" not in text


CODES = ["STALE_CONTEXT", "EPOCH_MISMATCH", "BINDING_DIGEST_MISMATCH", "DENIED_BINDING",
         "DEPENDENCY_UNAVAILABLE", "UNKNOWN_SCHEMA_VERSION",
         # governed credential release (#6946)
         "CREDENTIAL_REFERENCE_REQUIRED", "CREDENTIAL_REFERENCE_INACTIVE", "CREDENTIAL_GENERATION_MISMATCH",
         "DENIED_GRANT", "DENIED_RBAC", "MANDATE_INACTIVE", "MANDATE_EXPIRED", "RUNTIME_LIMIT_EXCEEDED", "RATE_LIMITED", "AUDIT_UNAVAILABLE",
         "WORKER_CAPABILITY_UNAVAILABLE", "UNSUPPORTED_AUTH_MODE", "STORE_DENIED", "STORE_SEALED",
         "STORE_UNAVAILABLE", "STORE_TLS_INVALID", "STORE_NOT_FOUND", "STORE_VERSION_CONFLICT",
         # a sub-agent version fetched on the run token (#7229)
         "DENIED_DEVELOPMENT_CHILD"]


@pytest.mark.parametrize("code", CODES)
def test_every_refusal_code_has_a_fixed_value_free_message(code):
    message = context_guard.rejection_message(code)
    assert message == context_guard.REJECTION_MESSAGES[code]
    assert message != context_guard.DEFAULT_REJECTION_MESSAGE
    assert not any(ch in message for ch in "{}%") and code not in message


def test_unknown_code_gets_the_default_message():
    assert context_guard.rejection_message("SOMETHING_NEW") == context_guard.DEFAULT_REJECTION_MESSAGE
    assert set(context_guard.REJECTION_MESSAGES) == set(CODES)


@pytest.mark.parametrize("rel, name", [
    ("methods/indexer_agent.py", "indexer_agent"),
    ("methods/indexer_predict_agent.py", "indexer_predict_agent"),
])
def test_hook_sits_between_event_node_and_client_creation(rel, name):
    source = (PLUGIN_ROOT / rel).read_text()
    node = source.index("local_event_node = setup_event_node(")
    guard = source.index("enforce_execution_context(", node)
    client = source.index("create_elitea_client(client_args", node)
    assert node < guard < client
    hook = source[guard:client]
    assert f'task_name="{name}"' in hook


@pytest.mark.parametrize("rel", ["methods/indexer_agent.py", "methods/indexer_predict_agent.py"])
def test_rejection_is_reported_as_a_readable_error_and_cleans_up(rel):
    source = (PLUGIN_ROOT / rel).read_text()
    start = source.index("except ContextRejected as rejected:")
    hook = source[start:source.index("create_elitea_client(client_args", start)]
    assert "raise" not in hook
    assert "return execution_error" not in hook and "result = execution_error(" in hook
    assert "human_readable=rejection_message(rejected.reason_code)" in hook
    assert 'result["reason_code"] = rejected.reason_code' in hook
    assert "node_interface.flush()" in hook
    assert 'tasknode_task.multiprocessing_context == "fork"' in hook and "local_event_node.stop()" in hook
    # cleanup runs on every exit of the readable path
    assert hook.index("finally:") < hook.index("node_interface.flush()") < hook.index("local_event_node.stop()")
