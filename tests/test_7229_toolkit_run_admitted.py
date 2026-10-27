"""C-1009 (#7229 item 2): the toolkit Test run and on-demand indexing are admitted runs. The runner takes the run token
out of the kwargs first, validates its context with Main before any tool work (a readable refusal otherwise), hangs the
token and the trusted resolver on its client so the SDK releases the credential slots, and closes both. A governed
dispatch Main did not admit (the platform's attachment read, no context, no credential) runs as before."""

import json
import pathlib
import sys
import types

import pytest

from utils import context_guard
from utils.context_guard import ContextRejected, enforce_execution_context, refusal_text, takes_unadmitted_read
from utils.task_envelope import attach_envelope

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (PLUGIN_ROOT / "methods" / "indexer_test_toolkit.py").read_text()
WRAPPER = SOURCE[SOURCE.index("def indexer_test_toolkit_tool("):SOURCE.index("def _indexer_test_toolkit_tool_task(")]
TASK = SOURCE[SOURCE.index("def _indexer_test_toolkit_tool_task("):SOURCE.index("def indexer_test_mcp_connection(")]


def test_the_run_token_is_taken_before_the_envelope_and_given_back_when_the_task_ends():
    assert WRAPPER.index("begin_run_token(kwargs)") < WRAPPER.index("enforce_envelope(kwargs)")
    assert "finally:\n            end_run_token(run_token)" in WRAPPER


def test_a_governed_run_is_validated_before_any_tool_work_and_refused_readably():
    validate = TASK.index("enforce_execution_context(")
    gate = TASK.index("if not takes_unadmitted_read(kwargs):")
    assert gate < validate < TASK.index("client.test_toolkit_tool(")
    assert 'task_name="indexer_test_toolkit_tool"' in TASK[validate:validate + 200]
    refused = TASK[TASK.index("except ContextRejected as rejected:"):]
    refused = refused[:refused.index("return result")]
    # emitted before the forked node stops (events sent after it are dropped), as a refusal, not a traceback
    assert refused.index("test_error(") < refused.index("run_rpc.close()") < refused.index("local_event_node.stop()")
    assert "refusal_code=rejected.reason_code" in refused


def test_a_refusal_is_the_readable_code_and_text_with_no_trace():
    body = SOURCE[SOURCE.index("def test_error("):SOURCE.index("def check_missing_index_data_status_event(")]
    assert "error_message = refusal_text(refusal_code, error_message)" in body
    assert "if not budget_error_code and not refusal_code:" in body   # no synthetic traceback tool call
    assert "result['reason_code'] = refusal_code" in body
    assert "refusal_code=governed_code" in TASK


def test_the_client_gets_the_run_token_and_the_trusted_resolver_and_both_are_closed():
    client = TASK.index("client = EliteAClient(")
    after = TASK[client:]
    assert after.index("attach_run_token(client)") < after.index("attach_credential_resolver(") \
        < after.index("client.test_toolkit_tool(")
    assert 'task_name="indexer_test_toolkit_tool"' in after[after.index("attach_credential_resolver("):][:300]
    tail = TASK[TASK.rindex("finally:"):]
    assert "credential_resolver.close()" in tail and "run_rpc.close()" in tail


def test_a_governed_credential_refusal_reads_as_its_reason():
    assert "error_msg = rejection_message(governed_code)" in TASK
    assert context_guard.rejection_message("NO_SUCH_CODE") == context_guard.DEFAULT_REJECTION_MESSAGE


# --- behaviour: the gate, the validation and the refusal text ----------------------------------------------------------

class FakeRpc:
    """arbiter's RpcNode: what Main answers to the validate call."""
    instances, response = [], {"ok": True, "context": {"run_id": "run-1", "attempt": 1}}

    def __init__(self, event_node, id_prefix=None, proxy_timeout=None):  # pylint: disable=W0613
        self.calls = []
        self.proxy = types.SimpleNamespace(restricted_validate_execution_context=self._call)
        FakeRpc.instances.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def _call(self, **payload):
        self.calls.append(payload)
        return FakeRpc.response


@pytest.fixture(autouse=True)
def fake_bus(monkeypatch):
    FakeRpc.instances = []
    FakeRpc.response = {"ok": True, "context": {"run_id": "run-1", "attempt": 1}}
    arbiter, rpcnode = types.ModuleType("arbiter"), types.ModuleType("arbiter.rpcnode")
    rpcnode.RpcNode, arbiter.rpcnode = FakeRpc, rpcnode
    monkeypatch.setitem(sys.modules, "arbiter", arbiter)
    monkeypatch.setitem(sys.modules, "arbiter.rpcnode", rpcnode)


def governed(context_ref=None, marked=False):
    kwargs = attach_envelope({"project_id": 7, context_guard.PREDICT_RUN_ID_KWARGS_KEY: "run-1"}, "governed")
    kwargs["envelope"]["context_ref"] = context_ref
    if marked is not None and marked is not False:
        kwargs[context_guard.UNADMITTED_KWARG] = marked
    return kwargs


def gate(kwargs):
    """What the runner does before any tool work: the unadmitted read runs as it is, anything else is validated."""
    if takes_unadmitted_read(kwargs):
        return "not validated"
    return enforce_execution_context(kwargs, object(), task_name="indexer_test_toolkit_tool", project_id=7)


def test_only_the_marked_read_without_a_context_is_not_validated_and_the_marker_never_reaches_a_tool():
    kwargs = governed(marked=True)
    assert gate(kwargs) == "not validated" and FakeRpc.instances == []
    assert context_guard.UNADMITTED_KWARG not in kwargs


def test_a_governed_toolkit_run_without_a_context_and_without_the_marker_is_refused_readably():
    for kwargs in (governed(), governed(marked="yes"), governed(marked=1)):   # only Main's own True counts
        with pytest.raises(ContextRejected) as err:
            gate(kwargs)
        assert err.value.reason_code == "STALE_CONTEXT" and context_guard.UNADMITTED_KWARG not in kwargs
    assert FakeRpc.instances == []   # refused before asking Main


def test_a_marker_on_a_run_that_carries_a_context_does_not_skip_its_validation():
    FakeRpc.response = {"ok": False, "reason_code": "EPOCH_MISMATCH"}
    with pytest.raises(ContextRejected) as err:
        gate(governed(context_ref="ctx-1", marked=True))
    assert err.value.reason_code == "EPOCH_MISMATCH"
    [rpc] = FakeRpc.instances
    assert rpc.calls[0]["caller"]["task_name"] == "indexer_test_toolkit_tool"
    FakeRpc.response = {"ok": True, "context": {"run_id": "run-1", "attempt": 1}}
    assert gate(governed(context_ref="ctx-1"))["attempt"] == 1


def test_the_refusal_text_is_the_code_and_the_readable_message_the_card_shows():
    assert json.loads(refusal_text("STALE_CONTEXT")) == {
        "code": "STALE_CONTEXT", "error": context_guard.rejection_message("STALE_CONTEXT")}
    assert json.loads(refusal_text("DENIED_GRANT", "own words")) == {"code": "DENIED_GRANT", "error": "own words"}
