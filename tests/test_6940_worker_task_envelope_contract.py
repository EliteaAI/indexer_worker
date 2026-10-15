"""C-1009 (#6940): the worker declines or rejects tasks whose envelope it cannot honour."""

import pathlib
import threading

import pytest

from utils import envelope_guard
from utils.task_envelope import (
    CAPABILITY_ENVELOPE_V1,
    EnvelopeRejected,
    attach_envelope,
)

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _governed(**overrides):
    kwargs = attach_envelope({}, "governed")
    kwargs["envelope"].update(overrides)
    return kwargs


def _query(kwargs):
    return {"name": "indexer_agent", "pool": "agents", "kwargs": kwargs}


def test_supported_capabilities_are_the_envelope_v1_only():
    assert envelope_guard.SUPPORTED_CAPABILITIES == frozenset({CAPABILITY_ENVELOPE_V1})


# --- node approver --------------------------------------------------------------------------

def test_approver_accepts_legacy_and_missing_envelope():
    approve = envelope_guard.envelope_task_approver()
    assert approve("task_start_query", _query({})) is True
    assert approve("task_start_query", _query(attach_envelope({}, "legacy"))) is True
    assert approve("task_start_query", {"name": "x"}) is True
    assert approve("task_start_query", {"kwargs": None}) is True


def test_approver_accepts_governed_payload_it_supports():
    assert envelope_guard.envelope_task_approver()("q", _query(_governed())) is True


def test_approver_declines_unknown_capability():
    kwargs = _governed(required_capabilities=[CAPABILITY_ENVELOPE_V1, "future.v9"])
    assert envelope_guard.envelope_task_approver()("q", _query(kwargs)) is False


@pytest.mark.parametrize("override", [{"schema_version": 2}, {"api_token": "x"}])
def test_approver_declines_unknown_version_or_extra_field(override):
    assert envelope_guard.envelope_task_approver()("q", _query(_governed(**override))) is False


def test_approver_declines_malformed_envelope():
    assert envelope_guard.envelope_task_approver()("q", _query({"envelope": "governed"})) is False


# --- entry-point guard ----------------------------------------------------------------------

def test_guard_passes_legacy_and_missing_envelope_without_counting():
    before = envelope_guard.WORKER_CAPABILITY_REJECT.total()
    envelope_guard.enforce_envelope({})
    envelope_guard.enforce_envelope(attach_envelope({}, "legacy"))
    envelope_guard.enforce_envelope(_governed())
    assert envelope_guard.WORKER_CAPABILITY_REJECT.total() == before


def test_guard_rejects_with_reason_code_and_counts_by_capability():
    kwargs = _governed(required_capabilities=[CAPABILITY_ENVELOPE_V1, "future.v9"])
    with pytest.raises(EnvelopeRejected) as err:
        envelope_guard.enforce_envelope(kwargs)
    assert err.value.reason_code == "CAPABILITY_MISSING"
    assert envelope_guard.WORKER_CAPABILITY_REJECT.total("future.v9") >= 1


def test_guard_counts_schema_rejection_without_a_capability_label():
    with pytest.raises(EnvelopeRejected) as err:
        envelope_guard.enforce_envelope(_governed(schema_version=9))
    assert err.value.reason_code == "UNKNOWN_SCHEMA_VERSION"
    assert envelope_guard.WORKER_CAPABILITY_REJECT.total("-") >= 1


def test_rejection_result_is_a_typed_failure_without_payload_echo():
    result = envelope_guard.rejection_result(EnvelopeRejected("CAPABILITY_MISSING", ("future.v9",)))
    assert result["success"] is False
    assert result["reason_code"] == "CAPABILITY_MISSING"
    assert "future.v9" in result["error"]


# --- capability advertisement ---------------------------------------------------------------

def test_capabilities_payload_shape():
    payload = envelope_guard.capabilities_payload("agents_1", "agents", now=12.5)
    assert payload == {
        "ident": "agents_1", "pool": "agents",
        "capabilities": [CAPABILITY_ENVELOPE_V1], "ts": 12.5,
    }


def test_heartbeat_emits_until_stopped():
    emitted = []
    ticked = threading.Event()

    def emit():
        emitted.append(1)
        ticked.set()

    stop = envelope_guard.start_capability_heartbeat(emit, interval_s=0.01)
    assert ticked.wait(2)
    stop()
    count = len(emitted)
    threading.Event().wait(0.05)
    assert len(emitted) == count


# --- wiring contract (module.py and entry points are not importable without pylon) -----------

def _read(rel):
    return (PLUGIN_ROOT / rel).read_text()


def test_agent_node_gets_node_level_approver_and_heartbeat():
    source = _read("module.py")
    node = source[source.index("self.agent_task_node = arbiter.TaskNode("):
                  source.index("self.agent_task_node.start()")]
    assert "task_approver=envelope_task_approver()" in node
    assert "CAPABILITIES_REQUEST_EVENT, self.task_node_capabilities_request" in source
    assert "start_capability_heartbeat(" in source
    # the light-node check_connection task is guarded by a per-task approver instead
    assert "self.indexer_configuration_check_connection, \"indexer_configuration_check_connection\"," in source


@pytest.mark.parametrize("rel, marker", [
    ("methods/indexer_agent.py", "def indexer_agent("),
    ("methods/indexer_predict_agent.py", "def indexer_predict_agent("),
    ("methods/indexer_test_toolkit.py", "def indexer_test_toolkit_tool("),
    ("methods/indexer_test_toolkit.py", "def indexer_test_mcp_connection("),
    ("methods/indexer_check_connection.py", "def indexer_configuration_check_connection("),
    ("methods/indexer_mcp_sync_tools.py", "def indexer_mcp_sync_tools("),
])
def test_every_entry_point_revalidates_the_envelope(rel, marker):
    source = _read(rel)
    body = source[source.index(marker):]
    nxt = body.find("    @web.method", 1)
    body = body[:nxt] if nxt != -1 else body
    assert "enforce_envelope(kwargs)" in body


def test_parallel_children_and_reconcile_carry_the_parent_envelope():
    source = _read("utils/agent_execution_common.py")
    assert "ENVELOPE_KEY," in source[source.index("def build_parent_reconcile_payload"):]
    child = source[source.index("child_payload = {"):source.index("new_spec['child_payload']")]
    assert "child_payload['envelope'] = parent_kwargs['envelope']" in child
