"""C-1009 (#6756): a governed refusal of an agent run -- the execution context refused at validation, or a credential
slot refused at release -- reaches the chat card as the readable ``{code, error, ref}``. The stack trace, and whatever
unexpected exception led to the refusal, stays in the server log under that support reference: nothing of it is sent
on the socket, stored on the message or returned to a blocking caller.

``execution_error`` is loaded by source so the suite runs without the pylon runtime.
"""

import json
import pathlib
import traceback
import types
import typing
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from utils import context_guard
from utils.context_guard import ContextRejected, enforce_execution_context, refusal_text
from utils.credential_resolver import governed_credential_code, governed_text
from utils.task_envelope import attach_envelope

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
LEAK = "/data/plugins/leak_marker_9f3c.py"   # stands for anything an unexpected exception may carry

STEP4_CODES = (
    "DENIED_SUBJECT_MISMATCH", "DENIED_MANDATE_DELEGATION_UNSUPPORTED", "DENIED_DELEGATION_UNSUPPORTED_CALLER",
    "DENIED_ASSERTION_UNAVAILABLE", "DENIED_ASSERTION_SOURCE_UNSUPPORTED", "DENIED_OAUTH_APP_UNAVAILABLE",
    "DELEGATION_EXCHANGE_FAILED",
)


class _Log:
    def __init__(self):
        self.lines = []

    def warning(self, message, *args):
        self.lines.append(message % args if args else message)


class _NodeEvent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def model_dump_json(self):
        return json.dumps(self.kwargs)


class _NodeInterface:
    """Records everything the run would put on the socket."""

    stream_id = "stream-1"
    payload_additional_kwargs = {}

    def __init__(self):
        self.events = []
        self.event_node = types.SimpleNamespace(emit=lambda name, payload: self.events.append(payload))

    def emit(self, **kwargs):
        self.events.append(kwargs)


def _load_execution_error(log):
    source = (PLUGIN_ROOT / "methods" / "agent_common.py").read_text()
    body = source[source.index("def execution_error("):source.index("class ToolCallPayload")]
    evidence = source[source.index("def envelope_token_source("):source.index("def build_mcp_auth_pause_result(")]
    namespace = {
        "json": json, "traceback": traceback, "uuid4": uuid4, "datetime": datetime, "timezone": timezone,
        "Optional": typing.Optional, "Dict": typing.Dict, "Any": typing.Any,
        "NodeEventInterface": object, "NodeEvent": _NodeEvent,
        "EventTypes": types.SimpleNamespace(
            agent_tool_start="agent_tool_start", agent_tool_end="agent_tool_end",
            agent_exception="agent_exception", full_message="full_message",
        ),
        "EVENTNODE_FULL_RESPONSE_NAME": "full_response",
        "is_fanout_child": lambda meta: False,
        "build_parallel_terminal_error": lambda **kwargs: kwargs,
        "_sanitize_input_for_event": lambda value: value,
        "refusal_text": refusal_text,
        "log": log,
    }
    exec(compile(evidence, "<build_execution_evidence>", "exec"), namespace)  # pylint: disable=W0122
    exec(compile(body, "<execution_error>", "exec"), namespace)  # pylint: disable=W0122
    return namespace["execution_error"]


def _refuse(code, human_readable, log):
    node = _NodeInterface()
    result = _load_execution_error(log)(
        node, "list the files", [], f"GovernedCredentialError: {code}", "thread-1", "message-1", {"project_id": 5},
        human_readable=human_readable, refusal_code=code,
    )
    return node, result


def _everything_sent(node, result):
    return json.dumps(node.events) + json.dumps(result)


def _assert_no_trace(sent):
    for leaked in ("Traceback", LEAK, "RuntimeError", "File \""):
        assert leaked not in sent, leaked


@pytest.fixture
def no_bus(monkeypatch):
    """Main cannot be asked: the bus call fails with an unexpected exception carrying internal detail."""
    def explode(*args, **kwargs):
        raise RuntimeError(f"bus exploded in {LEAK}")
    monkeypatch.setattr(context_guard, "_validate_with_main", explode)


def test_an_unexpected_exception_at_validation_reaches_the_card_as_a_coded_refusal_with_no_trace(no_bus):
    kwargs = attach_envelope({"project_id": 5}, "governed")
    kwargs["envelope"]["context_ref"] = "ctx-1"
    log = _Log()
    try:
        enforce_execution_context(kwargs, object(), task_name="indexer_agent", project_id=5)
    except ContextRejected as rejected:   # what indexer_agent does with it
        node, result = _refuse(rejected.reason_code, context_guard.rejection_message(rejected.reason_code), log)
    else:
        pytest.fail("the run was not refused")

    sent = _everything_sent(node, result)
    _assert_no_trace(sent)
    refusal = json.loads(result["error"])
    assert refusal["code"] == "DEPENDENCY_UNAVAILABLE"
    assert refusal["error"] == context_guard.REJECTION_MESSAGES["DEPENDENCY_UNAVAILABLE"]
    # the support reference names the server log line that holds the trace and the unexpected exception
    assert len(log.lines) == 1 and f"ref={refusal['ref']}" in log.lines[0]
    assert "Traceback" in log.lines[0] and LEAK in log.lines[0]


def test_an_unexpected_exception_behind_a_release_refusal_never_reaches_the_socket():
    class GovernedCredentialError(Exception):   # the SDK's, matched by shape
        def __init__(self, code):
            super().__init__("fixed text")
            self.reason_code = code

    log = _Log()
    try:
        try:
            raise RuntimeError(f"reply could not be opened in {LEAK}")
        except RuntimeError as unexpected:
            raise GovernedCredentialError("DENIED_ASSERTION_UNAVAILABLE") from unexpected
    except Exception as e:  # pylint: disable=W0718  # indexer_agent's last branch
        code = governed_credential_code(e)
        node, result = _refuse(code, governed_text(e, code), log)

    _assert_no_trace(_everything_sent(node, result))
    assert LEAK in log.lines[0]


def test_the_card_gets_code_text_and_reference_live_and_after_a_reload():
    node, result = _refuse("DENIED_ASSERTION_UNAVAILABLE", "readable", _Log())
    by_type = {event.get("type"): event for event in node.events}
    # no synthetic "Agent Exception Stacktrace" tool call
    assert "agent_tool_start" not in by_type and "agent_tool_end" not in by_type
    live = json.loads(by_type["agent_exception"]["content"])                       # the live socket event
    stored = json.loads(by_type["full_message"]["response_metadata"]["error"])     # the message meta after reload
    assert live == stored == json.loads(result["error"])
    assert set(live) == {"code", "error", "ref"} and live["code"] == "DENIED_ASSERTION_UNAVAILABLE"
    assert by_type["full_message"]["content"] == result["human_readable"] == "readable"


@pytest.mark.parametrize("code", STEP4_CODES)
def test_each_release_refusal_of_a_delegated_credential_has_its_own_readable_text(code):
    text = context_guard.rejection_message(code)
    assert text != context_guard.DEFAULT_REJECTION_MESSAGE
    assert "_" not in text and "Microsoft" in text   # no code names or ids, says which sign-in


def test_a_refusal_without_a_reference_keeps_the_two_keys():
    assert json.loads(refusal_text("DENIED_GRANT")) == {
        "code": "DENIED_GRANT", "error": context_guard.REJECTION_MESSAGES["DENIED_GRANT"]}


@pytest.mark.parametrize("module", ["indexer_agent.py", "indexer_predict_agent.py"])
def test_both_agent_runners_hand_the_code_of_every_governed_refusal_to_execution_error(module):
    source = (PLUGIN_ROOT / "methods" / module).read_text()
    rejected = source[source.index("except ContextRejected as rejected:"):]
    assert "refusal_code=rejected.reason_code" in rejected[:rejected.index("return result")]
    governed = source[source.index("governed_code = governed_credential_code(e)"):]
    assert "refusal_code=governed_code" in governed[:governed.index("return result")]
