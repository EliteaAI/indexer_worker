"""Next-input suggestion generation (best-effort, ephemeral, never persisted).

Pins the skip/happy-path matrix for maybe_emit_next_input_suggestion: every
failure mode (disabled, no sid, too-short reply, no low-tier model, timeout,
empty/NONE model output) must silently no-op and never raise, since this
runs after the primary response is already on the wire and must never affect
it. Loaded by source so the suite runs without the pylon runtime (same
approach as test_6024_budget_error_detection.py).

Run from this directory:
`cd tests && python3 -m pytest test_next_input_suggestion.py -q`
"""

import ast
import pathlib
import threading
import typing


def _load_target():
    """Exec just maybe_emit_next_input_suggestion + its prompt constant."""
    source = (pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'agent_execution_common.py').read_text()

    class _FakeLog:
        def warning(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

    namespace = {
        'threading': threading,
        'Optional': typing.Optional,
        'Dict': typing.Dict,
        'Any': typing.Any,
        'log': _FakeLog(),
    }

    for node in ast.parse(source).body:
        is_target_func = isinstance(node, ast.FunctionDef) and node.name == 'maybe_emit_next_input_suggestion'
        is_target_const = isinstance(node, ast.Assign) and any(
            getattr(target, 'id', '') == '_NEXT_INPUT_SUGGESTION_PROMPT' for target in node.targets
        )
        is_target_channel = isinstance(node, ast.Assign) and any(
            getattr(target, 'id', '') == '_EN_NEXT_INPUT_SUGGESTION_READY' for target in node.targets
        )
        if is_target_func or is_target_const or is_target_channel:
            exec(compile(ast.Module([node], []), '<target>', 'exec'), namespace)  # pylint: disable=W0122

    return namespace['maybe_emit_next_input_suggestion'], namespace['_EN_NEXT_INPUT_SUGGESTION_READY']


maybe_emit_next_input_suggestion, _EN_NEXT_INPUT_SUGGESTION_READY = _load_target()

LONG_REPLY = "x" * 200


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, response=None, exc=None, hang=False):
        self._response = response
        self._exc = exc
        self._hang = hang

    def invoke(self, _prompt):
        if self._hang:
            threading.Event().wait(5)
        if self._exc:
            raise self._exc
        return FakeMessage(self._response)


class FakeClient:
    def __init__(self, llm):
        self._llm = llm

    def get_low_tier_llm(self, max_tokens=64):  # noqa: ARG002
        return self._llm


class FakeEventNode:
    def __init__(self):
        self.emitted = []

    def emit(self, channel, payload):
        self.emitted.append((channel, payload))


BASE_CFG = {"enabled": True, "sid": "sid-1", "min_response_chars": 150, "timeout_seconds": 5}


def test_disabled_skips():
    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, FakeClient(FakeLLM("Sure, go ahead")), {**BASE_CFG, "enabled": False},
                                      LONG_REPLY, "s1", "m1")
    assert node.emitted == []


def test_missing_sid_skips():
    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, FakeClient(FakeLLM("Sure, go ahead")), {**BASE_CFG, "sid": None},
                                      LONG_REPLY, "s1", "m1")
    assert node.emitted == []


def test_reply_below_min_chars_skips():
    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, FakeClient(FakeLLM("Sure, go ahead")), BASE_CFG,
                                      "short reply", "s1", "m1")
    assert node.emitted == []


def test_no_low_tier_model_skips():
    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, FakeClient(None), BASE_CFG, LONG_REPLY, "s1", "m1")
    assert node.emitted == []


def test_timeout_skips():
    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, FakeClient(FakeLLM(hang=True)), {**BASE_CFG, "timeout_seconds": 0.05},
                                      LONG_REPLY, "s1", "m1")
    assert node.emitted == []


def test_model_exception_skips():
    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, FakeClient(FakeLLM(exc=RuntimeError("boom"))), BASE_CFG,
                                      LONG_REPLY, "s1", "m1")
    assert node.emitted == []


def test_none_marker_skips():
    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, FakeClient(FakeLLM("NONE")), BASE_CFG, LONG_REPLY, "s1", "m1")
    assert node.emitted == []


def test_empty_string_skips():
    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, FakeClient(FakeLLM("   ")), BASE_CFG, LONG_REPLY, "s1", "m1")
    assert node.emitted == []


def test_happy_path_emits():
    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, FakeClient(FakeLLM("Yes, please add a test.")), BASE_CFG,
                                      LONG_REPLY, "stream-1", "msg-1")
    assert len(node.emitted) == 1
    channel, payload = node.emitted[0]
    assert channel == _EN_NEXT_INPUT_SUGGESTION_READY
    assert payload == {
        "sid": "sid-1",
        "stream_id": "stream-1",
        "message_id": "msg-1",
        "suggestion": "Yes, please add a test.",
    }


def test_never_raises_on_client_blowup():
    class ExplodingClient:
        def get_low_tier_llm(self, max_tokens=64):  # noqa: ARG002
            raise RuntimeError("client is on fire")

    node = FakeEventNode()
    maybe_emit_next_input_suggestion(node, ExplodingClient(), BASE_CFG, LONG_REPLY, "s1", "m1")
    assert node.emitted == []
