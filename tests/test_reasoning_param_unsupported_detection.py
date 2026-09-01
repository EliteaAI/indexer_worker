"""Reasoning-param rejection detection (friendly message instead of a raw SDK error).

A model our own registry flags as reasoning-capable can still have its reasoning
param rejected by the gateway: LiteLLM's Bedrock Converse transformation converts
reasoning_effort -> thinking without re-checking that the target model supports it,
so the provider answers with a plain HTTP 400. Detection therefore keys off the
structured body's rejected-param signature rather than the model name, which is what
keeps it working for whichever model hits the same gap next.

These tests pin that contract: the two SDK body shapes we receive, the param aliases
across provider dialects, and -- most importantly -- the negatives that must NOT be
swallowed, since "unknown_parameter" is far too broad to branch on alone.

Loaded by source so the suite runs without the pylon runtime.

Run from this directory (`cd tests && python3 -m pytest test_reasoning_param_unsupported_detection.py`):
invoking pytest from the plugin root makes it import the plugin's module.py, which needs
pylon's `tools` and fails collection -- the pre-existing test files here behave the same way.
"""

import ast
import pathlib


def _load_detector():
    """Exec just the detector and its param-name tuple, avoiding pylon imports."""
    source = (pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'funcs.py').read_text()
    namespace = {}
    #
    for node in ast.parse(source).body:
        is_target_func = isinstance(node, ast.FunctionDef) and \
            node.name == 'is_reasoning_param_unsupported_error'
        is_target_const = isinstance(node, ast.Assign) and any(
            getattr(target, 'id', '') == 'REASONING_PARAM_NAMES' for target in node.targets
        )
        #
        if is_target_func or is_target_const:
            exec(compile(ast.Module([node], []), '<detector>', 'exec'), namespace)  # pylint: disable=W0122
    #
    return namespace['is_reasoning_param_unsupported_error'], namespace['REASONING_PARAM_NAMES']


is_reasoning_param_unsupported_error, REASONING_PARAM_NAMES = _load_detector()


class FakeSDKError(Exception):
    """Stands in for openai/anthropic BadRequestError, which expose .body."""

    def __init__(self, body, message="400"):
        super().__init__(message)
        self.body = body


def test_anthropic_shape_from_the_reported_incident_is_detected():
    # Verbatim body from the #6xxx traceback: the Anthropic SDK keeps the "error" wrapper
    error = FakeSDKError({"error": {
        "code": "unknown_parameter",
        "message": "Unknown parameter: 'thinking'.",
        "param": "thinking",
        "type": "invalid_request_error",
    }})
    assert is_reasoning_param_unsupported_error(error) is True


def test_openai_shape_is_detected():
    # The OpenAI SDK strips the "error" wrapper before storing .body
    error = FakeSDKError({
        "code": "unknown_parameter",
        "message": "Unknown parameter: 'thinking'.",
        "param": "thinking",
    })
    assert is_reasoning_param_unsupported_error(error) is True


def test_reasoning_effort_alias_is_detected():
    # The param can be rejected under either dialect's name depending on where it is caught
    error = FakeSDKError({"code": "unknown_parameter", "param": "reasoning_effort"})
    assert is_reasoning_param_unsupported_error(error) is True


def test_detected_from_message_when_param_field_is_absent():
    # Not every gateway populates "param", so the message alone must be enough
    error = FakeSDKError({"message": "Unknown parameter: 'thinking'."})
    assert is_reasoning_param_unsupported_error(error) is True


def test_param_matching_is_case_insensitive():
    # Bedrock spells it reasoningConfig; casing must not decide whether users get help
    error = FakeSDKError({"code": "unknown_parameter", "param": "reasoningConfig"})
    assert is_reasoning_param_unsupported_error(error) is True


def test_unknown_parameter_for_an_unrelated_param_is_not_claimed():
    # The regression that matters most: this must not become a catch-all for bad requests,
    # or an unrelated schema error would be reported to users as a reasoning problem
    error = FakeSDKError({
        "code": "unknown_parameter",
        "message": "Unknown parameter: 'top_q'.",
        "param": "top_q",
    })
    assert is_reasoning_param_unsupported_error(error) is False


def test_reasoning_param_named_in_an_unrelated_error_is_not_claimed():
    # Mentioning "thinking" is not enough; the error must actually be a rejected param
    error = FakeSDKError({
        "code": "context_length_exceeded",
        "message": "Too many tokens, including thinking blocks.",
    })
    assert is_reasoning_param_unsupported_error(error) is False


def test_other_400_is_not_treated_as_a_reasoning_error():
    error = FakeSDKError({"error": {"type": "invalid_request_error", "message": "bad model"}})
    assert is_reasoning_param_unsupported_error(error) is False


def test_exception_without_body_falls_back_to_its_string():
    # Some failures reach us already flattened to text, so the signature is still honoured
    assert is_reasoning_param_unsupported_error(
        Exception("400 unknown parameter: 'thinking'")
    ) is True
    assert is_reasoning_param_unsupported_error(Exception("boom")) is False


def test_non_dict_body_is_ignored():
    assert is_reasoning_param_unsupported_error(FakeSDKError("a string")) is False
    assert is_reasoning_param_unsupported_error(FakeSDKError(None)) is False


def test_message_exists_and_is_actionable_without_naming_a_model():
    # Copy must survive the next model hitting this, so it must not hardcode one
    source = (
        pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'agent_common.py'
    ).read_text()
    start = source.index('REASONING_PARAM_UNSUPPORTED_MESSAGE = (')
    block = source[start:source.index('\n)', start)].lower()
    #
    assert 'reasoning' in block
    for model_specific in ('gpt-5.6', 'sol', 'claude', 'bedrock', 'litellm'):
        assert model_specific not in block
