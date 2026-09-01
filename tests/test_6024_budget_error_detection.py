"""Budget-rejection detection for #6024 (friendly budget-exceeded messages).

A budget rejection arrives as a plain HTTP 400, indistinguishable by exception class
from an invalid model or a malformed request, so detection keys off the structured
response body instead. These tests pin that contract: the two SDK body shapes we
actually receive, and the negatives that must NOT be swallowed as budget errors.

Loaded by source so the suite runs without the pylon runtime.

Run from this directory (`cd tests && python3 -m pytest test_6024_budget_error_detection.py`):
invoking pytest from the plugin root makes it import the plugin's module.py, which needs
pylon's `tools` and fails collection -- the pre-existing test files here behave the same way.
"""

import ast
import pathlib
import typing


def _load_detector():
    """Exec just the detector and its codes tuple, avoiding pylon imports."""
    source = (pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'funcs.py').read_text()
    namespace = {'Optional': typing.Optional}
    #
    for node in ast.parse(source).body:
        is_target_func = isinstance(node, ast.FunctionDef) and \
            node.name == 'budget_exceeded_error_code'
        is_target_const = isinstance(node, ast.Assign) and any(
            getattr(target, 'id', '') == 'BUDGET_ERROR_CODES' for target in node.targets
        )
        #
        if is_target_func or is_target_const:
            exec(compile(ast.Module([node], []), '<detector>', 'exec'), namespace)  # pylint: disable=W0122
    #
    return namespace['budget_exceeded_error_code'], namespace['BUDGET_ERROR_CODES']


budget_exceeded_error_code, BUDGET_ERROR_CODES = _load_detector()

PROJECT_CODE, MEMBER_CODE = BUDGET_ERROR_CODES


class FakeSDKError(Exception):
    """Stands in for openai/anthropic BadRequestError, which expose .body."""

    def __init__(self, body):
        super().__init__("400")
        self.body = body


def test_openai_shape_is_detected():
    # The OpenAI SDK strips the "error" wrapper before storing .body
    error = FakeSDKError({"type": "budget_exceeded", "code": PROJECT_CODE})
    assert budget_exceeded_error_code(error) == PROJECT_CODE


def test_anthropic_shape_is_detected():
    # The Anthropic SDK keeps the wrapper, so both shapes must be handled
    error = FakeSDKError({"error": {"type": "budget_exceeded", "code": MEMBER_CODE}})
    assert budget_exceeded_error_code(error) == MEMBER_CODE


def test_member_scope_is_preserved():
    # Scope drives which message and usage link the user gets, so it must not collapse
    error = FakeSDKError({"type": "budget_exceeded", "code": MEMBER_CODE})
    assert budget_exceeded_error_code(error) == MEMBER_CODE


def test_unknown_code_falls_back_to_project_scope():
    # A future/unrecognised code should still produce a budget message, not a raw error
    error = FakeSDKError({"type": "budget_exceeded", "code": "something_new"})
    assert budget_exceeded_error_code(error) == PROJECT_CODE


def test_missing_code_falls_back_to_project_scope():
    error = FakeSDKError({"type": "budget_exceeded"})
    assert budget_exceeded_error_code(error) == PROJECT_CODE


def test_other_400_is_not_treated_as_budget_error():
    # The regression that matters: 400 is far too broad to branch on by class alone
    error = FakeSDKError({"error": {"type": "invalid_request_error", "message": "bad model"}})
    assert budget_exceeded_error_code(error) is None


def test_exception_without_body_is_ignored():
    assert budget_exceeded_error_code(Exception("boom")) is None


def test_non_dict_body_is_ignored():
    assert budget_exceeded_error_code(FakeSDKError("a string")) is None
    assert budget_exceeded_error_code(FakeSDKError(None)) is None


def test_messages_exist_for_every_code_and_are_period_neutral():
    source = (
        pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'agent_common.py'
    ).read_text()
    # Bounded to the dict's own closing brace: sibling constants have since been added
    # below it, and their copy must not be able to fail this budget-specific assertion
    start = source.index('BUDGET_EXCEEDED_MESSAGES = {')
    block = source[start:source.index('\n}', start)]
    #
    for code in BUDGET_ERROR_CODES:
        assert code in block
    #
    # Budgets are monthly today, but the copy must not need a rewrite if that changes
    for period in ('monthly', 'weekly', 'daily', 'this month'):
        assert period not in block.lower()
