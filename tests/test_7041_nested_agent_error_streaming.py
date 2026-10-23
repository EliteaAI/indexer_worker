"""Regression tests for #7041: nested agent LLM errors must surface immediately.

Three contracts verified by source inspection so the suite runs without the
pylon runtime:

1. on_llm_error emits a thinking step whose ``text`` is the error message, and
   also persists it via partial_message so run history survives a page reload.

2. on_tool_error sends a human-readable headline as ``content`` even when the
   underlying exception has an empty message (e.g. ``KeyError()``).

3. emit_subagent_invocation_chip sets both ``finish_reason`` and ``error`` from
   the parameters, so ``is_error`` is true in run history for failed children.

4. on_llm_end clears llm_error so a run that recovers from a failed LLM call is
   not incorrectly marked as errored.
"""

import ast
import pathlib


def _source():
    return (pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'agent_common.py').read_text()


def _ast_body(func_name):
    for node in ast.walk(ast.parse(_source())):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.unparse(node)
    raise AssertionError(f"function {func_name!r} not found in agent_common.py")


# ---------------------------------------------------------------------------
# Contract 1 — on_llm_error: error thinking step + partial_message persistence
# ---------------------------------------------------------------------------

def test_on_llm_error_builds_synthetic_thinking_step_with_error_text():
    """The step appended to thinking_steps must carry the LLM error message."""
    body = _ast_body('on_llm_error')
    # error_text is derived from self.llm_error and placed in the step dict
    assert 'error_text' in body
    assert '"text": error_text' in body or "text=error_text" in body or "'text': error_text" in body


def test_on_llm_error_emits_partial_message_for_persistence():
    """on_llm_error must emit a partial_message so the step persists to run history."""
    body = _ast_body('on_llm_error')
    assert 'EVENTNODE_PARTIAL_RESPONSE_NAME' in body
    assert 'partial_message' in body


def test_on_llm_error_uses_local_var_not_list_index():
    """Use a named local variable rather than self.thinking_steps[-1] to avoid fragility."""
    body = _ast_body('on_llm_error')
    assert 'decorated_error_step' in body
    # The old pattern is gone
    assert 'self.thinking_steps[-1]' not in body


def test_on_llm_error_thinking_steps_contains_decorated_step():
    """Both the agent_llm_end event and partial_message must use the same decorated step."""
    body = _ast_body('on_llm_error')
    # decorated_error_step is used in both the emit call and partial_message
    occurrences = body.count('decorated_error_step')
    assert occurrences >= 3  # assignment + agent_llm_end + partial_message


# ---------------------------------------------------------------------------
# Contract 2 — on_tool_error: human-readable content with empty-exc fallback
# ---------------------------------------------------------------------------

def test_on_tool_error_has_empty_exception_fallback():
    """str(exc) can be empty for e.g. KeyError(); fall back to the class name."""
    body = _ast_body('on_tool_error')
    # Must contain: str(tool_exception) or type(tool_exception).__name__
    assert 'type(tool_exception).__name__' in body


def test_on_tool_error_no_headline_in_response_metadata():
    """The unused ``headline`` key must not be included in response_metadata."""
    body = _ast_body('on_tool_error')
    assert '"headline"' not in body and "'headline'" not in body


def test_on_tool_error_content_is_error_headline():
    """content on agent_tool_error must be the human-readable headline, not the traceback."""
    body = _ast_body('on_tool_error')
    # content= must reference error_headline, not error_str
    content_assign = [line for line in body.splitlines() if 'content=error_headline' in line]
    assert content_assign, "content=error_headline not found in on_tool_error"


# ---------------------------------------------------------------------------
# Contract 3 — emit_subagent_invocation_chip: error field set for failed runs
# ---------------------------------------------------------------------------

def test_emit_subagent_invocation_chip_accepts_error_param():
    """The method must accept an ``error`` parameter."""
    body = _ast_body('emit_subagent_invocation_chip')
    assert 'error=None' in body


def test_emit_subagent_invocation_chip_passes_error_to_payload():
    """The ``error`` param must be forwarded to ToolCallPayload."""
    body = _ast_body('emit_subagent_invocation_chip')
    assert 'error=error' in body


# ---------------------------------------------------------------------------
# Contract 4 — on_llm_end: clears llm_error so recovered runs are not flagged
# ---------------------------------------------------------------------------

def test_on_llm_end_clears_llm_error():
    """A successful LLM call must clear any prior llm_error to prevent false positives."""
    body = _ast_body('on_llm_end')
    assert 'self.llm_error = None' in body
