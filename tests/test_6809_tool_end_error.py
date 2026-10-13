"""A tool exception the SDK middleware caught must reach the trajectory as an error (#6809).

The middleware returns its error prose instead of raising, so on_tool_end fires with a
success-shaped result and every such call was recorded finish_reason "stop", error None -
eval scored a failed read_file as ok. Re-breaks if on_tool_end stops consulting the recorded
outcome, or if the helper starts reading the output text instead of the envelope.

agent_common needs the pylon runtime to import, as its siblings note, so the helper is lifted
out of the source and run against the real SDK (skipped when the SDK lacks the sink reader).
"""

import ast
import pathlib
from typing import Optional

import pytest

tool_outcome = pytest.importorskip("elitea_sdk.runtime.tool_outcome")
if not hasattr(tool_outcome, "recorded_outcomes"):
    pytest.skip("SDK without recorded_outcomes", allow_module_level=True)

from langchain_core.messages import BaseMessage, ToolMessage  # noqa: E402

ToolOutcome = tool_outcome.ToolOutcome
ToolResultStatus = tool_outcome.ToolResultStatus


def _source():
    return (pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'agent_common.py').read_text()


def _function(name):
    for node in ast.walk(ast.parse(_source())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


@pytest.fixture(scope='module')
def tool_end_error():
    namespace = {
        'BaseMessage': BaseMessage,
        'Optional': Optional,
        'ToolResultStatus': ToolResultStatus,
        'recorded_outcomes': tool_outcome.recorded_outcomes,
    }
    exec(compile(ast.Module([_function('tool_end_error')], []), 'agent_common', 'exec'), namespace)
    return namespace['tool_end_error']


def test_recorded_error_outcome_is_the_error(tool_end_error):
    with tool_outcome.outcome_sink():
        tool_outcome.record_outcome(ToolOutcome(
            status=ToolResultStatus.ERROR, message='Error executing tool: File not found',
        ))
        assert tool_end_error('Tool read_file failed: file not found') == (
            'Error executing tool: File not found'
        )


def test_truncated_outcome_is_not_an_error(tool_end_error):
    with tool_outcome.outcome_sink():
        tool_outcome.record_outcome(ToolOutcome(status=ToolResultStatus.TRUNCATED, message='cut'))
        assert tool_end_error('partial') is None


def test_error_prose_without_an_outcome_is_not_an_error(tool_end_error):
    """The text is never the signal: a file that says "Error" is a successful read."""
    with tool_outcome.outcome_sink():
        assert tool_end_error('Error: this line is file content') is None
    assert tool_end_error('Error executing tool: x') is None


def test_error_status_message_is_the_error(tool_end_error):
    message = ToolMessage(content='Error executing read_file: boom', tool_call_id='c1', status='error')
    assert tool_end_error(message) == 'Error executing read_file: boom'
    assert tool_end_error(ToolMessage(content='ok', tool_call_id='c1')) is None


def test_on_tool_end_records_the_error_on_both_payload_paths():
    body = ast.unparse(_function('on_tool_end'))
    assert 'tool_end_error(raw_output)' in body
    # Read before the BaseMessage unwrap, or the status is gone.
    assert body.index('tool_end_error(raw_output)') < body.index('raw_output = raw_output.content')
    assert body.count('finish_reason=finish_reason') == 1
    assert '.finish_reason = finish_reason' in body
    assert body.count('error=tool_error') == 1
    assert '.error = tool_error' in body
    assert 'finish_reason="stop"' not in body.replace("'", '"')
