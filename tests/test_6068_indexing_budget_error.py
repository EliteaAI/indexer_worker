"""Indexing must show the friendly budget message, not the raw provider payload (#6068).

A budget rejection during indexing used to reach the user as
``Tool execution failed: Error code: 400 - {'error': {...'code': 'project_budget_exceeded'}}``.
The SDK returned the error as a dict, so its type was lost and nothing downstream could
recognise it.

The exception now propagates, which means it lands in the generic handler of
``_indexer_test_toolkit_tool_task``. Two separate sites there emit user-visible text, and
fixing only one moves the symptom rather than removing it:

* the ``agent_index_data_status`` event's ``error``, which is persisted as the index
  metadata the failure banner reads
* the message handed to ``test_error``, which is what the message list shows

These tests read the source rather than importing it: the module needs the pylon runtime,
as the sibling test files here already note.

Run from this directory: ``cd tests && python3 -m pytest test_6068_indexing_budget_error.py``
"""

import ast
import pathlib


def _source(name):
    return (pathlib.Path(__file__).resolve().parents[1] / 'methods' / name).read_text()


def _function(source, name):
    """Return the source text of one top-level or nested function by name."""
    tree = ast.parse(source)
    #
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    #
    raise AssertionError(f"function {name} not found")


TOOLKIT_SOURCE = _source('indexer_test_toolkit.py')


def test_detector_and_messages_are_imported_not_reimplemented():
    # The detector and the copy already exist for chat and pipelines; a second
    # implementation here would drift out of step with them
    assert 'budget_exceeded_error_code' in TOOLKIT_SOURCE
    assert 'BUDGET_EXCEEDED_MESSAGES' in TOOLKIT_SOURCE
    assert 'def budget_exceeded_error_code' not in TOOLKIT_SOURCE


def test_task_detects_the_budget_scope_once():
    task = _function(TOOLKIT_SOURCE, '_indexer_test_toolkit_tool_task')
    #
    assert task.count('budget_exceeded_error_code(e)') == 1


def test_persisted_index_error_prefers_the_friendly_message():
    # This is what the failure banner reads back; left as str(e) it holds the raw payload
    task = _function(TOOLKIT_SOURCE, '_indexer_test_toolkit_tool_task')
    #
    assert "'error': budget_message or f\"Failed to execute index_data tool: {str(e)}\"" in task


def test_message_list_text_prefers_the_friendly_message():
    task = _function(TOOLKIT_SOURCE, '_indexer_test_toolkit_tool_task')
    #
    assert 'error_msg = budget_message or f"Failed to test toolkit tool: {str(e)}"' in task


def test_scope_is_forwarded_to_test_error():
    task = _function(TOOLKIT_SOURCE, '_indexer_test_toolkit_tool_task')
    #
    assert 'budget_error_code=budget_code' in task


def test_test_error_accepts_the_scope():
    signature = _function(TOOLKIT_SOURCE, 'test_error')
    #
    assert 'budget_error_code' in signature.split('"""')[0]


def test_scope_reaches_the_persisted_message_meta():
    # A bare response_metadata key is dropped by elitea_core's whitelist, so the UI would
    # lose the code on reload even though the live socket update carried it
    handler = _function(TOOLKIT_SOURCE, 'test_error')
    #
    assert "'additional_response_meta'" in handler
    assert '"budget_error_code": budget_error_code' in handler


def test_scope_also_rides_the_live_exception_event():
    # Surfaces that read the stream never see full_message
    handler = _function(TOOLKIT_SOURCE, 'test_error')
    #
    assert 'exception_meta' in handler
    assert 'response_metadata=exception_meta' in handler


def test_no_synthetic_stacktrace_tool_for_a_budget_block():
    # Emitted as tool output it renders ABOVE the error frame, so a wall of provider
    # internals appears where the one actionable sentence should be. The trace stays
    # reachable from the frame's own "Error debugging info" expander.
    handler = _function(TOOLKIT_SOURCE, 'test_error')
    #
    assert 'if not budget_error_code:' in handler
    #
    gated = handler.split('if not budget_error_code:')[1]
    assert 'Toolkit Test Exception' in gated


def test_ordinary_failures_still_get_the_stacktrace_tool():
    # Only budget blocks lose it; a real toolkit failure still needs the trace up front
    handler = _function(TOOLKIT_SOURCE, 'test_error')
    #
    assert handler.count('Toolkit Test Exception') == 2
    assert 'traceback.format_exc()' in handler


def test_budget_block_sends_the_trace_as_the_exception_content():
    # The UI renders exception content inside the error frame's collapsed expander, so the
    # provider detail stays available for diagnosis while the headline comes from the scope
    handler = _function(TOOLKIT_SOURCE, 'test_error')
    #
    assert 'content=error if budget_error_code else error_message' in handler


def test_persisted_message_still_carries_the_friendly_text():
    # full_message is what renders after a reload; a trace there would be the original bug
    handler = _function(TOOLKIT_SOURCE, 'test_error')
    full_message = handler.split('EventTypes.full_message')[1]
    #
    assert 'content=error_message' in full_message


def test_non_budget_failures_keep_their_own_message():
    # The regression that matters: only a budget rejection is reworded
    task = _function(TOOLKIT_SOURCE, '_indexer_test_toolkit_tool_task')
    #
    assert 'Failed to test toolkit tool' in task
    assert 'Failed to execute index_data tool' in task
