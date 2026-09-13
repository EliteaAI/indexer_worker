"""The surfaced result and the synthesized status event must follow the tool's payload.

Re-breaks if `final_result` goes back to choosing on `success`, if the status-event fallback
stops being gated on a missing payload, or if `agent_response` moves off the success branch.
These read the source: the module needs the pylon runtime to import, as its siblings note, so
they pin structure rather than behaviour.
"""

import ast
import pathlib

import pytest

PAYLOAD_GUARD = 'tool_produced_a_payload'


def _source(name):
    return (pathlib.Path(__file__).resolve().parents[1] / 'methods' / name).read_text()


def _function(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    #
    raise AssertionError(f"function {name} not found")


def _assigned_value(function_node, target):
    assignments = [
        node.value
        for node in ast.walk(function_node)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == target for t in node.targets)
    ]
    #
    assert len(assignments) == 1, f"expected one assignment to {target}, found {len(assignments)}"
    return assignments[0]


def _emits(node, event_name):
    return any(
        isinstance(inner, ast.Attribute) and inner.attr == event_name
        for inner in ast.walk(node)
    )


TOOLKIT_SOURCE = _source('indexer_test_toolkit.py')


@pytest.fixture(scope='module')
def task():
    return _function(TOOLKIT_SOURCE, '_indexer_test_toolkit_tool_task')


def test_the_payload_guard_asks_whether_the_tool_returned_anything(task):
    guard = _assigned_value(task, PAYLOAD_GUARD)

    assert isinstance(guard, ast.Compare), ast.dump(guard)
    assert isinstance(guard.left, ast.Name) and guard.left.id == 'tool_result'
    assert isinstance(guard.ops[0], ast.IsNot)
    assert isinstance(guard.comparators[0], ast.Constant) and guard.comparators[0].value is None


def test_the_surfaced_result_is_chosen_by_the_payload_guard_not_by_success(task):
    final_result = _assigned_value(task, 'final_result')

    assert isinstance(final_result, ast.IfExp), ast.dump(final_result)
    assert isinstance(final_result.test, ast.Name) and final_result.test.id == PAYLOAD_GUARD


def test_a_payload_free_failure_still_falls_back_to_the_error_string(task):
    final_result = _assigned_value(task, 'final_result')

    assert isinstance(final_result.body, ast.Name) and final_result.body.id == 'tool_result'
    assert isinstance(final_result.orelse, ast.Name) and final_result.orelse.id == 'error_message'


def test_the_synthesized_status_event_is_reserved_for_runs_with_no_payload(task):
    calls = [
        node
        for node in ast.walk(task)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'check_missing_index_data_status_event'
    ]
    assert len(calls) == 1, f"expected one fallback call, found {len(calls)}"
    #
    guarded = [
        node
        for node in ast.walk(task)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == PAYLOAD_GUARD
        and any(call in list(ast.walk(node)) for call in calls)
    ]
    assert guarded, (
        f"the fallback call must stay gated on `not {PAYLOAD_GUARD}`: a returned failure has already "
        "run the tool's own terminal handling, and _build_aborted_result emits no status event on "
        "purpose because the row it would describe may belong to the run that superseded this one - "
        "pylon_main's failed-state writer guards a live run and a cancelled row but not a completed "
        "one, so synthesizing the event here stamps 'failed' on a healthy index"
    )


def test_the_response_event_still_belongs_to_successful_runs_only(task):
    branches = [
        node
        for node in ast.walk(task)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == 'success'
        and _emits(node, 'agent_response')
    ]
    assert len(branches) == 1, "no single `if success:` branch emits agent_response"
    #
    branch = branches[0]
    assert any(_emits(stmt, 'agent_response') for stmt in branch.body), \
        "agent_response is not on the success side"
    assert not any(_emits(stmt, 'agent_response') for stmt in branch.orelse), \
        "agent_response also reaches the failure side"
    assert any(_emits(stmt, 'agent_exception') for stmt in branch.orelse), \
        "the failure side no longer emits agent_exception"
    #
    assert "'finish_reason': 'stop' if success else 'error'" in ast.get_source_segment(TOOLKIT_SOURCE, task)


def test_the_persisted_message_carries_the_reason_beside_the_error_flag(task):
    source = ast.get_source_segment(TOOLKIT_SOURCE, task)

    assert "'is_error': not success," in source
    assert "'error': '' if success else error_message," in source
