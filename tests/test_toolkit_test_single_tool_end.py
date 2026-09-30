"""A toolkit test run must send the tested tool's agent_tool_end once.

EliteACallback emitted it from on_tool_end and indexer_test_toolkit emitted it again to add
content_type, so the tool output crossed the socket twice -- the second copy uncapped and repeated
as a raw `tool_output` dict, and without the tool_run_id the test panel matches on. Re-breaks if the
test run stops deferring the callback's root end, if the manual emit stops reusing the deferred
payload, or if it goes back to carrying `tool_output`.

These read the source: the modules need the pylon runtime and SDK to import, as their siblings
note. Run from this directory (`cd tests && python3 -m pytest test_toolkit_test_single_tool_end.py`).
"""

import ast
import pathlib


def _source(name):
    return (pathlib.Path(__file__).resolve().parents[1] / 'methods' / name).read_text()


def _function(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    #
    raise AssertionError(f"function {name} not found")


def _is_tool_end_emit(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'emit'
        and any(
            kw.arg == 'type' and isinstance(kw.value, ast.Attribute) and kw.value.attr == 'agent_tool_end'
            for kw in node.keywords
        )
    )


def test_the_callback_only_emits_the_root_end_when_not_deferred():
    on_tool_end = _function(_source('agent_common.py'), 'on_tool_end')
    guarded = [
        node for node in ast.walk(on_tool_end)
        if isinstance(node, ast.If) and 'defer_root_tool_end' in ast.unparse(node.test)
    ]
    assert len(guarded) == 1
    guard = guarded[0]
    assert "parent_run_id" in ast.unparse(guard.test)
    assert not any(_is_tool_end_emit(node) for stmt in guard.body for node in ast.walk(stmt))
    assert any(_is_tool_end_emit(node) for stmt in guard.orelse for node in ast.walk(stmt))
    # The guarded emit is the only one -- nothing else in on_tool_end sends the end.
    assert sum(_is_tool_end_emit(node) for node in ast.walk(on_tool_end)) == 1


def test_the_toolkit_test_run_defers_the_callback_root_end():
    tree = ast.parse(_source('indexer_test_toolkit.py'))
    callbacks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'EliteACallback'
    ]
    assert callbacks
    for call in callbacks:
        deferred = [kw for kw in call.keywords if kw.arg == 'defer_root_tool_end']
        assert deferred and isinstance(deferred[0].value, ast.Constant) and deferred[0].value.value is True


def test_the_manual_end_reuses_the_deferred_payload_without_the_raw_output():
    tree = ast.parse(_source('indexer_test_toolkit.py'))
    # The exception handler emits its own synthetic "Toolkit Test Exception" trace; the tested
    # tool's end is the one reading tool_end_metadata.
    emits = [
        node for node in ast.walk(tree)
        if _is_tool_end_emit(node) and 'tool_end_metadata' in ast.unparse(node)
    ]
    assert len(emits) == 1
    emit = emits[0]
    assert 'deferred_end' in ast.unparse(next(kw.value for kw in emit.keywords if kw.arg == 'content'))
    metadata = next(kw.value for kw in emit.keywords if kw.arg == 'response_metadata')
    assert isinstance(metadata, ast.Name) and metadata.id == 'tool_end_metadata'

    literals = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'tool_end_metadata' for t in node.targets)
    ]
    assert len(literals) == 1 and isinstance(literals[0], ast.Dict)
    keys = {key.value for key in literals[0].keys}
    assert 'content_type' in keys and 'tool_output' not in keys

    dumps = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'model_dump' and 'deferred_end' in ast.unparse(node.func.value)
    ]
    assert len(dumps) == 1
    included = {elt.value for elt in next(kw.value for kw in dumps[0].keywords if kw.arg == 'include').elts}
    assert 'tool_run_id' in included and 'tool_output' not in included
