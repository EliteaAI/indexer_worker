"""State-conflict detection for #7193.

A LangGraph add_messages conflict (another run rewrote the thread mid-turn) is a
plain ValueError, so it used to be reported as "agent is configured incorrectly".
These tests pin the narrow detector that routes it to a retry message instead,
and the negatives that must keep the configuration-error wording.

Loaded by source so the suite runs without the pylon runtime.
Run from this directory: `cd tests && python3 -m pytest test_7193_state_conflict_error.py`.
"""

import ast
import pathlib

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_detector():
    source = (PLUGIN_ROOT / 'utils' / 'funcs.py').read_text()
    names = {'is_state_conflict_error', '_STATE_CONFLICT_PREFIX', 'STATE_CONFLICT_MESSAGE'}
    namespace = {}
    for node in ast.parse(source).body:
        is_func = isinstance(node, ast.FunctionDef) and node.name in names
        is_const = isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in names for t in node.targets)
        if is_func or is_const:
            exec(compile(ast.Module([node], []), '<detector>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace['is_state_conflict_error']


is_state_conflict_error = _load_detector()


def test_reducer_conflict_detected():
    # Exact wording raised by langgraph.graph.message.add_messages
    exc = ValueError("Attempting to delete a message with an ID that doesn't exist ('3f889976-1')")
    assert is_state_conflict_error(exc)


def test_real_langgraph_error_detected():
    langgraph = __import__('pytest').importorskip('langgraph.graph.message')
    from langchain_core.messages import HumanMessage, RemoveMessage
    try:
        langgraph.add_messages([HumanMessage(content='x', id='a')], [RemoveMessage(id='missing')])
    except ValueError as exc:
        assert is_state_conflict_error(exc)
    else:
        raise AssertionError('add_messages no longer raises on unknown ID')


def test_other_value_errors_keep_configuration_wording():
    assert not is_state_conflict_error(ValueError('invalid model name'))
    assert not is_state_conflict_error(ValueError(''))


def test_non_value_error_ignored():
    assert not is_state_conflict_error(RuntimeError("Attempting to delete a message with an ID that doesn't exist"))


def test_both_entrypoints_check_conflict_before_generic_handler():
    for name in ('indexer_agent.py', 'indexer_predict_agent.py'):
        source = (PLUGIN_ROOT / 'methods' / name).read_text()
        handler = source.index('except ValueError as e:')
        assert source.index('is_state_conflict_error(e)', handler) < source.index('configured incorrectly', handler)
