"""#6898: per-model usage rollup sent with the final message for per-chat totals and cost.

Loaded by source so the suite runs without the pylon runtime.
"""

import ast
import pathlib
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _function(path, name):
    source = (ROOT / path).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function {name} not found")


def _load():
    namespace = {'USAGE_FIELDS': ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_creation_tokens')}
    source = "from typing import Dict\n" + _function('methods/agent_common.py', 'add_model_usage') \
        + "\n\n" + _function('utils/agent_execution_common.py', '_merge_usage_by_model')
    exec(compile(source, '<usage_by_model>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace['add_model_usage'], namespace['_merge_usage_by_model']


add_model_usage, merge_usage_by_model = _load()


def _response(usage):
    message = SimpleNamespace(usage_metadata=usage)
    return SimpleNamespace(generations=[[SimpleNamespace(message=message)]])


def test_provider_usage_with_cache_tokens():
    rollup = {}
    usage = {'input_tokens': 1200, 'output_tokens': 80,
             'input_token_details': {'cache_read': 1000, 'cache_creation': 50}}
    add_model_usage(rollup, 'claude-sonnet', _response(usage), {}, {'prompt_tokens': 1200, 'completion_tokens': 80})
    assert rollup == {'claude-sonnet': {'input_tokens': 1200, 'output_tokens': 80,
                                        'cache_read_tokens': 1000, 'cache_creation_tokens': 50}}


def test_estimate_path_uses_pending_counts():
    rollup = {}
    add_model_usage(rollup, 'gpt-x', _response(None), {'tokens_in': 300, 'tokens_out': 20}, None)
    assert rollup['gpt-x']['input_tokens'] == 300
    assert rollup['gpt-x']['output_tokens'] == 20
    assert rollup['gpt-x']['cache_read_tokens'] == 0


def test_error_path_has_no_response():
    rollup = {}
    add_model_usage(rollup, None, None, {'tokens_in': 50, 'tokens_out': 0}, None)
    assert rollup == {'unknown': {'input_tokens': 50, 'output_tokens': 0,
                                  'cache_read_tokens': 0, 'cache_creation_tokens': 0}}


def test_calls_to_the_same_model_accumulate():
    rollup = {}
    for _ in range(3):
        add_model_usage(rollup, 'gpt-x', _response(None), {}, {'prompt_tokens': 10, 'completion_tokens': 2})
    assert rollup['gpt-x']['input_tokens'] == 30
    assert rollup['gpt-x']['output_tokens'] == 6


def test_two_callbacks_merge_per_model():
    first = SimpleNamespace(usage_by_model={'a': {'input_tokens': 10, 'output_tokens': 1,
                                                  'cache_read_tokens': 0, 'cache_creation_tokens': 0}})
    second = SimpleNamespace(usage_by_model={'a': {'input_tokens': 5, 'output_tokens': 2,
                                                   'cache_read_tokens': 3, 'cache_creation_tokens': 0},
                                             'b': {'input_tokens': 7, 'output_tokens': 0,
                                                   'cache_read_tokens': 0, 'cache_creation_tokens': 0}})
    legacy = SimpleNamespace()
    merged = merge_usage_by_model([first, second, legacy])
    assert merged['a'] == {'input_tokens': 15, 'output_tokens': 3, 'cache_read_tokens': 3, 'cache_creation_tokens': 0}
    assert merged['b']['input_tokens'] == 7
