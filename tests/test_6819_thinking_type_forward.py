"""#6819: the model row's reasoning capabilities reach get_llm, and only for the model they describe.

Core puts ``thinking_type`` / ``supported_efforts`` / ``default_effort`` into ``llm.kwargs``; the
worker's get_llm call must copy them into model_config or the SDK falls back to its name list.
A child agent with its own model must not inherit the parent's values.
"""
import ast
from pathlib import Path
from typing import Any, Dict

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = ('thinking_type', 'supported_efforts', 'default_effort')


def _get_llm_model_config_keys():
    tree = ast.parse((PLUGIN_ROOT / 'methods/indexer_predict_agent.py').read_text())
    for node in ast.walk(tree):
        is_get_llm = isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'get_llm'
        if not is_get_llm:
            continue
        model_config = next(keyword.value for keyword in node.keywords if keyword.arg == 'model_config')
        return {
            key.value: ast.unparse(value)
            for key, value in zip(model_config.keys, model_config.values)
            if isinstance(key, ast.Constant)
        }
    raise AssertionError('get_llm call not found')


def test_predict_agent_copies_the_capability_fields_from_llm_kwargs():
    keys = _get_llm_model_config_keys()
    for capability in CAPABILITIES:
        assert keys.get(capability) == f"client_args.get('{capability}')", keys


def _child_builder():
    source = PLUGIN_ROOT / 'utils/agent_execution_common.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_child_launch_payloads']
    namespace = {'Dict': Dict, 'Any': Any, 'PREDICT_RUN_ID_KWARGS_KEY': '_elitea_predict_run_id',
                 'ROOT_ENTITY_KWARGS_KEY': 'root_entity', 'entity_from_application': lambda app: None}
    exec(compile(tree, str(source), 'exec'), namespace)
    return namespace['build_child_launch_payloads']


PARENT = {'llm': {'kwargs': {'model': 'claude-fable-5-1', 'reasoning_effort': 'high', 'thinking_type': 'always_on',
                             'supported_efforts': ['low', 'high'], 'default_effort': 'high'}}}


def test_embedded_child_inherits_the_parent_model_with_its_capabilities():
    child = _child_builder()(PARENT, [{'version_details': {'agent_type': 'agent', 'llm_settings': None}}])[0]['child_payload']
    assert child['llm']['kwargs']['model'] == 'claude-fable-5-1'
    assert {key: child['llm']['kwargs'][key] for key in CAPABILITIES} == {
        'thinking_type': 'always_on', 'supported_efforts': ['low', 'high'], 'default_effort': 'high'}


def test_child_with_its_own_model_does_not_inherit_the_parent_capabilities():
    spec = {'version_details': {'agent_type': 'agent', 'llm_settings': {'model_name': 'claude-haiku-4-5', 'reasoning_effort': 'low'}}}
    child = _child_builder()(PARENT, [spec])[0]['child_payload']
    assert child['llm']['kwargs']['model'] == 'claude-haiku-4-5'
    assert child['llm']['kwargs']['reasoning_effort'] == 'low'
    assert not set(CAPABILITIES) & set(child['llm']['kwargs'])
    assert PARENT['llm']['kwargs']['thinking_type'] == 'always_on'


def test_auto_child_does_not_inherit_the_parent_capabilities():
    spec = {'version_details': {'agent_type': 'agent', 'llm_settings': {'selection': {'mode': 'auto'}}}}
    child = _child_builder()(PARENT, [spec])[0]['child_payload']
    assert child['llm']['kwargs']['model'] is None
    assert not set(CAPABILITIES) & set(child['llm']['kwargs'])
