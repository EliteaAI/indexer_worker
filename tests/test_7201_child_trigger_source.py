"""#7201: dispatched sub-agents keep the parent's trigger_source on their usage events.

Run: cd tests && python3 -m pytest -q test_7201_child_trigger_source.py
"""
import ast
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'utils/agent_execution_common.py'
tree = ast.parse(source.read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_child_launch_payloads']
namespace = {'Dict': Dict, 'Any': Any, 'PREDICT_RUN_ID_KWARGS_KEY': '_elitea_predict_run_id',
             'ROOT_ENTITY_KWARGS_KEY': 'root_entity', 'entity_from_application': lambda app: None}
exec(compile(tree, str(source), 'exec'), namespace)  # pylint: disable=W0122
build = namespace['build_child_launch_payloads']


def _specs():
    return [{'application_id': 9, 'child_thread_id': ident, 'input': {'task': 'delegated task'},
             'version_details': {'agent_type': 'agent', 'llm_settings': {'model_name': 'm'}}}
            for ident in ('child1', 'child2')]


def test_literal_key_matches_the_usage_event_key():
    events = ast.parse((ROOT / 'utils/usage_tool_events.py').read_text())
    value = next(n.value.value for n in events.body if isinstance(n, ast.Assign)
                 and any(getattr(t, 'id', None) == 'TRIGGER_SOURCE_KWARGS_KEY' for t in n.targets))
    assert value == 'trigger_source'


@pytest.mark.parametrize('trigger', ['scheduled', 'webhook', 'index'])
def test_automated_parent_children_inherit_trigger_source(trigger):
    parent = {'llm': {'kwargs': {'model': 'parent-model'}}, 'trigger_source': trigger}
    before = deepcopy(parent)
    result = build(parent, _specs())
    assert len(result) == 2
    assert all(row['child_payload']['trigger_source'] == trigger for row in result)
    assert parent == before


@pytest.mark.parametrize('parent_extra', [{}, {'trigger_source': None}, {'trigger_source': ''}])
def test_manual_parent_children_have_no_trigger_source(parent_extra):
    parent = {'llm': {'kwargs': {'model': 'parent-model'}}, **parent_extra}
    result = build(parent, _specs())
    assert all('trigger_source' not in row['child_payload'] for row in result)


def test_grandchildren_inherit_through_the_child_payload():
    child = build({'llm': {'kwargs': {'model': 'p'}}, 'trigger_source': 'scheduled'}, _specs())[0]['child_payload']
    grandchild = build(child, _specs())[0]['child_payload']
    assert grandchild['trigger_source'] == 'scheduled'
