"""#7185 - the conversation the platform verified for a run's skill scripts reaches the SDK, a dispatched sub-agent and
the reconcile re-invoke, so all of them keep their files in the same place.

Run: cd tests && python3 -m pytest -q test_7185_skill_scratch_forwarding.py
"""
import ast
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = {'conversation_uuid': '6f1c2b8e-3d4a-4f5b-9c7d-0e1f2a3b4c5d', 'bucket': 'attachments'}


def _function(path, name):
    tree = ast.parse((ROOT / path).read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


@pytest.mark.parametrize("path", ["methods/indexer_predict_agent.py", "methods/indexer_agent.py"])
def test_both_agent_entry_points_hand_the_verified_conversation_to_the_sdk(path):
    assignments = [
        ast.unparse(node).replace('"', "'") for node in ast.walk(ast.parse((ROOT / path).read_text()))
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
    ]

    assert "invoke_config['configurable']['skill_scratch'] = kwargs.get('skill_scratch')" in assignments


def test_a_dispatched_sub_agent_keeps_the_parents_scratch():
    source = ROOT / 'utils/agent_execution_common.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_child_launch_payloads']
    namespace = {'Dict': Dict, 'Any': Any, 'PREDICT_RUN_ID_KWARGS_KEY': '_elitea_predict_run_id',
                 'ROOT_ENTITY_KWARGS_KEY': 'root_entity', 'entity_from_application': lambda app: None}
    exec(compile(tree, str(source), 'exec'), namespace)  # pylint: disable=W0122
    spec = {'application_id': 9, 'child_thread_id': 'child1', 'input': {'task': 'delegated task'},
            'version_details': {'agent_type': 'agent', 'llm_settings': {'model_name': 'm'}}}

    [child] = namespace['build_child_launch_payloads']({'llm': {'kwargs': {}}, 'skill_scratch': SCRATCH}, [spec])

    assert child['child_payload']['skill_scratch'] == SCRATCH


def test_the_reconcile_payload_carries_the_scratch():
    function = _function("utils/agent_execution_common.py", "build_parent_reconcile_payload")
    carry = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "carry_keys"
    )

    assert "'skill_scratch'" in ast.unparse(carry.value)
