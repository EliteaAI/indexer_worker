"""Worker pass-through contracts for the over-cap ~skill mentions (#7232).

The SDK refuses load_skill for these names only if they reach the configurable.
"""

import ast
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _subscript_key(node):
    return node.slice.value if isinstance(node.slice, ast.Constant) else None


def _is_kwargs_get(node, key):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "kwargs"
        and [a.value for a in node.args if isinstance(a, ast.Constant)] == [key]
    )


def _configurable_assignments(tree, key):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Subscript) and _subscript_key(target) == key):
            continue
        if isinstance(target.value, ast.Subscript) and _subscript_key(target.value) == "configurable":
            yield node.value


@pytest.mark.parametrize("relative_path", ["methods/indexer_agent.py", "methods/indexer_predict_agent.py"])
def test_both_chat_entrypoints_pass_the_capped_skills_to_the_sdk(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text())

    values = list(_configurable_assignments(tree, "capped_skills"))

    assert len(values) == 1
    value = values[0]
    assert isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or)
    assert _is_kwargs_get(value.values[0], "capped_skills")


def test_parent_reconcile_preserves_the_capped_skills():
    tree = ast.parse((ROOT / "utils/agent_execution_common.py").read_text())
    carry_keys = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "carry_keys" for t in node.targets)
    )

    assert "capped_skills" in {e.value for e in carry_keys.elts if isinstance(e, ast.Constant)}
