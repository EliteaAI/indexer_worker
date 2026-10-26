"""#7126 - project skills reach the SDK configurable and the name registries that attribute loads and chips.

Run: cd tests && PYTHONPATH=.. python3 -m pytest test_7126_project_skills_forwarding.py
"""
import ast
import pathlib

import pytest

from test_6926_skill_usage_rows import callback_module

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]

ATTACHED = [
    {"skill_id": 9, "skill_version_id": 90, "name": "Reviewer", "instructions": "Review."},
    {"skill_id": 3, "skill_version_id": 30, "skill_project_id": 7, "name": "pdf-report", "origin": "participant"},
]
PROJECT = [
    {"skill_id": 10, "skill_version_id": 100, "skill_project_id": 7, "name": "release-notes", "origin": "project"},
    {"skill_id": 12, "skill_version_id": 120, "skill_project_id": 7, "name": "PDF-Report", "origin": "project"},
    {"skill_id": 13, "name": "", "origin": "project"},
    "not a skill",
]


def test_attached_skills_come_first_and_shadow_a_project_copy():
    names = [s["name"] for s in callback_module.disclosed_skills({"attached_skills": ATTACHED, "project_skills": PROJECT})]

    assert names == ["Reviewer", "pdf-report", "release-notes"]


def test_project_skills_alone_are_disclosed():
    assert callback_module.disclosed_skills({"project_skills": PROJECT[:1]}) == PROJECT[:1]


def test_missing_lists_disclose_nothing():
    assert callback_module.disclosed_skills({}) == []
    assert callback_module.disclosed_skills({"attached_skills": None, "project_skills": None}) == []


def test_a_project_skill_identity_keeps_its_project_and_version():
    identities = callback_module.skill_identities(
        callback_module.disclosed_skills({"attached_skills": ATTACHED, "project_skills": PROJECT})
    )

    assert identities["release-notes"] == {"skill_id": 10, "skill_version_id": 100, "skill_project_id": 7}
    assert identities["pdf-report"] == {"skill_id": 3, "skill_version_id": 30, "skill_project_id": 7}


def _function(path, name):
    tree = ast.parse((PLUGIN_ROOT / path).read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _subscript_assignments(function):
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            yield ast.unparse(node)


@pytest.mark.parametrize("path", ["methods/indexer_predict_agent.py", "methods/indexer_agent.py"])
def test_both_agent_entry_points_forward_project_skills_and_register_them(path):
    source = (PLUGIN_ROOT / path).read_text()
    assignments = [line.replace('"', "'") for line in _subscript_assignments(ast.parse(source))]

    assert any(
        line.startswith("invoke_config['configurable']['project_skills'] = kwargs.get('project_skills')")
        for line in assignments
    )
    assert source.count("for s in disclosed_skills(kwargs)") == 1
    assert source.count("attached_skills=disclosed_skills(kwargs)") == 1
    assert "for s in (kwargs.get('attached_skills') or [])" not in source


def test_the_reconcile_payload_carries_project_skills():
    function = _function("utils/agent_execution_common.py", "build_parent_reconcile_payload")
    carry = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "carry_keys"
    )

    assert "'project_skills'" in ast.unparse(carry.value)


def test_the_usage_skill_callback_is_built_over_the_disclosed_skills():
    function = _function("utils/agent_execution_common.py", "create_usage_skill_callback")

    assert "disclosed_skills(kwargs)" in ast.unparse(function)
    assert "kwargs.get('attached_skills')" not in ast.unparse(function)
