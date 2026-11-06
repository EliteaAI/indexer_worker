"""A scheduled run asks the SDK to batch its sensitive-tool approvals into one pause.

The flag reaches the SDK call on the first start and on the re-start after a child
pause. It is passed only when set and when the SDK method has the parameter, so an
SDK build without it keeps the one-pause-per-tool behavior. The module needs the
pylon runtime to import, so the functions are loaded by source.
"""
import ast
import importlib.util
import inspect
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

WARNINGS = []
ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / 'utils' / 'agent_execution_common.py'


def _functions(names):
    tree = ast.parse(COMMON.read_text())
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]


def _exec(names, extra=None):
    namespace = {'Dict': typing.Dict, 'Any': typing.Any, 'Optional': typing.Optional, 'List': typing.List,
                 'inspect': inspect, 'log': SimpleNamespace(warning=lambda *a, **k: WARNINGS.append(a))}
    namespace.update(extra or {})
    exec(compile(ast.Module(body=_functions(names), type_ignores=[]), str(COMMON), 'exec'), namespace)  # noqa: S102 -- heavy module imports, only the helpers under test run
    return namespace


def sdk_with_flag(*, memory=None, batch_sensitive_approvals=False):
    return None


def sdk_without_flag(*, memory=None):
    return None


def sdk_with_var_kwargs(*, memory=None, **extra):
    return None


def batch_approval_sdk_kwargs(kwargs, sdk_method=sdk_with_flag):
    return _exec({'batch_approval_sdk_kwargs'})['batch_approval_sdk_kwargs'](kwargs, sdk_method)


reconcile = _exec(
    {'build_parent_reconcile_payload'},
    {'PREDICT_RUN_ID_KWARGS_KEY': 'run_id', 'ROOT_ENTITY_KWARGS_KEY': 'root_entity', 'ENTITY_KWARGS_KEY': 'entity',
     'TRIGGER_SOURCE_KWARGS_KEY': 'trigger', 'ENVELOPE_KEY': 'envelope'},
)['build_parent_reconcile_payload']


@pytest.mark.parametrize('kwargs,expected', [
    ({'batch_sensitive_approvals': True}, {'batch_sensitive_approvals': True}),
    ({'batch_sensitive_approvals': False}, {}),
    ({'batch_sensitive_approvals': 'false'}, {}),
    ({'batch_sensitive_approvals': None}, {}),
    ({}, {}),
], ids=['set', 'false', 'string', 'none', 'absent'])
def test_flag_is_passed_to_the_sdk_only_when_set(kwargs, expected):
    assert batch_approval_sdk_kwargs(kwargs) == expected


def test_flag_is_dropped_with_one_warning_when_the_sdk_method_lacks_the_parameter():
    WARNINGS.clear()

    assert batch_approval_sdk_kwargs({'batch_sensitive_approvals': True}, sdk_without_flag) == {}
    assert len(WARNINGS) == 1


@pytest.mark.parametrize('sdk_method', [sdk_with_flag, sdk_with_var_kwargs], ids=['named', 'var-kwargs'])
def test_flag_is_passed_when_the_sdk_method_accepts_it(sdk_method):
    WARNINGS.clear()

    assert batch_approval_sdk_kwargs({'batch_sensitive_approvals': True}, sdk_method) == {'batch_sensitive_approvals': True}
    assert WARNINGS == []


def test_a_method_without_a_readable_signature_gets_no_flag():
    assert batch_approval_sdk_kwargs({'batch_sensitive_approvals': True}, object()) == {}


def test_an_unset_flag_does_not_inspect_the_method():
    assert batch_approval_sdk_kwargs({}, object()) == {}


@pytest.mark.parametrize('path,sdk_call', [
    ('methods/indexer_predict_agent.py', 'predict_agent'),
    ('methods/indexer_agent.py', 'application'),
])
def test_entry_points_spread_the_flag_into_the_sdk_call(path, sdk_call):
    tree = ast.parse((ROOT / path).read_text())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == sdk_call
        and any(k.arg == 'auto_approve_sensitive_actions' for k in n.keywords)
    ]
    assert len(calls) == 1, f'{path}: expected one client.{sdk_call}(...) call with the approval keywords'
    spread = [ast.unparse(k.value) for k in calls[0].keywords if k.arg is None]
    assert f'batch_approval_sdk_kwargs(kwargs, client.{sdk_call})' in spread


def test_flag_survives_the_restart_after_a_child_pause():
    payload = reconcile({'batch_sensitive_approvals': True, 'user_input': 'go', 'hitl_resume': True})

    assert payload['batch_sensitive_approvals'] is True
    assert batch_approval_sdk_kwargs(payload) == {'batch_sensitive_approvals': True}


def test_restart_without_the_flag_adds_none():
    payload = reconcile({'user_input': 'go'})

    assert 'batch_sensitive_approvals' not in payload
    assert batch_approval_sdk_kwargs(payload) == {}
