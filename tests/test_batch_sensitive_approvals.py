"""A scheduled run asks the SDK to batch its sensitive-tool approvals into one pause.

The flag reaches the SDK call on the first start and on the re-start after a child
pause. It is passed only when set, so an SDK build without the parameter still works.
The aggregate pause of the SDK leaves the worker as a flat ``hitl_interrupts`` list.
The module needs the pylon runtime to import, so the functions are loaded by source.
"""
import ast
import importlib.util
import json
import re
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / 'utils' / 'agent_execution_common.py'


def _load_contract():
    spec = importlib.util.spec_from_file_location('parallel_dispatch_contract_under_test', ROOT / 'utils' / 'parallel_dispatch_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _functions(names):
    tree = ast.parse(COMMON.read_text())
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]


def _exec(names, extra=None):
    namespace = {'Dict': typing.Dict, 'Any': typing.Any, 'Optional': typing.Optional, 'List': typing.List}
    namespace.update(extra or {})
    exec(compile(ast.Module(body=_functions(names), type_ignores=[]), str(COMMON), 'exec'), namespace)  # noqa: S102 -- heavy module imports, only the helpers under test run
    return namespace


def batch_approval_sdk_kwargs(kwargs):
    return _exec({'batch_approval_sdk_kwargs'})['batch_approval_sdk_kwargs'](kwargs)


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
    assert 'batch_approval_sdk_kwargs(kwargs)' in spread


def test_flag_survives_the_restart_after_a_child_pause():
    payload = reconcile({'batch_sensitive_approvals': True, 'user_input': 'go', 'hitl_resume': True})

    assert payload['batch_sensitive_approvals'] is True
    assert batch_approval_sdk_kwargs(payload) == {'batch_sensitive_approvals': True}


def test_restart_without_the_flag_adds_none():
    payload = reconcile({'user_input': 'go'})

    assert 'batch_sensitive_approvals' not in payload
    assert batch_approval_sdk_kwargs(payload) == {}


class _Node:
    stream_id = 'stream'
    payload_additional_kwargs = {}

    def __init__(self):
        self.emitted = []
        self.event_node = SimpleNamespace(emit=lambda *_a, **_k: None)

    def emit(self, type, content, response_metadata=None):  # pylint: disable=W0622
        self.emitted.append({'type': type, 'content': content, 'response_metadata': response_metadata})


def _emit_pause(response):
    tree = ast.parse(COMMON.read_text())
    tree.body = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ('emit_response_events', 'tool_error_text', '_plain_name')
        or (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ('_TOOL_NAME_MAX_LEN', '_CONTROL_CHARS', '_PLAIN_NAME') for t in n.targets))
    ]
    namespace = {
        'Optional': typing.Optional, 'Dict': typing.Dict, 'Any': typing.Any, 'List': typing.List,
        'NodeEventInterface': object, 'EliteACallback': object, 'EliteACustomCallback': object,
        'HumanMessage': object, 're': re, 'json': json,
        'log': SimpleNamespace(warning=lambda *a, **k: None),
        'EventTypes': SimpleNamespace(
            mcp_authorization_required='mcp_auth', agent_hitl_interrupt='hitl', pipeline_finish='finish',
            agent_response='agent_response', full_message='full_message'),
        'is_fanout_child': contract.is_fanout_child,
        'split_mcp_auth_interrupts': contract.split_mcp_auth_interrupts,
        'normalize_hitl_pause': contract.normalize_hitl_pause,
    }
    exec(compile(tree, str(COMMON), 'exec'), namespace)  # noqa: S102 -- see _exec
    callback = SimpleNamespace(tokens_in=1, tokens_out=2, llm_start_timestamp=None, created_entities=None,
                               tool_calls=[], usage_run_id=None)
    custom = SimpleNamespace(tokens_in=0, tokens_out=0, additional_response_meta={}, modified_files=[], index_statuses=[])
    node = _Node()
    namespace['emit_response_events'](
        node, response, {'content': 'wait'}, 'thread', 'message', callback, custom,
        {'chat_project_id': 5}, {}, [], SimpleNamespace(content='q'))
    return node


def test_aggregate_pause_is_emitted_as_a_flat_list_with_interrupt_ids():
    # The shape the SDK returns for a batched pause: one entry per pending tool call.
    items = [
        {'interrupt_id': 'hitl_a', 'tool_call_id': 'c1', 'guardrail_type': 'sensitive_tool',
         'toolkit_name': 'Artifact', 'tool_name': 'create_file'},
        {'interrupt_id': 'hitl_b', 'tool_call_id': 'c2', 'guardrail_type': 'sensitive_tool',
         'toolkit_name': 'Artifact', 'tool_name': 'delete_file'},
        {'interrupt_id': 'hitl_c', 'tool_call_id': 'c3', 'guardrail_type': 'sensitive_tool',
         'toolkit_name': 'ADO', 'tool_name': 'list_indexes'},
    ]
    node = _emit_pause({'output': 'wait', 'execution_finished': False,
                        'hitl_interrupt': items[0], 'hitl_interrupts': items})

    events = [e for e in node.emitted if e['type'] == 'hitl']
    assert len(events) == 1, node.emitted
    meta = events[0]['response_metadata']
    assert [i['interrupt_id'] for i in meta['hitl_interrupts']] == ['hitl_a', 'hitl_b', 'hitl_c']
    assert all('pending' not in i for i in meta['hitl_interrupts'])
    assert meta['hitl_interrupt']['interrupt_id'] == 'hitl_a'
    assert meta['chat_project_id'] == 5
