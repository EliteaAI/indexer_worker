"""Swarm authorization pauses stay visible in the chat (#7470).

A tool that was already running when a sibling paused the run for authorization still finishes in the timeline.

A swarm batch runs its tools together. When one raises McpAuthorizationRequired, the pause gate
closes while the sibling is still running; the SDK keeps the sibling's result, so its end event
must reach the chat too. Tools that start after the pause stay hidden.

Loads the real module through a stub package so the suite runs without the pylon runtime or the SDK.
"""
import importlib
import pathlib
import sys
import types
from uuid import uuid4

import test_6677_evaluation_attribution  # noqa: F401  installs the pylon/elitea_sdk stubs

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = 'indexer_worker_7470'


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)


class McpAuthorizationRequired(Exception):
    pass


def _load_agent_common():
    _stub('elitea_sdk.tools')
    _stub('elitea_sdk.tools.utils')
    _stub('elitea_sdk.runtime.utils.trace_limits',
          TRACE_STEP_FIELD_MAX_CHARS=100000,
          cap_trace_json=lambda value, *_a, **_k: value,
          cap_trace_text=lambda value, *_a, **_k: value)
    _stub('elitea_sdk.tools.utils.serialization', to_json_primitive=lambda value: value)
    _stub('elitea_sdk.runtime.utils.mcp_oauth',
          McpAuthorizationRequired=McpAuthorizationRequired,
          infer_authorization_servers_from_realm=lambda *_a, **_k: [],
          build_mcp_auth_decision_result=lambda *_a, **_k: {})
    for name, path in ((PKG, ROOT), (f'{PKG}.methods', ROOT / 'methods'), (f'{PKG}.utils', ROOT / 'utils')):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules.setdefault(name, package)
    return importlib.import_module(f'{PKG}.methods.agent_common')


agent_common = _load_agent_common()


class _EventNode:
    def emit(self, name, payload):
        pass


class _NodeInterface:
    def __init__(self):
        self.event_node = _EventNode()
        self.stream_id = 'stream-1'
        self.payload_additional_kwargs = {}
        self.emitted = []

    def emit(self, **kwargs):
        self.emitted.append(kwargs)

    def decorate_thinking_step_for_persistence(self, step):
        return step

    def decorate_tool_call_for_persistence(self, tool_call):
        return tool_call


def _start(callback, name):
    run_id = uuid4()
    callback.on_tool_start({'name': name}, '{}', run_id=run_id, name=name)
    return run_id


def _ended(callback):
    return [
        event['response_metadata']['tool_name'] for event in callback.node_interface.emitted
        if event['type'] == agent_common.EventTypes.agent_tool_end
    ]


def _pause_by_sibling(callback):
    error = McpAuthorizationRequired('Authorization required')
    error.server_url = 'https://mcp.atlassian.com/v1/sse'
    error.toolkit_name = 'Atlassian'
    callback.on_tool_error(error, run_id=_start(callback, 'mcp_authorize_Atlassian'), name='mcp_authorize_Atlassian')


def test_tool_started_before_the_pause_still_finishes():
    callback = agent_common.EliteACallback(_NodeInterface())
    sibling = _start(callback, 'create_issue')

    _pause_by_sibling(callback)
    callback.on_tool_end('issue #1', run_id=sibling, name='create_issue')

    assert callback.mcp_auth_pause_payload is not None
    assert _ended(callback) == ['create_issue']


def test_tool_started_after_the_pause_stays_hidden():
    callback = agent_common.EliteACallback(_NodeInterface())
    _pause_by_sibling(callback)

    late = _start(callback, 'create_issue')
    callback.on_tool_end('issue #1', run_id=late, name='create_issue')

    assert _ended(callback) == []


def test_card_raised_outside_a_tool_run_has_an_id_the_chat_can_render():
    node_interface = _NodeInterface()
    error = McpAuthorizationRequired('Authorization required')
    error.to_dict = lambda: {'server_url': 'https://mcp.atlassian.com/v1/mcp', 'toolkit_name': 'Atlassian'}

    agent_common.build_mcp_auth_required_result(node_interface, error, 2, [])

    card = node_interface.emitted[-1]['response_metadata']
    assert card['tool_run_id']
