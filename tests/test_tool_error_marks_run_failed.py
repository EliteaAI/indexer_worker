"""A failed tool marks a non-interactive run as failed in the final message.

A scheduled or system run has no reader, so the stored run status must carry the
tool failure. A chat run is unchanged. The module needs the pylon runtime to
import, so the functions are loaded by source with small stand-ins.
"""
import ast
import json
import re
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

source = Path(__file__).resolve().parents[1] / 'utils/agent_execution_common.py'
tree = ast.parse(source.read_text())
tree.body = [
    n for n in tree.body
    if (isinstance(n, ast.FunctionDef) and n.name in ('tool_error_text', 'emit_response_events'))
    or (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in ('_TOOL_ERROR_MAX_LEN', '_CONTROL_CHARS') for t in n.targets))
]


class FakeNodeEvent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def model_dump_json(self):
        return json.dumps({'response_metadata': self.kwargs['response_metadata'],
                           'content': self.kwargs['content']})


class FakeInterface:
    stream_id = 'stream'
    payload_additional_kwargs = {}

    def __init__(self):
        self.emitted = []
        self.full_messages = []
        self.event_node = SimpleNamespace(emit=lambda name, payload: self.full_messages.append(payload))

    def emit(self, **kwargs):
        self.emitted.append(kwargs)


namespace = {
    'Optional': typing.Optional, 'Dict': typing.Dict, 'Any': typing.Any, 'List': typing.List,
    'NodeEventInterface': object, 'EliteACallback': object, 'EliteACustomCallback': object,
    'HumanMessage': object, 're': re, 'json': json,
    'NodeEvent': FakeNodeEvent,
    'EventTypes': SimpleNamespace(
        mcp_authorization_required='mcp_auth', agent_hitl_interrupt='hitl', pipeline_finish='finish',
        agent_response='agent_response', full_message='full_message'),
    'EVENTNODE_FULL_RESPONSE_NAME': 'full_response',
    'is_fanout_child': lambda meta: False,
    'split_mcp_auth_interrupts': lambda hitl, hitls: ([], hitl, hitls or []),
    '_merge_usage_by_model': lambda callbacks: {},
    '_collect_applied_skills': lambda applied, attached, calls: [],
}
exec(compile(tree, str(source), 'exec'), namespace)
tool_error_text = namespace['tool_error_text']


def emit(task_meta, tool_error, response=None):
    callback = SimpleNamespace(tokens_in=1, tokens_out=2, llm_start_timestamp=None, created_entities=None,
                               tool_calls=[], usage_run_id=None)
    custom = SimpleNamespace(tokens_in=0, tokens_out=0, additional_response_meta={}, modified_files=[],
                             index_statuses=[])
    interface = FakeInterface()
    namespace['emit_response_events'](
        interface, response or {}, {'content': 'answer'}, 'thread', 'message', callback, custom,
        task_meta, {}, [], SimpleNamespace(content='question'), tool_error=tool_error)
    return interface


def metadata(interface):
    assert len(interface.full_messages) == 1
    return interface.full_messages[0]['response_metadata']


MARKER = {'error': {'toolkit': 'github', 'tool': 'create_issue', 'reason': 'HTTP 403'}}


def test_marker_in_a_non_interactive_run_sets_is_error_and_error_text():
    meta = metadata(emit({'non_interactive': True}, MARKER))

    assert meta['is_error'] is True
    assert meta['error'] == 'github · create_issue: HTTP 403'


def test_marker_in_an_interactive_run_adds_neither_key():
    meta = metadata(emit({}, MARKER))

    assert 'is_error' not in meta
    assert 'error' not in meta


@pytest.mark.parametrize('empty', [None, {}])
def test_no_marker_adds_neither_key(empty):
    meta = metadata(emit({'non_interactive': True}, empty))

    assert 'is_error' not in meta
    assert 'error' not in meta


def test_hitl_pause_emits_no_full_message():
    interface = emit({'non_interactive': True}, MARKER, response={'hitl_interrupt': {'message': 'wait'}})

    assert interface.full_messages == []


def test_content_is_unchanged_by_a_marker():
    interface = emit({'non_interactive': True}, MARKER)

    assert interface.full_messages[0]['content'] == 'answer'


def test_error_text_is_one_short_line_without_control_characters():
    text = tool_error_text({'error': {'toolkit': 'git\nhub', 'tool': 'run\x00it', 'reason': 'a\nb\u2028c\r\td' + 'x' * 500}})

    assert len(text) <= 200
    assert text.startswith('git hub · run it: a b c d')
    assert not re.search(r'[\x00-\x1f\x7f-\x9f\u2028\u2029]', text)


def test_empty_toolkit_gives_the_tool_name_only():
    assert tool_error_text({'error': {'toolkit': '', 'tool': 'create_issue', 'reason': ''}}) == 'create_issue'


def test_empty_toolkit_keeps_the_reason():
    assert tool_error_text({'error': {'toolkit': None, 'tool': 'create_issue', 'reason': 'boom'}}) == 'create_issue: boom'


@pytest.mark.parametrize('marker', [{'error': {'toolkit': 'github', 'tool': '', 'reason': 'boom'}}, {'error': {'toolkit': 'github'}}, {'error': {}}])
def test_marker_with_an_empty_tool_gives_nothing(marker):
    assert tool_error_text(marker) is None
    meta = metadata(emit({'non_interactive': True}, marker))
    assert 'is_error' not in meta
