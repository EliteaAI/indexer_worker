"""Regression test for #6416: the curated failure message must survive the task boundary.

`execution_error` builds a user-facing `human_readable` string for every exception class it
handles, but only emitted it on the event stream. Blocking callers (`join_task`, used by the AI
draft generators) see the returned dict alone, so they were left with a raw traceback and
reported every failure as "LLM returned an empty response".

Loaded by source so the suite runs without the pylon runtime.
"""

import json
import pathlib
import traceback
import types
import typing
from datetime import datetime, timezone
from uuid import uuid4


def _load_execution_error():
    source = (pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'agent_common.py').read_text()
    body = source[source.index('def execution_error('):source.index('class ToolCallPayload')]

    class _NodeEvent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def model_dump_json(self):
            return '{}'

    namespace = {
        'json': json,
        'traceback': traceback,
        'uuid4': uuid4,
        'datetime': datetime,
        'timezone': timezone,
        'Optional': typing.Optional,
        'NodeEventInterface': object,
        'NodeEvent': _NodeEvent,
        'EventTypes': types.SimpleNamespace(
            agent_tool_start='agent_tool_start',
            agent_tool_end='agent_tool_end',
            agent_exception='agent_exception',
            full_message='full_message',
        ),
        'EVENTNODE_FULL_RESPONSE_NAME': 'full_response',
        'is_fanout_child': lambda meta: False,
        '_sanitize_input_for_event': lambda value: value,
    }
    exec(compile(body, '<execution_error>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace['execution_error']


execution_error = _load_execution_error()


class _NodeInterface:
    stream_id = 'stream-1'
    payload_additional_kwargs = {}
    event_node = types.SimpleNamespace(emit=lambda *a, **k: None)

    def emit(self, *args, **kwargs):
        pass


def _run(human_readable):
    try:
        raise RuntimeError('LLM Provider NOT provided')
    except RuntimeError:
        return execution_error(
            _NodeInterface(),
            'user input',
            [{'role': 'user', 'content': 'hi'}],
            'InternalSDKError on user input',
            'thread-1',
            'message-1',
            {'project_id': 2},
            human_readable=human_readable,
        )


def test_curated_message_is_returned_to_blocking_callers():
    result = _run('The selected model is not available for your team.')

    assert result['human_readable'] == 'The selected model is not available for your team.'
    assert 'Traceback' in result['error']


def test_technical_message_is_the_fallback():
    result = _run(None)

    assert result['human_readable'] == 'InternalSDKError on user input'


def test_chat_history_and_error_are_still_returned():
    result = _run('anything')

    assert result['chat_history'] == [{'role': 'user', 'content': 'hi'}]
    assert set(result) == {'chat_history', 'error', 'human_readable'}
