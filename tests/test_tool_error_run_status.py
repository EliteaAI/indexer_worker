"""A failed tool fails a non-interactive run only when the agent gives no final answer.

A scheduled or system run has no reader, so the stored run status must carry an
unrecovered tool failure. A recovered one leaves the run successful. A chat run is unchanged. The module needs the
pylon runtime to import, so the functions are loaded by source with small stand-ins.
"""
import ast
import json
import re
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage, SystemMessage, ToolMessage

source = Path(__file__).resolve().parents[1] / 'utils/agent_execution_common.py'
tree = ast.parse(source.read_text())
tree.body = [
    n for n in tree.body
    if (isinstance(n, ast.FunctionDef) and n.name in (
        'tool_error_text', 'emit_response_events', '_plain_name', '_has_final_answer', '_is_user_message',
        '_replies', '_kind_and_text', '_is_unfinished_reply',
        'normalize_response_content'))
    or (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in ('_TOOL_NAME_MAX_LEN', '_CONTROL_CHARS', '_PLAIN_NAME') for t in n.targets))
]


class FakeNodeEvent:
    def __init__(self, type, stream_id, message_id, response_metadata, content):
        self.response_metadata = response_metadata
        self.content = content

    def model_dump_json(self):
        return json.dumps({'response_metadata': self.response_metadata, 'content': self.content})


class FakeInterface:
    stream_id = 'stream'
    payload_additional_kwargs = {}

    def __init__(self):
        self.emitted = []
        self.full_messages = []
        self.event_node = SimpleNamespace(emit=lambda name, payload: self.full_messages.append(payload))

    def emit(self, type, content, response_metadata):
        self.emitted.append({'type': type, 'content': content, 'response_metadata': response_metadata})


namespace = {
    'Optional': typing.Optional, 'Dict': typing.Dict, 'Any': typing.Any, 'List': typing.List,
    'NodeEventInterface': object, 'EliteACallback': object, 'EliteACustomCallback': object,
    'HumanMessage': object, 're': re, 'json': json,
    'log': SimpleNamespace(warning=lambda *a, **k: None),
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
exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102 -- the module pulls in heavy imports, so only the helpers under test are run
tool_error_text = namespace['tool_error_text']


def emit(task_meta, tool_error, response=None, answer='answer'):
    callback = SimpleNamespace(tokens_in=1, tokens_out=2, llm_start_timestamp=None, created_entities=None,
                               tool_calls=[], usage_run_id=None)
    custom = SimpleNamespace(tokens_in=0, tokens_out=0, additional_response_meta={}, modified_files=[],
                             index_statuses=[])
    interface = FakeInterface()
    namespace['emit_response_events'](
        interface, response or {}, {'content': answer}, 'thread', 'message', callback, custom,
        task_meta, {}, [], SimpleNamespace(content='question'), tool_error=tool_error)
    return interface


def metadata(interface):
    assert len(interface.full_messages) == 1
    return interface.full_messages[0]['response_metadata']


MARKER = {'error': {'toolkit': 'github', 'tool': 'create_issue', 'reason': 'HTTP 403'}}


ERROR_TEXT = ("The step 'create_issue' of github could not be completed. "
              'Open the run log or ask an administrator for details.')


def test_recovered_tool_error_keeps_the_run_successful():
    messages = [HumanMessage(content='create it'), AIMessage(content='The issue could not be created, here is why.')]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages},
                         answer='The issue could not be created, here is why.'))

    assert 'is_error' not in meta
    assert 'error' not in meta


def test_recovered_run_with_an_ai_answer_as_last_message_stays_successful():
    messages = [HumanMessage(content='create it'), ToolMessage(content='HTTP 403', tool_call_id='1'),
                AIMessage(content='The issue could not be created, here is why.')]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages},
                         answer='The issue could not be created, here is why.'))

    assert 'is_error' not in meta
    assert 'error' not in meta


@pytest.mark.parametrize('last', [
    AIMessage(content=''),
    AIMessage(content=[]),
    AIMessage(content=[{'type': 'tool_use', 'id': '1', 'name': 'create_issue', 'input': {}}]),
    ToolMessage(content='HTTP 403', tool_call_id='1'),
    {'type': 'tool', 'content': 'HTTP 403'},
])
def test_output_filled_from_the_tool_error_does_not_count_as_an_answer(last):
    messages = [HumanMessage(content='create it'), ToolMessage(content='HTTP 403', tool_call_id='1'), last]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages}, answer='HTTP 403'))

    assert meta['is_error'] is True
    assert meta['error'] == ERROR_TEXT


@pytest.mark.parametrize('reply', [
    AIMessage(content='Working on it, the issue is created.',
              tool_calls=[{'name': 'create_issue', 'args': {}, 'id': '2'}]),
    {'role': 'assistant', 'content': 'Working on it.', 'tool_calls': [{'name': 'create_issue', 'id': '2'}]},
    AIMessage(content=[{'type': 'text', 'text': 'Working on it.'},
                       {'type': 'tool_use', 'id': '2', 'name': 'create_issue', 'input': {}}]),
], ids=['object-tool-calls', 'dict-tool-calls', 'tool-use-block'])
def test_ai_reply_with_text_and_pending_tool_calls_does_not_count_as_an_answer(reply):
    messages = [HumanMessage(content='create it'), ToolMessage(content='HTTP 403', tool_call_id='1'), reply]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages},
                         answer='Working on it.'))

    assert meta['is_error'] is True
    assert meta['error'] == ERROR_TEXT


@pytest.mark.parametrize('reply', [
    AIMessage(content='Error: the model call failed', response_metadata={'llm_node_error': True}),
    {'type': 'ai', 'content': 'Error: the model call failed', 'response_metadata': {'llm_node_error': True}},
], ids=['object', 'dict'])
def test_ai_reply_tagged_as_a_model_error_does_not_count_as_an_answer(reply):
    messages = [HumanMessage(content='create it'), ToolMessage(content='HTTP 403', tool_call_id='1'), reply]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages},
                         answer='Error: the model call failed'))

    assert meta['is_error'] is True
    assert meta['error'] == ERROR_TEXT


@pytest.mark.parametrize('last', [
    ToolMessage(content='HTTP 403', tool_call_id='1'),
    AIMessage(content='Done, with a note.'),
], ids=['tool-message', 'ai-message'])
def test_a_run_stopped_at_the_step_limit_after_a_failed_tool_has_no_answer(last):
    messages = [HumanMessage(content='create it'), ToolMessage(content='HTTP 403', tool_call_id='1'), last]
    response = {'messages': messages, 'tool_execution_limit_reached': True}
    meta = metadata(emit({'non_interactive': True}, MARKER, response=response,
                         answer='Tool step limit 25 reached for this run.'))

    assert meta['is_error'] is True
    assert meta['error'] == ERROR_TEXT


@pytest.mark.parametrize('last', [
    ToolMessage(content='HTTP 403', tool_call_id='1'),
    {'type': 'tool', 'content': 'HTTP 403'},
    {'role': 'tool', 'content': 'HTTP 403'},
])
def test_pipeline_ending_on_a_tool_message_with_a_different_output_counts_as_an_answer(last):
    messages = [HumanMessage(content='run it'), last]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages},
                         answer='Report: 3 items synced.'))

    assert 'is_error' not in meta
    assert 'error' not in meta


@pytest.mark.parametrize('answer', ['HTTP 403', '  HTTP 403\n', '', '   '])
def test_output_equal_to_the_tool_text_or_blank_does_not_count_as_an_answer(answer):
    messages = [HumanMessage(content='run it'), ToolMessage(content='HTTP 403', tool_call_id='1')]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages}, answer=answer))

    assert meta['is_error'] is True
    assert meta['error'] == ERROR_TEXT


def test_an_output_equal_to_an_earlier_tool_text_does_not_count_as_an_answer():
    messages = [HumanMessage(content='run it'), ToolMessage(content='HTTP 403', tool_call_id='1'),
                ToolMessage(content='', tool_call_id='2')]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages}, answer='HTTP 403'))

    assert meta['is_error'] is True


@pytest.mark.parametrize('response', [{}, {'messages': []}, {'messages': [HumanMessage(content='run it')]},
                                      {'messages': 'not a list'}])
def test_a_response_without_replies_has_no_answer_even_when_output_has_text(response):
    meta = metadata(emit({'non_interactive': True}, MARKER, response=response, answer='HTTP 403'))

    assert meta['is_error'] is True
    assert meta['error'] == ERROR_TEXT


@pytest.mark.parametrize('last', [
    FunctionMessage(content='HTTP 403', name='create_issue'),
    SystemMessage(content='The run was stopped.'),
    {'type': 'function', 'content': 'HTTP 403'},
    {'role': 'system', 'content': 'The run was stopped.'},
    {'content': 'no type at all'},
])
def test_a_function_or_system_message_with_text_does_not_count_as_an_answer(last):
    messages = [HumanMessage(content='create it'), ToolMessage(content='HTTP 403', tool_call_id='1'), last]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages}, answer='Report: done.'))

    assert meta['is_error'] is True
    assert meta['error'] == ERROR_TEXT


@pytest.mark.parametrize('last', [
    {'role': 'assistant', 'content': 'Done, with a note.'},
    {'type': 'ai', 'content': [{'type': 'text', 'text': 'Done, with a note.'}]},
])
def test_assistant_dict_replies_count_as_an_answer(last):
    messages = [HumanMessage(content='create it'), ToolMessage(content='HTTP 403', tool_call_id='1'), last]
    meta = metadata(emit({'non_interactive': True}, MARKER, response={'messages': messages},
                         answer='Done, with a note.'))

    assert 'is_error' not in meta


@pytest.mark.parametrize('answer', ['', '   \n', None])
def test_unrecovered_tool_error_without_an_answer_fails_the_run(answer):
    meta = metadata(emit({'non_interactive': True}, MARKER, answer=answer))

    assert meta['is_error'] is True
    assert meta['error'] == ERROR_TEXT
    assert 'HTTP 403' not in meta['error']


def test_marker_in_an_interactive_run_adds_neither_key():
    meta = metadata(emit({}, MARKER, answer=''))

    assert 'is_error' not in meta
    assert 'error' not in meta


@pytest.mark.parametrize('empty', [None, {}])
def test_no_marker_adds_neither_key(empty):
    meta = metadata(emit({'non_interactive': True}, empty, answer=''))

    assert 'is_error' not in meta
    assert 'error' not in meta


def test_hitl_pause_emits_no_full_message():
    interface = emit({'non_interactive': True}, MARKER, response={'hitl_interrupt': {'message': 'wait'}})

    assert interface.full_messages == []


def test_content_is_unchanged_by_a_marker():
    interface = emit({'non_interactive': True}, MARKER)

    assert interface.full_messages[0]['content'] == 'answer'


def test_error_text_is_a_plain_sentence_without_the_technical_reason():
    reason = "langchain_core.tools.base.ToolException: Node x failed at /data/plugins/a.py with token=abc123 https://u:p@h/x?k=1"
    text = tool_error_text({'error': {'toolkit': 'GitHub toolkit', 'tool': 'create_issue', 'reason': reason}})

    assert text == ("The step 'create_issue' of GitHub toolkit could not be completed. "
                    "Open the run log or ask an administrator for details.")
    for leak in ('langchain', 'ToolException', '/data', 'abc123', 'u:p', 'k=1'):
        assert leak not in text


def test_names_are_cleaned_of_control_direction_and_markup_characters():
    text = tool_error_text({'error': {'toolkit': 'git\nhub\u202e', 'tool': 'run\x00it<b>' + 'x' * 500, 'reason': ''}})

    assert len(text) < 250
    assert '<' not in text and '>' not in text
    assert not re.search(r'[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202e]', text)


def test_empty_toolkit_names_the_step_only():
    text = tool_error_text({'error': {'toolkit': '', 'tool': 'create_issue', 'reason': 'boom'}})

    assert text.startswith("The step 'create_issue' could not be completed.")


@pytest.mark.parametrize('marker', [{'error': {'toolkit': 'github', 'tool': '', 'reason': 'boom'}}, {'error': {'toolkit': 'github'}}])
def test_a_failure_without_a_tool_name_still_fails_a_run_without_an_answer(marker):
    assert tool_error_text(marker).startswith('A step could not be completed.')
    meta = metadata(emit({'non_interactive': True}, marker, answer=''))
    assert meta['is_error'] is True


def test_an_empty_holder_gives_nothing():
    assert tool_error_text({'error': {}}) is None
    assert tool_error_text({}) is None


@pytest.mark.parametrize('method_file', ['indexer_agent.py', 'indexer_predict_agent.py'])
def test_both_agent_methods_open_the_holder_and_pass_it_to_the_response_events(method_file):
    source = (Path(__file__).resolve().parents[1] / 'methods' / method_file).read_text()

    assert 'run_error_flag() as tool_error' in source
    assert 'tool_error=tool_error' in source
    # The holder has to be open around the executor call, not after it
    assert source.index('run_error_flag() as tool_error') < source.index('agent_executor.invoke(')


def test_without_run_error_flag_in_the_sdk_the_worker_runs_as_before_and_says_so():
    common = Path(__file__).resolve().parents[1] / 'utils' / 'agent_execution_common.py'
    tree = ast.parse(common.read_text())
    guards = [n for n in tree.body if isinstance(n, ast.Try)
              and any(isinstance(h.type, ast.Name) and h.type.id == 'ImportError' for h in n.handlers)
              and 'run_error_flag' in ast.unparse(n)]
    assert len(guards) == 1
    guard = guards[0]
    warned = []
    fallback = ast.Module(body=[
        ast.ImportFrom(module='contextlib', names=[ast.alias(name='contextmanager')], level=0),
        *guard.handlers[0].body,
    ], type_ignores=[])
    scope = {'log': SimpleNamespace(warning=lambda *a: warned.append(a))}
    exec(compile(ast.fix_missing_locations(fallback), 'fallback', 'exec'), scope)  # noqa: S102 -- only the fallback branch runs

    with scope['run_error_flag']() as holder:
        pass

    assert holder == {}
    assert warned and 'run_error_flag' in warned[0][0]
