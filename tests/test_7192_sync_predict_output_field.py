"""#7192: the sync predict result carries the final answer as a top-level ``output_text`` string.

Since perf #111 ``chat_history`` is only returned with ``return_chat_history=True``, so a caller
sending just ``user_input`` got a success envelope with no answer anywhere. ``output_text`` is a
single string, so the payload saving stays; ``chat_history`` is still gated by the flag.

``output_text`` is built by ``normalize_response_content``: text chunks are concatenated, while
image/tool/thinking blocks are dropped. Generated images travel separately (thumbnails).

Loads the real functions by AST so the suite runs without the pylon runtime.
"""

import ast
import json
import pathlib
import types
import typing

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    tree = ast.parse((ROOT / 'utils' / 'agent_execution_common.py').read_text())
    wanted = {'build_success_result', 'normalize_response_content', 'build_output_message'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    noop = lambda *_a, **_k: None  # noqa: E731
    namespace = {
        'Optional': typing.Optional, 'Dict': typing.Dict, 'Any': typing.Any, 'List': typing.List,
        'EliteACallback': object, 'json': json,
        'log': types.SimpleNamespace(warning=noop, debug=noop, info=noop),
        'build_execution_evidence': lambda *_a, **_k: {'thinking_steps': [], 'tool_calls_dict': {}},
        'normalize_hitl_pause': lambda single, plural: (single, plural),
        'AIMessage': _FakeAIMessage,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<builders>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace


class _FakeAIMessage:
    def __init__(self, content):
        self.content = content

    def dict(self):
        return {'content': self.content, 'type': 'ai'}


ns = _load()
build_success_result = ns['build_success_result']
build_output_message = ns['build_output_message']
normalize_response_content = ns['normalize_response_content']
HISTORY = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]


def _result(**kwargs):
    return build_success_result(HISTORY, None, 1, 2, **kwargs)


def test_output_text_is_returned_without_chat_history():
    result = _result(output_text='hello')
    assert result['output_text'] == 'hello'
    assert 'chat_history' not in result


def test_output_text_is_returned_alongside_chat_history_when_requested():
    result = _result(output_text='hello', return_chat_history=True)
    assert result['output_text'] == 'hello'
    assert result['chat_history'] == HISTORY


def test_output_text_key_is_always_present_even_when_empty():
    assert _result()['output_text'] == ''
    assert _result(output_text=None)['output_text'] == ''


def test_paused_hitl_run_has_empty_output_text_and_paused_flag():
    result = _result(output_text='', hitl_interrupt={'node': 'n1'})
    assert result['output_text'] == ''
    assert result['paused'] is True


def test_multi_chunk_text_is_concatenated_into_output():
    chunks = [{'type': 'text', 'text': 'Part one. '}, {'type': 'text', 'text': 'Part two.'}]
    message = build_output_message(normalize_response_content(chunks))
    assert _result(output_text=message['content'])['output_text'] == 'Part one. Part two.'


def test_multimodal_output_keeps_text_and_drops_image_blocks():
    content = [
        {'type': 'text', 'text': 'Here is the chart. '},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
        {'type': 'text', 'text': 'Done.'},
    ]
    output = _result(output_text=normalize_response_content(content))['output_text']
    assert output == 'Here is the chart. Done.'
    assert 'base64' not in output


def test_image_only_answer_gives_empty_output():
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]
    assert _result(output_text=normalize_response_content(content))['output_text'] == ''
