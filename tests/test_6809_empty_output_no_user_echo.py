"""An answerless run must not surface the user's own message as the reply.

The SDK returns output '' for a run that finished without an answer; the run's
messages then end in the user's input (or an empty AI message). The legacy
messages fallback in ``extract_response_content`` used to take ``messages[-1]``.
"""
import ast
import json
from pathlib import Path
from typing import Any, Dict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

source = Path(__file__).resolve().parents[1]/'utils/agent_execution_common.py'
tree = ast.parse(source.read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ('_is_user_message', 'normalize_response_content', 'extract_response_content')]
namespace = {'Dict': Dict, 'Any': Any, 'json': json, 'log': __import__('logging').getLogger('t')}
exec(compile(tree, str(source), 'exec'), namespace)
extract = namespace['extract_response_content']


@pytest.mark.parametrize('last_user', [
    HumanMessage(content='list the files'),
    {'role': 'user', 'content': 'list the files'},
    {'type': 'human', 'content': 'list the files'},
])
def test_user_message_is_never_the_reply(last_user):
    assert extract({'output': '', 'messages': [last_user]}) == ''


def test_empty_ai_reply_after_user_stays_empty():
    messages = [HumanMessage(content='list the files'), AIMessage(content=[])]
    assert extract({'output': '', 'messages': messages}) == ''


def test_last_non_user_message_is_still_used():
    messages = [AIMessage(content='earlier'), ToolMessage(content='tool result', tool_call_id='1'),
                HumanMessage(content='follow-up')]
    assert extract({'output': None, 'messages': messages}) == 'tool result'


def test_output_wins_when_present():
    assert extract({'output': 'answer', 'messages': [HumanMessage(content='q')]}) == 'answer'
