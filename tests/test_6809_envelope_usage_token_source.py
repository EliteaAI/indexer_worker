"""#6809/#6716: every result envelope carries the run's trajectory and usage evidence.

- ``ChatGeneration.model_dump()`` drops ``AIMessage.usage_metadata`` and ``tool_calls`` (the
  field is typed BaseMessage), so a thinking step lost the provider usage and the tool calls
  the LLM asked for. ``on_llm_end`` now carries both, plus a per-step ``token_source``.
- The envelope totals mix provider counts and tiktoken estimates; ``token_source`` says which
  (``provider`` / ``estimate`` / ``mixed`` / ``none``).
- Error, MCP-auth and parked results used to drop the evidence (or hard-code zeros), so a
  failed or paused case looked like a free, step-less run.

Loads the real module through a stub package so the suite runs without the pylon runtime.
"""

import ast
import importlib
import pathlib
import sys
import types
import typing
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = 'indexer_worker_6809'
EVIDENCE_KEYS = {
    'thinking_steps', 'tool_calls', 'tool_calls_dict',
    'chat_history_tokens_input', 'llm_response_tokens_output', 'token_source',
}


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
    return sys.modules[name]


def _stub_runtime():
    noop = lambda *_a, **_k: None  # noqa: E731
    _stub('pylon')
    _stub('pylon.core')
    _stub('pylon.core.tools', log=types.SimpleNamespace(
        debug=noop, info=noop, warning=noop, error=noop, exception=noop,
    ))
    try:
        importlib.import_module('elitea_sdk.runtime.utils.trace_limits')
        importlib.import_module('elitea_sdk.tools.utils.serialization')
        importlib.import_module('elitea_sdk.runtime.utils.mcp_oauth')
    except ImportError:
        for name in ('elitea_sdk', 'elitea_sdk.runtime', 'elitea_sdk.runtime.utils',
                     'elitea_sdk.tools', 'elitea_sdk.tools.utils'):
            _stub(name)
        _stub('elitea_sdk.runtime.utils.trace_limits',
              TRACE_STEP_FIELD_MAX_CHARS=100000,
              cap_trace_json=lambda value, *_a, **_k: value,
              cap_trace_text=lambda value, *_a, **_k: value)
        _stub('elitea_sdk.tools.utils.serialization', to_json_primitive=lambda value: value)

        class McpAuthorizationRequired(Exception):
            pass

        _stub('elitea_sdk.runtime.utils.mcp_oauth',
              McpAuthorizationRequired=McpAuthorizationRequired,
              infer_authorization_servers_from_realm=lambda *_a, **_k: [],
              build_mcp_auth_decision_result=lambda *_a, **_k: {})


def _load_agent_common():
    _stub_runtime()
    for name, path in ((PKG, ROOT), (f'{PKG}.methods', ROOT / 'methods'), (f'{PKG}.utils', ROOT / 'utils')):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules.setdefault(name, package)
    return importlib.import_module(f'{PKG}.methods.agent_common')


agent_common = _load_agent_common()


def _load_execution_common_builders():
    """The two builders that live in utils/agent_execution_common.py (heavy module imports)."""
    source = (ROOT / 'utils' / 'agent_execution_common.py').read_text()
    tree = ast.parse(source)
    wanted = {'build_success_result', 'build_parked_result'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {
        'Optional': typing.Optional, 'Dict': typing.Dict, 'Any': typing.Any, 'List': typing.List,
        'EliteACallback': agent_common.EliteACallback,
        'build_execution_evidence': agent_common.build_execution_evidence,
        'normalize_hitl_pause': importlib.import_module(
            f'{PKG}.utils.parallel_dispatch_contract'
        ).normalize_hitl_pause,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<builders>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace['build_success_result'], namespace['build_parked_result']


build_success_result, build_parked_result = _load_execution_common_builders()


class _EventNode:
    def __init__(self):
        self.events = []

    def emit(self, name, payload):
        self.events.append((name, payload))


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


def _callback():
    return agent_common.EliteACallback(_NodeInterface())


def _llm_end(callback, message, llm_output=None, pending_tokens=(0, 0)):
    run_id = uuid4()
    callback.pending_llm_requests[run_id] = {
        'tokens_in': pending_tokens[0], 'tokens_out': pending_tokens[1], 'timestamp_start': None,
    }
    callback.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=llm_output),
        run_id=run_id,
    )
    return callback.thinking_steps[-1]


TOOL_CALL = {'name': 'list_files', 'args': {'path': '/'}, 'id': 'call-1', 'type': 'tool_call'}
USAGE = {'input_tokens': 120, 'output_tokens': 7, 'total_tokens': 127}


# --- on_llm_end: the thinking step keeps usage and tool calls -------------------------------

def test_step_keeps_usage_metadata_and_tool_calls():
    callback = _callback()
    step = _llm_end(callback, AIMessage(content='', tool_calls=[TOOL_CALL], usage_metadata=USAGE))

    assert step['message']['usage_metadata'] == USAGE
    assert [c['name'] for c in step['message']['tool_calls']] == ['list_files']
    assert step['message']['tool_calls'][0]['args'] == {'path': '/'}
    assert step['token_source'] == 'provider'
    assert (callback.tokens_in, callback.tokens_out) == (120, 7)


def test_step_without_provider_usage_is_an_estimate():
    callback = _callback()
    step = _llm_end(callback, AIMessage(content='hello'), pending_tokens=(30, 2))

    assert step['message']['usage_metadata'] is None
    assert step['message']['tool_calls'] == []
    assert step['token_source'] == 'estimate'
    assert (callback.tokens_in, callback.tokens_out) == (30, 2)


# --- envelope token_source ----------------------------------------------------------------

def test_envelope_token_source_vocabulary():
    assert agent_common.envelope_token_source(None) == 'none'
    callback = _callback()
    assert agent_common.envelope_token_source(callback) == 'none'
    _llm_end(callback, AIMessage(content='a', usage_metadata=USAGE))
    assert agent_common.envelope_token_source(callback) == 'provider'
    _llm_end(callback, AIMessage(content='b'), pending_tokens=(5, 1))
    assert agent_common.envelope_token_source(callback) == 'mixed'

    estimate_only = _callback()
    _llm_end(estimate_only, AIMessage(content='c'), pending_tokens=(5, 1))
    assert agent_common.envelope_token_source(estimate_only) == 'estimate'


def test_failed_llm_call_counts_as_estimate():
    callback = _callback()
    run_id = uuid4()
    callback.pending_llm_requests[run_id] = {'tokens_in': 40, 'tokens_out': 0, 'timestamp_start': None}
    callback.on_llm_error(RuntimeError('boom'), run_id=run_id)
    assert agent_common.envelope_token_source(callback) == 'estimate'


# --- G1: all five builders carry the evidence -----------------------------------------------

def _run_with_one_provider_step():
    callback = _callback()
    _llm_end(callback, AIMessage(content='', tool_calls=[TOOL_CALL], usage_metadata=USAGE))
    return callback


def _assert_evidence(result, callback):
    assert EVIDENCE_KEYS <= set(result)
    assert len(result['thinking_steps']) == len(callback.thinking_steps) == 1
    assert result['thinking_steps'][0]['message']['usage_metadata'] == USAGE
    assert result['chat_history_tokens_input'] == 120
    assert result['llm_response_tokens_output'] == 7
    assert result['token_source'] == 'provider'


def test_success_result_carries_evidence():
    callback = _run_with_one_provider_step()
    _assert_evidence(build_success_result([], callback, callback.tokens_in, callback.tokens_out), callback)


def test_parked_result_carries_evidence_instead_of_zeros():
    callback = _run_with_one_provider_step()
    result = build_parked_result({'parallel_dispatch': [{'x': 1}]}, 's', 'm', elitea_callback=callback)
    _assert_evidence(result, callback)
    assert result['parallel_parked'] is True


def test_parked_result_without_callback_stays_empty():
    result = build_parked_result({})
    assert result['thinking_steps'] == [] and result['tool_calls_dict'] == {}
    assert (result['chat_history_tokens_input'], result['llm_response_tokens_output']) == (0, 0)
    assert result['token_source'] == 'none'


def test_execution_error_carries_evidence():
    callback = _run_with_one_provider_step()
    result = agent_common.execution_error(
        _NodeInterface(), 'hi', [], 'boom', 'thread-1', 'message-1', {}, elitea_callback=callback,
    )
    _assert_evidence(result, callback)
    assert result['error']


def test_mcp_auth_required_result_carries_evidence():
    callback = _run_with_one_provider_step()
    result = agent_common.build_mcp_auth_required_result(
        _NodeInterface(), RuntimeError('auth needed'), None, [], elitea_callback=callback,
    )
    _assert_evidence(result, callback)
    assert result['pause_type'] == 'mcp_auth'


def test_mcp_auth_pause_result_carries_evidence():
    callback = _run_with_one_provider_step()
    callback.mcp_auth_pause_payload = {'server': 'x'}
    callback.mcp_auth_pause_message = 'Authorize x'
    result = agent_common.build_mcp_auth_pause_result(callback, [])
    _assert_evidence(result, callback)
    assert result['error'] == 'Authorize x'
