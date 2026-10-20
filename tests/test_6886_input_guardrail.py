"""Coverage for the input guardrail (#6886) plumbing in the indexer worker.

The SDK raises InputGuardrailBlocked and dispatches an ``input_guardrail`` custom event;
the worker must turn the block into a curated, value-free error, keep the verdict on the
message meta, and configure the SDK from its own config without ever keeping a stale policy.

Loaded by source so the suite runs without the pylon runtime.
"""

import ast
import importlib
import json
import pathlib
import sys
import traceback
import types
import typing
from datetime import datetime, timezone
from uuid import uuid4

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAKE_SECRET = 'ghp_' + 'Q7wZ' * 9  # generated, matches the github_token rule shape only


def _load(path, names, namespace):
    source = (ROOT / path).read_text()
    nodes = [
        node for node in ast.parse(source).body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
        or (isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in names for t in node.targets))
    ]
    assert {getattr(n, 'name', None) or n.targets[0].id for n in nodes} == set(names), path
    exec(compile(ast.Module(nodes, []), f'<{path}>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace


_LOG = types.SimpleNamespace(
    debug=lambda *a, **k: None, info=lambda *a, **k: None,
    warning=lambda *a, **k: None, exception=lambda *a, **k: None,
)
_TYPING = {'Any': typing.Any, 'Dict': typing.Dict, 'Optional': typing.Optional,
           'List': typing.List, 'Iterable': typing.Iterable}

funcs = _load('utils/funcs.py', [
    'INPUT_GUARDRAIL_BLOCKED_CODE', '_INPUT_GUARDRAIL_CATEGORY_LABELS',
    'build_input_guardrail_error', 'build_parallel_terminal_error',
], dict(_TYPING))
guardrail = _load('utils/input_guardrail.py', [
    'INSPECTED_CONFIG_KEY', 'AUDIT_TRACER_NAME', 'AUDIT_OBSERVATION_TYPE', '_SEVERITY', '_META_KEYS',
    'inspected_segments', 'merge_verdict', 'emit_audit_spans', 'audit_blocked', '_audit_key',
    '_entity_attributes', '_redaction', 'set_redaction_policy', 'redact_span_text', 'register_tracing_redactor',
], {**_TYPING, 'log': _LOG})


class InputGuardrailBlocked(Exception):
    """Same name as the SDK class; the worker detects it by name to survive hot-reloads."""

    def __init__(self, categories, rules, segments):
        super().__init__('blocked')
        self.categories, self.rules, self.segments = categories, rules, segments


def _blocked():
    return InputGuardrailBlocked(['credentials'], ['credentials.github_token'], ['user_input'])


# --- curated error ------------------------------------------------------------------


def test_block_maps_to_curated_error():
    error = funcs['build_input_guardrail_error'](_blocked())

    assert error['code'] == 'input_guardrail_blocked'
    assert error['action'] == 'block'
    assert error['message_key'] == 'input_guardrail.blocked.credentials'
    assert 'a credential' in error['user_message']
    assert 'not sent to the model' in error['user_message']
    assert error['rules'] == ['credentials.github_token']


def test_instructions_block_asks_the_agent_owner():
    error = funcs['build_input_guardrail_error'](
        InputGuardrailBlocked(['credentials'], ['credentials.github_token'], ['instructions']),
    )

    assert error['message_key'] == 'input_guardrail.blocked.instructions'
    assert "agent's owner" in error['user_message']


def test_fail_closed_error_has_its_own_message():
    error = funcs['build_input_guardrail_error'](InputGuardrailBlocked(['error'], [], []))

    assert error['message_key'] == 'input_guardrail.blocked.error'
    assert 'could not be checked' in error['user_message']


def test_other_exceptions_are_not_guardrail_blocks():
    assert funcs['build_input_guardrail_error'](RuntimeError('boom')) is None
    assert funcs['build_input_guardrail_error'](ValueError(FAKE_SECRET)) is None


def test_parallel_child_reports_guardrail_code():
    error = funcs['build_input_guardrail_error'](_blocked())
    terminal = funcs['build_parallel_terminal_error'](
        error_message='Input blocked by the input guardrail',
        human_readable=error['user_message'],
        input_guardrail=error,
    )

    assert terminal['code'] == 'input_guardrail_blocked'
    assert terminal['categories'] == ['credentials']


# --- execution_error --------------------------------------------------------------


def _load_execution_error(captured):
    source = (ROOT / 'methods' / 'agent_common.py').read_text()
    body = source[source.index('def execution_error('):source.index('class ToolCallPayload')]
    evidence = source[source.index('def envelope_token_source('):source.index('def build_mcp_auth_pause_result(')]

    class _NodeEvent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def model_dump_json(self):
            return json.dumps(self.kwargs, default=str)

    namespace = {
        **_TYPING, 'json': json, 'traceback': traceback, 'uuid4': uuid4,
        'datetime': datetime, 'timezone': timezone,
        'NodeEventInterface': object, 'NodeEvent': _NodeEvent,
        'EventTypes': types.SimpleNamespace(
            agent_tool_start='agent_tool_start', agent_tool_end='agent_tool_end',
            agent_exception='agent_exception', full_message='full_message',
        ),
        'EVENTNODE_FULL_RESPONSE_NAME': 'full_response',
        'is_fanout_child': lambda meta: bool(meta.get('fanout')),
        'build_parallel_terminal_error': funcs['build_parallel_terminal_error'],
        '_sanitize_input_for_event': lambda value: value,
    }
    exec(compile(evidence, '<build_execution_evidence>', 'exec'), namespace)  # pylint: disable=W0122
    exec(compile(body, '<execution_error>', 'exec'), namespace)  # pylint: disable=W0122

    class _NodeInterface:
        stream_id = 'stream-1'
        payload_additional_kwargs = {}
        event_node = types.SimpleNamespace(emit=lambda name, payload: captured.append(('node', payload)))

        def emit(self, **kwargs):
            captured.append(('emit', kwargs))

    return namespace['execution_error'], _NodeInterface()


def _run_blocked(meta=None):
    captured = []
    execution_error, node_interface = _load_execution_error(captured)
    error = funcs['build_input_guardrail_error'](_blocked())
    try:
        raise _blocked()
    except InputGuardrailBlocked:
        result = execution_error(
            node_interface, f'deploy with {FAKE_SECRET}', [], 'Input blocked by the input guardrail',
            'thread-1', 'message-1', meta or {'project_id': 2},
            human_readable=error['user_message'], input_guardrail=error,
        )
    return result, captured


def test_block_skips_the_stacktrace_pin_that_echoes_the_input():
    _result, captured = _run_blocked()

    types_emitted = [kwargs.get('type') for kind, kwargs in captured if kind == 'emit']
    assert 'agent_tool_start' not in types_emitted
    assert 'agent_tool_end' not in types_emitted
    assert types_emitted == ['agent_exception']


def test_no_event_carries_the_blocked_value():
    _result, captured = _run_blocked()

    assert FAKE_SECRET not in json.dumps(captured, default=str)


def test_verdict_reaches_the_persisted_message_meta():
    result, captured = _run_blocked()

    full_message = next(payload for kind, payload in captured if kind == 'node')
    meta = full_message['response_metadata']['additional_response_meta']['input_guardrail']
    assert meta['code'] == 'input_guardrail_blocked'
    assert full_message['content'] == result['human_readable']
    assert 'a credential' in result['human_readable']


def test_fanout_child_returns_guardrail_terminal_error():
    result, captured = _run_blocked({'project_id': 2, 'fanout': True})

    assert result['parallel_terminal_error']['code'] == 'input_guardrail_blocked'
    assert not [1 for kind, _ in captured if kind == 'node']


def test_ordinary_errors_still_get_the_stacktrace_pin():
    captured = []
    execution_error, node_interface = _load_execution_error(captured)
    try:
        raise RuntimeError('boom')
    except RuntimeError:
        execution_error(node_interface, 'hi', [], 'Error', 'thread-1', 'message-1', {'project_id': 2})

    assert [k.get('type') for kind, k in captured if kind == 'emit'][:2] == ['agent_tool_start', 'agent_tool_end']


# --- verdict merge, inspected hashes, audit spans ------------------------------------


def _summary(action, rules, detections=1, **extra):
    return {
        'action': action, 'categories': sorted({r.split('.')[0] for r in rules}), 'rules': rules,
        'segments': ['user_input'], 'detection_count': detections, 'not_inspected': [], 'error': None,
        'on_error': None, 'elapsed_ms': 1.2, 'audit': [], **extra,
    }


def test_merge_keeps_the_most_severe_action_and_unions_rules():
    merge = guardrail['merge_verdict']
    verdict = merge(None, _summary('redact', ['personal_data.email']))
    verdict = merge(verdict, _summary('warn', ['credentials.github_token']))

    assert verdict['action'] == 'redact'
    assert verdict['rules'] == ['credentials.github_token', 'personal_data.email']
    assert verdict['categories'] == ['credentials', 'personal_data']
    assert verdict['detection_count'] == 2


def test_merge_drops_transport_only_keys():
    verdict = guardrail['merge_verdict'](None, _summary('warn', ['credentials.github_token']))

    assert set(verdict) == set(guardrail['_META_KEYS'])


def test_inspected_segments_only_accepts_a_list_of_hashes():
    inspected = guardrail['inspected_segments']

    assert inspected({'input_guardrail_inspected': ['a' * 64, 3]}) == ['a' * 64]
    assert inspected({'input_guardrail_inspected': 'a' * 64}) == []
    assert inspected({}) == []


def test_audit_spans_carry_action_and_owner(monkeypatch):
    spans = []

    class _Span:
        def end(self):
            pass

    class _Tracer:
        def start_span(self, name, attributes):
            spans.append((name, attributes))
            return _Span()

    trace = types.SimpleNamespace(get_tracer=lambda name: _Tracer())
    monkeypatch.setitem(sys.modules, 'opentelemetry', types.SimpleNamespace(trace=trace))
    monkeypatch.setitem(sys.modules, 'opentelemetry.trace', trace)

    count = guardrail['emit_audit_spans'](
        [{'action': 'input:credentials.github_token:block', 'segment_kind': 'user_input', 'is_error': False},
         {'action': ''}],
        user_id='7', project_id=2,
    )

    assert count == 1
    name, attrs = spans[0]
    assert name == 'input:credentials.github_token:block'
    assert attrs['audit.observation.type'] == 'guardrail'
    assert attrs['user.id'] == 7 and attrs['project.id'] == 2
    assert attrs['audit.tool.name'] == 'user_input'


def _capture_spans(monkeypatch):
    spans = []
    tracer = types.SimpleNamespace(
        start_span=lambda name, attributes: spans.append(name) or types.SimpleNamespace(end=lambda: None),
    )
    trace = types.SimpleNamespace(get_tracer=lambda name: tracer)
    monkeypatch.setitem(sys.modules, 'opentelemetry', types.SimpleNamespace(trace=trace))
    monkeypatch.setitem(sys.modules, 'opentelemetry.trace', trace)
    return spans


def test_sub_agent_block_is_audited_from_the_exception(monkeypatch):
    spans = _capture_spans(monkeypatch)
    exc = _blocked()
    exc.summary = {'audit': [{'action': 'input:credentials.github_token:block', 'segment_kind': 'subagent_task'}]}

    assert guardrail['audit_blocked'](exc, set(), user_id=7, project_id=2) == 1
    assert spans == ['input:credentials.github_token:block']


def test_block_already_audited_by_the_callback_is_not_repeated(monkeypatch):
    spans = _capture_spans(monkeypatch)
    entry = {'action': 'input:credentials.github_token:block', 'segment_kind': 'current_turn'}
    exc = _blocked()
    exc.summary = {'audit': [entry]}
    recorded = {guardrail['_audit_key'](entry)}

    assert guardrail['audit_blocked'](exc, recorded) == 0
    assert guardrail['audit_blocked'](RuntimeError('no summary'), recorded) == 0
    assert spans == []


def test_audit_spans_never_raise_without_opentelemetry(monkeypatch):
    monkeypatch.setitem(sys.modules, 'opentelemetry', None)

    assert guardrail['emit_audit_spans']([{'action': 'input:error.timeout:fail_open'}]) == 0


# --- event mapper and SDK configuration ---------------------------------------------


def test_event_type_and_mapper_entry_exist_without_the_value():
    source = (ROOT / 'utils' / 'node_interface.py').read_text()

    assert "agent_input_guardrail = 'agent_input_guardrail'" in source
    mapper = source[source.index('ELITEA_SDK_CUSTOM_EVENTS_MAPPER = {'):]
    entry = mapper[mapper.index('EventTypes.agent_input_guardrail.value'):]
    entry = entry[:entry.index('},')]
    for field in ('action', 'categories', 'rules', 'segments', 'detection_count', 'not_inspected', 'error'):
        assert f"'{field}'" in entry
    assert "'audit'" not in entry


def test_custom_event_handler_records_the_verdict():
    source = (ROOT / 'methods' / 'agent_common.py').read_text()

    assert 'if name == "input_guardrail" and isinstance(data, dict):' in source
    assert 'self._record_input_guardrail(data)' in source


def _load_apply(config, configure, monkeypatch):
    # module.py imports pylon at top level, so only the method is loaded
    source = (ROOT / 'module.py').read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Module')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_apply_input_guardrail')
    namespace = {'log': _LOG}
    exec(compile(ast.Module([method], []), '<_apply_input_guardrail>', 'exec'), namespace)  # pylint: disable=W0122
    security = types.SimpleNamespace(configure_input_guard=configure)
    monkeypatch.setitem(sys.modules, 'elitea_sdk.runtime.toolkits.security', security)
    owner = types.SimpleNamespace(descriptor=types.SimpleNamespace(config=config))
    return lambda: namespace['_apply_input_guardrail'](owner)


def test_apply_pushes_policy_and_disables_on_bad_config(monkeypatch):
    for name in ('elitea_sdk', 'elitea_sdk.runtime', 'elitea_sdk.runtime.toolkits'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    applied = []

    _load_apply({'input_guardrail': {'enabled': True}}, applied.append, monkeypatch)()
    _load_apply({'input_guardrail': 'yes'}, applied.append, monkeypatch)()
    _load_apply({}, applied.append, monkeypatch)()

    assert applied == [{'enabled': True}, {'enabled': False}, {}]


def test_apply_disables_when_sdk_rejects_policy(monkeypatch):
    for name in ('elitea_sdk', 'elitea_sdk.runtime', 'elitea_sdk.runtime.toolkits'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    applied = []

    def configure(policy):
        if policy.get('enabled'):
            raise ValueError('bad pattern')
        applied.append(policy)

    _load_apply({'input_guardrail': {'enabled': True}}, configure, monkeypatch)()

    assert applied == [{'enabled': False}]


# --- audit rows name the agent ---------------------------------------------------------


def _capture_attrs(monkeypatch):
    spans = []
    tracer = types.SimpleNamespace(
        start_span=lambda name, attributes: spans.append(attributes) or types.SimpleNamespace(end=lambda: None),
    )
    trace = types.SimpleNamespace(get_tracer=lambda name: tracer)
    monkeypatch.setitem(sys.modules, 'opentelemetry', types.SimpleNamespace(trace=trace))
    monkeypatch.setitem(sys.modules, 'opentelemetry.trace', trace)
    return spans


def test_audit_spans_carry_the_agent_as_entity_attributes(monkeypatch):
    spans = _capture_attrs(monkeypatch)
    entity = {'type': 'application', 'id': 42, 'version_id': 9, 'name': 'Refund Desk', 'project_id': 2}

    guardrail['emit_audit_spans']([{'action': 'input:credentials.github_token:block'}], entity=entity)

    assert spans[0]['entity.type'] == 'application'
    assert spans[0]['entity.id'] == '42'
    assert spans[0]['entity.name'] == 'Refund Desk'


def test_audit_spans_without_an_agent_have_no_entity_attributes(monkeypatch):
    spans = _capture_attrs(monkeypatch)

    guardrail['emit_audit_spans']([{'action': 'a'}])
    guardrail['emit_audit_spans']([{'action': 'a'}], entity={'type': 'application', 'id': None})
    guardrail['emit_audit_spans']([{'action': 'a'}], entity='nope')

    assert spans and not any(k.startswith('entity.') for attrs in spans for k in attrs)


def test_blocked_audit_forwards_the_agent(monkeypatch):
    spans = _capture_attrs(monkeypatch)
    exc = _blocked()
    exc.summary = {'audit': [{'action': 'input:credentials.github_token:block', 'segment_kind': 'current_turn'}]}

    guardrail['audit_blocked'](exc, set(), entity={'type': 'application', 'id': 5, 'name': 'A'})

    assert spans[0]['entity.id'] == '5'


def test_agent_tasks_hand_the_run_agent_to_the_audit_path():
    for name in ('indexer_agent.py', 'indexer_predict_agent.py'):
        source = (ROOT / 'methods' / name).read_text()
        assert "elitea_custom_callback.audit_entity = entity_from_application(kwargs.get('application'))" in source
        assert "entity=entity_from_application(kwargs.get('application'))" in source
    assert 'entity=self.audit_entity' in (ROOT / 'methods' / 'agent_common.py').read_text()


# --- tracing value redactor ------------------------------------------------------------


def _install_fake_detector(monkeypatch, inspect_fn):
    """A stand-in SDK detector so the contract is tested without the SDK version installed."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Segment:
        kind: str
        index: int
        text: str

    @dataclass(frozen=True)
    class Policy:
        categories: dict
        enabled: bool = True

        @property
        def active(self):
            return self.enabled and any(a != 'off' for a in self.categories.values())

    def policy_from_dict(raw):
        return Policy(categories=dict(raw.get('categories') or {'credentials': 'block'}),
                      enabled=bool(raw.get('enabled')))

    detector = types.SimpleNamespace(Segment=Segment, inspect=inspect_fn, policy_from_dict=policy_from_dict)
    for name in ('elitea_sdk', 'elitea_sdk.runtime', 'elitea_sdk.runtime.guardrails'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, 'elitea_sdk.runtime.guardrails.input_detector', detector)
    return detector


def _result(redacted=None, error=None):
    return types.SimpleNamespace(redacted=redacted or {}, error=error)


def _reset_redaction():
    guardrail['set_redaction_policy'](None)


def test_redactor_is_inert_while_the_guardrail_is_off(monkeypatch):
    calls = []
    _install_fake_detector(monkeypatch, lambda *a, **k: calls.append(1))

    assert guardrail['set_redaction_policy']({'enabled': False}) is False
    assert guardrail['set_redaction_policy']({'enabled': True, 'categories': {'credentials': 'off'}}) is False
    assert guardrail['set_redaction_policy']('not a mapping') is False
    assert guardrail['redact_span_text']('some text long enough') == 'some text long enough'
    assert calls == []


def test_redactor_replaces_detected_values_with_every_action_forced_to_redact(monkeypatch):
    seen = []

    def fake_inspect(segments, policy):
        seen.append(policy.categories)
        text = segments[0].text
        return _result({('span', 0): text.replace('SECRET', '[REDACTED:credentials]')})

    _install_fake_detector(monkeypatch, fake_inspect)
    try:
        assert guardrail['set_redaction_policy']({'enabled': True, 'categories': {'credentials': 'record'}}) is True

        assert guardrail['redact_span_text']('use SECRET now') == 'use [REDACTED:credentials] now'
        assert guardrail['redact_span_text']('nothing here') == 'nothing here'
        assert seen[0] == {'credentials': 'redact'}
    finally:
        _reset_redaction()


def test_redactor_returns_none_when_inspection_failed(monkeypatch):
    results = iter([_result(error='oversize'), _result(error='timeout'), RuntimeError('boom')])

    def fake_inspect(segments, policy):
        outcome = next(results)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    _install_fake_detector(monkeypatch, fake_inspect)
    try:
        guardrail['set_redaction_policy']({'enabled': True})

        assert guardrail['redact_span_text']('x' * 20) is None
        assert guardrail['redact_span_text']('x' * 20) is None
        assert guardrail['redact_span_text']('x' * 20) is None
    finally:
        _reset_redaction()


def test_switching_the_guardrail_off_clears_the_redactor(monkeypatch):
    _install_fake_detector(monkeypatch, lambda segments, policy: _result({('span', 0): 'changed'}))
    guardrail['set_redaction_policy']({'enabled': True})
    assert guardrail['redact_span_text']('original text') == 'changed'

    guardrail['set_redaction_policy']({'enabled': False})

    assert guardrail['redact_span_text']('original text') == 'original text'


def test_unavailable_sdk_detector_leaves_the_redactor_inactive(monkeypatch):
    monkeypatch.setitem(sys.modules, 'elitea_sdk.runtime.guardrails.input_detector', None)

    assert guardrail['set_redaction_policy']({'enabled': True}) is False
    assert guardrail['redact_span_text']('original text') == 'original text'


def test_real_detector_redacts_a_credential_and_keeps_the_sentence():
    try:
        detector = importlib.util.find_spec('elitea_sdk.runtime.guardrails.input_detector')
    except ImportError:
        detector = None
    if detector is None:
        import pytest
        pytest.skip('installed SDK has no input guardrail detector')
    try:
        assert guardrail['set_redaction_policy']({'enabled': True, 'categories': {'credentials': 'block'}}) is True

        out = guardrail['redact_span_text'](f'deploy with {FAKE_SECRET} now')

        assert FAKE_SECRET not in out
        assert out == 'deploy with [REDACTED:credentials] now'
        assert guardrail['redact_span_text']('plain sentence here') == 'plain sentence here'
        assert guardrail['redact_span_text'](' '.join(['word'] * 100_000)) is None  # over max_inspect_chars
    finally:
        _reset_redaction()


def test_register_hands_the_redactor_to_the_tracing_plugin(monkeypatch):
    registered = []
    tracing = types.SimpleNamespace(register_value_redactor=registered.append)
    tools = types.SimpleNamespace(this=types.SimpleNamespace(
        for_module=lambda name: types.SimpleNamespace(module=tracing) if name == 'tracing' else None,
    ))
    monkeypatch.setitem(sys.modules, 'tools', tools)

    assert guardrail['register_tracing_redactor']() is True
    assert registered == [guardrail['redact_span_text']]


def test_register_survives_a_missing_tracing_plugin(monkeypatch):
    def for_module(_name):
        raise RuntimeError('no such module')

    monkeypatch.setitem(sys.modules, 'tools', types.SimpleNamespace(this=types.SimpleNamespace(for_module=for_module)))

    assert guardrail['register_tracing_redactor']() is False


def test_module_registers_on_ready_and_retries_on_reconfig():
    source = (ROOT / 'module.py').read_text()
    assert 'def ready(self):' in source
    assert 'self._register_tracing_redactor()' in source
    assert 'set_redaction_policy(policy)' in source
