"""#7015: a pipeline run's node spans reach the result envelope.

Batch evaluation runs non-interactively, so the pipeline's edge and node custom events are
dropped by NoOpNodeEventInterface and its trajectory had only LLM and tool steps. The
callback now records one span per top-level node from LangGraph's chain callbacks and
``build_execution_evidence`` returns them as ``pipeline_trace``. Agents carry no such key.

Also covers the reconcile payload: a parked fan-out parent resumed for reconcile keeps the
evaluation entity and trigger source, so its spend is not relabelled as a plain app run.

The graph tests run a real LangGraph when it is importable (the indexer container) and are
skipped otherwise.
"""

import importlib
import pathlib
import sys
from uuid import uuid4

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from test_6809_envelope_usage_token_source import _NodeInterface, agent_common  # noqa: E402

pipeline_trace = importlib.import_module('indexer_worker_6809.utils.pipeline_trace')

PIPELINE_YAML = """
entry_point: Ask
nodes:
  - id: Ask
    type: llm
    transition: Route
  - id: Route
    type: router
  - id: Print me
    type: printer
    transition: END
  - id: Review
    type: hitl
"""


def _callback(version_details=None):
    callback = agent_common.EliteACallback(_NodeInterface())
    pipeline_trace.attach_pipeline_trace(
        callback, version_details if version_details is not None
        else {'agent_type': 'pipeline', 'instructions': PIPELINE_YAML},
    )
    return callback


def _node(callback, node, step, *, events=(), error=None, ns=None):
    run_id = uuid4()
    metadata = {'langgraph_node': node, 'langgraph_checkpoint_ns': ns or f'{node}:abc'}
    callback.on_chain_start({}, {}, run_id=run_id, name=node, metadata=metadata, tags=[f'graph:step:{step}'])
    for name, data in events:
        callback.on_custom_event(name, data, run_id=run_id, metadata=metadata)
    if error is None:
        callback.on_chain_end({}, run_id=run_id)
    else:
        callback.on_chain_error(error, run_id=run_id)


def _spans(callback):
    return agent_common.build_execution_evidence(callback)['pipeline_trace']['spans']


def test_node_types_come_from_the_pipeline_yaml_with_sdk_id_cleaning():
    types = pipeline_trace.pipeline_node_types({'instructions': PIPELINE_YAML})
    assert types == {'Ask': 'llm', 'Route': 'router', 'Printme': 'printer', 'Review': 'hitl'}
    assert pipeline_trace.pipeline_node_types({'instructions': '- not: [a mapping'}) == {}
    assert pipeline_trace.pipeline_node_types(None) == {}


def test_agents_carry_no_pipeline_trace():
    callback = _callback({'agent_type': 'openai', 'instructions': 'be nice'})
    _node(callback, 'agent', 1)
    assert callback.pipeline_trace is None
    assert 'pipeline_trace' not in agent_common.build_execution_evidence(callback)
    assert 'pipeline_trace' not in agent_common.build_execution_evidence(None)


def test_spans_are_recorded_in_order_with_type_timing_and_edge():
    callback = _callback()
    _node(callback, 'Ask', 1, events=[('on_transitional_edge', {'next_step': 'Route', 'state': {'secret': 'x'}})])
    _node(callback, 'Route', 2, events=[('on_conditional_edge', {'condition': '{{x}}', 'state': {}})])
    _node(callback, 'Printme', 3)

    spans = _spans(callback)
    assert [(s['node'], s['node_type'], s['step'], s['status']) for s in spans] == [
        ('Ask', 'llm', 1, 'ok'), ('Route', 'router', 2, 'ok'), ('Printme', 'printer', 3, 'ok'),
    ]
    assert spans[0]['edge'] == {'type': 'transitional', 'next': 'Route'}
    # The router's next node is the span that follows it
    assert spans[1]['edge'] == {'type': 'conditional', 'next': 'Printme'}
    assert all(s['duration_ms'] is not None and s['timestamp_finish'] for s in spans)
    # Pipeline state can hold secrets: never stored
    assert 'secret' not in str(spans)


def test_nodes_the_sdk_adds_are_not_spans():
    callback = _callback()
    _node(callback, 'StateDefaultNode', 1)
    _node(callback, 'Ask', 2)
    _node(callback, 'Printme_reset', 3)
    assert [s['node'] for s in _spans(callback)] == ['Ask']


def test_a_router_that_ends_the_run_routes_to_end():
    callback = _callback()
    _node(callback, 'Ask', 1)
    _node(callback, 'Route', 2)
    assert _spans(callback)[1]['edge'] == {'type': 'conditional', 'next': 'END'}


def test_inner_chains_and_subgraph_nodes_are_not_spans():
    callback = _callback()
    run_id = uuid4()
    # The graph itself and a node's inner runnable share no graph:step tag or node name
    callback.on_chain_start({}, {}, run_id=uuid4(), name='LangGraph', metadata={}, tags=[])
    callback.on_chain_start({}, {}, run_id=run_id, name='RunnableSequence',
                            metadata={'langgraph_node': 'Ask'}, tags=['seq:step:1'])
    callback.on_chain_end({}, run_id=run_id)
    # A sub-agent's own graph nodes run under the outer node's checkpoint namespace
    _node(callback, 'agent', 1, ns='Ask:abc|agent:def')
    _node(callback, 'Ask', 1)
    assert [s['node'] for s in _spans(callback)] == ['Ask']


def test_error_and_interrupt_statuses():
    class GraphInterrupt(Exception):
        pass

    callback = _callback()
    _node(callback, 'Ask', 1, error=ValueError('boom ' * 200))
    _node(callback, 'Review', 2, events=[('on_hitl_interrupt', {'node_name': 'Review'})], error=GraphInterrupt())
    spans = _spans(callback)
    assert spans[0]['status'] == 'error' and spans[0]['error'].startswith('boom')
    assert len(spans[0]['error']) == pipeline_trace.MAX_ERROR_CHARS
    assert spans[1]['status'] == 'interrupted'
    assert spans[1]['hitl'] == {'action': 'interrupt'}


def test_span_still_open_when_the_envelope_is_built_reports_running():
    callback = _callback()
    callback.on_chain_start({}, {}, run_id=uuid4(), name='Ask',
                            metadata={'langgraph_node': 'Ask'}, tags=['graph:step:1'])
    assert _spans(callback)[0]['status'] == 'running'


def test_span_cap_marks_the_trace_truncated(monkeypatch):
    monkeypatch.setattr(pipeline_trace, 'MAX_PIPELINE_SPANS', 3)
    callback = _callback()
    for step in range(1, 6):
        _node(callback, 'Ask', step)
    trace = agent_common.build_execution_evidence(callback)['pipeline_trace']
    assert len(trace['spans']) == 3 and trace['truncated'] is True


def test_hitl_events_no_longer_log_a_missing_mapper_key(monkeypatch):
    errors = []
    monkeypatch.setattr(agent_common.log, 'error', lambda msg, *a: errors.append(msg), raising=False)
    custom = agent_common.EliteACustomCallback(_NodeInterface())
    for name in ('on_hitl_interrupt', 'on_hitl_resume'):
        custom.on_custom_event(name, {'node_name': 'Review'}, run_id=uuid4(), metadata={})
    assert errors == []


def test_reconcile_payload_keeps_the_evaluation_entity_and_trigger_source():
    source = (ROOT / 'utils' / 'agent_execution_common.py').read_text()
    body = source[source.index('def build_parent_reconcile_payload'):]
    carry = body[body.index('carry_keys = ('):body.index('payload = {k: parent_kwargs[k]')]
    assert 'ENTITY_KWARGS_KEY,' in carry
    assert 'TRIGGER_SOURCE_KWARGS_KEY,' in carry


def test_real_langgraph_run_yields_one_span_per_node():
    graph = pytest.importorskip('langgraph.graph')
    from typing import TypedDict

    from langchain_core.callbacks import dispatch_custom_event

    class State(TypedDict):
        x: int

    def ask(state, config):
        dispatch_custom_event('on_transitional_edge', {'next_step': 'Route', 'state': state}, config=config)
        return {'x': state['x'] + 1}

    builder = graph.StateGraph(State)
    builder.add_node('Ask', ask)
    builder.add_node('Route', lambda s: {'x': s['x'] * 2})
    builder.add_node('Printme', lambda s: s)
    builder.set_entry_point('Ask')
    builder.add_edge('Ask', 'Route')
    builder.add_conditional_edges('Route', lambda s: 'Printme' if s['x'] > 0 else graph.END)
    builder.add_edge('Printme', graph.END)

    callback = _callback()
    builder.compile().invoke({'x': 1}, config={'callbacks': [callback]})

    spans = _spans(callback)
    assert [(s['node'], s['node_type'], s['step']) for s in spans] == [
        ('Ask', 'llm', 1), ('Route', 'router', 2), ('Printme', 'printer', 3),
    ]
    assert spans[0]['edge'] == {'type': 'transitional', 'next': 'Route'}
    assert all(s['status'] == 'ok' for s in spans)
