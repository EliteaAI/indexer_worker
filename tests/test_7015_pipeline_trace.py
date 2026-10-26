"""#7015: a pipeline run's node spans reach the result envelope.

Batch evaluation runs non-interactively, so the pipeline's edge and node custom events are
dropped by NoOpNodeEventInterface and its trajectory had only LLM and tool steps. The
callback now records one span per top-level node from LangGraph's chain callbacks and
``build_execution_evidence`` returns them as ``pipeline_trace``. Only evaluation runs record it;
agents and chat, scheduled or webhook pipeline runs carry no such key.

Also covers the reconcile payload: a parked fan-out parent resumed for reconcile keeps the
evaluation entity and trigger source, so its spend is not relabelled as a plain app run.

The graph tests run a real LangGraph (``tests/requirements-dev.txt`` installs it for CI).
"""

import ast
import contextvars
import importlib
import pathlib
import sys
import typing
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
  - id: Check
    type: llm
    condition:
      condition_definition: "{{x}}"
"""


PIPELINE = {'agent_type': 'pipeline', 'instructions': PIPELINE_YAML}
# What elitea_core's batch evaluation puts on the predict payload (usage_attribution, #6677)
EVAL_KWARGS = {'_elitea_entity': {'type': 'evaluation', 'id': 42}}


def _callback(version_details=None, kwargs=EVAL_KWARGS):
    callback = agent_common.EliteACallback(_NodeInterface())
    pipeline_trace.attach_pipeline_trace(
        callback, version_details if version_details is not None else PIPELINE, kwargs,
    )
    return callback


# The outer graph's run: LangGraph starts every node of a graph as a child of the graph's run
GRAPH = uuid4()


def _node(callback, node, step, *, events=(), error=None, ns=None, parent=GRAPH):
    run_id = uuid4()
    metadata = {'langgraph_node': node, 'langgraph_checkpoint_ns': ns or f'{node}:{run_id}'}
    callback.on_chain_start({}, {}, run_id=run_id, parent_run_id=parent, name=node, metadata=metadata,
                            tags=[f'graph:step:{step}'])
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
    assert types == {'Ask': 'llm', 'Route': 'router', 'Printme': 'printer', 'Review': 'hitl', 'Check': 'llm'}
    assert pipeline_trace.pipeline_node_types({'instructions': '- not: [a mapping'}) == {}
    assert pipeline_trace.pipeline_node_types(None) == {}


def test_routing_nodes_are_those_the_sdk_wires_a_runtime_edge_for():
    schema = PIPELINE_YAML + """
  - id: Pick
    type: tool
    decision:
      nodes: [Ask, Route]
  - id: Fixed
    type: llm
    transition: Ask
    condition:
      condition_definition: "{{x}}"
"""
    # transition wins over condition in the SDK, so Fixed is not a routing node
    assert pipeline_trace.pipeline_routing_nodes({'instructions': schema}) == {'Route', 'Check', 'Pick'}


@pytest.mark.parametrize('kwargs', [
    None,
    {},
    {'trigger_source': 'scheduled'},
    {'_elitea_entity': {'type': 'application', 'id': 7}},
    {'_elitea_entity': 'evaluation'},
], ids=['no-kwargs', 'chat', 'scheduled', 'other-entity', 'malformed-entity'])
def test_pipeline_runs_outside_an_evaluation_record_no_trace(kwargs):
    callback = _callback(kwargs=kwargs)
    assert callback.pipeline_trace is None
    _node(callback, 'Lookup', 1)
    assert 'pipeline_trace' not in agent_common.build_execution_evidence(callback)


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


def test_a_condition_on_a_non_router_node_infers_its_next():
    callback = _callback()
    _node(callback, 'Check', 1)
    _node(callback, 'Ask', 2)
    assert _spans(callback)[0]['edge'] == {'type': 'conditional', 'next': 'Ask'}


def test_parallel_branches_leave_a_routers_next_unknown():
    callback = _callback()
    _node(callback, 'Route', 1)
    _node(callback, 'Ask', 2)
    _node(callback, 'Printme', 2)
    assert 'edge' not in _spans(callback)[0]
    # Only a span of the next step counts: a later one is not where the router went
    callback = _callback()
    _node(callback, 'Route', 1)
    _node(callback, 'Ask', 3)
    assert 'edge' not in _spans(callback)[0]


def test_nodes_of_a_graph_run_inside_a_node_are_not_spans():
    # A sub-pipeline invoked from Ask runs as a root graph (no '|' in its namespace) under the
    # same callbacks, and its default node ids can match the outer pipeline's.
    callback = _callback()
    outer = uuid4()
    metadata = {'langgraph_node': 'Ask', 'langgraph_checkpoint_ns': f'Ask:{outer}'}
    callback.on_chain_start({}, {}, run_id=outer, parent_run_id=GRAPH, name='Ask', metadata=metadata,
                            tags=['graph:step:1'])
    child_graph = uuid4()
    callback.on_chain_start({}, {}, run_id=child_graph, parent_run_id=uuid4(), name='LangGraph',
                            metadata={}, tags=[])
    _node(callback, 'Route', 1, parent=child_graph,
          events=[('on_conditional_edge', {'next_step': 'Ask'})])
    callback.on_chain_end({}, run_id=child_graph)
    callback.on_chain_end({}, run_id=outer)
    _node(callback, 'Printme', 2)
    spans = _spans(callback)
    assert [s['node'] for s in spans] == ['Ask', 'Printme']
    assert 'edge' not in spans[0]


def test_an_edge_event_of_a_nested_graph_does_not_land_on_the_outer_node():
    callback = _callback()
    _node(callback, 'Ask', 1, events=[('on_transitional_edge', {'next_step': 'Printme'})])
    # Same node name, the nested graph's own namespace
    callback.on_custom_event('on_conditional_edge', {'next_step': 'Review'}, run_id=uuid4(),
                             metadata={'langgraph_node': 'Ask', 'langgraph_checkpoint_ns': 'Ask:other'})
    assert _spans(callback)[0]['edge'] == {'type': 'transitional', 'next': 'Printme'}


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
    from langgraph.errors import GraphInterrupt, ParentCommand
    from langgraph.types import Command

    callback = _callback()
    _node(callback, 'Ask', 1, error=ValueError('boom ' * 200))
    _node(callback, 'Review', 2, events=[('on_hitl_interrupt', {'node_name': 'Review'})], error=GraphInterrupt())
    _node(callback, 'Printme', 3, error=ParentCommand(Command(goto='Ask')))
    spans = _spans(callback)
    assert spans[0]['status'] == 'error' and spans[0]['error'].startswith('boom')
    assert len(spans[0]['error']) == pipeline_trace.MAX_ERROR_CHARS
    assert spans[1]['status'] == 'interrupted'
    assert spans[1]['hitl'] == {'action': 'interrupt'}
    # Control flow bubbling up to the parent graph is not a failure
    assert spans[2]['status'] == 'ok' and 'error' not in spans[2]


def test_interrupts_are_recognised_by_name_without_langgraph(monkeypatch):
    monkeypatch.setattr(pipeline_trace, 'GraphInterrupt', None)
    monkeypatch.setattr(pipeline_trace, 'GraphBubbleUp', None)

    class NodeInterrupt(Exception):
        pass

    callback = _callback()
    _node(callback, 'Review', 1, error=NodeInterrupt())
    assert _spans(callback)[0]['status'] == 'interrupted'


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


def _build_parent_reconcile_payload():
    """The real builder, out of utils/agent_execution_common.py (its module imports are heavy)."""
    tree = ast.parse((ROOT / 'utils' / 'agent_execution_common.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_parent_reconcile_payload')
    events = importlib.import_module('indexer_worker_6809.utils.usage_tool_events')
    namespace = {
        'Dict': typing.Dict, 'Any': typing.Any,
        'PREDICT_RUN_ID_KWARGS_KEY': '_elitea_predict_run_id',
        'ROOT_ENTITY_KWARGS_KEY': events.ROOT_ENTITY_KWARGS_KEY,
        'ENTITY_KWARGS_KEY': events.ENTITY_KWARGS_KEY,
        'TRIGGER_SOURCE_KWARGS_KEY': events.TRIGGER_SOURCE_KWARGS_KEY,
        'ENVELOPE_KEY': importlib.import_module('indexer_worker_6809.utils.task_envelope').ENVELOPE_KEY,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<reconcile>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace['build_parent_reconcile_payload']


def test_reconcile_payload_keeps_the_evaluation_entity_and_trigger_source():
    build = _build_parent_reconcile_payload()
    parent = {'thread_id': 't1', 'hitl_resume': True, '_elitea_entity': {'type': 'evaluation', 'id': 42},
              'trigger_source': 'scheduled', 'unrelated': 'x'}
    payload = build(parent)
    assert payload['_elitea_entity'] == {'type': 'evaluation', 'id': 42}
    assert payload['trigger_source'] == 'scheduled'
    assert 'unrelated' not in payload and payload['hitl_resume'] is False

    plain = build({'thread_id': 't1'})
    assert '_elitea_entity' not in plain and 'trigger_source' not in plain


def test_real_langgraph_run_yields_one_span_per_node():
    from langgraph import graph
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


def test_real_nested_graph_with_the_same_node_names_adds_no_spans():
    from langgraph import graph
    from typing import TypedDict

    from langchain_core.callbacks import dispatch_custom_event

    class State(TypedDict):
        x: int

    # The sub-pipeline: what Application._run invokes, a root graph with its own config
    inner = graph.StateGraph(State)

    def inner_ask(state, config):
        dispatch_custom_event('on_transitional_edge', {'next_step': 'Review'}, config=config)
        return {'x': state['x'] + 10}

    inner.add_node('Ask', inner_ask)
    inner.add_node('Route', lambda s: s)
    inner.set_entry_point('Ask')
    inner.add_edge('Ask', 'Route')
    inner.add_edge('Route', graph.END)
    inner_graph = inner.compile()

    def call_sub_pipeline(state, config):
        # Parent callbacks, no parent checkpoint namespace: the standalone child of
        # Application._run, a root graph whose nodes carry no '|' in their namespace
        return contextvars.Context().run(inner_graph.invoke, state, {'callbacks': config['callbacks']})

    outer = graph.StateGraph(State)
    outer.add_node('Ask', call_sub_pipeline)
    outer.add_node('Route', lambda s: s)
    outer.set_entry_point('Ask')
    outer.add_edge('Ask', 'Route')
    outer.add_edge('Route', graph.END)

    callback = _callback()
    outer.compile().invoke({'x': 1}, config={'callbacks': [callback]})

    spans = _spans(callback)
    assert [(s['node'], s['step']) for s in spans] == [('Ask', 1), ('Route', 2)]
    assert 'edge' not in spans[0]
    assert spans[1]['edge'] == {'type': 'conditional', 'next': 'END'}
