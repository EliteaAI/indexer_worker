"""Pipeline node spans for the result envelope (#7015).

A pipeline's LLM and tool calls already reach ``thinking_steps`` / ``tool_calls_dict``, but
which nodes ran, in what order, how long each took and which edge was taken was only sent as
live custom events, and non-interactive runs (batch evaluation) drop those. This records one
span per top-level node from LangGraph's chain callbacks, so it works whatever the node
interface does with the events. Only evaluation runs record it: nothing else reads it.

A node's run is the chain whose ``metadata.langgraph_node`` equals its run name and that
carries a ``graph:step:N`` tag. Only the outer graph's nodes are recorded: the first node run pins
its parent run (the outer graph), and a node run under any other parent is skipped. A
sub-agent or pipeline called from a node runs as a root graph of its own (``Application._run``
strips the checkpoint namespace), under the same callbacks, so neither a ``|`` in the namespace
nor the YAML ids would tell its nodes apart. Nodes the SDK adds on its own (``StateDefaultNode``,
a printer's ``*_reset``) are not in the YAML and are skipped. Pipeline ``state`` is never stored:
it is large and can hold secrets.

An edge or HITL event attaches to the span whose checkpoint namespace it carries, so an event
sent inside a nested graph never lands on the outer node. A router resolves its route without
dispatching ``on_conditional_edge`` (the SDK only sends it for a non-string result), so a routing
node's ``next`` is taken from the one span of the following graph step.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import yaml

try:
    from langgraph.errors import GraphBubbleUp, GraphInterrupt
except ImportError:  # pragma: no cover - the indexer always has langgraph
    GraphBubbleUp = GraphInterrupt = None

from .usage_tool_events import ENTITY_KWARGS_KEY, ENTITY_TYPE_EVALUATION

MAX_PIPELINE_SPANS = 500
MAX_ERROR_CHARS = 500

#: SDK custom events dispatched while an edge or node decides where to go next
EDGE_EVENTS = {
    'on_transitional_edge': 'transitional',
    'on_conditional_edge': 'conditional',
    'on_decision_edge': 'decision',
}
HITL_EVENTS = {'on_hitl_interrupt', 'on_hitl_resume'}
#: Node types whose outgoing edge is chosen at run time; any node with a ``condition`` or
#: ``decision`` block and no ``transition`` is one too (the SDK's edge wiring)
ROUTING_NODE_TYPES = {'router', 'decision'}
#: Fallback names when langgraph.errors cannot be imported
INTERRUPT_ERROR_NAMES = {'GraphInterrupt', 'NodeInterrupt'}

_NODE_ID_PATTERN = re.compile(r'[^a-zA-Z0-9_.-]')
_STEP_TAG = re.compile(r'^graph:step:(\d+)$')


def _clean_node_id(node_id: str) -> str:
    """Same normalisation the SDK applies to node ids before ``add_node``."""
    return _NODE_ID_PATTERN.sub('', node_id).replace('.', '_')


def _yaml_nodes(version_details: Optional[dict]) -> Dict[str, dict]:
    try:
        schema = yaml.safe_load((version_details or {}).get('instructions') or '') or {}
    except yaml.YAMLError:
        return {}
    nodes = schema.get('nodes') if isinstance(schema, dict) else None
    return {
        _clean_node_id(str(node['id'])): node
        for node in (nodes if isinstance(nodes, list) else [])
        if isinstance(node, dict) and node.get('id')
    }


def pipeline_node_types(version_details: Optional[dict]) -> Dict[str, str]:
    """Node id -> type (llm, tool, router, ...) from a pipeline's YAML; {} when unreadable."""
    return {node_id: str(node.get('type') or 'function') for node_id, node in _yaml_nodes(version_details).items()}


def pipeline_routing_nodes(version_details: Optional[dict]) -> Set[str]:
    """Ids of the nodes whose next node is chosen at run time, as the SDK wires their edges."""
    return {
        node_id for node_id, node in _yaml_nodes(version_details).items()
        if not node.get('transition') and (
            node.get('decision') or node.get('condition') or node.get('type') in ROUTING_NODE_TYPES
        )
    }


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _is_interrupt(error: Any) -> bool:
    if GraphInterrupt is not None and isinstance(error, GraphInterrupt):
        return True
    return type(error).__name__ in INTERRUPT_ERROR_NAMES


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _graph_step(tags: Optional[List[str]]) -> Optional[int]:
    for tag in tags or []:
        match = _STEP_TAG.match(str(tag))
        if match:
            return int(match.group(1))
    return None


class PipelineTraceRecorder:
    """Collects ``pipeline_trace`` spans from callback events, in start order."""

    def __init__(self, node_types: Optional[Dict[str, str]] = None, routing_nodes: Optional[Set[str]] = None):
        self.node_types = node_types or {}
        if routing_nodes is None:
            routing_nodes = {node for node, kind in self.node_types.items() if kind in ROUTING_NODE_TYPES}
        self.routing_nodes = set(routing_nodes)
        self.spans: List[dict] = []
        self.truncated = False
        self._graph_run_id: Optional[str] = None  # the outer graph's run, the parent of its nodes
        self._pinned = False
        self._open: Dict[str, dict] = {}  # chain run_id -> span
        self._started: Dict[str, datetime] = {}
        self._by_ns: Dict[str, dict] = {}  # node task's checkpoint namespace -> span

    def on_chain_start(self, run_id, name: Optional[str], metadata: Optional[dict], tags,
                       parent_run_id=None) -> None:
        metadata = metadata or {}
        node = metadata.get('langgraph_node')
        step = _graph_step(tags)
        if not node or node != name or step is None:
            return
        ns = str(metadata.get('langgraph_checkpoint_ns') or '')
        if '|' in ns:
            return
        parent = _optional_str(parent_run_id)
        if not self._pinned:
            # The first node to run is the outer graph's: a nested graph only runs inside one
            self._graph_run_id, self._pinned = parent, True
        if parent != self._graph_run_id:
            return  # a node of a graph run from inside an outer node
        if self.node_types and node not in self.node_types:
            return
        if len(self.spans) >= MAX_PIPELINE_SPANS:
            self.truncated = True
            return
        started = _now()
        span = {
            'node': node,
            'node_type': self.node_types.get(node),
            'step': step,
            'run_id': str(run_id),
            'timestamp_start': started.isoformat(),
            'timestamp_finish': None,
            'duration_ms': None,
            'status': 'running',
        }
        self.spans.append(span)
        self._open[str(run_id)] = span
        self._started[str(run_id)] = started
        if ns:
            self._by_ns[ns] = span

    def _close(self, run_id, status: str, error: Any = None) -> None:
        span = self._open.pop(str(run_id), None)
        if span is None:
            return
        finished = _now()
        started = self._started.pop(str(run_id), finished)
        span['timestamp_finish'] = finished.isoformat()
        span['duration_ms'] = int((finished - started).total_seconds() * 1000)
        span['status'] = status
        if error is not None:
            span['error'] = str(error)[:MAX_ERROR_CHARS]

    def on_chain_end(self, run_id) -> None:
        self._close(run_id, 'ok')

    def on_chain_error(self, run_id, error: BaseException) -> None:
        # A HITL node or MCP-auth guard pauses by raising GraphInterrupt: not a failure
        if _is_interrupt(error):
            self._close(run_id, 'interrupted')
        elif GraphBubbleUp is not None and isinstance(error, GraphBubbleUp):
            self._close(run_id, 'ok')  # control flow (e.g. ParentCommand), not a failure
        else:
            self._close(run_id, 'error', error)

    def _span_for(self, metadata: Optional[dict]) -> Optional[dict]:
        ns = (metadata or {}).get('langgraph_checkpoint_ns')
        return self._by_ns.get(str(ns)) if ns else None

    def on_custom_event(self, name: str, data: Any, metadata: Optional[dict]) -> None:
        if name not in EDGE_EVENTS and name not in HITL_EVENTS:
            return
        span = self._span_for(metadata)
        if span is None:
            return
        data = data if isinstance(data, dict) else {}
        if name in EDGE_EVENTS:
            edge = {'type': EDGE_EVENTS[name]}
            if data.get('next_step') is not None:
                edge['next'] = str(data['next_step'])
            span['edge'] = edge
        elif name == 'on_hitl_interrupt':
            span['hitl'] = {'action': 'interrupt'}
        else:
            span['hitl'] = {'action': _optional_str(data.get('action')), 'next': _optional_str(data.get('target_node'))}

    def _inferred_next(self, span: dict) -> Optional[str]:
        """The routing node's next node: the one span of the following step, or END when none ran."""
        if span['status'] != 'ok':
            return None
        following = [s['node'] for s in self.spans if s['step'] == span['step'] + 1]
        if len(following) == 1:
            return following[0]
        if following or self.truncated or any(s['step'] > span['step'] for s in self.spans):
            return None  # parallel branches: which one this node chose is not known
        return 'END'

    def snapshot(self) -> Dict[str, Any]:
        """JSON-safe copy for the envelope; spans still open are reported as ``running``."""
        spans = []
        for span in self.spans:
            copy = {key: dict(value) if isinstance(value, dict) else value for key, value in span.items()}
            if span['node'] in self.routing_nodes and not (span.get('edge') or {}).get('next'):
                following = self._inferred_next(span)
                if following is not None:
                    copy['edge'] = {'type': (span.get('edge') or {}).get('type') or 'conditional', 'next': following}
            spans.append(copy)
        return {'spans': spans, 'truncated': self.truncated}


def is_evaluation_run(kwargs: Optional[dict]) -> bool:
    """Whether the predict was started by a batch evaluation, which marks its leaf entity (#6677)."""
    entity = (kwargs or {}).get(ENTITY_KWARGS_KEY)
    return isinstance(entity, dict) and entity.get('type') == ENTITY_TYPE_EVALUATION


def attach_pipeline_trace(callback, version_details: Optional[dict], kwargs: Optional[dict]) -> None:
    """Start recording node spans on ``callback`` for a pipeline run of a batch evaluation.

    Evaluation is the only reader, so chat, scheduled and webhook runs record nothing and their
    envelope stays as it was, like an agent's."""
    if (version_details or {}).get('agent_type') == 'pipeline' and is_evaluation_run(kwargs):
        callback.pipeline_trace = PipelineTraceRecorder(
            pipeline_node_types(version_details), pipeline_routing_nodes(version_details),
        )
