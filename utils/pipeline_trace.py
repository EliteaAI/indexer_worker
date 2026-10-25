"""Pipeline node spans for the result envelope (#7015).

A pipeline's LLM and tool calls already reach ``thinking_steps`` / ``tool_calls_dict``, but
which nodes ran, in what order, how long each took and which edge was taken was only sent as
live custom events, and non-interactive runs (batch evaluation) drop those. This records one
span per top-level node from LangGraph's chain callbacks, so it works whatever the node
interface does with the events.

A node's run is the chain whose ``metadata.langgraph_node`` equals its run name and that
carries a ``graph:step:N`` tag. Only the outer graph is recorded (no ``|`` in the checkpoint
namespace): a sub-agent called from a node runs its own graph, and its nodes are not the
pipeline's. Nodes the SDK adds on its own (``StateDefaultNode``, a printer's ``*_reset``) are not
in the YAML and are skipped. Pipeline ``state`` is never stored: it is large and can hold secrets.

A router resolves its route without dispatching ``on_conditional_edge`` (the SDK only sends it
for a non-string result), so a router span's ``next`` is taken from the span that follows it.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

MAX_PIPELINE_SPANS = 500
MAX_ERROR_CHARS = 500

#: SDK custom events dispatched while an edge or node decides where to go next
EDGE_EVENTS = {
    'on_transitional_edge': 'transitional',
    'on_conditional_edge': 'conditional',
    'on_decision_edge': 'decision',
}
HITL_EVENTS = {'on_hitl_interrupt', 'on_hitl_resume'}
#: Node types whose outgoing edge is chosen at run time
ROUTING_NODE_TYPES = {'router', 'decision'}

_NODE_ID_PATTERN = re.compile(r'[^a-zA-Z0-9_.-]')
_STEP_TAG = re.compile(r'^graph:step:(\d+)$')


def _clean_node_id(node_id: str) -> str:
    """Same normalisation the SDK applies to node ids before ``add_node``."""
    return _NODE_ID_PATTERN.sub('', node_id).replace('.', '_')


def pipeline_node_types(version_details: Optional[dict]) -> Dict[str, str]:
    """Node id -> type (llm, tool, router, ...) from a pipeline's YAML; {} when unreadable."""
    try:
        schema = yaml.safe_load((version_details or {}).get('instructions') or '') or {}
    except yaml.YAMLError:
        return {}
    nodes = schema.get('nodes') if isinstance(schema, dict) else None
    types = {}
    for node in nodes if isinstance(nodes, list) else []:
        if isinstance(node, dict) and node.get('id'):
            types[_clean_node_id(str(node['id']))] = str(node.get('type') or 'function')
    return types


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _graph_step(tags: Optional[List[str]]) -> Optional[int]:
    for tag in tags or []:
        match = _STEP_TAG.match(str(tag))
        if match:
            return int(match.group(1))
    return None


class PipelineTraceRecorder:
    """Collects ``pipeline_trace`` spans from callback events, in start order."""

    def __init__(self, node_types: Optional[Dict[str, str]] = None):
        self.node_types = node_types or {}
        self.spans: List[dict] = []
        self.truncated = False
        self._open: Dict[str, dict] = {}  # chain run_id -> span
        self._started: Dict[str, datetime] = {}

    def on_chain_start(self, run_id, name: Optional[str], metadata: Optional[dict], tags) -> None:
        metadata = metadata or {}
        node = metadata.get('langgraph_node')
        step = _graph_step(tags)
        if not node or node != name or step is None:
            return
        if '|' in str(metadata.get('langgraph_checkpoint_ns') or ''):
            return
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
        if type(error).__name__ in {'GraphInterrupt', 'NodeInterrupt'}:
            self._close(run_id, 'interrupted')
        else:
            self._close(run_id, 'error', error)

    def _span_for(self, metadata: Optional[dict]) -> Optional[dict]:
        node = (metadata or {}).get('langgraph_node')
        if not node or '|' in str((metadata or {}).get('langgraph_checkpoint_ns') or ''):
            return None
        for span in reversed(self.spans):
            if span['node'] == node:
                return span
        return None

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
            span['hitl'] = {'action': data.get('action'), 'next': data.get('target_node')}

    def snapshot(self) -> Dict[str, Any]:
        """JSON-safe copy for the envelope; spans still open are reported as ``running``."""
        spans = [dict(span) for span in self.spans]
        for index, span in enumerate(spans):
            if span.get('node_type') not in ROUTING_NODE_TYPES or (span.get('edge') or {}).get('next'):
                continue
            if index + 1 < len(spans):
                span['edge'] = {'type': 'conditional', 'next': spans[index + 1]['node']}
            elif span['status'] == 'ok' and not self.truncated:
                span['edge'] = {'type': 'conditional', 'next': 'END'}
        return {'spans': spans, 'truncated': self.truncated}


def attach_pipeline_trace(callback, version_details: Optional[dict]) -> None:
    """Start recording node spans on ``callback`` when the run is a pipeline."""
    if (version_details or {}).get('agent_type') == 'pipeline':
        callback.pipeline_trace = PipelineTraceRecorder(pipeline_node_types(version_details))
