from datetime import datetime, timezone

import json
import logging
import time
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        ...
from typing import Optional, Any

from pydantic import BaseModel, Field
from pylon.core.tools import log


class InitiatorType(StrEnum):
    """Enum representing the initiator of an operation"""
    user = 'user'      # User-initiated (UI, API calls)
    llm = 'llm'        # LLM-initiated (agent operations)
    schedule = 'schedule'  # Schedule-initiated (cron jobs)


def clean_for_json_serialization(data: Any, fallback_message: str = "Could not serialize data") -> Any:
    """
    Clean data to ensure JSON serializability by filtering out non-serializable objects recursively.
    
    Args:
        data: Data to clean (can be dict, list, or any other type)
        fallback_message: Message to return in case of serialization errors
        
    Returns:
        Data containing only JSON-serializable values
    """

    # Keys that should be preserved even if they contain blocked keywords
    ALLOWED_KEYS = {'mcp_client_id', 'mcp_client_secret'}
    try:
        if isinstance(data, dict):
            cleaned = {}
            for k, v in data.items():
                # Skip keys that might contain client references or problematic objects
                # BUT allow specific keys like client_id, client_secret
                if isinstance(k, str) and k not in ALLOWED_KEYS and any(keyword in k.lower() for keyword in [
                    'client', 'instance', 'callback', 'llm_instance', 'events_dispatched'
                ]):
                    # For events_dispatched, try to extract just the event names/types if possible
                    if k.lower() == 'events_dispatched' and isinstance(v, list):
                        cleaned[k] = f"<{len(v)} events (cleaned for serialization)>"
                    continue

                # Recursively clean both key and value
                if isinstance(k, (str, int, float, bool, type(None))):
                    cleaned_value = clean_for_json_serialization(v, fallback_message)
                    if cleaned_value is not None:
                        cleaned[k] = cleaned_value
            return cleaned
        elif isinstance(data, list):
            return [clean_for_json_serialization(item, fallback_message) for item in data if clean_for_json_serialization(item, fallback_message) is not None]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            # For non-serializable objects, check if it's a known problematic type
            obj_type_name = type(data).__name__
            if any(keyword in obj_type_name.lower() for keyword in ['client', 'callback', 'handler', 'instance']):
                return f"<{obj_type_name} object (not serializable)>"
            # For other non-serializable objects, try to convert to string
            return str(data)
    except Exception:
        return fallback_message


class EventTypes(StrEnum):
    agent_start = 'agent_start'
    agent_response = 'agent_response'
    agent_exception = 'agent_exception'
    agent_tool_start = 'agent_tool_start'
    agent_tool_end = 'agent_tool_end'
    agent_tool_error = 'agent_tool_error'
    agent_llm_start = 'agent_llm_start'
    agent_llm_chunk = 'agent_llm_chunk'
    agent_llm_end = 'agent_llm_end'
    mcp_authorization_required = 'mcp_authorization_required'
    agent_on_tool_node = 'agent_on_tool_node'
    agent_on_function_tool_node = 'agent_on_function_tool_node'
    agent_on_loop_tool_node = 'agent_on_loop_tool_node'
    agent_on_loop_node = 'agent_on_loop_node'
    agent_on_conditional_edge = 'agent_on_conditional_edge'
    agent_on_decision_edge = 'agent_on_decision_edge'
    agent_on_transitional_edge = 'agent_on_transitional_edge'
    pipeline_finish = 'pipeline_finish'

    references = 'references'
    chunk = 'chunk'

    full_message = 'full_message'
    partial_message = 'partial_message'

    agent_thinking_step = 'agent_thinking_step'
    agent_thinking_step_update = 'agent_thinking_step_update'
    agent_requires_confirmation = 'agent_requires_confirmation'
    agent_file_modified = 'agent_file_modified'
    agent_index_data_status = 'agent_index_data_status'
    agent_index_data_removed = 'agent_index_data_removed'

    # Swarm mode events - for multi-agent collaboration visibility
    agent_swarm_agent_start = 'agent_swarm_agent_start'
    agent_swarm_agent_response = 'agent_swarm_agent_response'
    agent_swarm_handoff = 'agent_swarm_handoff'
    # Socket event for child agent messages (frontend expects this specific event type)
    swarm_child_message = 'swarm_child_message'

    # HITL events
    agent_hitl_interrupt = 'agent_hitl_interrupt'

    # Summarization events - for context management progress visibility
    summarization_started = 'summarization_started'
    summarization_finished = 'summarization_finished'


class NodeEvent(BaseModel):
    type: EventTypes
    stream_id: Optional[str] = None
    message_id: Optional[str] = None
    question_id: Optional[str] = None
    content: Any = None
    thinking: Optional[str] = None  # Extended thinking/reasoning content (streamed separately from content)
    response_metadata: Optional[dict] = {}
    references: Optional[list] = []
    sio_event: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    # Swarm child message fields
    parent_message_id: Optional[str] = None
    agent_name: Optional[str] = None


class NodeEventInterface:
    def __init__(self, event_node,
                 node_event_name: str,
                 stream_id: Optional[str] = None,
                 message_id: Optional[str] = None,
                 batch_enabled: bool = True,
                 batch_max_chars: int = 64,
                 batch_max_interval_ms: int = 80,
                 event_metadata_overlay: Optional[dict] = None,
                 **kwargs):
        self.event_node = event_node
        self.node_event_name = node_event_name
        self.stream_id = stream_id
        self.message_id = message_id
        # Fan-out child attribution (#4993 Track 2) merged into every event's
        # response_metadata.metadata so the UI groups a standalone child's live
        # chips + HITL card under its own sub-agent accordion. None for ordinary
        # tasks. Kept OUT of payload_additional_kwargs (those set top-level
        # NodeEvent fields); this targets the nested metadata dict only.
        self.event_metadata_overlay = event_metadata_overlay or None
        self.payload_additional_kwargs = kwargs
        self.event_log = []
        # agent_llm_chunk coalescing: buffer tiny token-deltas into fewer, larger
        # emits. Keyed by run_id (tool_run_id); separate content/thinking channels.
        self.batch_enabled = batch_enabled
        self.batch_max_chars = batch_max_chars
        self.batch_max_interval_ms = batch_max_interval_ms
        self._chunk_buffers: dict = {}

    def emit(self, **kwargs) -> None:
        if not self.batch_enabled:
            self._emit_now(**kwargs)
            return

        ev_type = kwargs.get("type")
        ev_type_val = ev_type.value if isinstance(ev_type, EventTypes) else ev_type
        if ev_type_val == EventTypes.agent_llm_chunk.value:
            # Buffer streamed token-deltas; flushed by size/time/channel-switch or
            # by any non-chunk event (below) / final flush().
            self._buffer_chunk(kwargs)
            return

        # Any non-chunk event flushes pending streamed text first so ordering is
        # preserved (text stays ahead of tool calls, agent_llm_end, references, ...).
        self._flush_all()
        self._emit_now(**kwargs)

    def _emit_now(self, **kwargs) -> None:
        # Stamp fan-out child attribution into response_metadata.metadata (the
        # nested dict the UI reads parent_agent_name/thread_id from) without
        # overwriting fields the producer already set (#4993 Track 2). No-op when
        # the overlay is None (ordinary, non-child task).
        if self.event_metadata_overlay:
            self._apply_event_metadata_overlay(kwargs)
        # Clean all data to ensure JSON serializability
        clean_additional_kwargs = clean_for_json_serialization(
            self.payload_additional_kwargs,
            "Could not serialize payload_additional_kwargs"
        )
        clean_kwargs = clean_for_json_serialization(
            kwargs,
            "Could not serialize emit kwargs"
        )

        try:
            e = NodeEvent(
                stream_id=self.stream_id,
                message_id=self.message_id,
                **clean_additional_kwargs,
                **clean_kwargs
            ).model_dump(mode="json")
        except Exception as exc:
            # Final fallback: create a minimal event with error information
            log.error(f"Failed to serialize NodeEvent: {exc}")
            e = {
                "stream_id": self.stream_id,
                "message_id": self.message_id,
                "type": kwargs.get("type", "unknown"),
                "content": f"Serialization failed: {str(exc)}",
                "response_metadata": {"serialization_error": True}
            }
        if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
            log.debug(f'NodeEventInterface emit {e=}')
        self.event_log.append(e)
        self.event_node.emit(self.node_event_name, e)

    def _apply_event_metadata_overlay(self, kwargs: dict) -> None:
        """Merge child attribution into kwargs['response_metadata']['metadata'].

        Only fills keys the producer left unset, so a genuine nested-sub-agent
        ``parent_agent_name`` (a child that itself fans out in-process) is never
        clobbered by this task-level overlay. Builds the nested dicts on demand.
        """
        rmeta = kwargs.get("response_metadata")
        if not isinstance(rmeta, dict):
            rmeta = {}
            kwargs["response_metadata"] = rmeta
        meta = rmeta.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            rmeta["metadata"] = meta
        for key, value in self.event_metadata_overlay.items():
            if value and not meta.get(key):
                meta[key] = value

    def _buffer_chunk(self, kwargs: dict) -> None:
        rmeta = kwargs.get("response_metadata") or {}
        run_id = str(rmeta.get("tool_run_id") or "")
        content = kwargs.get("content") or ""
        thinking = kwargs.get("thinking") or ""
        buf = self._chunk_buffers.setdefault(
            run_id,
            {"content": {"text": "", "ts": None},
             "thinking": {"text": "", "ts": None},
             "meta": rmeta},
        )
        buf["meta"] = rmeta
        now = time.monotonic()
        if content:
            # Switching channels: flush the other first to keep arrival order.
            if buf["thinking"]["text"]:
                self._flush_channel(run_id, "thinking")
            self._append_channel(run_id, "content", content, now)
        if thinking:
            if buf["content"]["text"]:
                self._flush_channel(run_id, "content")
            self._append_channel(run_id, "thinking", thinking, now)

    def _append_channel(self, run_id: str, channel: str, text: str, now: float) -> None:
        ch = self._chunk_buffers[run_id][channel]
        if not ch["text"]:
            ch["ts"] = now
        ch["text"] += text
        if (len(ch["text"]) >= self.batch_max_chars
                or (now - ch["ts"]) * 1000 >= self.batch_max_interval_ms):
            self._flush_channel(run_id, channel)

    def _flush_channel(self, run_id: str, channel: str) -> None:
        buf = self._chunk_buffers.get(run_id)
        if not buf:
            return
        ch = buf[channel]
        text = ch["text"]
        if not text:
            return
        ch["text"] = ""
        ch["ts"] = None
        if channel == "content":
            self._emit_now(type=EventTypes.agent_llm_chunk,
                           response_metadata=buf["meta"], content=text, thinking="")
        else:
            self._emit_now(type=EventTypes.agent_llm_chunk,
                           response_metadata=buf["meta"], content="", thinking=text)

    def _flush_run(self, run_id: str) -> None:
        buf = self._chunk_buffers.get(run_id)
        if not buf:
            return
        # Emit in arrival order when both channels hold a trailing partial.
        channels = [c for c in ("content", "thinking") if buf[c]["text"]]
        channels.sort(key=lambda c: buf[c]["ts"] or 0)
        for c in channels:
            self._flush_channel(run_id, c)

    def _flush_all(self) -> None:
        for run_id in list(self._chunk_buffers.keys()):
            self._flush_run(run_id)

    def flush(self) -> None:
        """Flush any buffered streamed text. Call at stream teardown."""
        self._flush_all()


class NoOpNodeEventInterface(NodeEventInterface):
    """Event interface for non-interactive predicts (scheduled pipelines, webhook
    triggers, blocking REST calls, indexer-internal predicts).

    These flows have no Socket.IO subscriber, so live-UI events (token chunks,
    tool start/end, thinking steps, graph edges, swarm start/handoff) are pure
    overhead — serialized and pushed to a room nobody is in.

    Only events that carry state or persistence semantics are still emitted.
    Note that `full_message` and `partial_message` ride separate event_node
    channels (`application_full_response` / `application_partial_response`) and
    never pass through this interface, so message-group DB persistence and
    incremental crash-safety are preserved regardless of this filter.
    """

    # Allowlist of state-bearing events. Everything else is dropped.
    _ALLOWED_EVENT_TYPES = frozenset({
        EventTypes.agent_index_data_status.value,    # drives index state machine
        EventTypes.agent_index_data_removed.value,   # drives index state machine
        EventTypes.mcp_authorization_required.value,  # pauses stream + DB row
        EventTypes.agent_hitl_interrupt.value,        # resumability
        EventTypes.agent_requires_confirmation.value,  # resumability
        EventTypes.agent_exception.value,             # error reporting
        EventTypes.agent_swarm_agent_response.value,  # triggers chat_child_message_save (DB)
        EventTypes.summarization_started.value,       # mid-turn context state
        EventTypes.summarization_finished.value,      # mid-turn context state
        EventTypes.pipeline_finish.value,             # completion signal
    })

    def emit(self, **kwargs) -> None:
        ev_type = kwargs.get("type")
        ev_type_val = ev_type.value if isinstance(ev_type, EventTypes) else ev_type
        if ev_type_val in self._ALLOWED_EVENT_TYPES:
            # Allowed events are never agent_llm_chunk, so bypass batching entirely.
            self._emit_now(**kwargs)

    def flush(self) -> None:
        # No chunk buffering happens in non-interactive mode; nothing to flush.
        return


ELITEA_SDK_CUSTOM_EVENTS_MAPPER = {
    EventTypes.agent_on_tool_node.value: {
        'state', 'input_variables', 'tool_result',
    },
    EventTypes.agent_on_function_tool_node.value: {
        'state', 'input_variables', 'input_mapping', 'tool_result',
    },
    EventTypes.agent_on_loop_tool_node.value: {
        'state', 'input_variables', 'tool_result'
    },
    EventTypes.agent_on_loop_node.value: {
        'state', 'input_variables', 'accumulated_response'
    },
    EventTypes.agent_on_conditional_edge.value: {
        'state', 'condition',
    },
    EventTypes.agent_on_decision_edge.value: {
        'state', 'decisional_inputs',
    },
    EventTypes.agent_on_transitional_edge.value: {
        'state', 'next_step',
    },
    EventTypes.agent_thinking_step.value: {
        'message', 'tool_name', 'toolkit'
    },
    EventTypes.agent_thinking_step_update.value: {
        'message', 'tool_name', 'toolkit', 'markdown'
    },
    EventTypes.agent_file_modified.value: {
        'message', 'filepath', 'tool_name', 'toolkit', 'operation_type', 'meta', 'media_type'
    },
    EventTypes.agent_index_data_status.value: {
        'id', 'index_name', 'state', 'error', 'reindex', 'indexed', 'updated',
        'created_at', 'updated_on', 'toolkit_id'
    },
    EventTypes.mcp_authorization_required.value: {
        'server_url', 'resource_metadata_url', 'www_authenticate', 'resource_metadata', 'authorization_servers', 'tool_run_id', 'tool_name'
    },
    EventTypes.agent_index_data_removed.value: {
        'index_name', 'toolkit_id', 'project_id'
    },
    # Swarm mode event mappings - for multi-agent collaboration visibility
    EventTypes.agent_swarm_agent_start.value: {
        'agent_name', 'is_parent', 'message_count',
    },
    EventTypes.agent_swarm_agent_response.value: {
        'agent_name', 'is_parent', 'content', 'has_tool_calls', 'tool_calls',
    },
    EventTypes.agent_swarm_handoff.value: {
        'from_agent', 'to_agent',
    },
    EventTypes.agent_hitl_interrupt.value: {
        'node_name', 'message', 'available_actions', 'routes', 'edit_state_key',
    },
}
