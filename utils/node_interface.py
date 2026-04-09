from datetime import datetime, timezone

import json
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
    agent_messages = 'agent_messages'
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
                 **kwargs):
        self.event_node = event_node
        self.node_event_name = node_event_name
        self.stream_id = stream_id
        self.message_id = message_id
        self.payload_additional_kwargs = kwargs
        self.event_log = []

    def emit(self, **kwargs) -> None:
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
            ).model_dump_json()
            e = json.loads(e)
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
        log.debug(f'NodeEventInterface emit {e=}')
        self.event_log.append(e)
        self.event_node.emit(self.node_event_name, e)


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
        'id', 'index_name', 'state', 'error', 'reindex', 'indexed', 'updated'
    },
    EventTypes.mcp_authorization_required.value: {
        'server_url', 'resource_metadata_url', 'www_authenticate', 'resource_metadata', 'tool_run_id', 'tool_name'
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
