#!/usr/bin/python3
# coding=utf-8

#   Copyright 2024 EPAM Systems
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Common infrastructure for agent execution.

This module provides shared utilities for both indexer_agent.py and indexer_predict_agent.py
to reduce code duplication and ensure consistent behavior.
"""

import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from langchain_core.messages import HumanMessage, AIMessage

from pylon.core.tools import log

from tools import worker_core

from .constants import DEFAULT_MEMORY_CONFIG
from .funcs import prepend_vision_system_message, prepend_attachment_system_message
from .image_helpers import (
    is_anthropic_model,
    strip_image_chunks_from_assistant_messages,
    strip_stale_filepath_image_chunks,
)
from .node_interface import NodeEventInterface, EventTypes, NodeEvent, InitiatorType
from .langfuse_callback import create_langfuse_callback, flush_langfuse_callback, langfuse_trace_context

from ..methods.agent_common import (
    execution_error,
    EliteACallback,
    EliteACustomCallback,
    EVENTNODE_EVENT_NAME,
    EVENTNODE_FULL_RESPONSE_NAME,
    _fetch_pgvector_connstr_with_retry,
    temp_elitea_client,
    fetch_langfuse_config,
)


# =============================================================================
# Summarization Callbacks
# =============================================================================

def create_summarization_callbacks(node_interface: NodeEventInterface) -> dict:
    """Create callbacks for summarization middleware to emit progress events."""
    def on_started(data):
        node_interface.emit(
            type=EventTypes.summarization_started,
            content=data
        )

    def on_summarized(data):
        node_interface.emit(
            type=EventTypes.summarization_finished,
            content=data
        )

    return {'started': on_started, 'summarized': on_summarized}


# =============================================================================
# Response Parsing
# =============================================================================

def normalize_response_content(content: Any) -> str:
    """
    Normalize response content to a string, handling various formats.

    This handles:
    - Plain strings (pass through)
    - Claude's list format: [{'type': 'text', 'text': '...'}]
    - Mixed list format with strings and dicts
    - None values
    - Other types (JSON serialized)

    Args:
        content: The response content in any format

    Returns:
        Normalized string content
    """
    if content is None:
        return ''

    if isinstance(content, str):
        # Check if string is JSON that contains tool_use blocks (shouldn't be shown to user)
        stripped = content.strip()
        if stripped.startswith('[') and 'tool_use' in stripped:
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    # Recursively handle the parsed list
                    return normalize_response_content(parsed)
            except json.JSONDecodeError:
                pass  # Not valid JSON, return as-is
        return content

    if isinstance(content, list):
        # Handle Claude's list format: [{'type': 'text', 'text': '...'}]
        text_parts = []
        has_only_tool_blocks = True
        for block in content:
            if isinstance(block, dict):
                # Standard Claude format with type='text'
                if block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
                    has_only_tool_blocks = False
                # Alternative format with just 'text' key
                elif 'text' in block and 'type' not in block:
                    text_parts.append(block.get('text', ''))
                    has_only_tool_blocks = False
                # Content blocks might have other types we should skip (tool_use, tool_result, thinking)
                elif block.get('type') in ('tool_use', 'tool_result', 'thinking'):
                    continue
                else:
                    # Unknown dict format - serialize it
                    text_parts.append(json.dumps(block, ensure_ascii=False))
                    has_only_tool_blocks = False
            elif isinstance(block, str):
                text_parts.append(block)
                has_only_tool_blocks = False
            else:
                # Unknown type in list - serialize it
                text_parts.append(str(block))
                has_only_tool_blocks = False

        # If list only had tool_use/tool_result blocks, return empty string (not JSON)
        if has_only_tool_blocks and not text_parts:
            return ''
        return ''.join(text_parts)

    # For any other type, JSON serialize
    return json.dumps(content, ensure_ascii=False)


def extract_response_content(response: Dict[str, Any], response_format: str = 'messages') -> str:
    """
    Extract and normalize content from agent response.

    Args:
        response: The raw response from agent invocation
        response_format: Either 'messages' (for predict_agent) or 'output' (for application agent)

    Returns:
        Normalized string content
    """
    if response_format == 'messages':
        # Predict agent format: {"messages": [...]}
        messages = response.get("messages", [])
        if isinstance(messages, list) and len(messages) > 0:
            last_message = messages[-1]
            # Handle both dict and message object
            if hasattr(last_message, 'content'):
                content = last_message.content
            elif isinstance(last_message, dict):
                content = last_message.get('content', '')
            else:
                content = str(last_message)
        else:
            # Fallback: swarm responses from SwarmResultAdapter have {"output": ...} but no "messages"
            # When predict agent runs in swarm mode, extract output directly
            if "output" in response and response.get("output"):
                content = response["output"]
            else:
                content = str(response)
    else:
        # Application agent format: {"output": ...}
        content = response.get("output", "")
        # Fallback: swarm responses may have messages but no output key
        if not content and "messages" in response:
            messages = response["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                from langchain_core.messages import HumanMessage
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and not isinstance(msg, HumanMessage):
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        break

    return normalize_response_content(content)


def build_output_message(content: str) -> Dict[str, Any]:
    """
    Build a standardized output message dict.

    Args:
        content: The normalized response content

    Returns:
        Dict with content and role='assistant'
    """
    output = AIMessage(content=content).dict()
    output['role'] = 'assistant'
    return output


# =============================================================================
# Memory Setup
# =============================================================================

def setup_memory(descriptor_config: Dict[str, Any], pgvector_connstr: Optional[str] = None):
    """
    Configure memory (checkpointer) based on config.

    Args:
        descriptor_config: The plugin descriptor config
        pgvector_connstr: Optional pgvector connection string from secrets

    Returns:
        Tuple of (memory_type, memory_config)
    """
    memory_config = descriptor_config.get("agent_memory_config", None)

    if memory_config is None:
        memory_config = DEFAULT_MEMORY_CONFIG

    # If pgvector_connstr is available and memory_config is postgres type,
    # always use pgvector_connstr (project secret takes precedence over local config)
    if pgvector_connstr and memory_config.get("type") == "postgres":
        memory_config = {
            **memory_config,
            "connection_string": pgvector_connstr.replace("postgresql+psycopg://", "postgresql://")
        }
        log.debug("Using pgvector_project_connstr for postgres memory")

    memory_type = memory_config.get("type", "memory")
    return memory_type, memory_config


def create_memory_saver(memory_type: str, memory_config: Dict[str, Any]):
    """
    Create the appropriate memory saver based on type.

    Args:
        memory_type: 'sqlite', 'postgres', or 'memory'
        memory_config: Configuration dict for the memory type

    Returns:
        Context manager that yields (memory_saver, cleanup_func)
    """
    if memory_type == "sqlite":
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver

        connection = sqlite3.connect(
            memory_config["path"],
            check_same_thread=False,
        )
        memory = SqliteSaver(connection)
        return memory, lambda: connection.close()

    if memory_type == "postgres":
        from psycopg import Connection
        from langgraph.checkpoint.postgres import PostgresSaver

        connection = Connection.connect(
            memory_config["connection_string"],
            **memory_config.get("connection_kwargs", {}),
        )
        memory = PostgresSaver(connection)
        memory.setup()
        return memory, lambda: connection.close()

    # Default: no persistence
    return None, lambda: None


# =============================================================================
# Client and Event Node Setup
# =============================================================================

def setup_event_node(multiprocessing_context: str):
    """
    Setup the event node based on multiprocessing context.

    Args:
        multiprocessing_context: The tasknode multiprocessing context ('fork' or other)

    Returns:
        Configured event node
    """
    if multiprocessing_context == "fork":
        local_event_node = worker_core.event_node.clone()
        local_event_node.start()
    else:
        local_event_node = worker_core.event_node
    return local_event_node


def create_elitea_client(client_args: Dict[str, Any], api_token: str, api_extra_headers: Dict[str, str]):
    """
    Create EliteAClient after fork to avoid pickling RLock objects.

    Args:
        client_args: Client configuration arguments
        api_token: API authentication token
        api_extra_headers: Additional API headers

    Returns:
        Configured EliteAClient instance
    """
    from ..utils.funcs import dev_reload_sdk
    dev_reload_sdk('elitea_sdk.runtime.clients')
    dev_reload_sdk('elitea_sdk.runtime.toolkits')
    dev_reload_sdk('elitea_sdk.runtime.tools')
    from elitea_sdk.runtime.clients.client import EliteAClient

    return EliteAClient(
        base_url=client_args.get("deployment", client_args.get("base_url", None)),
        project_id=client_args.get("project_id"),
        auth_token=api_token,
        api_extra_headers=api_extra_headers,
    )


def create_node_interface(
    local_event_node,
    stream_id: Optional[str],
    message_id: Optional[str],
    task_meta: Dict[str, Any]
) -> NodeEventInterface:
    """
    Create NodeEventInterface for emitting events.

    Args:
        local_event_node: The event node to use
        stream_id: Stream identifier
        message_id: Message identifier
        task_meta: Task metadata containing sio_event and question_id

    Returns:
        Configured NodeEventInterface
    """
    return NodeEventInterface(
        event_node=local_event_node,
        node_event_name=EVENTNODE_EVENT_NAME,
        stream_id=stream_id,
        message_id=message_id,
        sio_event=task_meta.get("sio_event"),
        question_id=task_meta.get("question_id")
    )


# =============================================================================
# Thread ID Management
# =============================================================================

def ensure_thread_id(thread_id: Optional[str], conversation_id: Optional[Any]) -> str:
    """
    Ensure thread_id is never None to prevent checkpointer state sharing.

    Args:
        thread_id: Provided thread ID (may be None)
        conversation_id: Fallback conversation ID

    Returns:
        Valid thread ID string
    """
    if thread_id is not None:
        return thread_id

    if conversation_id:
        return str(conversation_id)

    import uuid
    new_thread_id = str(uuid.uuid4())
    log.debug(f"Generated unique thread_id (no conversation_id): {new_thread_id}")
    return new_thread_id


# =============================================================================
# Callback Setup
# =============================================================================

def create_callbacks(
    node_interface: NodeEventInterface,
    thread_id: str,
    message_id: Optional[str],
    task_meta: Dict[str, Any],
    task_id: str,
    debug: bool = False
) -> Tuple[EliteACallback, EliteACustomCallback]:
    """
    Create EliteA callback handlers.

    Args:
        node_interface: Event interface for emitting events
        thread_id: Thread identifier
        message_id: Message identifier
        task_meta: Task metadata
        task_id: Task identifier
        debug: Enable debug mode

    Returns:
        Tuple of (EliteACallback, EliteACustomCallback)
    """
    elitea_callback = EliteACallback(
        node_interface,
        debug=debug,
        thread_id=thread_id,
        message_id=message_id,
        project_id=task_meta.get("project_id"),
        chat_project_id=task_meta.get("chat_project_id"),
    )

    elitea_custom_callback = EliteACustomCallback(
        node_interface,
        debug=debug,
        message_id=message_id,
        project_id=task_meta.get("project_id"),
        chat_project_id=task_meta.get("chat_project_id"),
        user_id=task_meta.get("user_context", {}).get("user_id"),
        initiator=task_meta.get("initiator", InitiatorType.llm),
        task_id=task_id,
    )

    return elitea_callback, elitea_custom_callback


def create_langfuse_callback_with_metadata(
    langfuse_config: Optional[Dict[str, Any]],
    application_name: str,
    thread_id: str,
    message_id: Optional[str],
    task_meta: Dict[str, Any]
) -> Tuple[Any, Any, Any]:
    """
    Create Langfuse callback for tracing.

    Args:
        langfuse_config: Langfuse configuration
        application_name: Name of the application/agent
        thread_id: Thread identifier
        message_id: Message identifier
        task_meta: Task metadata

    Returns:
        Tuple of (langfuse_client, langfuse_callback, langfuse_trace_attrs)
    """
    langfuse_user_id = str(task_meta.get("user_context", {}).get("user_id")) \
        if task_meta.get("user_context", {}).get("user_id") else None

    langfuse_metadata = {
        "project_id": str(task_meta.get("project_id", "")),
        "chat_project_id": str(task_meta.get("chat_project_id", "")),
        "application": application_name,
        "message_id": str(message_id or ""),
    }

    return create_langfuse_callback(
        langfuse_config,
        trace_name=application_name,
        session_id=thread_id,
        user_id=langfuse_user_id,
        metadata=langfuse_metadata,
    )


# =============================================================================
# Checkpoint Resume
# =============================================================================

def configure_checkpoint_resume(
    agent_executor,
    thread_id: str,
    checkpoint_id: Optional[str],
    invoke_input: Dict[str, Any],
    invoke_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Configure checkpoint resume if needed.

    Args:
        agent_executor: The agent executor with state history
        thread_id: Thread identifier
        checkpoint_id: Optional specific checkpoint to resume from
        invoke_input: Current invoke input dict
        invoke_config: Current invoke config dict

    Returns:
        Tuple of (modified_invoke_input, modified_invoke_config)
    """
    try:
        # If no checkpoint_id provided, get the last checkpoint from state history
        if not checkpoint_id:
            states = list(agent_executor.get_state_history({
                'configurable': {'thread_id': thread_id}
            }))
            if states:
                checkpoint_id = states[0].config['configurable']['checkpoint_id']

        if checkpoint_id:
            # Modify input for continuation
            invoke_input = {'input': 'Continue your last response'}
            invoke_config['configurable']['checkpoint_id'] = checkpoint_id
            invoke_config['should_continue'] = True
            log.debug(f'Resuming execution from checkpoint: {checkpoint_id}')
        else:
            log.warning('No checkpoint available to resume from, will start fresh')
    except Exception as e:
        log.error(f'Checkpoint configuration failed: {e}')

    return invoke_input, invoke_config


# =============================================================================
# Response Events
# =============================================================================

def emit_response_events(
    node_interface: NodeEventInterface,
    response: Dict[str, Any],
    output: Dict[str, Any],
    thread_id: str,
    message_id: Optional[str],
    elitea_callback: EliteACallback,
    elitea_custom_callback: EliteACustomCallback,
    task_meta: Dict[str, Any],
    application_details: Dict[str, Any],
    chat_history: List[Dict[str, Any]],
    user_message: HumanMessage,
    should_continue: bool = False,
    hitl_resume: bool = False,
    hitl_action: Optional[str] = None,
    hitl_value: str = '',
    image_thumbnails: Optional[Dict[str, str]] = None,
    context_info: Optional[Dict[str, Any]] = None
):
    """
    Emit all response-related events.

    Args:
        node_interface: Event interface
        response: Raw agent response
        output: Normalized output dict
        thread_id: Thread identifier
        message_id: Message identifier
        elitea_callback: Main callback handler
        elitea_custom_callback: Custom callback handler
        task_meta: Task metadata
        application_details: Application configuration
        chat_history: Chat history list
        user_message: The user's message
        should_continue: Whether this was a continuation
        context_info: Context info with message/token counts and summarization details if summarization occurred
    """
    thread_id_response = response.get('thread_id', thread_id)

    # Emit a dedicated event when execution paused at a HITL node.
    hitl_interrupt = response.get('hitl_interrupt')
    if hitl_interrupt:
        node_interface.emit(
            type=EventTypes.agent_hitl_interrupt,
            content=hitl_interrupt.get('message', 'Awaiting human review...'),
            response_metadata={
                'thread_id': thread_id_response,
                'message': hitl_interrupt.get('message', 'Awaiting human review...'),
                'hitl_interrupt': hitl_interrupt,
                'node_name': hitl_interrupt.get('node_name'),
                'available_actions': hitl_interrupt.get('available_actions', []),
                'routes': hitl_interrupt.get('routes', {}),
                'edit_state_key': hitl_interrupt.get('edit_state_key'),
            }
        )

    # Emit pipeline_finish if execution completed
    if response.get('execution_finished'):
        node_interface.emit(
            type=EventTypes.pipeline_finish,
            content=output['content'],
            response_metadata={
                'finish_reason': 'finished',
                'next_step': 'END',
                'thread_id': thread_id_response
            }
        )

    # Calculate tokens
    elitea_callbacks = [elitea_callback, elitea_custom_callback]
    total_tokens_in = sum(map(lambda cb: cb.tokens_in, elitea_callbacks))
    total_tokens_out = sum(map(lambda cb: cb.tokens_out, elitea_callbacks))

    # Skip normal message events for HITL pauses. The frontend renders the
    # pause state from the dedicated interrupt event and should not receive a
    # duplicate assistant message or a synthetic chat history turn.
    if not hitl_interrupt:
        node_interface.emit(
            type=EventTypes.agent_response,
            content=output['content'],
            response_metadata={
                'finish_reason': 'stop',
                'thread_id': thread_id_response
            }
        )

        full_message_metadata = {
            'project_id': task_meta.get("project_id"),
            'chat_project_id': task_meta.get("chat_project_id"),
            'application_details': application_details,
            'thread_id': thread_id_response,
            'thinking_steps': elitea_callback.thinking_steps,
            'tool_calls': {
                run_id: tool_call.model_dump()
                for run_id, tool_call in elitea_callback.tool_calls.items()
            },
            'llm_start_timestamp': elitea_callback.llm_start_timestamp,
            'additional_response_meta': elitea_custom_callback.additional_response_meta,
            'files_modified': elitea_custom_callback.modified_files,
            'image_thumbnails': image_thumbnails or {},
            'index_statuses': elitea_custom_callback.index_statuses,
            'chat_history_tokens_input': total_tokens_in,
            'llm_response_tokens_output': total_tokens_out,
            'should_continue': should_continue,
            'context_info': context_info,
        }

        msg_event_node = NodeEvent(
            type=EventTypes.full_message,
            stream_id=node_interface.stream_id,
            message_id=message_id,
            response_metadata=full_message_metadata,
            content=output['content'],
            **node_interface.payload_additional_kwargs
        ).model_dump_json()
        msg_event_node = json.loads(msg_event_node)
        node_interface.event_node.emit(EVENTNODE_FULL_RESPONSE_NAME, msg_event_node)

        if hitl_resume:
            if hitl_action == 'edit':
                chat_history.append({'role': 'user', 'content': hitl_value})
        else:
            chat_history.append({'role': 'user', 'content': user_message.content})

        chat_history.append(output)

        node_interface.emit(
            type=EventTypes.agent_messages,
            response_metadata={
                'chat_history': chat_history,
                'thread_id': thread_id_response,
                'finish_reason': 'stop'
            }
        )

    return total_tokens_in, total_tokens_out, thread_id_response


# =============================================================================
# Tracing Context
# =============================================================================

def with_tracing_span(
    traceparent: Optional[str],
    span_name: str,
    trace_stream_id: Optional[str],
    trace_message_id: Optional[str],
    project_id: str,
    func,
    *args,
    **kwargs
):
    """
    Execute function with optional OpenTelemetry tracing span.

    Args:
        traceparent: Traceparent header from parent context
        span_name: Name for the span
        trace_stream_id: Stream identifier (for tracing attributes)
        trace_message_id: Message identifier (for tracing attributes)
        project_id: Project identifier
        func: Function to execute
        *args, **kwargs: Arguments to pass to function

    Returns:
        Result of func execution
    """
    if traceparent:
        try:
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind
            from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

            tracer = trace.get_tracer('pylon-indexer', '1.0.0')
            propagator = TraceContextTextMapPropagator()
            ctx = propagator.extract(carrier={'traceparent': traceparent})

            with tracer.start_as_current_span(
                span_name,
                context=ctx,
                kind=SpanKind.CONSUMER,
                attributes={
                    'task.name': span_name,
                    'stream.id': trace_stream_id or '',
                    'message.id': trace_message_id or '',
                    'project.id': str(project_id),
                }
            ):
                return func(*args, **kwargs)
        except Exception as e:
            log.debug(f"Tracing setup failed, continuing without trace: {e}")

    return func(*args, **kwargs)


# =============================================================================
# Result Building
# =============================================================================

def build_success_result(
    chat_history: List[Dict[str, Any]],
    elitea_callback: EliteACallback,
    tokens_in: int,
    tokens_out: int,
    context_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build successful execution result dict.

    Args:
        chat_history: Updated chat history
        elitea_callback: Callback with thinking steps and tool calls
        tokens_in: Input token count
        tokens_out: Output token count
        context_info: Optional context info with message/token counts and summarization details

    Returns:
        Result dict
    """
    sanitized_steps = json.loads(json.dumps(elitea_callback.thinking_steps, default=str))
    result = {
        'chat_history': chat_history,
        'error': None,
        'thinking_steps': sanitized_steps,
        'tool_calls': [
            i for i in sanitized_steps
            if i.get('generation_info') and i['generation_info'].get('finish_reason') == 'tool_calls'
        ],
        'tool_calls_dict': {k: v.model_dump() for k, v in elitea_callback.tool_calls.items()},
        'chat_history_tokens_input': tokens_in,
        'llm_response_tokens_output': tokens_out,
    }
    if context_info:
        result['context_info'] = context_info
    return result


# =============================================================================
# Invoke Input Preparation
# =============================================================================

def _strip_all_image_chunks(messages: list) -> None:
    """Remove all image_url chunks from messages in-place (non-vision models)."""
    for message in messages:
        if isinstance(message, dict):
            content = message.get('content')
        else:
            content = getattr(message, 'content', None)
        if not isinstance(content, list):
            continue
        filtered = [c for c in content if not (isinstance(c, dict) and c.get('type') == 'image_url')]
        if len(filtered) != len(content):
            new_content = filtered if filtered else '[Image content removed - model does not support vision]'
            if isinstance(message, dict):
                message['content'] = new_content
            else:
                message.content = new_content


def has_images_in_messages(
    chat_history: List[Dict[str, Any]],
    user_message: HumanMessage
) -> bool:
    """
    Check if any message in chat_history or user_message contains image_url chunks.

    Args:
        chat_history: Chat history messages
        user_message: Current user message

    Returns:
        True if at least one image_url chunk is found, False otherwise
    """
    for message in chat_history + [user_message]:
        content = message.get('content') if isinstance(message, dict) else getattr(message, 'content', None)

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'image_url':
                    return True

    return False


def prepare_invoke_input(
    chat_history: List[Dict[str, Any]],
    user_message: HumanMessage,
    conversation_id: Optional[str] = None,
    include_attachment_system_message: bool = True,
    model_name: Optional[str] = None,
    supports_vision: bool = True,
) -> Dict[str, Any]:
    """Prepare unified invoke input using messages format.

    Strips stale ``filepath:`` image chunks from *chat_history* before
    building the messages list.  For Anthropic models, also removes all
    ``image_url`` chunks from assistant-role messages because Anthropic
    does not permit image content blocks inside assistant turns.
    """
    # Remove unresolved filepath: image refs from older turns
    strip_stale_filepath_image_chunks(chat_history)

    # Anthropic rejects image blocks in assistant messages — strip them
    # while preserving sibling text chunks (image file description).
    if model_name and is_anthropic_model(model_name):
        strip_image_chunks_from_assistant_messages(chat_history)

    # Strip all image_url chunks when model does not support vision — must run
    # before has_images_in_messages() to avoid prepending a vision system prompt.
    if not supports_vision:
        _strip_all_image_chunks(chat_history)
        if isinstance(user_message.content, list):
            filtered = [c for c in user_message.content if not (isinstance(c, dict) and c.get('type') == 'image_url')]
            user_message = HumanMessage(content=filtered if filtered else '[Image content removed - model does not support vision]')

    # Start with original chat history
    invoke_messages = list(chat_history)

    # Prepend attachment system message only when the attachment toolkit is
    # actually present in this invocation — avoids polluting the LLM context
    # with bucket names that have no corresponding tool.
    if conversation_id and include_attachment_system_message:
        invoke_messages = prepend_attachment_system_message(invoke_messages, str(conversation_id))
    
    # Prepend vision system message if images are present
    if has_images_in_messages(chat_history, user_message):
        invoke_messages = prepend_vision_system_message(invoke_messages)
    
    # Add user message
    invoke_messages = invoke_messages + [user_message]

    return {"messages": invoke_messages}


# Backwards compatibility alias
prepare_invoke_input_messages = prepare_invoke_input
