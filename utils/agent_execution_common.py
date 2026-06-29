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
from .node_interface import NodeEventInterface, NoOpNodeEventInterface, EventTypes, NodeEvent, InitiatorType
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

    All SDK agent types (LangGraphAgentRunnable, SwarmResultAdapter, Application)
    now return a standardized format with 'output' key always present.

    Args:
        response: The raw response from agent invocation
        response_format: Either 'messages' (for predict_agent) or 'output' (for application agent)
                        Note: Both formats now support 'output' key as primary source.

    Returns:
        Normalized string content
    """
    # Primary extraction: use 'output' key (always present in standardized SDK responses)
    content = response.get("output", "")

    # Fallback for legacy responses that may only have 'messages' key
    if not content and "messages" in response:
        messages = response.get("messages", [])
        if isinstance(messages, list) and len(messages) > 0:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
            elif isinstance(last_message, dict):
                content = last_message.get('content', '')
            else:
                content = str(last_message)

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
    dev_reload_sdk('elitea_sdk.runtime.langchain')
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
    task_meta: Dict[str, Any],
    batch_config: Optional[Dict[str, Any]] = None,
) -> NodeEventInterface:
    """
    Create NodeEventInterface for emitting events.

    Args:
        local_event_node: The event node to use
        stream_id: Stream identifier
        message_id: Message identifier
        task_meta: Task metadata containing sio_event and question_id
        batch_config: Optional llm_chunk_batching config (enabled/max_chars/max_interval_ms).
            Missing keys fall back to NodeEventInterface defaults.

    Returns:
        Configured NodeEventInterface. For non-interactive flows
        (``task_meta["non_interactive"]`` truthy) a NoOpNodeEventInterface is
        returned, which suppresses live-UI events and keeps only state-bearing
        ones. See NoOpNodeEventInterface for the allowlist.
    """
    batch_config = batch_config or {}
    batch_kwargs = {}
    if "enabled" in batch_config:
        batch_kwargs["batch_enabled"] = batch_config["enabled"]
    if "max_chars" in batch_config:
        batch_kwargs["batch_max_chars"] = batch_config["max_chars"]
    if "max_interval_ms" in batch_config:
        batch_kwargs["batch_max_interval_ms"] = batch_config["max_interval_ms"]

    interface_cls = (
        NoOpNodeEventInterface
        if task_meta.get("non_interactive")
        else NodeEventInterface
    )
    return interface_cls(
        event_node=local_event_node,
        node_event_name=EVENTNODE_EVENT_NAME,
        stream_id=stream_id,
        message_id=message_id,
        sio_event=task_meta.get("sio_event"),
        question_id=task_meta.get("question_id"),
        event_metadata_overlay=_child_event_metadata_overlay(task_meta),
        **batch_kwargs,
    )


def _child_event_metadata_overlay(task_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attribution to stamp onto every event when this task is a fan-out child.

    A parked-fan-out child (#4993 Track 2) is a standalone ``indexer_agent`` that
    streams onto the PARENT's message, so — unlike the in-process gather path —
    its events carry no ``parent_agent_name``/``checkpoint_ns`` and the UI cannot
    attribute its live chips + HITL card to a sub-agent accordion. pylon_main puts
    the child's identity in the task meta (``parallel_dispatch_launch_children``);
    we surface it as a metadata overlay merged into every event's
    ``response_metadata.metadata`` so the existing UI bucketing (which keys on
    ``parent_agent_name`` / ``thread_id``) groups the child unchanged.

    Returns None for an ordinary (non-child) task, leaving event metadata as-is.
    """
    if not isinstance(task_meta, dict):
        return None
    child_thread_id = task_meta.get("child_thread_id")
    if not (child_thread_id and task_meta.get("parent_thread_id")):
        return None
    overlay = {
        "parent_agent_name": task_meta.get("subagent_name") or "",
        "child_thread_id": child_thread_id,
        "thread_id": child_thread_id,
    }
    if task_meta.get("tool_call_id"):
        overlay["tool_call_id"] = task_meta.get("tool_call_id")
    return overlay


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
        # Durable fan-out child (#4993 Track 2): present only when this run is a
        # parked sub-agent (set in parallel_dispatch_launch_children). Lets the
        # callback tag the child's steps with its own name so they group under its
        # sub-agent accordion in the parent's thinking view on history replay.
        subagent_name=task_meta.get("subagent_name"),
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

    project_id = task_meta.get("project_id")
    langfuse_environment = f"project-{project_id}" if project_id else None

    return create_langfuse_callback(
        langfuse_config,
        trace_name=application_name,
        session_id=thread_id,
        user_id=langfuse_user_id,
        metadata=langfuse_metadata,
        environment=langfuse_environment,
    )


# =============================================================================
# Checkpoint Resume
# =============================================================================

def configure_checkpoint_resume(
    agent_executor,
    thread_id: str,
    checkpoint_id: Optional[str],
    invoke_input: Dict[str, Any],
    invoke_config: Dict[str, Any],
    user_input: Optional[str] = None,
    user_declined_mcp_servers: Optional[List[Dict[str, Any]]] = None,
    mcp_tokens: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Configure checkpoint resume if needed.

    Args:
        agent_executor: The agent executor with state history
        thread_id: Thread identifier
        checkpoint_id: Optional specific checkpoint to resume from
        invoke_input: Current invoke input dict
        invoke_config: Current invoke config dict
        user_input: Original user question (used to build continuation message)
        user_declined_mcp_servers: List of MCP servers the user skipped auth for.
            When non-empty AND no new tokens were provided, a skip continuation
            message is generated so the LLM knows auth was declined.
        mcp_tokens: Dict of newly-authorized MCP server tokens for this resume.
            When non-empty it means the user just authorized at least one server,
            so the "authorization completed" message is used even if
            user_declined_mcp_servers also contains other servers that were skipped.

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
            declined = user_declined_mcp_servers or []
            authorized = bool(mcp_tokens)  # True if the user just authorized at least one server

            if declined and not authorized:
                # User clicked Skip for all servers — tell the LLM auth was declined.
                # This prevents "auth completed" confusion and allows the LLM to
                # acknowledge the skip and answer without MCP tools.
                skip_details = []
                for s in declined:
                    if not isinstance(s, dict):
                        continue
                    reason = (s.get("skip_reason") or s.get("denial_reason") or "").strip()
                    server = (s.get("server_url") or "").strip()
                    if reason and server:
                        skip_details.append(f"{server}: {reason}")
                    elif reason:
                        skip_details.append(reason)
                if skip_details:
                    reason_text = "; ".join(skip_details)
                    continuation_message = (
                        f"The user declined MCP authorization for this session. "
                        f"Reason: {reason_text}. "
                        f"Please proceed with the original request without using the unavailable MCP tools, "
                        f"or explain that you cannot complete it without them: {user_input}"
                        if user_input else
                        f"The user declined MCP authorization for this session. "
                        f"Reason: {reason_text}. "
                        f"Please proceed with the original request without using the unavailable MCP tools, "
                        f"or explain that you cannot complete it without them."
                    )
                else:
                    continuation_message = (
                        f"The user chose to skip MCP authorization for this session. "
                        f"Please proceed with the original request without using the unavailable MCP tools, "
                        f"or explain that you cannot complete it without them: {user_input}"
                        if user_input else
                        "The user chose to skip MCP authorization for this session. "
                        "Please proceed with the original request without using the unavailable MCP tools, "
                        "or explain that you cannot complete it without them."
                    )
            else:
                # User completed authorization for at least one server — proceed with the tools.
                # Repeating the original user question avoids the LLM treating
                # the stale "auth initiated" checkpoint messages as still pending.
                # If some other servers were also declined in this session, mention it
                # so the LLM knows those specific tools are still unavailable.
                declined_note = ""
                if declined:
                    skip_details = []
                    for s in declined:
                        if not isinstance(s, dict):
                            continue
                        server = (s.get("server_url") or "").strip()
                        if server:
                            skip_details.append(server)
                    if skip_details:
                        declined_note = (
                            f" Note: authorization for the following servers was declined and those tools "
                            f"remain unavailable: {', '.join(skip_details)}."
                        )
                if user_input:
                    continuation_message = (
                        f"The required authorization has been completed.{declined_note} "
                        f"Please proceed with the original request: {user_input}"
                    )
                else:
                    continuation_message = (
                        f"The required authorization has been completed.{declined_note} "
                        "Please proceed with the original request using the newly available tools."
                    )
            invoke_input = {'input': continuation_message}
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
    context_info: Optional[Dict[str, Any]] = None,
    invoked_skills: Optional[List[Dict[str, Any]]] = None
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

    # Parked fan-out child (#4993 Track 2): this standalone indexer_agent streams
    # onto the PARENT's message. Its live events (chunks, tool chips) flow so the
    # UI animates the child's accordion, and its HITL pause MUST surface its card
    # — but its TERMINAL/final-answer events are suppressed: the real answer is
    # emitted by the reconciled parent (which reads each child's checkpoint), so
    # a child emitting agent_response/full_message/pipeline_finish here would
    # prematurely end the parent message and overwrite it with one child's output.
    is_fanout_child = bool(
        isinstance(task_meta, dict)
        and task_meta.get('child_thread_id')
        and task_meta.get('parent_thread_id')
    )

    # Emit a dedicated event when execution paused at a HITL node.
    hitl_interrupt = response.get('hitl_interrupt')
    # Parallel sub-agent fan-out (#4993): the SDK aggregates one entry per
    # paused child into hitl_interrupts (plural). Forward the full list so the
    # UI can render N stacked approval cards; fall back to the single interrupt
    # for the ordinary one-pause case.
    hitl_interrupts = response.get('hitl_interrupts')
    if not hitl_interrupts and hitl_interrupt:
        hitl_interrupts = [hitl_interrupt]
    # Bidirectional fallback: a parallel fan-out aggregate may arrive as the
    # plural list only. Derive the singular from the first entry so BOTH the
    # interrupt-emission guard below and the skip-normal-message guard further
    # down (which key off the singular) fire correctly on a plural-only pause —
    # otherwise a paused child would leak a premature assistant message (#4993).
    if not hitl_interrupt and hitl_interrupts:
        hitl_interrupt = hitl_interrupts[0]
    if hitl_interrupt:
        node_interface.emit(
            type=EventTypes.agent_hitl_interrupt,
            content=hitl_interrupt.get('message', 'Awaiting human review...'),
            response_metadata={
                'thread_id': thread_id_response,
                'message': hitl_interrupt.get('message', 'Awaiting human review...'),
                'hitl_interrupt': hitl_interrupt,
                'hitl_interrupts': hitl_interrupts,
                'node_name': hitl_interrupt.get('node_name'),
                'available_actions': hitl_interrupt.get('available_actions', []),
                'routes': hitl_interrupt.get('routes', {}),
                'edit_state_key': hitl_interrupt.get('edit_state_key'),
            }
        )

    # Emit pipeline_finish if execution completed. Suppressed for a fan-out child
    # (it must not signal END on the parent's stream — only the reconciled parent
    # does that once every child has settled).
    if response.get('execution_finished') and not is_fanout_child:
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
    # Also skip for a fan-out child: its final answer must NOT land on the
    # parent's message — the reconciled parent emits the real answer after
    # reading each child's checkpoint (#4993 Track 2).
    if not hitl_interrupt and not is_fanout_child:
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
            'invoked_skills': [
                {'skill_id': s.get('skill_id'), 'name': s.get('name')}
                for s in (invoked_skills or [])
                if isinstance(s, dict) and s.get('name')
            ],
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
    context_info: Optional[Dict[str, Any]] = None,
    return_chat_history: bool = False,
    hitl_interrupt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build successful execution result dict.

    Args:
        chat_history: Updated chat history
        elitea_callback: Callback with thinking steps and tool calls
        tokens_in: Input token count
        tokens_out: Output token count
        context_info: Optional context info with message/token counts and summarization details
        return_chat_history: Include chat_history in result (only needed for blocking API callers
            that call join_task(); socket-based flows never read this field so it defaults to False
            to avoid serializing potentially large base64 payloads into the task store).
        hitl_interrupt: Set when the run PAUSED at a HITL node (not a completed answer).
            A parked fan-out child (#4993 Track 2) fires the arbiter ``stopped`` event on
            a HITL pause just like a completion, so the reconcile gate
            (parallel_dispatch_on_child_terminal) MUST be able to tell the two apart:
            a paused child is NOT terminal and must not advance the gate. Surfacing the
            interrupt in the task result is how the gate detects "still open".

    Returns:
        Result dict
    """
    sanitized_steps = json.loads(json.dumps(elitea_callback.thinking_steps, default=str))
    result = {
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
    if return_chat_history:
        result['chat_history'] = chat_history
    if context_info:
        result['context_info'] = context_info
    if hitl_interrupt:
        result['hitl_interrupt'] = hitl_interrupt
    return result


# =============================================================================
# Parallel sub-agent dispatch (Track 2, issue #4993)
# =============================================================================

# A forked agent process cannot safely call start_task (shared-Redis-socket
# hazard), so the SDK's child_dispatcher is a pure presence-sentinel: the SDK
# only checks `is not None` to choose park-by-returning (Track 2) over the
# in-process asyncio.gather fan-out (Track 1). pylon_main owns the real child
# launch. We hand the SDK a module-level marker — never called, only injected.
_PARALLEL_DISPATCH_SENTINEL = object()


def get_child_dispatcher(descriptor_config: Dict[str, Any]) -> Optional[Any]:
    """Return a non-None sentinel to enable Track 2 park-mode, else None.

    Gated on the indexer config flag ``parallel_subagent_dispatch`` so Track 1
    (in-process gather) stays the default until an operator opts in. The SDK
    only presence-checks the value, so the marker object's identity is
    irrelevant — its non-None-ness is the whole switch.
    """
    if descriptor_config.get('parallel_subagent_dispatch', False):
        return _PARALLEL_DISPATCH_SENTINEL
    return None


def detect_parked_dispatch(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detect the SDK park-by-returning sentinel in an agent response.

    When the parent LLMNode fans out to 2+ Application sub-agents with a
    child_dispatcher present, the SDK writes child specs to the parallel_tasks
    channel and returns a parked shape instead of running them. Returns the
    dispatch payload (specs + parent thread_id) for pylon_main to read via
    get_task_result, or None for an ordinary run.
    """
    if not isinstance(response, dict) or not response.get('parallel_parked'):
        return None
    return {
        'parallel_parked': True,
        'parallel_dispatch': response.get('parallel_dispatch') or [],
        'thread_id': response.get('thread_id'),
    }


def build_parked_result(
    parked: Dict[str, Any],
    stream_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the task-result payload for a parked parent.

    Carries the child dispatch specs (plus the parent's own reconcile re-invoke
    payload) out to pylon_main (read via get_task_result) without the normal
    agent_response/full_message events. The parent task then goes terminal
    (``stopped``); pylon_main's stopped-handler launches one durable
    indexer_agent per child and, once all settle, re-invokes the parent with
    parallel_reconcile using the carried reconcile_payload.

    ``stream_id``/``message_id`` are the parent's own — carried so pylon_main
    re-invokes the reconcile on the SAME stream/message and the final answer
    lands on the original user message.
    """
    return {
        'error': None,
        'parallel_parked': True,
        'parallel_dispatch': parked.get('parallel_dispatch') or [],
        'reconcile_payload': parked.get('reconcile_payload'),
        'parent_stream_id': stream_id,
        'parent_message_id': message_id,
        'thread_id': parked.get('thread_id'),
        'thinking_steps': [],
        'tool_calls': [],
        'tool_calls_dict': {},
        'chat_history_tokens_input': 0,
        'llm_response_tokens_output': 0,
    }


def build_child_launch_payloads(
    parent_kwargs: Dict[str, Any],
    parked_dispatch: list,
) -> list:
    """Turn parked dispatch specs into self-contained child indexer_agent payloads.

    A forked agent cannot call start_task and pylon_main cannot safely mint a
    fresh predict token outside request scope, so the launch payload is built
    HERE — the parent indexer task holds a valid token/base_url/headers in its
    own ``llm.kwargs`` (same project + user as the child). Each child is a saved
    Application: its identity (id/version_id) and already-fetched
    ``version_details`` (carrying its own llm_settings + tools) ride in the SDK
    spec, so the child re-resolves its model/tools without any extra round-trip.

    Inherited from the parent: transport credentials (base_url, api_key,
    api_extra_headers, project_id), conversation_id, mcp_tokens,
    ignored_mcp_servers, persona, supports_vision, debug. Child-specific: model
    (the sub-agent's own, falling back to the parent's), thread_id, user_input
    (the tool-call ``task``), variables, and a fresh/empty chat_history.

    pylon_main reads the returned specs from the parked task result and calls
    start_task("indexer_agent", kwargs=spec["child_payload"], ...) verbatim.
    """
    parent_llm_kwargs = (parent_kwargs.get('llm') or {}).get('kwargs') or {}
    enriched = []
    for spec in parked_dispatch or []:
        version_details = spec.get('version_details') or {}
        llm_settings = version_details.get('llm_settings') or {}

        # Child uses its OWN configured model; embedded sub-agents with null
        # llm_settings inherit the parent's model. Token/base_url/headers are
        # always the parent's (valid for the same project + user).
        child_llm_kwargs = dict(parent_llm_kwargs)
        child_llm_kwargs['model'] = llm_settings.get('model_name') or parent_llm_kwargs.get('model')
        child_llm_kwargs['openai_compatible'] = llm_settings.get(
            'openai_compatible', parent_llm_kwargs.get('openai_compatible', False)
        )

        # Variables: defaults overlaid with any non-task inputs from the call.
        task_input = spec.get('input') if isinstance(spec.get('input'), dict) else {}
        variables = dict(spec.get('variable_defaults') or {})
        for key, value in task_input.items():
            if key not in ('task', 'chat_history') and value is not None:
                variables[key] = value
        payload_variables = (
            {k: {'name': k, 'value': v} for k, v in variables.items()} or None
        )

        child_payload = {
            'llm': {'kwargs': child_llm_kwargs},
            'chat_history': [],
            'user_input': task_input.get('task') or '',
            'thread_id': spec.get('child_thread_id'),
            'conversation_id': parent_kwargs.get('conversation_id'),
            'mcp_tokens': parent_kwargs.get('mcp_tokens') or {},
            'ignored_mcp_servers': parent_kwargs.get('ignored_mcp_servers') or [],
            'context_settings': {},
            'supports_vision': parent_kwargs.get('supports_vision', True),
            'debug': parent_kwargs.get('debug', False),
            'persona': parent_kwargs.get('persona', 'generic'),
            'should_continue': False,
            'hitl_resume': False,
            'is_regenerate': False,
            'meta': version_details.get('meta', {}),
            'application': {
                'id': spec.get('application_id'),
                'name': spec.get('name'),
                'version_id': spec.get('application_version_id'),
                'variables': payload_variables,
                'version_details': version_details,
            },
        }

        # Keep the spec light for the task result: version_details now lives
        # inside child_payload, no need to carry it twice across the RPC.
        new_spec = {k: v for k, v in spec.items() if k != 'version_details'}
        new_spec['child_payload'] = child_payload
        enriched.append(new_spec)
    return enriched


def build_parent_reconcile_payload(parent_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build the self-contained payload to re-invoke a parked parent for reconcile.

    Once every child task is terminal, pylon_main re-invokes the parked parent
    as a FRESH ``indexer_agent`` task (not a checkpoint resume) so the SDK can
    read each child's own checkpoint, append one ToolMessage per child, and
    finish the tool loop. The re-invoke must land on the SAME thread_id with the
    SAME agent identity and a valid token. The parent's own ``llm.kwargs`` token
    is a long-lived user/system API token (no near-term expiry), so it survives
    the human-think-time between park and reconcile — we carry the parent's
    launch fields verbatim and only flip control flags.

    Built HERE (not in pylon_main) for the same reason as the child payloads:
    pylon_main cannot safely mint a predict token outside request scope, but the
    parent indexer task already holds a valid one. pylon_main stashes this
    payload (transiently, keyed by parent_thread_id+epoch) and replays it with
    ``parallel_reconcile`` stamped in once the reconcile gate opens.

    Only JSON-safe launch fields are carried; transient resume/HITL flags are
    forced off so the re-invoke is a clean reconcile, not a continuation.
    """
    carry_keys = (
        'llm', 'application', 'thread_id', 'conversation_id', 'user_input',
        'chat_history', 'tools', 'internal_tools', 'mcp_tokens',
        'ignored_mcp_servers', 'persona', 'supports_vision', 'debug',
        'debug_mode', 'steps_limit', 'meta', 'context_settings',
        'exception_handling_enabled', 'auto_approve_sensitive_actions',
        'return_chat_history',
    )
    payload = {k: parent_kwargs[k] for k in carry_keys if k in parent_kwargs}
    # context_settings is mutated in place at task entry to attach live
    # summarization-callback closures (create_summarization_callbacks.<locals>.*),
    # which are NOT picklable — and the parked result is pickled to the arbiter
    # tasknode .bin. Drop the callbacks: the reconcile re-invoke rebuilds its own
    # fresh from node_interface, exactly as the original invoke does, so this is a
    # clean shallow copy, not a loss of configuration.
    if isinstance(payload.get('context_settings'), dict):
        payload['context_settings'] = {
            k: v for k, v in payload['context_settings'].items() if k != 'callbacks'
        }
    # Force a clean reconcile invocation: not a HITL/continue/regenerate resume.
    payload['should_continue'] = False
    payload['hitl_resume'] = False
    payload['is_regenerate'] = False
    return payload


def apply_parallel_reconcile(invoke_input: Dict[str, Any], kwargs: Dict[str, Any]) -> Optional[Any]:
    """Thread the reconcile epoch from task kwargs into the SDK invoke input.

    pylon_main re-invokes the parked parent with ``parallel_reconcile=<epoch>``
    once every child task is terminal. The SDK runnable consumes
    ``invoke_input['parallel_reconcile']`` to switch into reconcile-assembly
    (read each child's own checkpoint, append one ToolMessage per child, resume
    the agent node). Returns the epoch when present, else None.
    """
    epoch = kwargs.get('parallel_reconcile')
    if epoch:
        invoke_input['parallel_reconcile'] = epoch
    return epoch


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
