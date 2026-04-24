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

"""Common utilities and classes for agent methods"""

import json
import re
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import requests
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired
from langchain_core.callbacks import BaseCallbackHandler  # pylint: disable=E0401
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, LLMResult
from pydantic import BaseModel
from pylon.core.tools import log  # pylint: disable=E0611,E0401

from ..utils.exceptions import InternalSDKError
from ..utils.funcs import (
    extract_finish_reason,
    extract_token_usage,
    num_tokens_from_messages,
)
from ..utils.node_interface import (
    ELITEA_SDK_CUSTOM_EVENTS_MAPPER,
    EventTypes,
    NodeEvent,
    NodeEventInterface,
)

# Event node names
EVENTNODE_EVENT_NAME = "application_stream_response"
EVENTNODE_FULL_RESPONSE_NAME = "application_full_response"
EVENTNODE_PARTIAL_RESPONSE_NAME = "application_partial_response"

# Secret name for project PostgreSQL connection string
PGVECTOR_PROJECT_CONNSTR_SECRET = "pgvector_project_connstr"


@contextmanager
def temp_elitea_client(
    client_args: dict, api_token: str = None, api_extra_headers: dict = None
):
    """
    Context manager for creating temporary EliteAClient instances to fetch project secrets before fork.

    Args:
        client_args: Client configuration dictionary containing deployment/base_url and project_id
        api_token: API token for authentication
        api_extra_headers: Additional headers for API requests

    Yields:
        EliteAClient: Configured temporary client instance

    Example:
        with temp_elitea_client(client_args, api_token, api_extra_headers) as temp_client:
            pgvector_connstr = _fetch_pgvector_connstr_with_retry(temp_client)
    """
    from ..utils.funcs import dev_reload_sdk

    dev_reload_sdk("elitea_sdk.runtime.clients")
    from elitea_sdk.runtime.clients.client import (
        EliteAClient,  # pylint: disable=E0401,C0415
    )

    temp_client = EliteAClient(
        base_url=client_args.get("deployment", client_args.get("base_url", None)),
        project_id=client_args.get("project_id"),
        auth_token=api_token,
        api_extra_headers=api_extra_headers or {},
    )

    try:
        yield temp_client
    finally:
        # Clean up temp client to avoid pickling issues
        del temp_client


def _fetch_pgvector_connstr_with_retry(
    client, max_retries: int = 3, base_delay: float = 0.5
) -> Optional[str]:
    """
    Fetch pgvector_project_connstr secret from the project vault with retry logic.

    Args:
        client: EliteAClient instance with unsecret capability
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds between retries, uses exponential backoff (default: 0.5)

    Returns:
        Connection string if successful, None if secret doesn't exist or all retries failed
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            conn_str = client.unsecret(PGVECTOR_PROJECT_CONNSTR_SECRET)
            if conn_str:
                log.info(
                    f"Successfully fetched {PGVECTOR_PROJECT_CONNSTR_SECRET} secret"
                )
                return conn_str
            else:
                log.warning(
                    f"{PGVECTOR_PROJECT_CONNSTR_SECRET} secret not found or empty in project vault"
                )
                return None
        except Exception as e:  # pylint: disable=W0718
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff
                log.warning(
                    f"Attempt {attempt + 1}/{max_retries} to fetch {PGVECTOR_PROJECT_CONNSTR_SECRET} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                log.warning(
                    f"All {max_retries} attempts to fetch {PGVECTOR_PROJECT_CONNSTR_SECRET} failed. "
                    f"Last error: {last_error}. Planning toolkit will use filesystem storage."
                )
    return None


def _unsecret_vault_references(data: dict, client) -> dict:
    """
    Unsecret any vault references ({{secret.xxx}}) in the data dict.

    Args:
        data: Dict that may contain vault reference strings
        client: EliteAClient instance with unsecret method

    Returns:
        Dict with vault references replaced by actual values
    """
    import re

    secret_pattern = re.compile(r"^\{\{secret\.([A-Za-z0-9_]+)\}\}$")

    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            match = secret_pattern.match(value)
            if match:
                secret_name = match.group(1)
                try:
                    unsecreted = client.unsecret(secret_name)
                    result[key] = unsecreted if unsecreted else value
                except Exception as e:
                    log.warning(f"Failed to unsecret {key}: {e}")
                    result[key] = value
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def unsecret_mcp_tools(tools: list, client) -> list:
    """
    Resolve {{secret.xxx}} placeholders in MCP toolkit settings (headers, url, etc.)
    at the indexer level before passing to the SDK.

    Args:
        tools: List of tool configuration dicts from the task payload
        client: EliteAClient instance with unsecret capability

    Returns:
        New list with {{secret.xxx}} patterns replaced by actual secret values
    """
    secret_pattern = re.compile(r"\{\{secret\.([A-Za-z0-9_]+)\}\}")

    def _resolve(value):
        if isinstance(value, str):
            def _replacer(match):
                secret_name = match.group(1)
                try:
                    resolved = client.unsecret(secret_name)
                    return resolved if resolved is not None else match.group(0)
                except Exception as e:  # pylint: disable=W0718
                    log.warning(f"[MCP] Failed to unsecret '{secret_name}': {e}")
                    return match.group(0)
            return secret_pattern.sub(_replacer, value)
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        return value

    result = []
    for tool in tools:
        if (
            isinstance(tool, dict)
            and isinstance(tool.get("type"), str)
            and (tool["type"] == "mcp" or tool["type"].startswith("mcp_"))
            and "settings" in tool
        ):
            tool = {**tool, "settings": _resolve(tool["settings"])}
        result.append(tool)
    return result


def fetch_langfuse_config(client) -> Optional[Dict[str, Any]]:
    """
    Fetch langfuse configuration from project credentials.

    Args:
        client: EliteAClient instance

    Returns:
        Dict with langfuse config (base_url, public_key, secret_key) or None
    """
    try:
        # Fetch configurations with type=langfuse and section=credentials
        url = f"{client.base_url}/api/v2/configurations/configurations/{client.project_id}?type=langfuse&section=credentials"

        response = requests.get(url, headers=client.headers, verify=False, timeout=10)
        response.raise_for_status()

        result = response.json()

        # Extract items from the response
        items = result.get("items", [])
        if not items:
            log.debug("No langfuse configuration found in project")
            return None

        # Return the data from the first langfuse configuration found
        config = items[0]
        data = config.get("data", {})
        if data:
            # Unsecret any vault references before returning
            data = _unsecret_vault_references(data, client)
            log.debug("Langfuse configuration found and unsecreted")
            return data

        log.debug("Langfuse configuration data is empty")
        return None
    except Exception as e:
        log.warning(f"Failed to fetch langfuse config: {e}")
        return None


def execution_error(
    node_interface: NodeEventInterface,
    user_input: str,
    chat_history: list,
    error_message: str,
    thread_id: str,
    message_id: str,
    tasknode_task_meta: dict,
    human_readable: str = None,
    execution_start_time: Optional[datetime] = None,
) -> dict:
    """
    Handle execution errors by emitting appropriate events and returning error response.

    Args:
        node_interface: The node event interface for emitting events
        user_input: The original user input that caused the error
        chat_history: Current chat history
        error_message: Technical error message for logging
        thread_id: Thread ID for the conversation
        message_id: Message ID
        tasknode_task_meta: Task metadata containing project info
        human_readable: Human-readable error message (optional)
        execution_start_time: Execution start timestamp for duration calculation (optional)

    Returns:
        Dict containing chat_history and error information
    """
    exception_uid = str(uuid4())
    error = str(traceback.format_exc())

    execution_time_seconds = None
    if execution_start_time:
        execution_end_time = datetime.now(tz=timezone.utc)
        execution_time_seconds = (
            execution_end_time - execution_start_time
        ).total_seconds()

    node_interface.emit(
        type=EventTypes.agent_tool_start,
        response_metadata={
            "tool_name": "Agent Exception Stacktrace",
            "tool_run_id": exception_uid,
            "tool_meta": user_input,
            "tool_inputs": "",
        },
    )
    node_interface.emit(
        type=EventTypes.agent_tool_end,
        response_metadata={
            "tool_name": "Agent Exception Stacktrace",
            "tool_run_id": exception_uid,
            "finish_reason": "stop",
        },
        content=error,
    )

    node_interface.emit(type=EventTypes.agent_exception, content=error_message)

    # build response metadata with execution_time_seconds if available
    response_metadata = {
        "project_id": tasknode_task_meta.get("project_id"),
        "chat_project_id": tasknode_task_meta.get("chat_project_id"),
        # chat conversations need it
        "thread_id": thread_id,
        "is_error": True,
        "error": error,
    }

    # Add execution_time_seconds for accurate duration calculation (Issue #3134)
    if execution_time_seconds is not None:
        response_metadata["execution_time_seconds"] = execution_time_seconds

    msg_event_node = NodeEvent(
        type=EventTypes.full_message,
        stream_id=node_interface.stream_id,
        message_id=message_id,
        response_metadata=response_metadata,
        content=human_readable or error_message,
        **node_interface.payload_additional_kwargs,
    ).model_dump_json()
    msg_event_node = json.loads(msg_event_node)
    node_interface.event_node.emit(EVENTNODE_FULL_RESPONSE_NAME, msg_event_node)
    return {"chat_history": chat_history, "error": error}


class ToolCallPayload(BaseModel):
    """Payload model for tool call information"""

    tool_name: str
    tool_run_id: str
    run_id: str
    tool_meta: Optional[Any] = None
    tool_inputs: Optional[Any] = None
    metadata: Optional[dict] = None
    agent_type: Optional[str] = None
    content: Optional[str] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    tool_output: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_finish: Optional[str] = None


class EliteACallback(BaseCallbackHandler):
    """EliteA agent callback handler"""

    def __init__(
        self,
        node_interface: NodeEventInterface,
        debug: bool = False,
        thread_id: str = None,
        message_id: str = None,
        project_id: int = None,
        chat_project_id: int = None,
        toolkit_metadata: dict = None,
    ):
        log.debug(f"EliteACallback init debug={debug}")
        self.node_interface = node_interface
        self.event_node = node_interface.event_node
        self.stream_id = node_interface.stream_id
        self.debug = debug
        self.thread_id: str = thread_id
        self.thinking_steps: list[dict] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.pending_llm_requests = defaultdict(
            lambda: {"tokens_in": 0, "tokens_out": 0}
        )
        # Track last sent content/thinking per run_id to send only deltas (some providers send cumulative)
        self._last_sent_content: Dict[str, str] = {}
        self._last_sent_thinking: Dict[str, str] = {}
        self.current_model_name = "gpt-4"
        self.tool_calls: Dict[str, ToolCallPayload] = {}  # tool_run_id -> payload
        self.llm_start_timestamp: str | None = None
        self.message_id: str = message_id
        self.project_id: int = project_id
        self.chat_project_id: int = chat_project_id
        self.llm_error: Optional[InternalSDKError] = None
        self.toolkit_metadata: dict = toolkit_metadata or {}
        # Extract and cache toolkit_name and toolkit_type from toolkit_metadata for injection
        self.cached_toolkit_name = None
        self.cached_toolkit_type = None
        if self.toolkit_metadata:
            self.cached_toolkit_name = toolkit_metadata.get(
                "toolkit_name"
            ) or toolkit_metadata.get("name")
            self.cached_toolkit_type = toolkit_metadata.get(
                "toolkit_type"
            ) or toolkit_metadata.get("type")
            log.debug(
                f"EliteACallback cached_toolkit_name: {self.cached_toolkit_name}, cached_toolkit_type: {self.cached_toolkit_type}"
            )
        super().__init__()

    #
    # Chain
    #

    def on_chain_start(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_chain_start(%s, %s)", args, kwargs)

    def on_chain_end(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_chain_end(%s, %s)", args, kwargs)

    def on_chain_error(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_chain_error(%s, %s)", args, kwargs)
        #
        # exception = args[0]
        # FIXME: should we emit an error here too?

    #
    # Tool
    #

    def on_tool_start(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_tool_start(%s, %s)", args, kwargs)
        tool_name = args[0].get("name")
        tool_metadata_from_kwargs = kwargs.get("metadata", {})
        if (
            isinstance(tool_metadata_from_kwargs, dict)
            and "original_name" in tool_metadata_from_kwargs
        ):
            tool_name = tool_metadata_from_kwargs["original_name"]
        now = datetime.now(tz=timezone.utc).isoformat()

        # Extract tool metadata (includes MCP session info if available)
        tool_metadata = kwargs.get("metadata", {})

        # Extract metadata from tool if available (from BaseAction.metadata)
        # Try multiple sources for metadata with toolkit_name
        tool_meta = args[0].copy()

        # Source 1: kwargs['serialized']['metadata'] - LangChain's full tool serialization
        if "serialized" in kwargs and "metadata" in kwargs["serialized"]:
            tool_meta["metadata"] = kwargs["serialized"]["metadata"]
            # Also merge into tool_metadata for backward compatibility.
            # Preserve original_name from the execution context (kwargs["metadata"]) because it
            # carries the PARENT agent's name when a tool runs inside a nested Application.
            # tool.metadata["original_name"] = the tool's own name (set at construction time),
            # while kwargs["metadata"]["original_name"] = the outer/parent agent's name (injected
            # via Application._run nested_config propagation). Using dict.update() would overwrite
            # the parent name with the tool's own name, breaking parent_agent_name detection.
            if isinstance(kwargs["serialized"]["metadata"], dict):
                _context_original_name = tool_metadata.get("original_name")
                tool_metadata.update(kwargs["serialized"]["metadata"])
                if _context_original_name is not None:
                    tool_metadata["original_name"] = _context_original_name

        # Source 2: Check if metadata is directly in args[0] (some LangChain versions)
        elif "metadata" in args[0]:
            tool_meta["metadata"] = args[0]["metadata"]
            if isinstance(args[0]["metadata"], dict):
                _context_original_name = tool_metadata.get("original_name")
                tool_metadata.update(args[0]["metadata"])
                if _context_original_name is not None:
                    tool_metadata["original_name"] = _context_original_name

        # Copy metadata fields from tool_metadata to tool_meta["metadata"]
        # This handles the case where LangGraph puts these in execution metadata
        metadata_fields = ["toolkit_name", "toolkit_type", "agent_type", "display_name"]
        for field in metadata_fields:
            if field in tool_metadata:
                if "metadata" not in tool_meta:
                    tool_meta["metadata"] = {}
                if field not in tool_meta["metadata"]:
                    tool_meta["metadata"][field] = tool_metadata[field]

        own_display_name = tool_meta.get("metadata", {}).get("display_name")

        # Primary: dedicated parent_agent_name key injected by Application._run() into
        # nested_config['metadata']. LangGraph propagates it unchanged (via merge_configs)
        # to every per-step config inside the nested graph. No tool's own serialized metadata
        # carries this key, so tool_metadata.update() above never overwrites it — making it
        # a reliable, collision-free channel for identifying the parent Application.
        _parent_agent_name = tool_metadata.get("parent_agent_name")
        if _parent_agent_name and _parent_agent_name != own_display_name:
            if "metadata" not in tool_meta:
                tool_meta["metadata"] = {}
            tool_meta["metadata"]["parent_agent_name"] = _parent_agent_name
        else:
            # Secondary fallback: original_name in kwargs["metadata"] carries the parent
            # Application's name when Application._run()'s nested_config propagation is intact
            # and our update()-preservation fix kept it from being overwritten.
            context_original_name = tool_metadata.get("original_name")
            if context_original_name and context_original_name != own_display_name:
                if "metadata" not in tool_meta:
                    tool_meta["metadata"] = {}
                tool_meta["metadata"]["parent_agent_name"] = context_original_name

        # Extract icon_meta from tool_metadata (kwargs['metadata']) and add directly to tool_meta
        # This is where LangGraph passes execution context metadata including icon_meta
        if "icon_meta" in tool_metadata:
            tool_meta["icon_meta"] = tool_metadata["icon_meta"]

        if not tool_metadata.get("toolkit_name") and self.cached_toolkit_name:
            log.debug(
                f"[METADATA] Adding cached toolkit_name to tool_metadata: {self.cached_toolkit_name}"
            )
            tool_metadata["toolkit_name"] = self.cached_toolkit_name

        if not tool_metadata.get("toolkit_type") and self.cached_toolkit_type:
            log.debug(
                f"[METADATA] Adding cached toolkit_type to tool_metadata: {self.cached_toolkit_type}"
            )
            tool_metadata["toolkit_type"] = self.cached_toolkit_type

        # For MCP tools, construct metadata from serialized fields if not already present
        if not tool_metadata.get("mcp_session_id"):
            session_id = args[0].get("session_id") if args else None
            server_url = args[0].get("server_url") if args else None
            if session_id and server_url:
                tool_metadata["mcp_session_id"] = session_id
                tool_metadata["mcp_server_url"] = server_url
                log.debug(
                    f"[MCP] Constructed metadata from tool fields: session={session_id}, url={server_url}"
                )

        # Build payload with optional agent_type field
        payload = {
            "tool_name": tool_name,
            "tool_run_id": str(run_id),
            "tool_meta": tool_meta,
            "tool_inputs": kwargs.get("inputs"),
            "metadata": tool_metadata,  # Include session_id and other metadata
            "timestamp_start": now,
            "agent_type": tool_metadata.get("agent_type"),  # Optional field for nested agents/pipelines
        }

        tool_call = ToolCallPayload(**payload, run_id=str(run_id))
        self.tool_calls[str(run_id)] = tool_call

        # Include agent_type in emit only if present
        include_fields = {
            "tool_name",
            "tool_run_id",
            "tool_meta",
            "tool_inputs",
            "metadata",
            "timestamp_start",
            "agent_type",
        }

        self.node_interface.emit(
            type=EventTypes.agent_tool_start,
            response_metadata=tool_call.model_dump(include=include_fields),
        )

    def on_tool_end(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_tool_end(%s, %s)", args, kwargs)
        tool_run_id = str(run_id)
        # Use JSON serialization for non-string types to preserve proper formatting
        raw_output = args[0]
        # LangChain wraps tool results in a ToolMessage (BaseMessage) when a
        # tool_call_id is provided (e.g. via LangGraph's ToolNode for published
        # agents with toolkits). Extract the actual content, otherwise the
        # non-serializable ToolMessage falls back to its pydantic __str__
        # ("content='...' name='...' tool_call_id='...'") and is surfaced to
        # the end user / LLM context instead of the plain tool result.
        if isinstance(raw_output, BaseMessage):
            raw_output = raw_output.content
        tool_output = (
            raw_output
            if isinstance(raw_output, str)
            else json.dumps(
                raw_output, 
                ensure_ascii=False, 
                default=lambda o: str(o)
            )
        )
        now = datetime.now(tz=timezone.utc).isoformat()
        if tool_run_id in self.tool_calls:
            self.tool_calls[tool_run_id].finish_reason = "stop"
            self.tool_calls[tool_run_id].content = tool_output
            self.tool_calls[tool_run_id].tool_output = tool_output
            self.tool_calls[tool_run_id].timestamp_finish = now
            tool_call = self.tool_calls[tool_run_id]
        else:
            tool_call = ToolCallPayload(
                tool_name=kwargs.get("name"),
                tool_run_id=tool_run_id,
                tool_output=tool_output,
                finish_reason="stop",
                timestamp_start=now,
                timestamp_finish=now,
                run_id=str(run_id),
            )
            self.tool_calls[tool_run_id] = tool_call

        # Include agent_type field (will be None if not applicable)
        include_fields = {
            "tool_name",
            "tool_run_id",
            "tool_meta",
            "finish_reason",
            "tool_output",
            "timestamp_start",
            "timestamp_finish",
            "metadata",
            "agent_type",
        }

        self.node_interface.emit(
            type=EventTypes.agent_tool_end,
            response_metadata=tool_call.model_dump(include=include_fields),
            content=tool_output,
        )

        # necessary for partial message saving
        msg_event_node = NodeEvent(
            type=EventTypes.partial_message,
            stream_id=self.node_interface.stream_id,
            message_id=self.message_id,
            response_metadata={
                "project_id": self.project_id,
                "chat_project_id": self.chat_project_id,
                "thread_id": self.thread_id,
                "application_details": kwargs.get("application", {}),
                "thinking_steps": self.thinking_steps,
                "tool_calls": {
                    run_id: tool_call.model_dump()
                    for run_id, tool_call in self.tool_calls.items()
                },
                "llm_start_timestamp": self.llm_start_timestamp,
                "additional_response_meta": {},
            },
            content=None,
            **self.node_interface.payload_additional_kwargs,
        ).model_dump_json()
        msg_event_node = json.loads(msg_event_node)
        self.node_interface.event_node.emit(
            EVENTNODE_PARTIAL_RESPONSE_NAME, msg_event_node
        )

    def on_tool_error(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_tool_error(%s, %s)", args, kwargs)
        tool_run_id = str(run_id)
        tool_exception = args[0]
        now = datetime.now(tz=timezone.utc).isoformat()

        if isinstance(tool_exception, McpAuthorizationRequired):
            error_str = (
                tool_exception.args[0]
                if tool_exception.args
                else "Authorization required"
            )
            if tool_run_id in self.tool_calls:
                self.tool_calls[tool_run_id].finish_reason = "action_required"
                self.tool_calls[tool_run_id].error = error_str
                self.tool_calls[tool_run_id].tool_output = None
                self.tool_calls[tool_run_id].timestamp_finish = now
                tool_call = self.tool_calls[tool_run_id]
            else:
                tool_call = ToolCallPayload(
                    tool_name=kwargs.get("name"),
                    tool_run_id=tool_run_id,
                    error=error_str,
                    run_id=str(run_id),
                    finish_reason="action_required",
                    timestamp_start=now,
                    timestamp_finish=now,
                )
                self.tool_calls[tool_run_id] = tool_call

            auth_payload = tool_exception.to_dict()
            auth_payload.update(
                {
                    "tool_name": tool_call.tool_name,
                    "tool_run_id": tool_run_id,
                    "chat_project_id": self.chat_project_id,  # Include for DB update
                }
            )

            self.node_interface.emit(
                type=EventTypes.mcp_authorization_required,
                response_metadata=auth_payload,
                content=error_str,
            )
            return

        error_str = "".join(traceback.format_exception(tool_exception))
        if tool_run_id in self.tool_calls:
            self.tool_calls[tool_run_id].finish_reason = "error"
            self.tool_calls[tool_run_id].error = error_str
            self.tool_calls[tool_run_id].tool_output = None
            self.tool_calls[tool_run_id].timestamp_finish = now
            tool_call = self.tool_calls[tool_run_id]
        else:
            tool_call = ToolCallPayload(
                tool_name=kwargs.get("name"),
                tool_run_id=tool_run_id,
                run_id=str(run_id),
                error=error_str,
                finish_reason="error",
                timestamp_start=now,
                timestamp_finish=now,
            )
            self.tool_calls[tool_run_id] = tool_call

        # Include agent_type field (will be None if not applicable)
        include_fields = {
            "tool_name",
            "tool_run_id",
            "finish_reason",
            "error",
            "timestamp_start",
            "timestamp_finish",
            "agent_type",
        }

        self.node_interface.emit(
            type=EventTypes.agent_tool_error,
            response_metadata=tool_call.model_dump(include=include_fields),
            content=error_str,
        )

    #
    # Agent
    #

    def on_agent_action(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_agent_action(%s, %s)", args, kwargs)

    def on_agent_finish(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_agent_finish(%s, %s)", args, kwargs)

    #
    # LLM
    #

    def _handle_llm_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]] | List[List[str]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if self.debug:
            log.debug(f"on_llm_start run_id={run_id}, node={metadata.get('langgraph_node') if metadata else 'N/A'}")

        now = datetime.now(tz=timezone.utc).isoformat()
        if not self.llm_start_timestamp:
            self.llm_start_timestamp = now

        self.current_model_name = metadata.get("ls_model_name", self.current_model_name)
        for msg_list in messages:
            try:
                tokens_in = num_tokens_from_messages(
                    msg_list, model=self.current_model_name
                )
            except NotImplementedError:
                tokens_in = num_tokens_from_messages(msg_list, model="gpt-4")

            self.pending_llm_requests[run_id]["tokens_in"] += tokens_in
            self.pending_llm_requests[run_id]["timestamp_start"] = now
            # Store langgraph_node and parent_agent_name for use in on_llm_end
            if metadata and metadata.get("langgraph_node"):
                self.pending_llm_requests[run_id]["langgraph_node"] = metadata.get(
                    "langgraph_node"
                )
            if metadata and metadata.get("parent_agent_name"):
                self.pending_llm_requests[run_id]["parent_agent_name"] = metadata.get(
                    "parent_agent_name"
                )

        # Use langgraph_node as tool_name if available (for pipeline LLM nodes), otherwise fallback to 'Thinking step'
        llm_tool_name = metadata.get("langgraph_node") if metadata else None
        self.node_interface.emit(
            type=EventTypes.agent_llm_start,
            response_metadata={
                "tool_name": llm_tool_name or "Thinking step",
                "tool_run_id": str(run_id),
                "metadata": metadata,
                "thinking_steps": self.thinking_steps,
                "timestamp_start": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    def on_llm_start(self, *args, **kwargs):
        """Callback"""
        self._handle_llm_start(*args, **kwargs)

    def on_chat_model_start(self, *args, **kwargs):
        """Callback"""
        self._handle_llm_start(*args, **kwargs)

    def on_llm_new_token(
        self, *args, run_id: UUID, parent_run_id: UUID = None, **kwargs
    ):
        """Callback"""
        if self.debug:
            log.info("on_llm_new_token(%s, %s)", args, kwargs)

        chunk: ChatGenerationChunk = kwargs.get("chunk")
        content = None
        thinking = None
        if chunk:
            content = chunk.text

            # Normalize content - extract from provider-specific formats if chunk.text is empty
            if hasattr(chunk, "message") and chunk.message:
                msg_content = chunk.message.content
                # Anthropic format: content is array with {type: "text/thinking", ...} items
                if isinstance(msg_content, list):
                    # Extract text items
                    if not content:
                        text_items = [
                            item.get("text", "")
                            for item in msg_content
                            if isinstance(item, dict)
                            and item.get("type") == "text"
                            and item.get("text")
                        ]
                        if text_items:
                            content = "".join(text_items)
                    # Extract thinking items (extended thinking / reasoning)
                    thinking_items = []
                    for item in msg_content:
                        if not isinstance(item, dict):
                            continue
                        item_type = item.get("type")
                        # Anthropic extended thinking
                        if item_type == "thinking" and item.get("thinking"):
                            thinking_items.append(item.get("thinking"))
                        # OpenAI reasoning models - summary array format
                        elif item_type == "reasoning" and item.get("summary"):
                            for summary_item in item.get("summary", []):
                                if isinstance(summary_item, dict) and summary_item.get(
                                    "text"
                                ):
                                    thinking_items.append(summary_item.get("text"))
                        # OpenAI reasoning models - direct reasoning field
                        elif item_type == "reasoning" and item.get("reasoning"):
                            thinking_items.append(item.get("reasoning"))
                    if thinking_items:
                        thinking = "\n".join(thinking_items)
                # OpenAI format: content is a string
                elif (
                    isinstance(msg_content, str) and msg_content.strip() and not content
                ):
                    content = msg_content

            # DEBUG: Log chunk details for streaming troubleshooting
            if not content and not thinking:
                log.debug(
                    f"[STREAM_DEBUG] Empty content - chunk.text={repr(chunk.text)}, "
                    f"has_message={hasattr(chunk, 'message') and chunk.message is not None}, "
                    f"msg_content_type={type(chunk.message.content).__name__ if hasattr(chunk, 'message') and chunk.message else 'N/A'}, "
                    f"msg_content={repr(chunk.message.content)[:200] if hasattr(chunk, 'message') and chunk.message else 'N/A'}"
                )

            # Count output tokens from chunk (will be used as fallback if API doesn't provide counts)
            try:
                chunk_tokens = num_tokens_from_messages(
                    [chunk], model=self.current_model_name, is_chunk=True
                )
                self.pending_llm_requests[run_id]["tokens_out"] += chunk_tokens
            except Exception as e:
                log.warning(f"Failed to count chunk tokens: {e}")

        # Calculate deltas - some providers send cumulative content instead of deltas
        run_id_str = str(run_id)
        content_delta = None
        thinking_delta = None

        if content:
            last_content = self._last_sent_content.get(run_id_str, "")
            if last_content and content.startswith(last_content):
                # Content is cumulative - extract only the new part
                if len(content) > len(last_content):
                    content_delta = content[len(last_content) :]
                # else: same content, no delta - skip
                # Store the full cumulative content for next comparison
                self._last_sent_content[run_id_str] = content
            elif last_content and last_content.startswith(content):
                # Content received is shorter - likely a new stream or reset, skip
                self._last_sent_content[run_id_str] = content
            else:
                # Content is a delta (doesn't start with previous) or first chunk
                content_delta = content
                # Build up cumulative from deltas
                self._last_sent_content[run_id_str] = last_content + content

        if thinking:
            last_thinking = self._last_sent_thinking.get(run_id_str, "")
            if last_thinking and thinking.startswith(last_thinking):
                # Thinking is cumulative - extract only the new part
                if len(thinking) > len(last_thinking):
                    thinking_delta = thinking[len(last_thinking) :]
                # Store the full cumulative thinking for next comparison
                self._last_sent_thinking[run_id_str] = thinking
            elif last_thinking and last_thinking.startswith(thinking):
                # Thinking received is shorter - likely a new stream or reset, skip
                self._last_sent_thinking[run_id_str] = thinking
            else:
                # Thinking is a delta or first chunk
                thinking_delta = thinking
                self._last_sent_thinking[run_id_str] = last_thinking + thinking

        # Only emit if there's actual non-empty content to send
        # Ensure content_delta and thinking_delta are valid strings (not None, not empty, not "null")
        has_content = (
            content_delta and isinstance(content_delta, str) and content_delta.strip()
        )
        has_thinking = (
            thinking_delta
            and isinstance(thinking_delta, str)
            and thinking_delta.strip()
        )

        if has_content or has_thinking:
            self.node_interface.emit(
                type=EventTypes.agent_llm_chunk,
                response_metadata={
                    "tool_run_id": str(run_id),
                },
                content=content_delta if has_content else "",
                thinking=thinking_delta if has_thinking else "",
            )

    def _parse_llm_error_message(self, error_body: dict) -> str:
        """Parse nested error messages from LLM providers

        Args:
            error_body: Error body dictionary from LLM provider

        Returns:
            Human-readable error message
        """
        # Handle nested error structure (e.g., {'error': {'message': '...'}})
        if "error" in error_body and isinstance(error_body["error"], dict):
            raw_message = error_body["error"].get("message", "Unknown error")
        else:
            raw_message = error_body.get("message", "Unknown error")

        # Try to parse nested JSON in error message
        if isinstance(raw_message, str):
            # Handle JSON-encoded messages (common in Anthropic errors)
            # Message might be like: '{"message":"..."}. Additional text...'
            # Extract JSON part if it exists at the beginning
            if raw_message.strip().startswith('{'):
                try:
                    # Find the end of the JSON object
                    json_end = raw_message.find('}') + 1
                    if json_end > 0:
                        json_str = raw_message[:json_end]
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and "message" in parsed:
                            raw_message = parsed["message"]
                except (json.JSONDecodeError, ValueError):
                    pass

            # Extract specific error patterns for user-friendly messages

            # Anthropic image size limit error
            if "image exceeds" in raw_message.lower() and "mb maximum" in raw_message.lower():
                # Extract size info if available
                size_match = re.search(r'(\d+)\s*MB maximum', raw_message, re.IGNORECASE)
                if size_match:
                    max_size = size_match.group(1)
                    return f"Image exceeds the {max_size} MB maximum size limit for this model."
                return "Image exceeds the maximum size limit for this model."

            # Anthropic rate limit errors
            if "rate limit" in raw_message.lower() or "rate_limit" in raw_message.lower():
                return "Rate limit exceeded. Please try again in a moment."

            # Token limit errors
            if "maximum context length" in raw_message.lower() or "token limit" in raw_message.lower():
                return "The request exceeds the model's token limit. Please reduce the input size."

            # Generic message cleanup - extract the most relevant part
            # Remove technical prefixes like "messages.0.content.0.image.source.base64:"
            cleaned = re.sub(r'^[\w\.]+:\s*', '', raw_message)
            return cleaned

        return raw_message

    def on_llm_error(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_llm_error(%s, %s)", args, kwargs)

        # Track tokens consumed before error occurred
        pending = self.pending_llm_requests.get(run_id, {})
        self.tokens_in += pending.get("tokens_in", 0)
        self.tokens_out += pending.get("tokens_out", 0)
        self.pending_llm_requests.pop(run_id, None)
        #
        if args:
            try:
                status_code: int = args[0].status_code
                error_message = self._parse_llm_error_message(args[0].body)
                self.llm_error = InternalSDKError(
                    f"status code: {status_code}, message: {error_message}"
                )
            except (AttributeError, TypeError, KeyError):
                self.llm_error = InternalSDKError(str(args[0]))
        else:
            self.llm_error = InternalSDKError("Unknown LLM error occurred")
        # exception = args[0]
        # FIXME: should we emit an error here too?

    #
    # Misc
    #

    def on_text(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.info("on_text(%s, %s)", args, kwargs)

    def on_llm_end(self, response: LLMResult, run_id: UUID, **kwargs) -> None:
        if self.debug:
            log.debug("on_llm_end(%s, %s)", response, kwargs)

        # Try to get token usage from API response
        token_usage = extract_token_usage(response)

        # Get pending request data (contains tokens and timestamp_start)
        pending = self.pending_llm_requests.get(run_id, {})

        if token_usage:
            # Use API-provided token counts (authoritative)
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            self.tokens_in += prompt_tokens
            self.tokens_out += completion_tokens
            log.debug(
                f"Token counting strategy: API-provided (run_id={run_id}, prompt={prompt_tokens}, completion={completion_tokens})"
            )
        else:
            # Fallback: use our estimated tokens from pending
            tokens_in = pending.get("tokens_in", 0)
            tokens_out = pending.get("tokens_out", 0)
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            log.debug(
                f"Token counting strategy: tiktoken estimation (run_id={run_id}, prompt={tokens_in}, completion={tokens_out})"
            )

        # Get the timestamp_start, langgraph_node, and parent_agent_name before popping
        llm_timestamp_start = pending.get("timestamp_start")
        langgraph_node = pending.get("langgraph_node")
        parent_agent_name = pending.get("parent_agent_name")
        self.pending_llm_requests.pop(run_id, None)

        for generation in response.generations:
            for generation_item in generation:
                generation_chunk = {
                    **generation_item.model_dump(),
                    "timestamp_start": llm_timestamp_start,
                    "timestamp_finish": datetime.now(tz=timezone.utc).isoformat(),
                }
                # Add langgraph_node as tool_name in message.response_metadata for frontend display
                if langgraph_node and "message" in generation_chunk:
                    if "response_metadata" not in generation_chunk["message"]:
                        generation_chunk["message"]["response_metadata"] = {}
                    generation_chunk["message"]["response_metadata"]["tool_name"] = (
                        langgraph_node
                    )
                # Extract text and thinking from message.content (always runs)
                msg_content = generation_chunk.get("message", {}).get("content")

                # Anthropic format: content is array with {type: "text/thinking", ...} items
                # OpenAI reasoning format: content is array with {type: "reasoning", summary: [...]} items
                if isinstance(msg_content, list) and not generation_chunk.get(
                    "thinking"
                ):
                    text_items = []
                    thinking_items = []
                    for item in msg_content:
                        if not isinstance(item, dict):
                            continue
                        item_type = item.get("type")
                        if item_type == "text" and item.get("text"):
                            text_items.append(item.get("text"))
                        # Anthropic extended thinking
                        elif item_type == "thinking" and item.get("thinking"):
                            thinking_items.append(item.get("thinking"))
                        # OpenAI reasoning models (gpt-5, o1, o3) - format 1: summary array
                        elif item_type == "reasoning" and item.get("summary"):
                            # GPT-5 returns reasoning in summary array with type="summary_text"
                            summary_items = item.get("summary", [])
                            for summary_item in summary_items:
                                if isinstance(summary_item, dict) and summary_item.get(
                                    "text"
                                ):
                                    thinking_items.append(summary_item.get("text"))
                        # OpenAI reasoning models - format 2: direct reasoning field
                        elif item_type == "reasoning" and item.get("reasoning"):
                            thinking_items.append(item.get("reasoning"))
                    # Set text only if not already set
                    if text_items and not generation_chunk.get("text"):
                        generation_chunk["text"] = "\n".join(text_items)
                    # Set thinking (primary extraction)
                    if thinking_items:
                        generation_chunk["thinking"] = "\n".join(thinking_items)
                # OpenAI format: content is a string - set text only if not already set
                elif (
                    isinstance(msg_content, str)
                    and msg_content.strip()
                    and not generation_chunk.get("text")
                ):
                    generation_chunk["text"] = msg_content

                # OpenAI/GPT reasoning: also check content_blocks for reasoning models
                # LangChain returns reasoning in response.content_blocks with type="reasoning"
                content_blocks = generation_chunk.get("message", {}).get(
                    "content_blocks", []
                )
                if content_blocks and not generation_chunk.get("thinking"):
                    reasoning_items = []
                    for block in content_blocks:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "reasoning"
                            and block.get("reasoning")
                        ):
                            reasoning_items.append(block.get("reasoning"))
                    if reasoning_items:
                        generation_chunk["thinking"] = "\n".join(reasoning_items)

                # OpenAI/GPT thinking: fallback to additional_kwargs['thinking']
                additional_kwargs = generation_chunk.get("message", {}).get(
                    "additional_kwargs", {}
                )
                if additional_kwargs.get("thinking") and not generation_chunk.get(
                    "thinking"
                ):
                    generation_chunk["thinking"] = additional_kwargs["thinking"]

                if not generation_chunk.get("text"):
                    # Fallback: extract tool call decisions if still no text
                    if not generation_chunk.get("text"):
                        decisions = []
                        try:
                            for tool_call in (
                                generation_chunk.get("message", {})
                                .get("additional_kwargs", {})
                                .get("tool_calls", [])
                            ):
                                tool_name = tool_call.get("function", {}).get("name")
                                tool_args = tool_call.get("function", {}).get(
                                    "arguments", {}
                                )
                                decisions.append(
                                    f"Planned to call tool '{tool_name}' with inputs {tool_args}"
                                )
                        except Exception:
                            pass
                        generation_chunk["text"] = "\n".join(decisions)

                # Add normalized tool_run_id for UI matching (works for both Anthropic and OpenAI)
                generation_chunk["tool_run_id"] = str(run_id)
                # Propagate parent_agent_name so history replay can show the nested agent context
                if parent_agent_name:
                    generation_chunk["parent_agent_name"] = parent_agent_name
                self.thinking_steps.append(generation_chunk)

        self.node_interface.emit(
            type=EventTypes.agent_llm_end,
            response_metadata={
                "tool_run_id": str(run_id),
                "thinking_steps": self.thinking_steps,
                "llm_start_timestamp": self.llm_start_timestamp,
            },
        )

        # necessary for partial message saving
        msg_event_node = NodeEvent(
            type=EventTypes.partial_message,
            stream_id=self.node_interface.stream_id,
            message_id=self.message_id,
            response_metadata={
                "project_id": self.project_id,
                "chat_project_id": self.chat_project_id,
                "thread_id": self.thread_id,
                "application_details": kwargs.get("application", {}),
                "thinking_steps": self.thinking_steps,
                "tool_calls": {
                    run_id: tool_call.model_dump()
                    for run_id, tool_call in self.tool_calls.items()
                },
                "llm_start_timestamp": self.llm_start_timestamp,
                "additional_response_meta": {},
            },
            content=None,
            **self.node_interface.payload_additional_kwargs,
        ).model_dump_json()
        msg_event_node = json.loads(msg_event_node)
        self.node_interface.event_node.emit(
            EVENTNODE_PARTIAL_RESPONSE_NAME, msg_event_node
        )
        # Check if the last thinking step
        # was truncated (finish_reason == 'length')
        if self.thinking_steps:
            last_step = self.thinking_steps[-1]
            finish_reason = extract_finish_reason(response, generation_chunk=last_step)
            if finish_reason == "length":
                self.node_interface.emit(
                    type=EventTypes.agent_requires_confirmation,
                    content="Continue",
                    response_metadata={
                        "tool_run_id": str(run_id),
                        "finish_reason": finish_reason,
                    },
                )


class EliteACustomCallback(BaseCallbackHandler):
    """EliteA custom agent callback handler"""

    def __init__(
        self,
        node_interface: NodeEventInterface,
        debug: bool = False,
        message_id: str = None,
        project_id: int = None,
        chat_project_id: int = None,
        user_id: int = None,
        initiator: str = None,
        task_id: str = None,
        toolkit_metadata: dict = None,
    ):
        log.debug(f"EliteACustomCallback init debug={debug}")
        self.node_interface = node_interface
        self.event_node = node_interface.event_node
        self.debug = debug

        self.tokens_in = 0
        self.tokens_out = 0
        self.message_id: str = message_id
        self.project_id: int = project_id
        self.chat_project_id: int = chat_project_id
        self.user_id: int = user_id
        self.initiator: str = initiator
        self.task_id: str = task_id
        self.toolkit_metadata: dict = toolkit_metadata or {}
        self.additional_response_meta = {}
        self.modified_files = []  # List to store modified file information
        self.generated_image_filepaths = []  # Filepaths of tool-generated images for thumbnail resolution
        self.index_statuses = []  # List to store index operation statuses
        # self.pending_llm_requests = defaultdict(int)
        # self.current_model_name = 'gpt-4'
        # self.stream_id = node_interface.stream_id

        super().__init__()

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Callback containing a group of custom events"""
        if self.debug:
            log.debug(f"on_custom_event name={name}, data_keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        event_key = f"agent_{name}"
        fields = ELITEA_SDK_CUSTOM_EVENTS_MAPPER.get(event_key, set())

        if self.debug:
            log.debug(f"{fields=}")

        if fields:
            payload = {
                "name": name,
                "run_id": str(run_id),
                "tool_run_id": str(run_id),  # compatibility
                "metadata": metadata,
                "datetime": str(datetime.now(tz=timezone.utc)),
                **{field: data.get(field) for field in fields if field in data},
            }
            payload = json.loads(
                json.dumps(payload, ensure_ascii=False, default=lambda o: str(o))
            )
            event_type_value = next(
                (event.value for event in EventTypes if event.name == event_key), None
            )

            if self.debug:
                log.debug(f"{event_type_value=}")

            if event_type_value:
                if event_type_value in {
                    EventTypes.agent_thinking_step.value,
                    EventTypes.agent_thinking_step_update.value,
                }:
                    self.additional_response_meta[
                        EventTypes.agent_thinking_step.value
                    ] = payload

                if event_type_value == EventTypes.agent_index_data_status.value:
                    # Add all required fields matching indexer_test_toolkit.py event structure
                    payload["task_id"] = self.task_id
                    payload["initiator"] = str(self.initiator)
                    payload["project_id"] = self.project_id
                    payload["user_id"] = self.user_id
                    # Add toolkit_config, tool_params, and toolkit_id from toolkit_metadata if not already in payload
                    if (
                        "toolkit_config" not in payload
                        and "toolkit_config" in self.toolkit_metadata
                    ):
                        payload["toolkit_config"] = self.toolkit_metadata[
                            "toolkit_config"
                        ]
                    if (
                        "tool_params" not in payload
                        and "tool_params" in self.toolkit_metadata
                    ):
                        payload["tool_params"] = self.toolkit_metadata["tool_params"]
                    if (
                        "toolkit_id" not in payload
                        and "toolkit_id" in self.toolkit_metadata
                    ):
                        payload["toolkit_id"] = self.toolkit_metadata["toolkit_id"]
                    # Collect index info for local storage
                    index_info = {
                        "id": payload.get("id"),
                        "task_id": self.task_id,
                        "index_name": payload.get("index_name"),
                        "state": payload.get("state"),
                        "error": payload.get("error"),
                        "reindex": payload.get("reindex"),
                        "indexed": payload.get("indexed"),
                        "updated": payload.get("updated"),
                        "created_at": payload.get("created_at"),
                        "updated_on": payload.get("updated_on"),
                        "datetime": payload.get("datetime"),
                        "toolkit_config": payload.get("toolkit_config"),
                        "tool_params": payload.get("tool_params"),
                        "toolkit_id": payload.get("toolkit_id"),
                        "initiator": str(self.initiator),
                        "project_id": self.project_id,
                        "user_id": self.user_id,
                    }
                    self.index_statuses.append(index_info)

                # Skip emitting agent_thinking_step for LLM reasoning since it's already
                # included in agent_llm_end.thinking_steps - avoids duplicate UI chips
                if (
                    event_type_value
                    in {
                        EventTypes.agent_thinking_step.value,
                        EventTypes.agent_thinking_step_update.value,
                    }
                    and payload.get("toolkit") == "reasoning"
                ):
                    log.debug(
                        f"Skipping {event_type_value} with toolkit=reasoning (handled by llm_end)"
                    )
                else:
                    # Add chat_project_id for swarm events (needed for persistence)
                    emit_payload = payload
                    if (
                        event_type_value == EventTypes.agent_swarm_agent_response.value
                        and self.chat_project_id
                    ):
                        emit_payload = {
                            **payload,
                            "chat_project_id": self.chat_project_id,
                        }
                    self.node_interface.emit(
                        type=event_type_value, response_metadata=emit_payload
                    )

                # Special handling for file modification events - collect file info
                if event_type_value == EventTypes.agent_file_modified.value:
                    file_info = {
                        "filepath": payload.get("filepath"),
                        "tool_name": payload.get("tool_name"),
                        "toolkit": payload.get("toolkit"),
                        "message": payload.get("message"),
                        "user_id": self.user_id,
                        "operation_type": payload.get(
                            "operation_type"
                        ),  # 'create' or 'modify'
                        "media_type": payload.get(
                            "media_type"
                        ),  # 'image', 'audio', 'video' or None
                        "meta": payload.get("meta", {}),  # Toolkit-specific metadata
                        "updated_at": payload.get(
                            "datetime"
                        ),  # Use datetime from callback
                    }
                    self.modified_files.append(file_info)

                    # Track generated image filepaths for thumbnail resolution at stream end
                    if file_info.get('media_type') == 'image' and file_info.get('filepath'):
                        self.generated_image_filepaths.append(file_info['filepath'])

            else:
                log.error(f"No such {event_type_value} in EventTypes")
        else:
            log.error(
                f"No such key {event_key} was found in ELITEA_SDK_CUSTOM_EVENTS_MAPPER"
            )


class Method:  # To make pylon happy
    pass
