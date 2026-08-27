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
from langchain_core.callbacks import BaseCallbackHandler  # pylint: disable=E0401
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, LLMResult
from elitea_sdk.runtime.utils.trace_limits import cap_trace_json, cap_trace_text
from pydantic import BaseModel
from pylon.core.tools import log  # pylint: disable=E0611,E0401

try:
    from elitea_sdk.runtime.langchain.constants import (
        LOAD_SKILL_ALREADY_ACTIVE_RE,
        LOADED_SKILL_PREFIX_RE,
    )
except ImportError:
    # Shim for SDKs predating the shared patterns; such an SDK says "is already
    # active for this turn", hence the alternation. Duplicated, not imported from
    # utils.agent_execution_common: that module imports this one.
    LOADED_SKILL_PREFIX_RE = re.compile(r'^Skill "([^"]+)" is now active')
    LOAD_SKILL_ALREADY_ACTIVE_RE = re.compile(r'^Skill "([^"]+)" is already (?:loaded|active)')

from ..utils.constants import DEFAULT_MEMORY_CONFIG
from ..utils.exceptions import InternalSDKError
from ..utils.funcs import (
    _is_mcp_authorization_required_error,
    _is_unresolved_mcp_type,
    _mcp_auth_error_to_metadata,
    budget_exceeded_error_code,
    is_mcp_authorization_required_error,
    extract_finish_reason,
    extract_token_usage,
    num_tokens_from_messages,
    should_emit_output_limit_confirmation,
)
from ..utils.node_interface import (
    ELITEA_SDK_CUSTOM_EVENTS_MAPPER,
    EventTypes,
    NodeEvent,
    NodeEventInterface,
)
from ..utils.parallel_dispatch_contract import is_fanout_child

# Event node names
EVENTNODE_EVENT_NAME = "application_stream_response"
EVENTNODE_FULL_RESPONSE_NAME = "application_full_response"
EVENTNODE_PARTIAL_RESPONSE_NAME = "application_partial_response"

HIERARCHY_METADATA_KEYS = (
    "langgraph_node",
    "parent_agent_name",
    "parent_agent_call_id",
    "parent_agent_path",
    "sibling_ordinal",
    "child_thread_id",
    "thread_id",
)

# Deliberately period-neutral, so the copy holds whatever the budget period is
BUDGET_EXCEEDED_MESSAGES = {
    "project_budget_exceeded": (
        "This project's budget has been reached. AI requests are unavailable until "
        "the budget resets or a project admin increases the limit."
    ),
    "member_budget_exceeded": (
        "Your budget for this project has been reached. Your AI requests are unavailable "
        "until the budget resets or a project admin increases your limit."
    ),
}

# Secret name for project PostgreSQL connection string
PGVECTOR_PROJECT_CONNSTR_SECRET = "pgvector_project_connstr"

# Per-worker cache: project_id -> connstr (immutable after project creation, safe to hold forever)
_pgvector_connstr_cache: dict = {}



def build_mcp_auth_pause_result(
    elitea_callback,
    chat_history: list,
    fallback_error: str = "Authorization required",
    node_interface: Optional[NodeEventInterface] = None,
) -> Optional[dict]:
    """Emit/return the legacy exception-based pause when no durable guard exists."""
    # A modern SDK can report the underlying McpAuthorizationRequired through
    # on_tool_error and then publish the authoritative checkpoint-backed
    # parallel_hitl_interrupt a moment later. Callback ordering is not a safe
    # way to distinguish those paths: a late on_tool_error can repopulate the
    # legacy cache after the durable event cleared it. Once this invocation has
    # exposed any durable auth interrupt, never synthesize the legacy UUID card
    # at turn end; the exact interrupt id and nested caller already own resume.
    durable_interrupt_seen = getattr(
        elitea_callback, "mcp_auth_durable_interrupt_seen", False,
    ) or getattr(elitea_callback, "parallel_hitl_run_state", {}).get(
        "mcp_auth_durable_interrupt_seen", False,
    )
    if durable_interrupt_seen:
        return None
    if getattr(elitea_callback, "mcp_auth_pause_payload", None):
        if node_interface is not None:
            node_interface.emit(
                type=EventTypes.mcp_authorization_required,
                content=(
                    getattr(elitea_callback, "mcp_auth_pause_message", None)
                    or fallback_error
                ),
                response_metadata=elitea_callback.mcp_auth_pause_payload,
            )
        return {
            "chat_history": chat_history,
            "error": getattr(elitea_callback, "mcp_auth_pause_message", None) or fallback_error,
            # Durable-child reconcile must not mistake an auth pause for completion. Resume parity
            # for an MCP-auth child is intentionally gated/deferred; this explicit state keeps the
            # parent epoch open instead of producing a partial final answer.
            "paused": True,
            "pause_type": "mcp_auth",
        }
    return None


def build_mcp_auth_required_result(
    node_interface: NodeEventInterface,
    exc: Exception,
    chat_project_id: Optional[int],
    chat_history: list,
) -> dict:
    """Emit mcp_authorization_required with normalized metadata and return stop payload."""
    auth_metadata = _mcp_auth_error_to_metadata(exc)
    provided_settings = getattr(exc, 'provided_settings', None)
    if provided_settings:
        auth_metadata['provided_settings'] = provided_settings
    if chat_project_id is not None:
        auth_metadata["chat_project_id"] = chat_project_id
    node_interface.emit(
        type=EventTypes.mcp_authorization_required,
        content=str(exc),
        response_metadata=auth_metadata,
    )
    return {
        "chat_history": chat_history,
        "error": str(exc),
        "paused": True,
        "pause_type": "mcp_auth",
    }


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
    client, project_id=None, max_retries: int = 3, base_delay: float = 0.5
) -> Optional[str]:
    """
    Fetch pgvector_project_connstr secret from the project vault with retry logic.
    Result is cached per project_id for the lifetime of the worker process.

    Args:
        client: EliteAClient instance with unsecret capability
        project_id: Project ID used as cache key (skips cache if None)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds between retries, uses exponential backoff (default: 0.5)

    Returns:
        Connection string if successful, None if secret doesn't exist or all retries failed
    """
    if project_id is not None and project_id in _pgvector_connstr_cache:
        log.debug(
            "pgvector_connstr cache hit for project_id=%s", project_id
        )
        return _pgvector_connstr_cache[project_id]

    log.debug(
        "pgvector_connstr cache miss for project_id=%s — fetching from vault", project_id
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            conn_str = client.unsecret(PGVECTOR_PROJECT_CONNSTR_SECRET)
            if conn_str:
                log.debug(
                    "Successfully fetched %s secret for project_id=%s",
                    PGVECTOR_PROJECT_CONNSTR_SECRET, project_id
                )
                if project_id is not None:
                    _pgvector_connstr_cache[project_id] = conn_str
                return conn_str
            else:
                log.warning(
                    "%s secret not found or empty in project vault for project_id=%s",
                    PGVECTOR_PROJECT_CONNSTR_SECRET, project_id
                )
                if project_id is not None:
                    _pgvector_connstr_cache[project_id] = None
                return None
        except Exception as e:  # pylint: disable=W0718
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff
                log.warning(
                    "Attempt %d/%d to fetch %s failed: %s. Retrying in %.1fs...",
                    attempt + 1, max_retries, PGVECTOR_PROJECT_CONNSTR_SECRET, e, delay
                )
                time.sleep(delay)
            else:
                log.warning(
                    "All %d attempts to fetch %s failed. Last error: %s. "
                    "Planning toolkit will use filesystem storage.",
                    max_retries, PGVECTOR_PROJECT_CONNSTR_SECRET, last_error
                )
    return None


def pgvector_connstr_needed(descriptor_config: dict) -> bool:
    """ True only when the checkpointer is postgres; otherwise the connstr is discarded """
    memory_config = descriptor_config.get("agent_memory_config") or DEFAULT_MEMORY_CONFIG
    return memory_config.get("type") == "postgres"


def resolve_pgvector_connstr(
    descriptor_config: dict, client_args: dict, api_token: str, api_extra_headers: dict,
    prefetched: Optional[str] = None,
) -> Optional[str]:
    """ Resolve the pgvector connstr, avoiding post-fork DNS where possible """
    # Cheapest first: not-postgres, then parent-resolved, then inherited cache.
    # The vault call is last because post-fork getaddrinfo can hang forever (#6245).
    if not pgvector_connstr_needed(descriptor_config):
        return None

    if prefetched is not None:
        return prefetched

    project_id = client_args.get("project_id")
    if project_id is not None and project_id in _pgvector_connstr_cache:
        return _pgvector_connstr_cache[project_id]

    log.warning(
        "pgvector_connstr not prefetched for project_id=%s — falling back to a post-fork "
        "vault call, which is exposed to the fork/getaddrinfo hang", project_id
    )
    with temp_elitea_client(client_args, api_token, api_extra_headers) as temp_client:
        return _fetch_pgvector_connstr_with_retry(temp_client, project_id=project_id)


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


def _sanitize_input_for_event(user_input):
    """Strip resolved base64 image data from user_input before it goes into an event.

    ``resolve_filepath_images`` rewrites each ``filepath:`` image chunk in-place to
    a full-resolution ``data:<mime>;base64,...`` URL for the LLM/persistence. That
    inflated payload must not ride along in error events to the browser — the UI
    already has the image. Replace each image_url chunk with a small placeholder;
    leave text chunks (and plain-string inputs) untouched. Returns a copy; never
    mutates the original list.
    """
    if not isinstance(user_input, list):
        return user_input
    sanitized = []
    for chunk in user_input:
        if isinstance(chunk, dict) and chunk.get("type") == "image_url":
            url = (chunk.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and ";" in url:
                mime = url[5:url.find(";")]
            else:
                mime = "image"
            sanitized.append({"type": "image_url", "image_url": {"url": f"[{mime}]"}})
        else:
            sanitized.append(chunk)
    return sanitized


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
    budget_error_code: Optional[str] = None,
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
        budget_error_code: Budget scope that blocked the call, for the UI to link to usage (optional)

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
            "tool_meta": _sanitize_input_for_event(user_input),
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

    # Carry the user-facing text and budget scope on the exception event too. Surfaces
    # that consume the stream directly (skill test panel) never see full_message, which
    # only reaches persisted chat conversations, so error_message alone would leave them
    # showing an internal label like "InternalSDKError on user input".
    exception_meta = {}
    #
    if human_readable:
        exception_meta["human_readable"] = human_readable
    #
    if budget_error_code:
        exception_meta["budget_error_code"] = budget_error_code
    #
    node_interface.emit(
        type=EventTypes.agent_exception,
        content=error_message,
        response_metadata=exception_meta,
    )

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

    # additional_response_meta is the supported way onto the persisted message meta;
    # a bare response_metadata key is dropped by elitea_core's whitelist, and the UI
    # needs this after a reload, not just on the live socket update
    if budget_error_code:
        response_metadata["additional_response_meta"] = {"budget_error_code": budget_error_code}

    if not is_fanout_child(tasknode_task_meta):
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
        subagent_name: str = None,
    ):
        log.debug(f"EliteACallback init debug={debug}")
        self.node_interface = node_interface
        self.event_node = node_interface.event_node
        self.stream_id = node_interface.stream_id
        self.debug = debug
        self.thread_id: str = thread_id
        # Durable parallel fan-out child (#4993 Track 2): this callback runs in a
        # standalone indexer_agent process for a sub-agent that streams onto the
        # PARENT's message. Unlike the in-process gather path, no parent callback
        # is in scope to tag the child's steps, so its thinking_steps/tool_calls
        # would persist into the parent meta with parent_agent_name=None and the
        # UI (partitionIntoBlocks, keyed on parent_agent_name) would scatter them
        # onto the coordinator. Stamp this name as the FALLBACK parent_agent_name
        # so each child's steps group under its own sub-agent accordion in the
        # thinking view — matching the sequential (in-process) render. None for an
        # ordinary top-level run, leaving attribution untouched.
        self.subagent_name: str = subagent_name
        # The child's REAL kind ('pipeline' or the agent kind e.g. 'openai'), read
        # from its own version_details. A pipeline child's internal LLM node emits
        # self-named chips stamped with agent_type=<model provider> (e.g. 'openai')
        # — wrong for the icon, which needs 'pipeline'. Stamped as a fallback onto
        # the child's tool chips so the UI renders the pipeline (flow) icon, the
        # same way subagent_name fixes parent_agent_name. None for ordinary runs.
        # Set post-construction by indexer_agent (which has the child's
        # version_details in scope); defaults None for an ordinary top-level run.
        self.subagent_agent_type: str = None
        self.applied_skills: list = []
        self.skills_by_name: dict = {}
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
        # If an MCP auth-required tool error is observed in callback flow,
        # capture it so caller can pause execution and avoid emitting normal completion.
        self.mcp_auth_pause_payload: Optional[dict] = None
        self.mcp_auth_durable_interrupt_seen = False
        self.parallel_hitl_run_state: dict = {}
        self.mcp_auth_pause_message: Optional[str] = None
        self.created_entities: list = []  # Entities created via MCP tools during this run
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

    def _is_mcp_auth_paused(self) -> bool:
        """Return True only while legacy exception-based auth owns this run.

        A durable nested auth interrupt is reported by EliteACustomCallback,
        while tool lifecycle events are handled by this callback instance.  The
        original exception payload can remain cached here after the durable
        supervisor takes ownership; treating that cache as an active pause
        suppresses every later child on_tool_start/on_tool_end and leaves the UI
        accordions empty.  Shared run state is therefore authoritative.
        """
        if self.parallel_hitl_run_state.get(
            "mcp_auth_durable_interrupt_seen", False,
        ):
            return False
        return self.mcp_auth_pause_payload is not None

    def emit_subagent_invocation_chip(self, task_text, response, agent_type=None):
        """Emit the PARENT's bare sub-agent invocation chip for a durable child (#4993).

        In the sequential (in-process) path the orchestrator runs the sub-agent
        as a TOOL, so its on_tool_start/on_tool_end fire and produce a BARE chip:
        tool name == the sub-agent name, ``parent_agent_name`` ABSENT (so it
        renders without the "(Name Resolver)" parenthesis), grouped under the
        sub-agent's own accordion via ``original_name``, and crucially carrying
        BOTH the task (tool_inputs) and the sub-agent's final answer (tool_output)
        — see group 568. For a pipeline child a second, parenthesized chip comes
        from its embedded agent; for a simple-agent child this is the only one.

        In the durable Track 2 park path the parent returns child specs WITHOUT
        invoking the tool in-process, so this bare chip never exists — it is the
        one missing from the parallel view. Reproduce it from the child's own
        end-of-run state: its name, the task it was launched with, its full
        AgentResponse, and its REAL agent_type for the icon. Persist via the same
        partial_message path the child's real tool chips already use (full_message
        is suppressed for a fan-out child), so it lands in the parent group's meta.

        ``response`` is the raw AgentResponse dict the sub-agent returns (keys:
        output, messages, name_meaning, …). The in-process path serializes this
        WHOLE dict as the tool_output, so the UI shows the same structured JSON;
        we must do the same here, not just the plain ``output`` string (#4993).

        No-op for an ordinary top-level run (``subagent_name`` is None).
        """
        if not self.subagent_name:
            return
        if isinstance(response, str):
            output_text = response
        else:
            try:
                output_text = json.dumps(response, ensure_ascii=False, default=lambda o: str(o))
            except (TypeError, ValueError):
                output_text = str(response)
        now = datetime.now(tz=timezone.utc).isoformat()
        run_id = str(uuid4())
        _agent_type = agent_type or self.subagent_agent_type or "agent"
        # original_name groups the chip under the sub-agent's accordion;
        # parent_agent_name is intentionally ABSENT so it renders bare (no
        # parenthetical), matching the sequential parent-invocation chip.
        metadata = {
            "display_name": self.subagent_name,
            "toolkit_name": self.subagent_name,
            "toolkit_type": "application",
            "agent_type": _agent_type,
            "original_name": self.subagent_name,
        }
        metadata = self.node_interface.apply_metadata_overlay(metadata)
        # This is the orchestrator's own invocation/result chip. Its owner is
        # conveyed by the canonical path; leaving parent_agent_name unset keeps
        # the chip bare rather than labelling it as its own child.
        metadata.pop("parent_agent_name", None)
        tool_meta = {"name": self.subagent_name, "metadata": dict(metadata)}
        self.tool_calls[run_id] = ToolCallPayload(
            tool_name=self.subagent_name,
            tool_run_id=run_id,
            run_id=run_id,
            tool_meta=tool_meta,
            tool_inputs={"task": task_text} if task_text else {},
            metadata=metadata,
            agent_type=_agent_type,
            tool_output=output_text,
            finish_reason="stop",
            timestamp_start=now,
            timestamp_finish=now,
        )
        persisted_invocation = self.node_interface.decorate_tool_call_for_persistence(
            self.tool_calls[run_id].model_dump()
        )
        persisted_invocation.get("metadata", {}).pop("parent_agent_name", None)
        if isinstance(persisted_invocation.get("tool_meta"), dict):
            persisted_invocation["tool_meta"].get("metadata", {}).pop(
                "parent_agent_name", None
            )
        msg_event_node = NodeEvent(
            type=EventTypes.partial_message,
            stream_id=self.node_interface.stream_id,
            message_id=self.message_id,
            response_metadata={
                "project_id": self.project_id,
                "chat_project_id": self.chat_project_id,
                "thread_id": self.thread_id,
                "thinking_steps": [],
                "tool_calls": {run_id: persisted_invocation},
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

    #
    # Chain
    #

    def on_chain_start(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.debug("on_chain_start(%s, %s)", args, kwargs)

    def on_chain_end(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.debug("on_chain_end(%s, %s)", args, kwargs)

    def on_chain_error(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.debug("on_chain_error(%s, %s)", args, kwargs)
        #
        # exception = args[0]
        # FIXME: should we emit an error here too?

    #
    # Tool
    #

    def on_tool_start(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        if self._is_mcp_auth_paused():
            return
        if self.debug:
            log.debug("on_tool_start(%s, %s)", args, kwargs)
        # The serialized tool name is the action that actually ran. Execution
        # metadata.original_name can be the enclosing Application name and must
        # never replace it (that made configurations/artifact calls look like
        # repeated sub-orchestrator invocations after reload).
        tool_name = args[0].get("name")
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

        # Durable fan-out child fallback (#4993 Track 2): if no nested-agent name
        # was resolved above, attribute this tool to the child sub-agent this
        # whole run represents, so its chip groups under the child's accordion in
        # the parent's thinking view. Only fills when unset — a genuine deeper
        # nested agent_name (set above) is never overwritten.
        #
        # Write BOTH dicts: the UI's deriveSubAgentName reads parent_agent_name
        # from the TOP-LEVEL `metadata` (tool_metadata -> persisted tool_call
        # `metadata`, also carried on the live event's response_metadata.metadata);
        # the nested tool_meta["metadata"] is merged in only on the live socket
        # path. A simple sub-agent (single `agent:` node) has no original_name /
        # checkpoint_ns to fall back on, so without the top-level stamp its tool
        # chip leaks to the coordinator block both live and on reload.
        if self.subagent_name and not (
            tool_metadata.get("parent_agent_name")
            or tool_meta.get("metadata", {}).get("parent_agent_name")
        ):
            if "metadata" not in tool_meta:
                tool_meta["metadata"] = {}
            tool_meta["metadata"]["parent_agent_name"] = self.subagent_name
            tool_metadata["parent_agent_name"] = self.subagent_name

        # Trace rows drop tool_inputs, so without this stamp a reload degrades the
        # chip to the generic toolkit label. The icon rides tool_metadata: the
        # live event's toolMeta is built from `metadata`, not tool_meta.
        if tool_name == "load_skill":
            requested = (kwargs.get("inputs") or {}).get("skill")
            if isinstance(requested, str) and requested.strip():
                registered = self.skills_by_name.get(requested.strip().lower()) or {}
                tool_meta["loaded_skill"] = registered.get("name") or requested.strip()
                if registered.get("icon_meta"):
                    tool_metadata["icon_meta"] = registered["icon_meta"]

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

        # Fallback: get agent_type from tool's serialized metadata if not in execution metadata
        if not tool_metadata.get("agent_type"):
            serialized_agent_type = tool_meta.get("metadata", {}).get("agent_type")
            if serialized_agent_type:
                log.debug(f"[METADATA] Adding agent_type from serialized metadata: {serialized_agent_type}")
                tool_metadata["agent_type"] = serialized_agent_type

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
            "tool_inputs": cap_trace_json(kwargs.get("inputs")),
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

    def _applied_skills_for_partial(self, tool_call=None) -> list:
        """Applied skills to ride a partial save.

        ``full_message`` is the only other writer and never fires on Stop or at a
        HITL pause. The meta writer unions partial saves, so emitting the
        dispatch-time set plus this step's own load_skill accumulates the same list
        one step at a time.
        """
        applied = [
            {
                'skill_id': skill.get('skill_id'),
                'name': skill.get('name'),
                'icon_meta': skill.get('icon_meta'),
            }
            for skill in (self.applied_skills or [])
            if isinstance(skill, dict) and skill.get('name')
        ]
        if tool_call is not None and getattr(tool_call, 'tool_name', '') == 'load_skill':
            tool_output = getattr(tool_call, 'tool_output', '') or ''
            # An "already loaded" answer still means the skill is in effect this turn.
            match = LOADED_SKILL_PREFIX_RE.match(tool_output) or LOAD_SKILL_ALREADY_ACTIVE_RE.match(tool_output)
            if match:
                name = match.group(1)
                seen = {(entry['name'] or '').strip().lower() for entry in applied}
                key = name.strip().lower()
                if key not in seen:
                    registered = self.skills_by_name.get(key) or {}
                    applied.append({
                        'skill_id': registered.get('skill_id'),
                        'name': name,
                        'icon_meta': registered.get('icon_meta'),
                    })
        return applied

    def on_tool_end(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        if self._is_mcp_auth_paused():
            return
        if self.debug:
            log.debug("on_tool_end(%s, %s)", args, kwargs)
        tool_run_id = str(run_id)
        # Use JSON serialization for non-string types to preserve proper formatting
        raw_output = args[0]
        # Parallel sub-agent HITL pause (#5378): a child that pauses for a
        # sensitive-action approval returns a deferred sentinel
        # ({"__hitl_deferred__": True, ...}) instead of a real result. Its
        # invocation wrapper's on_tool_end still fires (the sentinel is the _run
        # return value), which would otherwise mark the sub-agent "done" in the UI
        # and drop its shimmer mid-run — before the aggregate approval card even
        # surfaces. Detect it here (where the sentinel is structurally available,
        # pre-serialization), stamp a clean flag the UI can read, and DON'T leak
        # the raw sentinel as the tool's visible result.
        hitl_deferred = isinstance(raw_output, dict) and bool(
            raw_output.get("__hitl_deferred__")
        )
        # LangChain wraps tool results in a ToolMessage (BaseMessage) when a
        # tool_call_id is provided (e.g. via LangGraph's ToolNode for published
        # agents with toolkits). Extract the actual content, otherwise the
        # non-serializable ToolMessage falls back to its pydantic __str__
        # ("content='...' name='...' tool_call_id='...'") and is surfaced to
        # the end user / LLM context instead of the plain tool result.
        if isinstance(raw_output, BaseMessage):
            raw_output = raw_output.content
        if hitl_deferred:
            tool_output = ""
        else:
            tool_output = cap_trace_text(
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

        # Surface the deferred-pause flag (#5378) on the event metadata the UI
        # reads (response_metadata.metadata). Lets the UI keep the sub-agent's
        # shimmer alive through the approval gap instead of marking it "done".
        if hitl_deferred:
            tool_call.metadata = {**(tool_call.metadata or {}), "hitl_deferred": True}

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

        _tool_name_for_entity = kwargs.get("name") or (
            self.tool_calls[tool_run_id].tool_name if tool_run_id in self.tool_calls else None
        )
        _ENTITY_TOOL_NAMES = frozenset({
            "post_elitea_core_applications",
            "post_elitea_core_versions",
            "post_elitea_core_skills",
            "post_elitea_core_toolkits",
            "post_elitea_core_tools",
            "put_project_context_project-context",
        })
        if _tool_name_for_entity in _ENTITY_TOOL_NAMES:
            try:
                _resp = json.loads(tool_output) if isinstance(tool_output, str) else tool_output
                _entity_id = _resp.get("id")
                _entity_payload = None

                if _tool_name_for_entity == "post_elitea_core_applications":
                    _entity_name = _resp.get("name", "")
                    _vd = _resp.get("version_details") or {}
                    _version_id = _vd.get("id")
                    _agent_type = _vd.get("agent_type") or _resp.get("agent_type") or "openai"
                    if _entity_id and _version_id:
                        _etype = "pipeline" if _agent_type == "pipeline" else "agent"
                        _entity_payload = {
                            "entity_type": _etype,
                            "entity_id": _entity_id,
                            "version_id": _version_id,
                            "entity_name": _entity_name,
                        }

                elif _tool_name_for_entity == "post_elitea_core_versions":
                    _entity_name = _resp.get("name", "")
                    _version_id = _resp.get("id")
                    _app_id = _resp.get("application_id")
                    _agent_type = _resp.get("agent_type") or "openai"
                    if _app_id and _version_id:
                        _etype = "pipeline" if _agent_type == "pipeline" else "agent"
                        _entity_payload = {
                            "entity_type": _etype,
                            "entity_id": _app_id,
                            "version_id": _version_id,
                            "entity_name": _entity_name,
                        }

                elif _tool_name_for_entity == "post_elitea_core_skills":
                    _entity_name = _resp.get("name", "")
                    _vd = _resp.get("version_details") or {}
                    _version_id = _vd.get("id")
                    if _entity_id and _version_id:
                        _entity_payload = {
                            "entity_type": "skill",
                            "entity_id": _entity_id,
                            "version_id": _version_id,
                            "entity_name": _entity_name,
                        }

                elif _tool_name_for_entity in ("post_elitea_core_toolkits", "post_elitea_core_tools"):
                    _entity_name = _resp.get("name", "")
                    _toolkit_type = _resp.get("type", "")
                    _is_mcp = bool(_toolkit_type and (
                        _toolkit_type == "mcp" or _toolkit_type.startswith("mcp_")
                    ))
                    if _entity_id:
                        _entity_payload = {
                            "entity_type": "toolkit",
                            "entity_id": _entity_id,
                            "version_id": None,
                            "entity_name": _entity_name,
                            "is_mcp": _is_mcp,
                        }

                elif _tool_name_for_entity == "put_project_context_project-context":
                    if _entity_id:
                        _entity_payload = {
                            "entity_type": "project_context",
                            "entity_id": _entity_id,
                            "version_id": None,
                            "entity_name": "Project Context",
                        }

                if _entity_payload:
                    self.node_interface.emit(
                        type=EventTypes.agent_entity_created,
                        response_metadata=_entity_payload,
                    )
                    self.created_entities.append(_entity_payload)
                    log.info(
                        "[ENTITY_CREATED] Detected %s creation via MCP tool: id=%s version_id=%s name=%r",
                        _entity_payload["entity_type"], _entity_payload["entity_id"],
                        _entity_payload["version_id"], _entity_payload["entity_name"],
                    )
            except Exception as _exc:
                log.debug("[ENTITY_CREATED] Could not parse entity from tool output: %s", _exc)

        # necessary for partial message saving — send only the single updated entry (delta)
        msg_event_node = NodeEvent(
            type=EventTypes.partial_message,
            stream_id=self.node_interface.stream_id,
            message_id=self.message_id,
            response_metadata={
                "project_id": self.project_id,
                "chat_project_id": self.chat_project_id,
                "thread_id": self.thread_id,
                "application_details": kwargs.get("application", {}),
                "thinking_steps": [],
                "tool_calls": {
                    tool_run_id: self.node_interface.decorate_tool_call_for_persistence(
                        tool_call.model_dump()
                    )
                },
                "llm_start_timestamp": self.llm_start_timestamp,
                "additional_response_meta": {},
                "invoked_skills": self._applied_skills_for_partial(tool_call),
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
        if self._is_mcp_auth_paused():
            return
        if self.debug:
            log.debug("on_tool_error(%s, %s)", args, kwargs)
        tool_run_id = str(run_id)
        tool_exception = args[0]
        now = datetime.now(tz=timezone.utc).isoformat()

        if _is_mcp_authorization_required_error(tool_exception):
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

            auth_payload = _mcp_auth_error_to_metadata(tool_exception)

            # Some SDK/client paths raise McpAuthorizationRequired without a resolved
            # toolkit_type (or with the generic placeholder "mcp"/"mcp_config"). Backfill
            # from the currently running tool metadata so FE receives a stable discriminator.
            if _is_unresolved_mcp_type(auth_payload.get("toolkit_type")):
                resolved_toolkit_type = None

                callback_metadata = kwargs.get("metadata")
                if isinstance(callback_metadata, dict):
                    resolved_toolkit_type = (
                        callback_metadata.get("toolkit_type")
                        or callback_metadata.get("type")
                    )

                tool_call_metadata = getattr(tool_call, "metadata", None)
                if not resolved_toolkit_type and isinstance(tool_call_metadata, dict):
                    resolved_toolkit_type = (
                        tool_call_metadata.get("toolkit_type")
                        or tool_call_metadata.get("type")
                    )

                tool_meta = getattr(tool_call, "tool_meta", None)
                if not resolved_toolkit_type and isinstance(tool_meta, dict):
                    nested_meta = tool_meta.get("metadata")
                    if isinstance(nested_meta, dict):
                        resolved_toolkit_type = (
                            nested_meta.get("toolkit_type")
                            or nested_meta.get("type")
                        )

                if not resolved_toolkit_type:
                    resolved_toolkit_type = self.cached_toolkit_type

                if resolved_toolkit_type:
                    auth_payload["toolkit_type"] = resolved_toolkit_type
                    try:
                        setattr(tool_exception, "toolkit_type", resolved_toolkit_type)
                    except Exception:  # pragma: no cover - defensive only
                        pass

            if not auth_payload.get("toolkit_name"):
                resolved_toolkit_name = None

                callback_metadata = kwargs.get("metadata")
                if isinstance(callback_metadata, dict):
                    resolved_toolkit_name = callback_metadata.get("toolkit_name")

                tool_call_metadata = getattr(tool_call, "metadata", None)
                if not resolved_toolkit_name and isinstance(tool_call_metadata, dict):
                    resolved_toolkit_name = tool_call_metadata.get("toolkit_name")

                tool_meta = getattr(tool_call, "tool_meta", None)
                if not resolved_toolkit_name and isinstance(tool_meta, dict):
                    nested_meta = tool_meta.get("metadata")
                    if isinstance(nested_meta, dict):
                        resolved_toolkit_name = (
                            nested_meta.get("toolkit_name")
                            or nested_meta.get("display_name")
                        )

                if not resolved_toolkit_name:
                    resolved_toolkit_name = self.cached_toolkit_name

                if resolved_toolkit_name:
                    auth_payload["toolkit_name"] = resolved_toolkit_name
                    if not auth_payload.get("tool_name"):
                        auth_payload["tool_name"] = resolved_toolkit_name
                    try:
                        setattr(tool_exception, "toolkit_name", resolved_toolkit_name)
                    except Exception:  # pragma: no cover - defensive only
                        pass

            if auth_payload.get("toolkit_name") and not auth_payload.get("tool_name"):
                auth_payload["tool_name"] = auth_payload["toolkit_name"]

            provided_settings = getattr(tool_exception, 'provided_settings', None)
            if provided_settings:
                auth_payload['provided_settings'] = provided_settings
            auth_tool_name = auth_payload.get("tool_name") or tool_call.tool_name
            auth_payload.update(
                {
                    "tool_name": auth_tool_name,
                    "tool_run_id": tool_run_id,
                    "chat_project_id": self.chat_project_id,  # Include for DB update
                }
            )

            # Mark this run as paused for MCP auth so caller can stop post-invoke flow.
            self.mcp_auth_pause_payload = auth_payload
            self.mcp_auth_pause_message = error_str

            # Do not emit here.  Modern SDKs catch this exception and return a
            # checkpoint-backed mcp_auth interrupt with exact nested routing.
            # The post-invoke path emits that authoritative payload.  Older SDK
            # paths fall back to build_mcp_auth_pause_result(), which emits this
            # cached metadata once after invoke/exception handling.
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
            log.debug("on_agent_action(%s, %s)", args, kwargs)

    def on_agent_finish(self, *args, **kwargs):
        """Callback"""
        if self.debug:
            log.debug("on_agent_finish(%s, %s)", args, kwargs)

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
        if self._is_mcp_auth_paused():
            return
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
            if metadata:
                self.pending_llm_requests[run_id]["hierarchy_metadata"] = {
                    key: metadata[key]
                    for key in HIERARCHY_METADATA_KEYS
                    if metadata.get(key) is not None
                }

        # Use langgraph_node as tool_name if available (for pipeline LLM nodes), otherwise fallback to 'Thinking step'
        llm_tool_name = metadata.get("langgraph_node") if metadata else None
        self.node_interface.emit(
            type=EventTypes.agent_llm_start,
            response_metadata={
                "tool_name": llm_tool_name or "Thinking step",
                "tool_run_id": str(run_id),
                "metadata": metadata,
                "thinking_steps": [self.thinking_steps[-1]] if self.thinking_steps else [],
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
        if self._is_mcp_auth_paused():
            return
        if self.debug:
            log.debug("on_llm_new_token(%s, %s)", args, kwargs)

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
                        text_items = []
                        for item in msg_content:
                            # Anthropic adaptive thinking returns the final answer
                            # as a bare string list item; capture it as text.
                            if isinstance(item, str):
                                if item:
                                    text_items.append(item)
                            elif (
                                isinstance(item, dict)
                                and item.get("type") == "text"
                                and item.get("text")
                            ):
                                text_items.append(item.get("text"))
                        if text_items:
                            content = "\n".join(text_items)
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
                    "metadata": self.pending_llm_requests.get(run_id, {}).get(
                        "hierarchy_metadata", {}
                    ),
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
            log.debug("on_llm_error(%s, %s)", args, kwargs)

        # Track tokens consumed before error occurred
        pending = self.pending_llm_requests.get(run_id, {})
        self.tokens_in += pending.get("tokens_in", 0)
        self.tokens_out += pending.get("tokens_out", 0)
        self.pending_llm_requests.pop(run_id, None)
        #
        if args:
            # Wrapping loses the structured body, so carry the budget scope across it —
            # a budget block needs a friendly message, not a raw provider error
            budget_code = budget_exceeded_error_code(args[0])
            #
            try:
                status_code: int = args[0].status_code
                error_message = self._parse_llm_error_message(args[0].body)
                self.llm_error = InternalSDKError(
                    f"status code: {status_code}, message: {error_message}"
                )
            except (AttributeError, TypeError, KeyError):
                self.llm_error = InternalSDKError(str(args[0]))
            #
            self.llm_error.budget_error_code = budget_code
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
            log.debug("on_text(%s, %s)", args, kwargs)

    def on_llm_end(self, response: LLMResult, run_id: UUID, **kwargs) -> None:
        if self._is_mcp_auth_paused():
            return
        if self.debug:
            log.debug("on_llm_end(%s, %s)", response, kwargs)

        # Track which steps belong to this callback. ``thinking_steps`` spans
        # the whole agent run and may already contain earlier LLM calls.
        previous_step_count = len(self.thinking_steps)

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
        hierarchy_metadata = pending.get("hierarchy_metadata", {})
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
                        # Anthropic adaptive thinking returns the final answer as
                        # a bare string list item (content=['', {thinking...},
                        # 'answer text']); capture non-empty strings as text.
                        if isinstance(item, str):
                            if item:
                                text_items.append(item)
                            continue
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

                generation_chunk["text"] = cap_trace_text(generation_chunk.get("text"))
                generation_chunk["thinking"] = cap_trace_text(generation_chunk.get("thinking"))

                # Add normalized tool_run_id for UI matching (works for both Anthropic and OpenAI)
                generation_chunk["tool_run_id"] = str(run_id)
                # Propagate parent_agent_name so history replay can show the nested agent context
                if parent_agent_name:
                    generation_chunk["parent_agent_name"] = parent_agent_name
                # Durable fan-out child fallback (#4993 Track 2): attribute this
                # reasoning step to the child sub-agent this run represents when no
                # deeper nested name applies, so it groups under the child's
                # accordion in the parent's thinking view. Only fills when unset.
                elif self.subagent_name:
                    generation_chunk["parent_agent_name"] = self.subagent_name
                for key, value in hierarchy_metadata.items():
                    if value is not None:
                        generation_chunk[key] = value
                # MUST run after all extraction above and before append.
                _msg = generation_chunk.get("message")
                if isinstance(_msg, dict):
                    _msg.pop("content", None)
                self.thinking_steps.append(
                    self.node_interface.decorate_thinking_step_for_persistence(
                        generation_chunk
                    )
                )

        self.node_interface.emit(
            type=EventTypes.agent_llm_end,
            response_metadata={
                "tool_run_id": str(run_id),
                "thinking_steps": (
                    [self.thinking_steps[-1]]
                    if len(self.thinking_steps) > previous_step_count else []
                ),
                "llm_start_timestamp": self.llm_start_timestamp,
            },
        )

        # necessary for partial message saving — send only the new thinking step (delta)
        new_thinking_step = (
            self.thinking_steps[-1]
            if len(self.thinking_steps) > previous_step_count else None
        )
        msg_event_node = NodeEvent(
            type=EventTypes.partial_message,
            stream_id=self.node_interface.stream_id,
            message_id=self.message_id,
            response_metadata={
                "project_id": self.project_id,
                "chat_project_id": self.chat_project_id,
                "thread_id": self.thread_id,
                "application_details": kwargs.get("application", {}),
                "thinking_steps": [new_thinking_step] if new_thinking_step else [],
                "tool_calls": {},
                "llm_start_timestamp": self.llm_start_timestamp,
                "additional_response_meta": {},
                "invoked_skills": self._applied_skills_for_partial(),
            },
            content=None,
            **self.node_interface.payload_additional_kwargs,
        ).model_dump_json()
        msg_event_node = json.loads(msg_event_node)
        self.node_interface.event_node.emit(
            EVENTNODE_PARTIAL_RESPONSE_NAME, msg_event_node
        )
        # A provider can report token exhaustion with no generation at all
        # (notably reasoning output through the OpenAI Responses API). Detect
        # the terminal reason from the complete response, not only from the
        # most recently persisted step.
        last_step = new_thinking_step
        finish_reason = extract_finish_reason(response, generation_chunk=last_step)
        if should_emit_output_limit_confirmation(finish_reason, hierarchy_metadata):
            self.node_interface.emit(
                type=EventTypes.agent_requires_confirmation,
                content="Continue",
                response_metadata={
                    "tool_run_id": str(run_id),
                    "thread_id": self.thread_id,
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
        # create_callbacks() replaces this with the same dict owned by the
        # regular callback.  LangChain dispatches custom events and tool errors
        # to separate callback objects, so durable-auth suppression must be
        # shared across both objects for the lifetime of this run.
        self.parallel_hitl_run_state: dict = {}
        # self.pending_llm_requests = defaultdict(int)
        # self.current_model_name = 'gpt-4'
        # self.stream_id = node_interface.stream_id

        super().__init__()

    def _persist_injection_marker(self, payload: dict) -> None:
        """Persist a consumed injection as a trace-step delta on the partial_message channel.

        Rides the SDK's existing accumulator path (partial_message -> sync_trace_steps) so the
        table keeps a single writer. Writing the row from pylon_main's inject endpoint instead
        would race that accumulator, which deletes any row it cannot reconstruct from its own
        delta.

        Shaped as a thinking step with an injection marker in generation_info: the pin renders
        via the existing thinking branch, and the marker is what tells the UI to draw it as a
        user interjection.
        """
        text = payload.get("text")
        injection_id = payload.get("injection_id")
        if not text or not injection_id:
            return
        now = datetime.now(tz=timezone.utc).isoformat()
        step = {
            # run_id is the writer's natural key; prefixing keeps it clear of tool run ids.
            "tool_run_id": f"injection_{injection_id}",
            "type": "midturn_injection",
            "text": text,
            "thinking": "",
            "timestamp_start": now,
            "timestamp_finish": now,
            "generation_info": {"midturn_injection_id": injection_id},
            "message": {"response_metadata": {"midturn_injection_id": injection_id}},
        }
        try:
            event = NodeEvent(
                type=EventTypes.partial_message,
                stream_id=self.node_interface.stream_id,
                message_id=self.message_id,
                response_metadata={
                    "project_id": self.project_id,
                    "chat_project_id": self.chat_project_id,
                    "thinking_steps": [step],
                    "tool_calls": {},
                    "additional_response_meta": {},
                },
                content=None,
                **self.node_interface.payload_additional_kwargs,
            ).model_dump_json()
            self.node_interface.event_node.emit(
                EVENTNODE_PARTIAL_RESPONSE_NAME, json.loads(event)
            )
        except Exception as e:
            log.warning(f"Failed to persist injection marker {injection_id}: {e}")

    def _persist_mcp_auth_decision(
        self,
        payload: dict,
        event_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a synthetically completed auth call as a normal tool step.

        SDK auth resume closes the original ``mcp_auth_control`` call with a
        structured ToolMessage instead of executing the tool again. That keeps
        checkpoint semantics correct but bypasses ``on_tool_end``. Mirror its
        delta contract here so live and reloaded child accordions retain the
        LLM -> auth tool -> LLM timeline.
        """
        tool_output = cap_trace_text(payload.get("tool_output") or "")
        tool_name = str(payload.get("tool_name") or "mcp_auth_control")
        tool_run_id = str(payload.get("tool_call_id") or uuid4())
        if not tool_output:
            return

        now = datetime.now(tz=timezone.utc).isoformat()
        # Unlike ordinary on_tool_end callbacks, this synthetic completion is
        # dispatched from inside the SDK leaf graph. Its canonical in-process
        # ancestry therefore arrives in the custom event's Runnable metadata,
        # not in the worker's static NodeInterface overlay. Persist the selected
        # hierarchy fields as well, otherwise a reload moves the auth chip from
        # its Surname Resolver accordion back to the root timeline.
        lineage = {
            key: event_metadata.get(key)
            for key in HIERARCHY_METADATA_KEYS
            if (
                isinstance(event_metadata, dict)
                and event_metadata.get(key) is not None
            )
        }
        tool_metadata = self.node_interface.apply_metadata_overlay({
            "toolkit_name": payload.get("toolkit_name") or "",
            "toolkit_type": payload.get("toolkit_type") or "",
            **lineage,
        })
        tool_call = ToolCallPayload(
            tool_name=tool_name,
            tool_run_id=tool_run_id,
            run_id=tool_run_id,
            tool_meta={
                "name": tool_name,
                "metadata": dict(tool_metadata),
            },
            tool_inputs={"action": payload.get("action") or "skip"},
            metadata=tool_metadata,
            tool_output=tool_output,
            finish_reason="stop",
            timestamp_start=now,
            timestamp_finish=now,
        )
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

        try:
            event = NodeEvent(
                type=EventTypes.partial_message,
                stream_id=self.node_interface.stream_id,
                message_id=self.message_id,
                response_metadata={
                    "project_id": self.project_id,
                    "chat_project_id": self.chat_project_id,
                    "thinking_steps": [],
                    "tool_calls": {
                        tool_run_id: self.node_interface.decorate_tool_call_for_persistence(
                            tool_call.model_dump()
                        )
                    },
                    "additional_response_meta": {},
                },
                content=None,
                **self.node_interface.payload_additional_kwargs,
            ).model_dump_json()
            self.node_interface.event_node.emit(
                EVENTNODE_PARTIAL_RESPONSE_NAME, json.loads(event)
            )
        except Exception as exc:
            log.warning(
                "Failed to persist MCP auth decision tool step %s: %s",
                tool_run_id,
                exc,
            )

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

        # The SDK supervisor publishes one typed early pause event regardless of
        # guard kind. Split it here into the established transport contracts so
        # Core/UI keep using the same auth and HITL persistence/rendering paths.
        if name == "parallel_hitl_interrupt" and isinstance(data, dict):
            entries = data.get("hitl_interrupts") or [data.get("hitl_interrupt")]
            entries = [item for item in entries if isinstance(item, dict)]
            common = {
                "chat_project_id": self.chat_project_id,
                "root_thread_id": data.get("root_thread_id"),
                "resume_strategy": "supervised_child",
            }
            auth_entries = [
                item for item in entries
                if item.get("guardrail_type") == "mcp_auth"
            ]
            if auth_entries:
                # The SDK has converted the callback exception into an exact,
                # checkpoint-backed interrupt and the supervisor now owns it.
                # Do not let the callback cache survive until the outer invoke
                # returns: build_mcp_auth_pause_result() would otherwise emit a
                # second legacy UUID/tool-run authorization after the durable
                # child already consumed its structured decision.
                self.mcp_auth_pause_payload = None
                self.mcp_auth_pause_message = None
                self.mcp_auth_durable_interrupt_seen = True
                self.parallel_hitl_run_state[
                    "mcp_auth_durable_interrupt_seen"
                ] = True
            for item in auth_entries:
                self.node_interface.emit(
                    type=EventTypes.mcp_authorization_required,
                    content=item.get("message", "Toolkit authorization required"),
                    # Keep the checkpoint-owning leaf thread on the flattened
                    # authorization event.  ``data['thread_id']`` is the
                    # supervisor/root mailbox and must never replace it: doing
                    # so renders a SurnameResolver authorization at the root and
                    # makes Continue restart the orchestrator instead of
                    # resuming the exact child checkpoint.
                    response_metadata={**item, **common},
                )
            hitl_entries = [item for item in entries if item not in auth_entries]
            if hitl_entries:
                self.node_interface.emit(
                    type=EventTypes.agent_hitl_interrupt,
                    content=hitl_entries[0].get("message", "Awaiting human review..."),
                    response_metadata={
                        **common,
                        "message": hitl_entries[0].get("message"),
                        "hitl_interrupt": hitl_entries[0],
                        "hitl_interrupts": hitl_entries,
                    },
                )
            return

        if name == "mcp_auth_decision" and isinstance(data, dict):
            self._persist_mcp_auth_decision(data, metadata)
            return

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
                    if (
                        event_type_value == EventTypes.agent_parallel_hitl_state.value
                        and self.chat_project_id
                    ):
                        emit_payload = {
                            **payload,
                            "chat_project_id": self.chat_project_id,
                        }
                    self.node_interface.emit(
                        type=event_type_value, response_metadata=emit_payload
                    )

                # Place a consumed mid-turn injection in the turn's timeline, so it
                # renders among the tool/thinking pins at the point it was folded in
                # rather than only inside the user's (scrolled-away) bubble.
                if event_type_value == EventTypes.agent_midturn_injection_consumed.value:
                    self._persist_injection_marker(payload)

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
