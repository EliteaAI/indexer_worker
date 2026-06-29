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

""" Method for Application Agent """
from copy import deepcopy
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel as PydanticBaseModel

from pylon.core.tools import log
from pylon.core.tools import web

from ..utils.exceptions import InternalSDKError, PipelineConfigurationError
from ..utils.node_interface import EventTypes

# Import shared components
from .agent_common import (
    execution_error,
    _fetch_pgvector_connstr_with_retry,
    temp_elitea_client,
    fetch_langfuse_config,
    is_mcp_authorization_required_error,
    build_mcp_auth_pause_result,
    build_mcp_auth_required_result,
)
from ..utils.checkpoint_utils import (
    compute_pipeline_state_hash,
    reset_checkpoint_if_state_changed,
    delete_checkpoints_by_thread_ids,
)
from ..utils.agent_execution_common import (
    setup_memory,
    create_memory_saver,
    setup_event_node,
    create_elitea_client,
    create_node_interface,
    ensure_thread_id,
    create_callbacks,
    create_langfuse_callback_with_metadata,
    configure_checkpoint_resume,
    emit_response_events,
    with_tracing_span,
    build_success_result,
    prepare_invoke_input,
    extract_response_content,
    build_output_message,
    create_summarization_callbacks,
    get_child_dispatcher,
    detect_parked_dispatch,
    build_parked_result,
    build_child_launch_payloads,
    build_parent_reconcile_payload,
    apply_parallel_reconcile,
)
from ..utils.langfuse_callback import flush_langfuse_callback, langfuse_trace_context
from ..utils.image_helpers import resolve_filepath_images, resolve_generated_image_thumbnails
from ..utils.funcs import (
    _extract_mcp_server_url,
    _is_http_url,
    get_mcp_server_settings,
    normalize_mcp_server_url,
    expand_mcp_token_aliases,
)

from pydantic import ValidationError
from elitea_sdk.runtime.utils.mcp_oauth import (
    McpAuthorizationRequired,
    build_mcp_auth_decision_result,
    has_active_mcp_token,
)
from openai import InternalServerError

# Collect LLM authentication/authorization error types
_LLM_AUTH_ERRORS_APP = []
try:
    from anthropic import AuthenticationError as AnthropicAuthError
    _LLM_AUTH_ERRORS_APP.append(AnthropicAuthError)
except ImportError:
    pass
try:
    from openai import AuthenticationError as OpenAIAuthError
    _LLM_AUTH_ERRORS_APP.append(OpenAIAuthError)
except ImportError:
    pass
LLM_AUTH_ERRORS_APP = tuple(_LLM_AUTH_ERRORS_APP) if _LLM_AUTH_ERRORS_APP else None




def _mcp_discovery_url(server_url: str) -> str:
    """Return the MCP endpoint URL to use for tool discovery/auth challenge."""
    if not _is_http_url(server_url):
        return server_url
    # Use the URL as-is. normalize_mcp_server_url already converts the deprecated
    # /v1/sse to /v1/mcp/authv2 before this function is called, so we do not
    # reverse that mapping here.
    return server_url.strip()


def _build_mcp_server_alias_map(tool_configs: list) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Build alias maps for MCP URL resolution and toolkit metadata lookup."""
    alias_map: Dict[str, str] = {}
    alias_meta_map: Dict[str, Dict[str, Any]] = {}

    def _register(
        alias: Optional[str],
        url: Optional[str],
        toolkit_type: Optional[str],
        toolkit_name: Optional[str],
    ) -> None:
        if not alias or not url or not _is_http_url(url):
            return
        key = alias.strip().lower()
        alias_map[key] = normalize_mcp_server_url(url)
        alias_meta_map[key] = {
            **alias_meta_map.get(key, {}),
            "toolkit_type": toolkit_type,
            "tool_name": toolkit_name or alias,
        }

    for tool in tool_configs or []:
        if not isinstance(tool, dict):
            continue

        tool_type = str(tool.get("type") or "").strip()
        if not (tool_type == "mcp" or tool_type == "mcp_config" or tool_type.startswith("mcp_")):
            continue

        settings = tool.get("settings") if isinstance(tool.get("settings"), dict) else {}
        toolkit_name = str(tool.get("toolkit_name") or "").strip()
        aliases = {
            toolkit_name,
            str(settings.get("server_name") or "").strip(),
            tool_type,
        }
        if tool_type.startswith("mcp_") and tool_type != "mcp_config":
            aliases.add(tool_type[4:])

        direct_url = _extract_mcp_server_url(settings)
        if _is_http_url(direct_url):
            for alias in aliases:
                _register(alias, direct_url, tool_type, toolkit_name)
            continue

        for alias in list(aliases):
            if not alias:
                continue
            server_cfg = get_mcp_server_settings(alias) or {}
            cfg_url = _extract_mcp_server_url(server_cfg)
            _register(alias, cfg_url, tool_type, toolkit_name)

    return alias_map, alias_meta_map


def _make_mcp_auth_tools(
    declined_servers: list,
    tool_configs: Optional[list] = None,
    mcp_tokens: Optional[Dict[str, Any]] = None,
) -> list[StructuredTool]:
    """Create MCP auth tools aligned with sensitive-tool style guidance.

    Exposes:
    - `mcp_auth_control`: primary control tool with structured decision output
    - `request_mcp_authorization`: compatibility alias for legacy prompts/checkpoints
    """
    server_metadata: Dict[str, Dict[str, Any]] = {
        normalize_mcp_server_url(s["server_url"]): {
            **s,
            "server_url": normalize_mcp_server_url(s.get("server_url")),
        }
        for s in declined_servers
        if s.get("server_url")
    }
    alias_to_server_url, alias_to_tool_meta = _build_mcp_server_alias_map(tool_configs or [])

    def _decline_reason(meta: Dict[str, Any]) -> str:
        reason = str(meta.get("skip_reason") or meta.get("denial_reason") or "").strip()
        return reason or "user skipped MCP login for this run"

    server_list = "\n".join(
        (
            f"- {s.get('server_url', '')} ({s.get('tool_name', '')}) "
            f"reason: {_decline_reason(s)}"
        )
        for s in declined_servers
    )

    class _McpAuthControlInput(PydanticBaseModel):
        action: str = "authorize"
        server_url: str
        tool_name: Optional[str] = None
        reason: Optional[str] = None

    class _RequestMcpAuthorizationInput(PydanticBaseModel):
        server_url: str

    def _alias_candidates(server_url: str, tool_name: Optional[str]) -> list[str]:
        candidates = [server_url, tool_name or ""]
        normalized_tool_name = (tool_name or "").strip()
        if normalized_tool_name.startswith("mcp_authorize_"):
            candidates.append(normalized_tool_name[len("mcp_authorize_"):])
        return [candidate for candidate in candidates if str(candidate).strip()]

    def _resolve_server_meta(server_url: str, tool_name: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
        """Resolve (url, metadata) with exact match then fuzzy fallback."""
        server_url = (server_url or "").strip()
        if server_url in server_metadata:
            return server_url, server_metadata[server_url]

        for alias in _alias_candidates(server_url, tool_name):
            alias_key = str(alias).strip().lower()
            resolved_url = alias_to_server_url.get(alias_key)
            if resolved_url:
                resolved_meta = dict(alias_to_tool_meta.get(alias_key) or {})
                if not resolved_meta.get("tool_name"):
                    resolved_meta["tool_name"] = tool_name or alias
                return resolved_url, resolved_meta

        if _is_http_url(server_url):
            resolved_meta: Dict[str, Any] = {}
            for alias in _alias_candidates("", tool_name):
                alias_key = str(alias).strip().lower()
                if alias_key in alias_to_tool_meta:
                    resolved_meta = dict(alias_to_tool_meta.get(alias_key) or {})
                    break
            if not resolved_meta.get("tool_name"):
                resolved_meta["tool_name"] = tool_name or server_url
            return normalize_mcp_server_url(server_url), resolved_meta

        # Last-chance resolution for symbolic server aliases (for example
        # "atlassian3") that may not be present in tool-config alias map.
        # Query global MCP server settings directly and extract an HTTP URL.
        fallback_cfg = get_mcp_server_settings(server_url) or {}
        fallback_url = _extract_mcp_server_url(fallback_cfg)
        if _is_http_url(fallback_url):
            fallback_meta: Dict[str, Any] = {}
            for alias in _alias_candidates(server_url, tool_name):
                alias_key = str(alias).strip().lower()
                if alias_key in alias_to_tool_meta:
                    fallback_meta = dict(alias_to_tool_meta.get(alias_key) or {})
                    break
            if not fallback_meta.get("tool_name"):
                fallback_meta["tool_name"] = tool_name or server_url
            return normalize_mcp_server_url(fallback_url), fallback_meta

        needle = server_url.lower()
        tool_needle = (tool_name or "").lower().strip()
        for key, val in server_metadata.items():
            candidate_tool = (val.get("tool_name") or "").lower()
            key_l = key.lower()
            if needle and (needle in key_l or key_l in needle):
                return key, val
            if needle and candidate_tool and (needle in candidate_tool or candidate_tool in needle):
                return key, val
            if tool_needle and candidate_tool and (tool_needle in candidate_tool or candidate_tool in tool_needle):
                return key, val

        return server_url, {}

    def _mcp_auth_control(
        action: str,
        server_url: str,
        tool_name: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        normalized_action = (action or "authorize").strip().lower()
        resolved_url, meta = _resolve_server_meta(server_url, tool_name)
        resolved_tool_name = meta.get("tool_name") or (tool_name or "")
        toolkit_type = meta.get("toolkit_type") or "mcp"

        if normalized_action == "authorize":
            if not resolved_url:
                return build_mcp_auth_decision_result(
                    status="error",
                    server_url="",
                    tool_name=resolved_tool_name,
                    toolkit_type=toolkit_type,
                    message="No MCP server was provided. Choose a valid server and retry authorization.",
                    next_step="ask_user",
                    denial_reason=reason or None,
                )

            # Early return if the server was already declined in this conversation.
            # Do NOT call discover_mcp_tools — that would raise McpAuthorizationRequired
            # again and trigger another auth dialog even though the user already skipped.
            if resolved_url in server_metadata:
                decline_reason = _decline_reason(meta)
                return build_mcp_auth_decision_result(
                    status="declined",
                    server_url=resolved_url,
                    tool_name=resolved_tool_name,
                    toolkit_type=toolkit_type,
                    message=(
                        "This MCP server was already skipped in this conversation. "
                        f"Skip reason: {decline_reason}. "
                        "Do NOT request authorization again. "
                        "Continue the task without this capability, or explain to the user "
                        "why the task cannot be completed."
                    ),
                    next_step="use_other_tool",
                    denial_reason=decline_reason,
                    resource_metadata_url=meta.get("resource_metadata_url"),
                    www_authenticate=meta.get("www_authenticate"),
                )

            # Follow the same MCP auth discovery process used by other flows
            # (sync tools / toolkit discovery) so metadata shape is consistent.
            from ..utils.funcs import dev_reload_sdk
            dev_reload_sdk('elitea_sdk.runtime.utils')
            from elitea_sdk.runtime.utils.mcp_tools_discovery import discover_mcp_tools

            normalized_url = normalize_mcp_server_url(resolved_url)
            discovery_url = _mcp_discovery_url(normalized_url)

            # Extract OAuth token from mcp_tokens if available for this server.
            # mcp_tokens is already expanded with URL aliases by expand_mcp_token_aliases.
            _auth_headers: dict = {}
            _token_session_id = None
            if mcp_tokens:
                for _key in [k for k in [normalized_url, discovery_url] if k]:
                    _token_data = mcp_tokens.get(_key)
                    if _token_data:
                        if isinstance(_token_data, dict):
                            _access_token = _token_data.get('access_token')
                            _token_session_id = _token_data.get('session_id')
                        else:
                            _access_token = _token_data
                        if _access_token:
                            _auth_headers['Authorization'] = f'Bearer {_access_token}'
                        break

            try:
                discover_mcp_tools(
                    url=discovery_url,
                    headers=_auth_headers,
                    timeout=30,
                    session_id=_token_session_id,
                    ssl_verify=True,
                )
            except McpAuthorizationRequired as exc:
                setattr(exc, "server_url", normalized_url)
                if not getattr(exc, "tool_name", None):
                    exc.tool_name = resolved_tool_name
                if not getattr(exc, "toolkit_type", None):
                    exc.toolkit_type = meta.get("toolkit_type")
                raise
            except Exception as exc:
                if is_mcp_authorization_required_error(exc):
                    setattr(exc, "server_url", normalized_url)
                    if not getattr(exc, "tool_name", None):
                        setattr(exc, "tool_name", resolved_tool_name)
                    if not getattr(exc, "toolkit_type", None):
                        setattr(exc, "toolkit_type", meta.get("toolkit_type"))
                    raise
                log.warning("MCP auth discovery failed for %s: %s", normalized_url, exc)
                return build_mcp_auth_decision_result(
                    status="error",
                    server_url=normalized_url,
                    tool_name=resolved_tool_name,
                    toolkit_type=meta.get("toolkit_type") or toolkit_type,
                    message=(
                        "Unable to start MCP authorization flow. "
                        f"Details: {str(exc)}"
                    ),
                    next_step="ask_user",
                    denial_reason=reason or None,
                )

            # Discovery succeeded. If we injected an OAuth token, that means the
            # token was accepted — real MCP tools are now loaded. Direct the LLM
            # to proceed with the original tool call instead of skipping it.
            if _auth_headers.get("Authorization"):
                return build_mcp_auth_decision_result(
                    status="authorized",
                    server_url=normalized_url,
                    tool_name=resolved_tool_name,
                    toolkit_type=meta.get("toolkit_type") or toolkit_type,
                    message=(
                        "MCP authorization succeeded. The server is accessible with the provided credentials. "
                        f"Proceed immediately with the original tool call using the {resolved_tool_name or toolkit_type} "
                        "tools that are now available."
                    ),
                    next_step="use_mcp_tool",
                    denial_reason=reason or None,
                )
            # No token was injected — server did not require auth.
            return build_mcp_auth_decision_result(
                status="not_needed",
                server_url=normalized_url,
                tool_name=resolved_tool_name,
                toolkit_type=meta.get("toolkit_type") or toolkit_type,
                message="MCP authorization challenge was not required for this server.",
                next_step="respond_without_tool",
                denial_reason=reason or None,
            )

        if normalized_action == "status":
            status = "declined" if resolved_url in server_metadata else "not_needed"
            next_step = "use_other_tool" if status == "declined" else "respond_without_tool"
            decline_reason = _decline_reason(meta) if status == "declined" else None
            message = (
                "This MCP server was already skipped in this conversation. "
                f"Skip reason: {decline_reason}. "
                "Do NOT request authorization again. "
                "Continue the task without this capability, or explain to the user "
                "why the task cannot be completed."
                if status == "declined"
                else "No declined MCP auth state is tracked for this server in the current conversation."
            )
            return build_mcp_auth_decision_result(
                status=status,
                server_url=resolved_url or server_url,
                tool_name=resolved_tool_name,
                toolkit_type=toolkit_type,
                message=message,
                next_step=next_step,
                denial_reason=reason or decline_reason,
                resource_metadata_url=meta.get("resource_metadata_url"),
                www_authenticate=meta.get("www_authenticate"),
            )

        if normalized_action == "explain_skip":
            decline_reason = reason or _decline_reason(meta)
            return build_mcp_auth_decision_result(
                status="skipped",
                server_url=resolved_url or server_url,
                tool_name=resolved_tool_name,
                toolkit_type=toolkit_type,
                message=(
                    "Authorization was declined for THIS invocation. Do not retry the same blocked call now. "
                    "Continue with other available tools. If no alternative remains and the task fails, "
                    f"explicitly mention this skip reason: {decline_reason}."
                ),
                next_step="use_other_tool",
                denial_reason=decline_reason,
                resource_metadata_url=meta.get("resource_metadata_url"),
                www_authenticate=meta.get("www_authenticate"),
            )

        return build_mcp_auth_decision_result(
            status="error",
            server_url=resolved_url or server_url,
            tool_name=resolved_tool_name,
            toolkit_type=toolkit_type,
            message=f"Unsupported action '{normalized_action}'. Use one of: authorize, status, explain_skip.",
            next_step="ask_user",
            denial_reason=reason or None,
        )

    def _request_mcp_authorization(server_url: str) -> str:
        # Legacy compatibility alias: same behavior as mcp_auth_control(action='authorize').
        return _mcp_auth_control(action="authorize", server_url=server_url)

    mcp_auth_control_tool = StructuredTool.from_function(
        func=_mcp_auth_control,
        name="mcp_auth_control",
        description=(
            "Control MCP authorization flow with structured decisions. "
            "Use action='authorize' when an MCP capability is required and user auth is needed. "
            "Use action='status' to check declined state in this conversation. "
            "Use action='explain_skip' to generate structured skip guidance and continue with alternatives.\n"
            "This is NOT a stop signal by itself: if auth is not granted, continue with other tools when possible.\n"
            "If a required MCP capability was skipped and no alternative works, the assistant response must "
            "explicitly state the skip reason.\n"
            f"Known declined servers in this conversation:\n{server_list or '- (none)'}"
        ),
        args_schema=_McpAuthControlInput,
        handle_tool_error=False,
    )

    legacy_alias_tool = StructuredTool.from_function(
        func=_request_mcp_authorization,
        name="request_mcp_authorization",
        description=(
            "Legacy alias for MCP authorization. Prefer mcp_auth_control(action='authorize', server_url=...). "
            "Calling this tool requests MCP OAuth authorization and shows the user auth dialog."
        ),
        args_schema=_RequestMcpAuthorizationInput,
        handle_tool_error=False,
    )

    return [mcp_auth_control_tool, legacy_alias_tool]


def _has_mcp_toolkits(tool_configs: list) -> bool:
    """Return True when application tool configs include MCP toolkits."""
    for tool in tool_configs or []:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "")
        if tool_type == "mcp" or tool_type == "mcp_config" or tool_type.startswith("mcp_"):
            return True
    return False


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource for Application Agent

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def indexer_agent(  # pylint: disable=R0914,W1113
            self,
            stream_id: Optional[str] = None,
            message_id: Optional[str] = None,
            *args,
            **kwargs,
    ):
        """ Run task target """
        self.indexer_enable_logging()
        #
        log.debug(f'indexer_agent start stream_id={stream_id}, message_id={message_id}')
        #
        try:
            # Extract client args - will be used to create EliteAClient after fork
            client_args = kwargs.get("llm", {}).get("kwargs", {})
            api_token = kwargs.get("api_token", client_args.get("api_key", None))
            api_extra_headers = kwargs.get("api_extra_headers", client_args.get("api_extra_headers", {}))

            # Fetch pgvector connection string for memory (PostgresSaver) and cleanup using context manager
            with temp_elitea_client(client_args, api_token, api_extra_headers) as temp_client:
                pgvector_connstr = _fetch_pgvector_connstr_with_retry(
                    temp_client, project_id=client_args.get("project_id")
                )

            # Setup memory configuration
            memory_type, memory_config = setup_memory(
                self.descriptor.config,
                pgvector_connstr
            )

            # Create memory saver and execute task
            memory, cleanup = create_memory_saver(memory_type, memory_config)
            try:
                return self._indexer_agent_task(
                    memory,
                    memory_config,
                    client_args,
                    api_token,
                    api_extra_headers,
                    *args,
                    stream_id=stream_id,
                    message_id=message_id,
                    **kwargs,
                )
            finally:
                cleanup()
        #
        except:  # pylint: disable=W0702
            log.exception("indexer_agent failed to start")
            raise

    @web.method()
    def _indexer_agent_task(  # pylint: disable=R0914,R0915
            self,
            memory,
            memory_config,
            client_args,
            api_token,
            api_extra_headers,
            *args,
            stream_id: Optional[str] = None,
            message_id: Optional[str] = None,
            **kwargs,
    ):
        _ = args
        import tasknode_task  # pylint: disable=E0401,C0415

        # Extract traceparent for distributed tracing
        traceparent = tasknode_task.meta.get('traceparent') if tasknode_task.meta else None

        # Execute with optional tracing span
        return with_tracing_span(
            traceparent,
            'indexer_agent',
            stream_id,
            message_id,
            str(tasknode_task.meta.get('project_id', '')),
            self._indexer_agent_task_inner,
            memory, memory_config, client_args, api_token, api_extra_headers,
            stream_id=stream_id, message_id=message_id, **kwargs
        )

    @web.method()
    def _indexer_agent_task_inner(
            self,
            memory,
            memory_config,
            client_args,
            api_token,
            api_extra_headers,
            stream_id: Optional[str] = None,
            message_id: Optional[str] = None,
            **kwargs,
    ):
        """Inner agent task execution (separated for tracing)."""
        import tasknode_task  # pylint: disable=E0401,C0415
        from datetime import datetime, timezone

        # Setup event node
        local_event_node = setup_event_node(tasknode_task.multiprocessing_context)

        # Create EliteAClient AFTER fork
        client = create_elitea_client(client_args, api_token, api_extra_headers)

        should_continue = kwargs.get('should_continue', False)
        hitl_resume = kwargs.get('hitl_resume', False)
        hitl_action = kwargs.get('hitl_action', 'approve')
        hitl_value = kwargs.get('hitl_value', '')
        # Parallel sub-agent fan-out (#4993): per-child decisions for resuming
        # multiple paused sub-agents in one turn (keyed by tool_call_id).
        hitl_decisions = kwargs.get('hitl_decisions') or None
        # Parallel sub-agent reconcile (#4993 Track 2): pylon_main re-invokes the
        # parked parent with this epoch once all children settle. It is a resume
        # of the existing parent checkpoint — treat like should_continue so the
        # checkpoint-reset logic does not wipe the parked state we must read back.
        parallel_reconcile = kwargs.get('parallel_reconcile') or None
        # Set when THIS run paused at a HITL node. Surfaced in the task result so a
        # parked fan-out child's HITL pause is distinguishable from a completion —
        # the reconcile gate must NOT treat a paused child as terminal (#4993).
        paused_hitl_interrupt = None

        # Fetch Langfuse config for tracing
        langfuse_config = fetch_langfuse_config(client)
        langfuse_client = None
        langfuse_callback = None
        langfuse_trace_attrs = None

        # Create node interface
        node_interface = create_node_interface(
            local_event_node,
            stream_id,
            message_id,
            tasknode_task.meta
        )

        node_interface.emit(type=EventTypes.agent_start)

        execution_start_time = datetime.now(tz=timezone.utc)

        # Extract common parameters
        chat_history = kwargs.get("chat_history", [])
        user_input = kwargs.get("user_input")
        thread_id = kwargs.get("thread_id")
        conversation_id = kwargs.get("conversation_id")
        supports_vision = kwargs.get("supports_vision", True)

        # Ensure thread_id is valid
        thread_id = ensure_thread_id(thread_id, conversation_id)

        try:
            version_details = kwargs.get("application", {}).get('version_details', {})
            # version details is missed for requests from API endpoint
            meta = version_details.get('meta', {}) if version_details else kwargs.get("meta", {})
            is_regenerate = kwargs.get('is_regenerate', False)

            # Reset stale LangGraph checkpoint when pipeline state defaults have changed.
            # Skip during HITL/continue flows to avoid disrupting an in-progress run.
            current_state_hash = None
            if not hitl_resume and not should_continue and not parallel_reconcile:
                current_state_hash = compute_pipeline_state_hash(version_details)
                if is_regenerate and current_state_hash is not None:
                    # On user-initiated regenerate: unconditionally clear the pipeline checkpoint
                    # so LangGraph restarts from scratch instead of resuming from interruption.
                    delete_checkpoints_by_thread_ids(memory_config, [thread_id])
                else:
                    reset_checkpoint_if_state_changed(
                        memory, thread_id, current_state_hash, memory_config
                    )

            # Get error_handling_enabled from kwargs (runtime override) or config (default)
            exception_handling_enabled = kwargs.get(
                "exception_handling_enabled",
                self.descriptor.config.get("exception_handling_enabled", False)
            )
            log.debug('exception_handling_enabled "%s"', exception_handling_enabled)

            # Prepare context_settings with summarization callbacks
            context_settings = kwargs.get("context_settings", {})
            context_settings['callbacks'] = create_summarization_callbacks(node_interface)

            # Always provide mcp_auth_control in chat runs that include MCP toolkits.
            # This guarantees a single, explicit auth flow for all MCP tool usage.
            user_declined = kwargs.get('user_declined_mcp_servers') or []
            app_tool_configs = (
                (kwargs.get("application") or {}).get("version_details", {}).get("tools")
                or kwargs.get("tools")
                or []
            )
            has_mcp_toolkits = _has_mcp_toolkits(app_tool_configs)
            raw_mcp_tokens = kwargs.get("mcp_tokens", None)
            mcp_tokens = expand_mcp_token_aliases(raw_mcp_tokens)
            additional_tools = (
                _make_mcp_auth_tools(user_declined, app_tool_configs, mcp_tokens=mcp_tokens)
                if has_mcp_toolkits
                else []
            )

            # Create application agent
            _child_dispatcher = get_child_dispatcher(self.descriptor.config)
            elitea_callback = None  # guard: McpAuthorizationRequired may be raised before create_callbacks
            agent_executor = client.application(
                application_id=kwargs.get("application", {})["id"],
                application_version_id=kwargs.get("application", {})["version_id"],
                memory=memory,
                application_variables=kwargs.get("application", {}).get('variables'),
                version_details=deepcopy(version_details),
                mcp_tokens=mcp_tokens,
                conversation_id=conversation_id,
                ignored_mcp_servers=kwargs.get("ignored_mcp_servers", None),
                exception_handling_enabled=exception_handling_enabled,
                context_settings=context_settings,
                auto_approve_sensitive_actions=kwargs.get("auto_approve_sensitive_actions", False),
                openai_compatible=client_args.get('openai_compatible', False),
                # Parallel sub-agent dispatch seam (#4993 Track 2): non-None when
                # the operator enabled parallel_subagent_dispatch — switches the
                # SDK from in-process gather to park-by-returning. None = Track 1.
                child_dispatcher=_child_dispatcher,
                tools=additional_tools if additional_tools else None,
            )

            # Create callbacks
            elitea_callback, elitea_custom_callback = create_callbacks(
                node_interface,
                thread_id,
                message_id,
                tasknode_task.meta,
                tasknode_task.id,
                debug=kwargs.get("debug", False)
            )

            # Durable fan-out child (#4993 Track 2): give the callback this child's
            # REAL kind (pipeline vs agent) from its own version_details so a
            # pipeline child's self-named chips render the pipeline (flow) icon
            # instead of the model-provider-derived application icon. Only set for
            # a parked sub-agent; harmless (None) for an ordinary run.
            if tasknode_task.meta.get("subagent_name"):
                elitea_callback.subagent_agent_type = version_details.get("agent_type")

            # Create Langfuse callback
            application_name = kwargs.get("application", {}).get("name", "agent")
            langfuse_client, langfuse_callback, langfuse_trace_attrs = create_langfuse_callback_with_metadata(
                langfuse_config,
                application_name,
                thread_id,
                message_id,
                tasknode_task.meta
            )

            callbacks = [elitea_callback, elitea_custom_callback]
            if langfuse_callback:
                callbacks.append(langfuse_callback)

            # Resolve filepath: image URLs in current turn — single S3 read per image
            user_input, image_thumbnails = resolve_filepath_images(user_input, client)

            # Extract pipeline attachment filepaths from user_input chunks before LLM invocation.
            # The 'attachment_filepath' key is a structural marker set by pipeline_process() in
            # pylon_main attachments.py. Strip it here so clean {type, text} dicts reach the
            # model API, then inject filepaths into the pipeline graph state as input_attachments.
            attachment_filepaths = []
            if isinstance(user_input, list):
                for chunk in user_input:
                    if isinstance(chunk, dict) and 'attachment_filepath' in chunk:
                        attachment_filepaths.append(chunk.pop('attachment_filepath'))

            user_message_content = hitl_value if hitl_resume and hitl_action == 'edit' else user_input
            user_message = HumanMessage(content=user_message_content or '')
            log.debug(f'invoke payload thread_id={thread_id}')

            # Safe nested access - handle None values at each level
            application = kwargs.get('application') or {}
            version_details = application.get('version_details') or {}
            _all_tools = kwargs.get('tools') or version_details.get('tools') or []
            invoke_input = prepare_invoke_input(
                chat_history,
                user_message,
                conversation_id,
                include_attachment_system_message=any(
                    t.get('name') == 'Attachments' for t in _all_tools
                ),
                model_name=client_args.get('model', ''),
                supports_vision=supports_vision,
            )

            if attachment_filepaths:
                invoke_input['input_attachments'] = attachment_filepaths
                
            invoke_config = {
                'callbacks': callbacks,
                'configurable': {'thread_id': thread_id},
                'recursion_limit': meta.get("step_limit", 25),
                'metadata': (
                    {'pipeline_state_defaults_hash': current_state_hash}
                    if not hitl_resume and not should_continue and current_state_hash
                    else {}
                ),
            }

            invoke_config['configurable']['invoked_skills'] = kwargs.get('invoked_skills') or []

            # HITL resume takes precedence over generic checkpoint continuation.
            if hitl_resume:
                invoke_input['hitl_resume'] = True
                invoke_input['hitl_action'] = hitl_action
                invoke_input['hitl_value'] = hitl_value
                # Parallel multi-interrupt resume (#4993): the SDK routes each
                # decision to its paused sub-agent by tool_call_id.
                if hitl_decisions:
                    invoke_input['hitl_decisions'] = hitl_decisions
                    log.info(f'[HITL] Resume with {len(hitl_decisions)} parallel decision(s)')
                log.info(f'[HITL] Resume action: {invoke_input["hitl_action"]}')
            elif should_continue:
                invoke_input, invoke_config = configure_checkpoint_resume(
                    agent_executor,
                    thread_id,
                    kwargs.get('checkpoint_id'),
                    invoke_input,
                    invoke_config,
                    user_input=user_input,
                    user_declined_mcp_servers=kwargs.get('user_declined_mcp_servers') or [],
                    mcp_tokens=mcp_tokens,
                )

            # Parallel sub-agent reconcile (#4993 Track 2): pylon_main re-invokes
            # the parked parent with this epoch once every child task is
            # terminal. The SDK reads each child's own checkpoint, appends one
            # ToolMessage per child, and resumes the agent node to synthesize the
            # final answer. Independent of hitl_resume/should_continue.
            apply_parallel_reconcile(invoke_input, kwargs)

            # Invoke the agent executor with Langfuse trace context
            with langfuse_trace_context(langfuse_trace_attrs, langfuse_client):
                response = agent_executor.invoke(invoke_input, invoke_config)

            # Callback-path MCP auth interruption: pause immediately and do not
            # emit regular response completion events for this run.
            pause_result = build_mcp_auth_pause_result(elitea_callback, chat_history)
            if pause_result:
                return pause_result

            # Parallel sub-agent park (#4993 Track 2): the parent fanned out to
            # 2+ sub-agents and returned parked (specs written, children NOT run
            # in-process). Surface the dispatch specs in the task result so
            # pylon_main's stopped-handler can launch one durable child per spec;
            # skip the normal response events (no final answer exists yet).
            _parked = detect_parked_dispatch(response)
            if _parked is not None:
                # Enrich each spec with a self-contained child launch payload
                # (inherits this parent's valid token/base_url; child model +
                # tools come from the sub-agent's own version_details). Also
                # carry the parent's own reconcile re-invoke payload so pylon_main
                # can replay this same parent (same token, same thread) once the
                # children settle.
                _parked['parallel_dispatch'] = build_child_launch_payloads(
                    kwargs, _parked['parallel_dispatch']
                )
                _parked['reconcile_payload'] = build_parent_reconcile_payload(kwargs)
                log.info(
                    f'[PARALLEL] parent parked, dispatching '
                    f'{len(_parked["parallel_dispatch"])} child(ren) thread_id={thread_id}'
                )
                # The finally block flushes Langfuse and stops the fork-pool
                # event node on this return — no manual cleanup needed here.
                return build_parked_result(_parked, stream_id, message_id)

            # Extract and normalize response content using unified parsing
            response_content = extract_response_content(response, response_format='output')
            output = build_output_message(response_content)

            # Durable fan-out child (#4993 Track 2): emit the PARENT's bare
            # sub-agent invocation chip — the one the parked orchestrator never
            # produced because it returned specs instead of running the sub-agent
            # tool in-process. In the sequential path this chip is born from the
            # parent's on_tool_end and carries the task AND the sub-agent's final
            # answer (group 568). Reproduce it here from THIS child's end-of-run
            # state, now that `output` (the final answer) exists. Only on genuine
            # completion: skip while paused at HITL (no final answer yet) — the
            # reconcile re-run will emit it once the child truly finishes.
            if (
                tasknode_task.meta.get("subagent_name")
                and not response.get("hitl_interrupt")
            ):
                _task_text = user_input if isinstance(user_input, str) else None
                elitea_callback.emit_subagent_invocation_chip(
                    task_text=_task_text,
                    response=response,
                    agent_type=version_details.get("agent_type"),
                )

            # Extract context info (includes summarization details when summarization occurred)
            context_info = response.get('context_info')

            # Resolve thumbnails for tool-generated images and merge with user-upload thumbnails
            resolve_generated_image_thumbnails(elitea_custom_callback, image_thumbnails, client)

            # Emit response events
            application_details = kwargs.get("application", {})
            total_tokens_in, total_tokens_out, thread_id_response = emit_response_events(
                node_interface,
                response,
                output,
                thread_id,
                message_id,
                elitea_callback,
                elitea_custom_callback,
                tasknode_task.meta,
                application_details,
                chat_history,
                user_message,
                should_continue,
                hitl_resume=hitl_resume,
                hitl_action=hitl_action,
                hitl_value=hitl_value,
                image_thumbnails=image_thumbnails,
                context_info=context_info,
                invoked_skills=invoke_config['configurable'].get('invoked_skills'),
            )

            # Capture a HITL pause so the final task result carries it: the
            # reconcile gate keys off this to keep a paused child OPEN (#4993).
            paused_hitl_interrupt = response.get('hitl_interrupt')

        except InternalSDKError as e:
            return execution_error(
                node_interface, user_input, chat_history,
                f"InternalSDKError on user input",
                thread_id, message_id, tasknode_task.meta,
                human_readable=f"Internal SDK error occurred while processing your request, {e}",
                execution_start_time=execution_start_time
            )
        except ValidationError:
            return execution_error(
                node_interface, user_input, chat_history, f"ValidationError on user input: {user_input}",
                thread_id, message_id, tasknode_task.meta,
                human_readable="A validation error occurred while processing your request",
                execution_start_time=execution_start_time
            )
        except AssertionError:
            return execution_error(
                node_interface, user_input, chat_history, f"AssertionError on user input: {user_input}",
                thread_id, message_id, tasknode_task.meta,
                human_readable="Assertion error occurred while processing your request",
                execution_start_time=execution_start_time
            )
        except PipelineConfigurationError as e:
            return execution_error(
                node_interface, user_input, chat_history,
                f"PipelineConfigurationError: {e}",
                thread_id, message_id, tasknode_task.meta,
                human_readable=str(e),
                execution_start_time=execution_start_time
            )
        except ValueError:
            return execution_error(
                node_interface, user_input, chat_history,
                f"Seems like your agent is missconfigured on user input: {user_input}",
                thread_id, message_id, tasknode_task.meta,
                human_readable="Seems like your agent is configured incorrectly",
                execution_start_time=execution_start_time
            )
        except McpAuthorizationRequired as e:
            pause_result = build_mcp_auth_pause_result(elitea_callback, chat_history, fallback_error=str(e))
            if pause_result:
                return pause_result
            return build_mcp_auth_required_result(
                node_interface,
                e,
                tasknode_task.meta.get('chat_project_id'),
                chat_history,
            )
        except InternalServerError:
            return execution_error(
                node_interface, user_input, chat_history,
                f"OpenAI Responded with Internal Server Error on User Input: {user_input}",
                thread_id, message_id, tasknode_task.meta,
                human_readable="The AI service is currently experiencing issues",
                execution_start_time=execution_start_time
            )
        except Exception as e:
            # Dev-reload-safe fallback: class identity can differ across reloaded SDK modules.
            if is_mcp_authorization_required_error(e):
                return build_mcp_auth_required_result(
                    node_interface,
                    e,
                    tasknode_task.meta.get('chat_project_id'),
                    chat_history,
                )

            # Check for LLM authentication/authorization errors (model access denied, invalid keys, etc.)
            if LLM_AUTH_ERRORS_APP and isinstance(e, LLM_AUTH_ERRORS_APP):
                error_body = getattr(e, 'body', {}) or {}
                error_detail = error_body.get('error', {}) if isinstance(error_body, dict) else {}
                error_type = error_detail.get('type', '') if isinstance(error_detail, dict) else ''
                if error_type == 'team_model_access_denied':
                    human_msg = (
                        f"The selected model is not available for your team. "
                        f"Please choose a different model in the chat settings. "
                        f"Details: {error_detail.get('message', str(e))}"
                    )
                else:
                    human_msg = (
                        f"Authentication error with the AI provider: {e}. "
                        f"Please check your model configuration and API credentials."
                    )
                return execution_error(
                    node_interface, user_input, chat_history,
                    f"LLM AuthenticationError on user input",
                    thread_id, message_id, tasknode_task.meta,
                    human_readable=human_msg,
                    execution_start_time=execution_start_time
                )
            return execution_error(
                node_interface, user_input, chat_history, f"Error on user input: {user_input}",
                thread_id, message_id, tasknode_task.meta,
                human_readable="An unexpected error occurred while processing your request",
                execution_start_time=execution_start_time
            )
        finally:
            # Flush any buffered streamed text before tearing down the event node.
            # NodeEventInterface coalesces agent_llm_chunk token-deltas into a
            # buffer that only empties on flush(); stopping the node without this
            # drops the last partial tokens — especially on the early parked-parent
            # return path (#4993). Mirrors indexer_predict_agent's teardown.
            node_interface.flush()

            # Flush Langfuse traces
            flush_langfuse_callback(langfuse_client, langfuse_callback)

            if tasknode_task.multiprocessing_context == "fork":
                local_event_node.stop()

        return_chat_history = kwargs.get('return_chat_history', False)
        return build_success_result(chat_history, elitea_callback, total_tokens_in, total_tokens_out, context_info, return_chat_history=return_chat_history, hitl_interrupt=paused_hitl_interrupt)
