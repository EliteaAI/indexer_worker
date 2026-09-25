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

"""Method for Syncing/Fetching Tools from Remote MCP Server"""
import json
import traceback
from typing import Optional, Dict, Any, List

from pylon.core.tools import log
from pylon.core.tools import web

from tools import worker_core

from ..utils.funcs import normalize_mcp_auth_metadata_urls, normalize_mcp_server_url, backfill_mcp_auth_metadata
from ..utils.node_interface import NodeEventInterface, EventTypes, NodeEvent

# Import shared components from the agent common module
from .agent_common import (
    EVENTNODE_EVENT_NAME,
    EVENTNODE_FULL_RESPONSE_NAME,
)


CACHE_NOT_RETIRED_WARNING = (
    "Tools were fetched, but cached tool lists could not be retired; "
    "runs may use the previous list until the Cache TTL expires."
)


def _describe_error_without_sdk(exc: Exception, headers: Optional[Dict[str, Any]] = None) -> str:
    return str(exc)


def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """Safely serialize data to JSON string."""
    try:
        return json.dumps(data, indent=indent)
    except (TypeError, ValueError) as e:
        log.warning(f"JSON serialization failed: {e}, falling back to str()")
        return str(data)


def _sync_tools_failed(  # pylint: disable=R0913
    e, describe_error, connection_headers, normalized_url, stream_id, message_id, local_event_node,
):
    user_error_message = describe_error(e, connection_headers)
    error_msg = f"Failed to sync MCP tools: {user_error_message}"

    log.error(f"{error_msg}\n{traceback.format_exc()}")
    
    error_event = NodeEvent(
        type=EventTypes.agent_exception,
        stream_id=stream_id,
        message_id=message_id,
        content=error_msg,
        response_metadata={
            'error': error_msg,
            'server_url': normalized_url,
        }
    ).model_dump_json()
    error_event = json.loads(error_event)
    local_event_node.emit(EVENTNODE_FULL_RESPONSE_NAME, error_event)
    
    return {
        'success': False,
        'error': error_msg,
        'server_url': normalized_url,
    }


class Method:
    @web.method("indexer_mcp_sync_tools")
    def indexer_mcp_sync_tools(
        self,
        stream_id: str,
        message_id: str,
        url: str,
        project_id: int,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        mcp_tokens: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        ssl_verify: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Sync/fetch tools from a remote MCP server.
        
        This method discovers available tools from a remote MCP server.
        If the server requires OAuth authorization, it emits an
        'mcp_authorization_required' socket event with the OAuth metadata.
        
        Args:
            stream_id: Stream ID for socket communication
            message_id: Message ID for tracking
            url: MCP server HTTP URL
            project_id: Project ID
            headers: HTTP headers for authentication
            timeout: Request timeout in seconds
            mcp_tokens: MCP OAuth tokens (keyed by server URL)
            user_id: User ID for monitoring
            
        Returns:
            Dictionary with tools list or authorization requirement
        """
        import tasknode_task  # pylint: disable=E0401,C0415
        #
        normalized_url = normalize_mcp_server_url(url)
        log.debug(f"MCP sync tools task started: url={normalized_url}, project_id={project_id}")

        # Set up event node for socket communication
        if tasknode_task.multiprocessing_context == "fork":
            local_event_node = worker_core.event_node.clone()
            local_event_node.start()
        else:
            local_event_node = worker_core.event_node
        
        # Create node interface for emitting events
        node_interface = NodeEventInterface(
            local_event_node,
            EVENTNODE_EVENT_NAME,
            stream_id=stream_id,
            message_id=message_id,
        )

        from ..utils.funcs import dev_reload_sdk, is_mcp_authorization_required_error

        connection_headers = {}
        describe_error = _describe_error_without_sdk
        try:
            # The SDK imports sit inside the handler's reach: on SDK skew an ImportError must
            # become a "Failed to sync" response, not a raw 500. The auth-required check goes
            # by class name for the same reason, and because dev reload re-creates the class.
            dev_reload_sdk('elitea_sdk.runtime.utils')
            from elitea_sdk.runtime.utils.mcp_tools_discovery import discover_mcp_tools
            from elitea_sdk.runtime.utils.mcp_oauth import (
                canonical_resource,
                drop_unusable_authorization,
                extract_user_friendly_mcp_error,
                has_authorization_on_the_wire,
                has_configured_authorization,
                merge_oauth_authorization,
            )
            from elitea_sdk.runtime.utils.mcp_discovery_cache import invalidate_server_discovery
            describe_error = extract_user_friendly_mcp_error

            connection_headers = dict(headers or {})
            configured_auth = has_configured_authorization(connection_headers)
            access_token = None
            session_id = None

            if mcp_tokens and not configured_auth:
                server_key = canonical_resource(normalized_url)
                original_server_key = canonical_resource(url)
                # can be None or type for pre-built mcp, e.g. "mcp_github"
                toolkit_type = kwargs.get('toolkit_type')
                log.debug(f"Looking for token with server_key: {server_key} or toolkit_type: {toolkit_type}")
                log.debug(f"Available mcp_tokens keys: {list(mcp_tokens.keys())}")

                token_data = mcp_tokens.get(server_key) if not toolkit_type else mcp_tokens.get(toolkit_type)
                if not token_data and original_server_key != server_key:
                    token_data = mcp_tokens.get(original_server_key)
                # Try exact URL match if canonical didn't work
                if not token_data:
                    token_data = mcp_tokens.get(url)
                    if token_data:
                        log.debug(f"Found token using exact URL match: {url}")

                if token_data:
                    access_token = token_data.get('access_token')
                    session_id = token_data.get('session_id')
                    if session_id:
                        log.debug(f"Using session_id for MCP server: {server_key}")
                else:
                    log.warning(f"No token found for server_key: {server_key} or url: {url}")
            elif mcp_tokens:
                log.info(f"Configured Authorization header present for {normalized_url}; skipping OAuth token lookup")

            connection_headers, oauth_token_injected = merge_oauth_authorization(connection_headers, access_token)
            if oauth_token_injected:
                log.debug(f"Added OAuth token for MCP server: {normalized_url}")
            else:
                connection_headers = drop_unusable_authorization(connection_headers)

            # Discover tools from the MCP server
            log.debug(f"Discovering tools from MCP server: {normalized_url} (ssl_verify={ssl_verify})")
            tools_list = discover_mcp_tools(
                url=normalized_url,
                headers=connection_headers,
                timeout=timeout,
                session_id=session_id,
                ssl_verify=ssl_verify,
                configured_auth=has_authorization_on_the_wire(connection_headers, oauth_token_injected),
            )
            # Load Tools is the explicit refresh: no user's next run may serve the list it replaced
            cache_retired = invalidate_server_discovery(normalized_url)

            log.debug(f"Successfully discovered {len(tools_list)} tools from {normalized_url}")
            
            # Build success response
            result = {
                'success': True,
                'tools': tools_list,
                'count': len(tools_list),
                'server_url': normalized_url,
            }
            
            # Build response metadata
            response_metadata = {
                'tool_output': tools_list,
                'success': True,
                'count': len(tools_list),
                'server_url': normalized_url,
            }
            if not cache_retired:
                log.warning(f"Tools fetched from {normalized_url} but cached tool lists could not be retired")
                result['warning'] = CACHE_NOT_RETIRED_WARNING
                response_metadata['warning'] = CACHE_NOT_RETIRED_WARNING
            
            # Emit success response via socket
            response_event = NodeEvent(
                type=EventTypes.agent_response,
                stream_id=stream_id,
                message_id=message_id,
                content=safe_json_dumps(result),
                response_metadata=response_metadata
            ).model_dump_json()
            response_event = json.loads(response_event)
            local_event_node.emit(EVENTNODE_FULL_RESPONSE_NAME, response_event)
            
            return result
        
        except Exception as e:  # pylint: disable=W0718
            if not is_mcp_authorization_required_error(e):
                return _sync_tools_failed(
                    e, describe_error, connection_headers, normalized_url, stream_id, message_id, local_event_node,
                )
            log.info(f"MCP authorization required for server: {url}")

            # Get OAuth metadata from the exception
            response_metadata = normalize_mcp_auth_metadata_urls(e.to_dict()) or {}
            backfill_mcp_auth_metadata(response_metadata, kwargs)
            response_metadata['chat_project_id'] = tasknode_task.meta.get('chat_project_id')
            
            # Emit the mcp_authorization_required event
            node_interface.emit(
                type=EventTypes.mcp_authorization_required,
                content=str(e),
                response_metadata=response_metadata,
            )
            
            # Return response indicating authorization is needed
            return {
                'success': False,
                'error': str(e),
                'server_url': normalized_url,
                'requires_authorization': True,
                'response_metadata': response_metadata,
            }
            
        finally:
            # Stop event node if forked
            if tasknode_task.multiprocessing_context == "fork":
                local_event_node.stop()
