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

from ..utils.node_interface import NodeEventInterface, EventTypes, NodeEvent

# Import shared components from the agent common module
from .agent_common import (
    EVENTNODE_EVENT_NAME,
    EVENTNODE_FULL_RESPONSE_NAME,
)


def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """Safely serialize data to JSON string."""
    try:
        return json.dumps(data, indent=indent)
    except (TypeError, ValueError) as e:
        log.warning(f"JSON serialization failed: {e}, falling back to str()")
        return str(data)


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
        log.debug(f"MCP sync tools task started: url={url}, project_id={project_id}")
        
        # Get the task for monitoring
        import tasknode_task  # pylint: disable=E0401,C0415
        
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

        # Import the discovery function from elitea-sdk
        # Note: We must import McpAuthorizationRequired AFTER dev_reload_sdk
        # to ensure we catch the same class that will be raised (not a stale reference)
        from ..utils.funcs import dev_reload_sdk
        dev_reload_sdk('elitea_sdk.runtime.utils')
        from elitea_sdk.runtime.utils.mcp_tools_discovery import discover_mcp_tools
        from elitea_sdk.runtime.utils.mcp_oauth import canonical_resource, McpAuthorizationRequired, extract_user_friendly_mcp_error

        try:
            # Prepare connection configuration (secrets already substituted by caller)
            connection_headers = headers or {}
            session_id = None
            
            # Add OAuth token if available
            if mcp_tokens:
                server_key = canonical_resource(url)
                # can be None or type for pre-built mcp, e.g. "mcp_github"
                toolkit_type = kwargs.get('toolkit_type')
                log.debug(f"Looking for token with server_key: {server_key} or toolkit_type: {toolkit_type}")
                log.debug(f"Available mcp_tokens keys: {list(mcp_tokens.keys())}")
                
                token_data = mcp_tokens.get(server_key) if not toolkit_type else mcp_tokens.get(toolkit_type)
                # Try exact URL match if canonical didn't work
                if not token_data:
                    token_data = mcp_tokens.get(url)
                    if token_data:
                        log.debug(f"Found token using exact URL match: {url}")
                
                if token_data:
                    access_token = token_data.get('access_token')
                    session_id = token_data.get('session_id')
                    if access_token:
                        connection_headers['Authorization'] = f'Bearer {access_token}'
                        log.debug(f"Added OAuth token for MCP server: {server_key}")
                    if session_id:
                        log.debug(f"Using session_id for MCP server: {session_id}")
                else:
                    log.warning(f"No token found for server_key: {server_key} or url: {url}")
            
            # Discover tools from the MCP server
            log.debug(f"Discovering tools from MCP server: {url} (ssl_verify={ssl_verify})")
            tools_list = discover_mcp_tools(
                url=url,
                headers=connection_headers,
                timeout=timeout,
                session_id=session_id,
                ssl_verify=ssl_verify,
            )
            
            log.debug(f"Successfully discovered {len(tools_list)} tools from {url}")
            
            # Build success response
            result = {
                'success': True,
                'tools': tools_list,
                'count': len(tools_list),
                'server_url': url,
            }
            
            # Build response metadata
            response_metadata = {
                'tool_output': tools_list,
                'success': True,
                'count': len(tools_list),
                'server_url': url,
            }
            
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
        
        except McpAuthorizationRequired as e:
            log.info(f"MCP authorization required for server: {url}")
            
            # Get OAuth metadata from the exception
            response_metadata = e.to_dict()
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
                'server_url': url,
                'requires_authorization': True,
                'response_metadata': response_metadata,
            }
            
        except Exception as e:
            # Use shared SDK utility to extract user-friendly error message
            user_error_message = extract_user_friendly_mcp_error(e, connection_headers)
            error_msg = f"Failed to sync MCP tools: {user_error_message}"

            log.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Emit error response
            error_event = NodeEvent(
                type=EventTypes.agent_exception,
                stream_id=stream_id,
                message_id=message_id,
                content=error_msg,
                response_metadata={
                    'error': error_msg,
                    'server_url': url,
                }
            ).model_dump_json()
            error_event = json.loads(error_event)
            local_event_node.emit(EVENTNODE_FULL_RESPONSE_NAME, error_event)
            
            return {
                'success': False,
                'error': error_msg,
                'server_url': url,
            }
            
        finally:
            # Stop event node if forked
            if tasknode_task.multiprocessing_context == "fork":
                local_event_node.stop()
