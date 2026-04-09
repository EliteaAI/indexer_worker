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

""" Method for Testing Toolkit Tools """
from copy import deepcopy as copy
import json
import traceback
from typing import Optional, Any
from uuid import uuid4

from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web  # pylint: disable=E0611,E0401

from tools import worker_core  # pylint: disable=E0401

from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired, extract_user_friendly_mcp_error

from ..utils.node_interface import NodeEventInterface, EventTypes, NodeEvent, InitiatorType

# Import shared components from the agent common module
from .agent_common import (
    EVENTNODE_EVENT_NAME,
    EVENTNODE_FULL_RESPONSE_NAME,
    EVENTNODE_PARTIAL_RESPONSE_NAME,
    EliteACallback,
    EliteACustomCallback,
)

def build_mcp_auth_metadata(
    exception: 'McpAuthorizationRequired',
    toolkit_config: dict,
    chat_project_id: Optional[int] = None
) -> dict:
    """
    Build authorization metadata for MCP authorization required events.

    This function centralizes the logic for building auth_metadata that includes
    provided_settings information for pre-built MCP toolkits.

    Args:
        exception: The McpAuthorizationRequired exception
        toolkit_config: Toolkit configuration dict
        chat_project_id: Optional chat project ID to include in metadata

    Returns:
        Dictionary containing authorization metadata with provided_settings if applicable
    """
    auth_metadata = exception.to_dict()

    if chat_project_id is not None:
        auth_metadata['chat_project_id'] = chat_project_id

    # For pre-built MCP toolkits (type starts with 'mcp_'), add provided_settings info
    toolkit_type = toolkit_config.get('type', '')
    if toolkit_type.startswith('mcp_') or toolkit_config.get('settings', {}).get('server_name'):
        settings = toolkit_config.get('settings', {})
        provided_settings = {}

        # Only include client_id, client_secret, and scopes
        if settings.get('client_id'):
            provided_settings['mcp_client_id'] = settings['client_id']

        if settings.get('client_secret'):
            from ..utils.funcs import mask_secret
            provided_settings['mcp_client_secret'] = mask_secret(settings['client_secret'])

        if settings.get('scopes'):
            provided_settings['scopes'] = settings['scopes']

        if provided_settings:
            auth_metadata['provided_settings'] = provided_settings

    return auth_metadata


def safe_json_dumps(data: Any, indent: int = 2, fallback_prefix: str = "Serialization failed: ") -> str:
    """
    Safely serialize data to JSON string with fallback to str() if JSON serialization fails.

    Args:
        data: Data to serialize
        indent: JSON indentation
        fallback_prefix: Prefix to add when falling back to str()

    Returns:
        JSON string or string representation if JSON fails
    """
    try:
        return json.dumps(data, indent=indent)
    except (TypeError, ValueError) as e:
        log.warning(f"JSON serialization failed: {e}, falling back to str()")
        return f"{fallback_prefix}{str(data)}"


def clean_for_json_serialization(data: Any, fallback_message: str = "Could not serialize data") -> Any:
    """
    Clean data to ensure JSON serializability by filtering out non-serializable objects recursively.

    Args:
        data: Data to clean (can be dict, list, or any other type)
        fallback_message: Message to return in case of serialization errors

    Returns:
        Data containing only JSON-serializable values
    """
    try:
        if isinstance(data, dict):
            cleaned = {}
            for k, v in data.items():
                # Skip keys that might contain client references or problematic objects
                if isinstance(k, str) and any(keyword in k.lower() for keyword in [
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
            return [clean_for_json_serialization(item, fallback_message) for item in data if
                    clean_for_json_serialization(item, fallback_message) is not None]
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


def test_error(
        node_interface: NodeEventInterface, toolkit_config: dict, tool_name: str,
        error_message: str, message_id: str, tasknode_task_meta: dict
) -> dict:
    """Handle test execution errors with proper event emission"""
    exception_uid = str(uuid4())
    error = str(traceback.format_exc())

    # Clean toolkit_config to ensure JSON serializability
    clean_toolkit_config = clean_for_json_serialization(
        toolkit_config,
        "Could not serialize toolkit_config"
    )

    node_interface.emit(
        type=EventTypes.agent_tool_start,
        response_metadata={
            "tool_name": "Toolkit Test Exception",
            "tool_run_id": exception_uid,
            "tool_meta": {"toolkit_config": clean_toolkit_config, "tool_name": tool_name},
            "tool_inputs": ''
        }
    )
    node_interface.emit(
        type=EventTypes.agent_tool_end,
        response_metadata={
            "tool_name": "Toolkit Test Exception",
            "tool_run_id": exception_uid,
            'finish_reason': 'error'
        },
        content=error
    )

    node_interface.emit(
        type=EventTypes.agent_exception,
        content=error_message
    )

    # Clean the additional kwargs to avoid serialization issues
    clean_additional_kwargs = clean_for_json_serialization(
        node_interface.payload_additional_kwargs,
        "Could not serialize payload_additional_kwargs"
    )

    msg_event_node = NodeEvent(
        type=EventTypes.full_message,
        stream_id=node_interface.stream_id,
        message_id=message_id,
        response_metadata={
            'project_id': tasknode_task_meta.get("project_id"),
            'chat_project_id': tasknode_task_meta.get("chat_project_id"),
            'toolkit_config': clean_toolkit_config,
            'tool_name': tool_name,
            'is_error': True
        },
        content=error_message,
        **clean_additional_kwargs
    ).model_dump_json()
    msg_event_node = json.loads(msg_event_node)
    node_interface.event_node.emit(EVENTNODE_FULL_RESPONSE_NAME, msg_event_node)

    return {
        'success': False,
        'error': error_message,
        'toolkit_config': clean_toolkit_config,
        'tool_name': tool_name
    }


def check_missing_index_data_status_event(
        node_interface: NodeEventInterface,
        tool_name: str,
        tool_params: dict,
        toolkit_config: dict,
        elitea_callback,
        tasknode_task_meta: dict,
        tasknode_task_id: str,
        error_message: str
):
    """
    Check if index_data status event was missed and emit it if needed.

    This handles cases where index_data tool fails but SDK doesn't emit
    the status event (e.g., early validation failures, SDK bugs, etc.).

    Args:
        node_interface: Interface for emitting events
        tool_name: Name of the tool that was executed
        tool_params: Parameters passed to the tool
        toolkit_config: Toolkit configuration
        elitea_callback: Callback that tracks index statuses
        tasknode_task_meta: Task metadata
        tasknode_task_id: Current task ID
        error_message: Error message from the failure
    """
    # Only handle index_data tool
    if tool_name != 'index_data':
        return

    # Check if SDK already recorded a failed status via EliteACustomCallback
    index_statuses = getattr(elitea_callback, 'index_statuses', [])
    if any(s.get('state') == 'failed' for s in index_statuses):
        # SDK already emitted the failure status event, nothing to do
        return

    log.warning(
        f"index_data tool failed but no status event was emitted by SDK. "
        f"Manually emitting event for task_id={tasknode_task_id}"
    )

    # Emit the missing index_data failure event
    node_interface.emit(
        type=EventTypes.agent_index_data_status,
        response_metadata={
            'task_id': tasknode_task_id,
            'index_name': tool_params.get('index_name'),
            'state': 'failed',
            'error': error_message,
            'toolkit_config': clean_for_json_serialization(
                toolkit_config,
                "Could not serialize toolkit_config"
            ),
            'tool_params': clean_for_json_serialization(
                tool_params,
                "Could not serialize tool_params"
            ),
            'indexed': 0,
            'updated': 0,
            'toolkit_id': toolkit_config.get('id'),
            'initiator': str(tasknode_task_meta.get("initiator", InitiatorType.user)),
            'project_id': tasknode_task_meta.get("project_id"),
            'user_id': tasknode_task_meta.get("user_context", {}).get("user_id"),
        }
    )


def detect_content_type(content: Any) -> tuple[str, Any]:
    """
    Detect the content type for toolkit testing page formatting and prepare content for UI.

    This is specifically for the toolkit testing UI to determine how to display
    tool results. Returns a tuple of (content_type, formatted_content).

    The formatted_content is prepared based on the detected type:
    - For JSON: Returns the raw dict/list (UI will wrap in ```json block)
    - For markdown/text: Returns the string as-is

    Args:
        content: The content to analyze (can be string, dict, list, etc.)

    Returns:
        Tuple of (content_type, formatted_content) where:
        - content_type: 'markdown', 'json', or 'text'
        - formatted_content: Content prepared for UI consumption
    """
    # If content is already a dict or list, it's JSON - return raw for UI to format
    if isinstance(content, (dict, list)):
        return ('json', content)

    # Convert to string for analysis
    content_str = str(content) if not isinstance(content, str) else content

    # Early return for empty or very short content
    if not content_str or len(content_str.strip()) < 10:
        return ('text', content_str)

    # Try to parse as JSON first (pure JSON strings)
    try:
        parsed_json = json.loads(content_str)
        return ('json', parsed_json)  # Return parsed object for UI to format
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Check for explicit markdown patterns
    markdown_indicators = [
        content_str.startswith('#'),  # Headers at start
        '\n#' in content_str,  # Headers in content
        '\n## ' in content_str,  # Level 2 headers
        '\n### ' in content_str,  # Level 3 headers
        '\n- ' in content_str,  # Unordered lists
        '\n* ' in content_str,  # Alternative unordered lists
        '\n+ ' in content_str,  # Another list variant
        '\n```' in content_str,  # Code blocks
        '**' in content_str and content_str.count('**') >= 2,  # Bold text
        '__' in content_str and content_str.count('__') >= 2,  # Alternative bold
        '*' in content_str and content_str.count('*') >= 2,  # Emphasis
        '`' in content_str and content_str.count('`') >= 2,  # Inline code
        '\n>' in content_str,  # Blockquotes
        '[' in content_str and '](' in content_str,  # Links
        '|' in content_str and '\n|' in content_str,  # Tables
    ]

    # Check for numbered lists with more variations
    numbered_list_patterns = [
        '\n1. ' in content_str,
        '\n2. ' in content_str,
        '\n3. ' in content_str,
        '\n1) ' in content_str,
        '\n2) ' in content_str,
    ]

    if any(markdown_indicators) or any(numbered_list_patterns):
        return ('markdown', content_str)

    # Heuristic: Content with multiple line breaks is likely prose/narrative
    # This catches LLM-generated summaries and explanations
    line_breaks = content_str.count('\n')
    if line_breaks >= 3:
        # Check if it looks like structured prose (multiple sentences)
        sentences = content_str.count('. ') + content_str.count('.\n')
        if sentences >= 3:
            return ('markdown', content_str)

        # Check for paragraph breaks (double newlines)
        if '\n\n' in content_str:
            return ('markdown', content_str)

    # Check for specific tool output patterns
    if content_str.strip().startswith(('Found ', 'No documents found')):
        # search_index/stepback_search_index pattern: "Found X documents..."
        # Check if it's followed by JSON-like content
        if '[' in content_str and ']' in content_str:
            return ('text', content_str)  # Mixed format, keep as plain text

    # Long single-paragraph text (>200 chars with few line breaks)
    # is likely narrative prose from an LLM
    if len(content_str) > 200 and line_breaks < 3:
        # Check if it's sentence-like (has punctuation)
        if '. ' in content_str or '! ' in content_str or '? ' in content_str:
            return ('markdown', content_str)

    # Default to text for everything else
    return ('text', content_str)


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource for Testing Toolkit Tools

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def indexer_test_toolkit_tool(  # pylint: disable=R0914,W1113
            self,
            stream_id: Optional[str] = None,
            message_id: Optional[str] = None,
            *args,
            **kwargs,
    ):
        """ Test a single toolkit tool """
        self.indexer_enable_logging()
        #
        log.debug(f'indexer_test_toolkit_tool start stream_id={stream_id}, message_id={message_id}')
        #
        try:
            return self._indexer_test_toolkit_tool_task(
                *args,
                stream_id=stream_id,
                message_id=message_id,
                **kwargs,
            )
        except:  # pylint: disable=W0702
            log.exception("indexer_test_toolkit_tool failed to start")
            raise

    @web.method()
    def _indexer_test_toolkit_tool_task(  # pylint: disable=R0914,R0915
            self,
            *args,
            stream_id: Optional[str] = None,
            message_id: Optional[str] = None,
            **kwargs,
    ):
        """ Internal task for testing toolkit tool """
        _ = args
        #
        #
        import tasknode_task  # pylint: disable=E0401,C0415
        #
        if tasknode_task.multiprocessing_context == "fork":
            local_event_node = worker_core.event_node.clone()
            local_event_node.start()
        else:
            local_event_node = worker_core.event_node
        #
        node_interface = NodeEventInterface(
            event_node=local_event_node,
            node_event_name=EVENTNODE_EVENT_NAME,
            stream_id=stream_id,
            message_id=message_id,
            sio_event=tasknode_task.meta.get("sio_event"),
            question_id=tasknode_task.meta.get("question_id")
        )
        #
        node_interface.emit(
            type=EventTypes.agent_start
        )
        #
        # Extract required parameters
        toolkit_config = kwargs.get("toolkit_config", {})
        tool_name = kwargs.get("tool_name", "")
        tool_params = kwargs.get("tool_params", {})
        runtime_config = kwargs.get("runtime_config", {})
        llm_model = kwargs.get("llm_model")
        if not llm_model:
            llm_model = kwargs.get("llm_settings", {}).get("model_name")
        llm_settings = kwargs.get("llm_settings", {})
        project_id = kwargs.get("project_id")
        project_auth_token = kwargs.get("project_auth_token")
        deployment_url = kwargs.get("deployment_url")
        mcp_tokens = kwargs.get("mcp_tokens")

        # Clean toolkit_config to ensure JSON serializability
        clean_toolkit_config = clean_for_json_serialization(
            toolkit_config,
            "Could not serialize toolkit_config"
        )

        # Clean other configs for safety
        clean_runtime_config = clean_for_json_serialization(
            runtime_config,
            "Could not serialize runtime_config"
        )
        clean_llm_settings = clean_for_json_serialization(
            llm_settings,
            "Could not serialize llm_settings"
        )

        if not toolkit_config:
            error_msg = "toolkit_config is required"
            log.error(error_msg)
            return test_error(
                node_interface, clean_toolkit_config, tool_name, error_msg,
                message_id, tasknode_task.meta
            )

        if not tool_name:
            error_msg = "tool_name is required"
            log.error(error_msg)
            return test_error(
                node_interface, clean_toolkit_config, tool_name, error_msg,
                message_id, tasknode_task.meta
            )

        # Resolve MCP credentials from pylon config if toolkit is MCP type and credentials are missing
        from ..utils.funcs import resolve_mcp_credentials
        toolkit_config = resolve_mcp_credentials(toolkit_config)
        # Update clean_toolkit_config after credential resolution
        clean_toolkit_config = clean_for_json_serialization(
            toolkit_config,
            "Could not serialize toolkit_config after credential resolution"
        )

        try:
            # Create EliteAClient with authentication
            from ..utils.funcs import dev_reload_sdk
            dev_reload_sdk('elitea_sdk.runtime.clients')
            from elitea_sdk.runtime.clients.client import EliteAClient  # pylint: disable=E0401,C0415

            # Initialize EliteAClient with proper authentication
            client = EliteAClient(
                project_id=project_id,
                auth_token=project_auth_token,
                base_url=deployment_url
            )
            # Seems like not used
            # Generate persistent tool_run_id for this execution
            # tool_run_id = str(uuid4())

            # # Clean tool_params to ensure JSON serializability
            # clean_tool_params = clean_for_json_serialization(
            #     tool_params,
            #     "Could not serialize tool_params"
            # )
            thread_id = kwargs.get("thread_id")
            # Ensure thread_id is never None for consistent checkpointer behavior
            if thread_id is None:
                import uuid
                thread_id = str(uuid.uuid4())
                log.info(f"[THREAD_ID] Generated unique thread_id for toolkit test: {thread_id}")
            # EliteACallback and EliteACustomCallback are imported from agent_common at module level
            elitea_callback = EliteACallback(
                node_interface,
                debug=kwargs.get("debug", False),
                thread_id=thread_id,
                message_id=message_id,
                project_id=tasknode_task.meta.get("project_id"),
                chat_project_id=tasknode_task.meta.get("chat_project_id"),
                toolkit_metadata=clean_toolkit_config,
            )
            elitea_custom_callback = EliteACustomCallback(
                node_interface,
                debug=kwargs.get("debug", False),
                message_id=message_id,
                project_id=tasknode_task.meta.get("project_id"),
                chat_project_id=tasknode_task.meta.get("chat_project_id"),
                user_id=tasknode_task.meta.get("user_context", {}).get("user_id"),
                initiator=tasknode_task.meta.get("initiator", InitiatorType.user),
                task_id=tasknode_task.id,
                toolkit_metadata={
                    'toolkit_config': clean_toolkit_config,
                    'tool_params': clean_for_json_serialization(tool_params, "Could not serialize tool_params"),
                    'toolkit_id': toolkit_config.get('id')
                },
            )
            callbacks = [elitea_callback, elitea_custom_callback]

            # Add callbacks to params
            runtime_config.update({"callbacks": callbacks})

            # Call the test_toolkit_tool method
            # Note: agent_tool_start/end events are automatically emitted by EliteACallback
            test_result = client.test_toolkit_tool(
                toolkit_config=copy(toolkit_config),  # Use original config for the actual call
                tool_name=tool_name,
                tool_params=copy(tool_params),
                runtime_config=runtime_config,  # Use original config for the actual call
                llm_model=llm_model,
                llm_config=copy(llm_settings),  # Use original config for the actual call
                mcp_tokens=mcp_tokens
            )

            log.info(f"Test result: {test_result}")

            # Clean the test_result to remove any non-serializable objects (including EliteAClient references)
            clean_test_result = clean_for_json_serialization(
                test_result,
                "Could not serialize test_result"
            )
            clean_test_result.pop("toolkit_config", None)

            # Extract values from the cleaned result (use SDK response as source of truth)
            success = clean_test_result.get("success", False)
            tool_result = clean_test_result.get("result")
            error_message = clean_test_result.get("error", "")
            execution_time = clean_test_result.get("execution_time_seconds", 0.0)
            log.debug(f"Tool clean result: {clean_test_result}")
            final_result = tool_result if success else error_message
            log.debug(f"Tool result (success - '{success}'): {final_result}")
            # Detect content type and prepare formatted content for UI
            content_type, formatted_content = detect_content_type(final_result)

            # For JSON content, serialize it to a JSON string so it can be transmitted via Socket.IO
            # The frontend will parse it back when content_type === 'json'
            if content_type == 'json':
                formatted_content = safe_json_dumps(formatted_content) if formatted_content else "Tool executed successfully"
            elif not formatted_content:
                formatted_content = "Tool executed successfully"

            # Manually emit agent_tool_end to include content_type metadata
            # (instead of relying on EliteACallback's automatic emission)
            node_interface.emit(
                type=EventTypes.agent_tool_end,
                content=formatted_content,
                response_metadata={
                    'tool_name': tool_name,
                    'tool_run_id': test_result.get('tool_run_id'),
                    'tool_output': final_result,
                    'timestamp_finish': test_result.get('timestamp_finish'),
                    'finish_reason': 'stop' if success else 'error',
                    'execution_time_seconds': execution_time,
                    'content_type': content_type  # For toolkit testing page formatting
                }
            )

            # Emit response event
            if success:
                node_interface.emit(
                    type=EventTypes.agent_response,
                    content=formatted_content,
                    response_metadata={
                        'finish_reason': 'stop',
                        'execution_time_seconds': execution_time,
                        'content_type': content_type  # For toolkit testing page formatting
                    }
                )
            else:
                node_interface.emit(
                    type=EventTypes.agent_exception,
                    content=error_message
                )

                # Check if index_data status event is missing and emit it if needed
                # Pass elitea_custom_callback (has index_statuses), not elitea_callback
                check_missing_index_data_status_event(
                    node_interface=node_interface,
                    tool_name=tool_name,
                    tool_params=tool_params,
                    toolkit_config=toolkit_config,
                    elitea_callback=elitea_custom_callback,
                    tasknode_task_meta=tasknode_task.meta,
                    tasknode_task_id=tasknode_task.id,
                    error_message=error_message
                )

            # Create full message event using data from SDK (avoid duplication)
            # Clean the additional kwargs to avoid serialization issues
            clean_additional_kwargs = clean_for_json_serialization(
                node_interface.payload_additional_kwargs,
                "Could not serialize payload_additional_kwargs"
            )

            msg_event_node = NodeEvent(
                type=EventTypes.full_message,
                stream_id=node_interface.stream_id,
                message_id=message_id,
                response_metadata={
                    'project_id': tasknode_task.meta.get("project_id"),
                    'chat_project_id': tasknode_task.meta.get("chat_project_id"),
                    'toolkit_config': clean_toolkit_config,
                    'tool_name': tool_name,
                    # Use cleaned SDK result directly instead of duplicating data
                    'test_result': clean_test_result,
                    'execution_time_seconds': execution_time,
                    'is_error': not success,
                    'content_type': content_type  # For toolkit testing page formatting
                },
                # Use formatted content (already serialized for JSON types)
                content=formatted_content,
                **clean_additional_kwargs
            ).model_dump_json()
            msg_event_node = json.loads(msg_event_node)
            node_interface.event_node.emit(EVENTNODE_FULL_RESPONSE_NAME, msg_event_node)

            log.info("Test toolkit tool task completed successfully")
            return clean_test_result

        except McpAuthorizationRequired as e:
            log.info(f"MCP authorization required for toolkit test: {str(e)}")
            # Build authorization metadata using shared helper function
            auth_metadata = build_mcp_auth_metadata(
                exception=e,
                toolkit_config=toolkit_config,
                chat_project_id=tasknode_task.meta.get('chat_project_id')
            )
            log.debug(f"Pre-built MCP toolkit provided_settings [test_toolkit_tool_task]: {auth_metadata}")

            node_interface.emit(
                type=EventTypes.mcp_authorization_required,
                content=str(e),
                response_metadata=auth_metadata,
            )
            # Return a response indicating authorization is needed
            return {
                'success': False,
                'error': str(e),
                'toolkit_config': clean_toolkit_config,
                'tool_name': tool_name,
                'requires_authorization': True
            }

        except Exception as e:
            # For index_data tool, emit special event for metadata handling
            # Only emit if SDK didn't already record a failed status via EliteACustomCallback
            _sdk_recorded_failure = any(
                s.get('state') == 'failed'
                for s in getattr(elitea_custom_callback, 'index_statuses', [])
            )
            if tool_name == 'index_data' and not _sdk_recorded_failure:
                # Emit index_data failure event with all required fields matching indexer_agent.py format
                node_interface.emit(
                    type=EventTypes.agent_index_data_status,
                    response_metadata={
                        'task_id': tasknode_task.id,
                        'index_name': tool_params.get('index_name'),
                        'state': 'failed',
                        'error': f"Failed to execute index_data tool: {str(e)}",
                        'toolkit_config': clean_toolkit_config,
                        'tool_params': clean_for_json_serialization(tool_params, "Could not serialize tool_params"),
                        'indexed': 0,
                        'updated': 0,
                        'toolkit_id': toolkit_config.get('id'),
                        'initiator': str(tasknode_task.meta.get("initiator", InitiatorType.user)),
                        'project_id': tasknode_task.meta.get("project_id"),
                        'user_id': tasknode_task.meta.get("user_context", {}).get("user_id"),
                    }
                )

            # For all exceptions (including index_data after emitting event), handle as errors
            error_msg = f"Failed to test toolkit tool: {str(e)}"
            log.exception(error_msg)
            return test_error(
                node_interface, clean_toolkit_config, tool_name, error_msg,
                message_id, tasknode_task.meta
            )
        finally:
            # Stop event node if forked (following indexer_agent.py pattern)
            if tasknode_task.multiprocessing_context == "fork":
                local_event_node.stop()

    @web.method()
    def indexer_test_mcp_connection(  # pylint: disable=R0914,W1113
            self,
            stream_id: Optional[str] = None,
            message_id: Optional[str] = None,
            *args,
            **kwargs,
    ):
        """
        Test MCP server connection using protocol-level list_tools.

        This method verifies MCP server connectivity and authentication by calling
        the protocol-level tools/list JSON-RPC method (NOT executing a tool).
        This is ideal for auth checks as it validates the connection without
        requiring any tool execution.
        """
        self.indexer_enable_logging()
        #
        log.debug(f'indexer_test_mcp_connection start stream_id={stream_id}, message_id={message_id}')
        #
        try:
            return self._indexer_test_mcp_connection_task(
                *args,
                stream_id=stream_id,
                message_id=message_id,
                **kwargs,
            )
        except:  # pylint: disable=W0702
            log.exception("indexer_test_mcp_connection failed to start")
            raise

    @web.method()
    def _indexer_test_mcp_connection_task(  # pylint: disable=R0914,R0915
            self,
            *args,
            stream_id: Optional[str] = None,
            message_id: Optional[str] = None,
            **kwargs,
    ):
        """ Internal task for testing MCP connection """
        _ = args
        import tasknode_task  # pylint: disable=E0401,C0415
        #
        if tasknode_task.multiprocessing_context == "fork":
            local_event_node = worker_core.event_node.clone()
            local_event_node.start()
        else:
            local_event_node = worker_core.event_node
        #
        node_interface = NodeEventInterface(
            event_node=local_event_node,
            node_event_name=EVENTNODE_EVENT_NAME,
            stream_id=stream_id,
            message_id=message_id,
            sio_event=tasknode_task.meta.get("sio_event"),
            question_id=tasknode_task.meta.get("question_id")
        )
        #
        node_interface.emit(
            type=EventTypes.agent_start
        )
        #
        # Extract required parameters
        toolkit_config = kwargs.get("toolkit_config", {})
        project_id = kwargs.get("project_id")
        project_auth_token = kwargs.get("project_auth_token")
        deployment_url = kwargs.get("deployment_url")
        mcp_tokens = kwargs.get("mcp_tokens")
        log.info(f"Tokens: {mcp_tokens}")

        # Clean toolkit_config to ensure JSON serializability
        clean_toolkit_config = clean_for_json_serialization(
            toolkit_config,
            "Could not serialize toolkit_config"
        )

        if not toolkit_config:
            error_msg = "toolkit_config is required"
            log.error(error_msg)
            return mcp_connection_error(
                node_interface, clean_toolkit_config, error_msg,
                message_id, tasknode_task.meta, local_event_node, tasknode_task
            )

        # Verify this is an MCP toolkit
        toolkit_type = toolkit_config.get('type')
        if toolkit_type != 'mcp' and not toolkit_type.startswith('mcp_'):
            error_msg = f"test_mcp_connection only works with MCP toolkits, got type: {toolkit_type}"
            log.error(error_msg)
            return mcp_connection_error(
                node_interface, clean_toolkit_config, error_msg,
                message_id, tasknode_task.meta, local_event_node, tasknode_task
            )

        # Resolve MCP credentials from pylon config if not in settings
        from ..utils.funcs import resolve_mcp_credentials
        toolkit_config = resolve_mcp_credentials(toolkit_config)
        # Update clean_toolkit_config after credential resolution
        clean_toolkit_config = clean_for_json_serialization(
            toolkit_config,
            "Could not serialize toolkit_config after credential resolution"
        )

        try:
            # Create EliteAClient with authentication
            from ..utils.funcs import dev_reload_sdk
            dev_reload_sdk('elitea_sdk.runtime.clients')
            from elitea_sdk.runtime.clients.client import EliteAClient  # pylint: disable=E0401,C0415

            # Initialize EliteAClient with proper authentication
            client = EliteAClient(
                project_id=project_id,
                auth_token=project_auth_token,
                base_url=deployment_url
            )

            # Call the test_mcp_connection method (uses protocol-level list_tools)
            test_result = client.test_mcp_connection(
                toolkit_config=toolkit_config,
                mcp_tokens=mcp_tokens
            )

            log.info(f"MCP connection test result: {test_result}")

            # Clean the test_result to remove any non-serializable objects
            clean_test_result = clean_for_json_serialization(
                test_result,
                "Could not serialize test_result"
            )

            success = clean_test_result.get("success", False)
            tools = clean_test_result.get("tools", [])
            tools_count = clean_test_result.get("tools_count", 0)
            error_message = clean_test_result.get("error", "")
            execution_time = clean_test_result.get("execution_time_seconds", 0.0)

            if success:
                # Emit success response
                content = f"MCP connection successful. Found {tools_count} tools: {', '.join(tools[:10])}"
                if tools_count > 10:
                    content += f"... and {tools_count - 10} more"

                node_interface.emit(
                    type=EventTypes.agent_response,
                    content=content,
                    response_metadata={
                        'finish_reason': 'stop',
                        'execution_time_seconds': execution_time,
                        'tools_count': tools_count,
                        'tools': tools
                    }
                )
            else:
                toolkit_settings = toolkit_config.get('settings', {})
                mcp_headers = toolkit_settings.get('headers', {})
                friendly_error = extract_user_friendly_mcp_error(Exception(error_message), mcp_headers)
                node_interface.emit(
                    type=EventTypes.agent_exception,
                    content=friendly_error
                )

            # Clean the additional kwargs to avoid serialization issues
            clean_additional_kwargs = clean_for_json_serialization(
                node_interface.payload_additional_kwargs,
                "Could not serialize payload_additional_kwargs"
            )

            msg_event_node = NodeEvent(
                type=EventTypes.full_message,
                stream_id=node_interface.stream_id,
                message_id=message_id,
                response_metadata={
                    'project_id': tasknode_task.meta.get("project_id"),
                    'chat_project_id': tasknode_task.meta.get("chat_project_id"),
                    'toolkit_config': clean_toolkit_config,
                    'test_result': clean_test_result,
                    'execution_time_seconds': execution_time,
                    'is_error': not success
                },
                content=content if success else error_message,
                **clean_additional_kwargs
            ).model_dump_json()
            msg_event_node = json.loads(msg_event_node)
            node_interface.event_node.emit(EVENTNODE_FULL_RESPONSE_NAME, msg_event_node)

            log.info("Test MCP connection task completed successfully")
            return clean_test_result

        except McpAuthorizationRequired as e:
            log.info(f"MCP authorization required for connection test: {str(e)}")
            # Build authorization metadata using shared helper function
            auth_metadata = build_mcp_auth_metadata(
                exception=e,
                toolkit_config=toolkit_config,
                chat_project_id=tasknode_task.meta.get('chat_project_id')
            )
            log.debug(f"Pre-built MCP toolkit provided_settings [test_mcp_connection]: {auth_metadata}")
            node_interface.emit(
                type=EventTypes.mcp_authorization_required,
                content=str(e),
                response_metadata=auth_metadata,
            )
            # Return a response indicating authorization is needed
            return {
                'success': False,
                'error': str(e),
                'toolkit_config': clean_toolkit_config,
                'requires_authorization': True,
                'tools': [],
                'tools_count': 0
            }

        except Exception as e:
            # Extract toolkit headers to check for auth token
            toolkit_settings = toolkit_config.get('settings', {})
            headers = toolkit_settings.get('headers', {})

            # Use shared SDK utility to extract user-friendly error message
            user_error_message = extract_user_friendly_mcp_error(e, headers)
            error_msg = f"Failed to test MCP connection: {user_error_message}"

            log.exception(f"[MCP Connection] {error_msg}")
            return mcp_connection_error(
                node_interface, clean_toolkit_config, error_msg,
                message_id, tasknode_task.meta, local_event_node, tasknode_task
            )
        finally:
            # Stop event node if forked (following indexer_agent.py pattern)
            if tasknode_task.multiprocessing_context == "fork":
                local_event_node.stop()


def mcp_connection_error(
        node_interface: NodeEventInterface, toolkit_config: dict,
        error_message: str, message_id: str, tasknode_task_meta: dict,
        local_event_node, tasknode_task
) -> dict:
    """Handle MCP connection test errors with proper event emission"""
    node_interface.emit(
        type=EventTypes.agent_exception,
        content=error_message
    )

    # Clean the additional kwargs to avoid serialization issues
    clean_additional_kwargs = clean_for_json_serialization(
        node_interface.payload_additional_kwargs,
        "Could not serialize payload_additional_kwargs"
    )

    msg_event_node = NodeEvent(
        type=EventTypes.full_message,
        stream_id=node_interface.stream_id,
        message_id=message_id,
        response_metadata={
            'project_id': tasknode_task_meta.get("project_id"),
            'chat_project_id': tasknode_task_meta.get("chat_project_id"),
            'toolkit_config': toolkit_config,
            'is_error': True
        },
        content=error_message,
        **clean_additional_kwargs
    ).model_dump_json()
    msg_event_node = json.loads(msg_event_node)
    node_interface.event_node.emit(EVENTNODE_FULL_RESPONSE_NAME, msg_event_node)

    # Stop event node if forked
    if tasknode_task.multiprocessing_context == "fork":
        local_event_node.stop()

    return {
        'success': False,
        'error': error_message,
        'toolkit_config': toolkit_config,
        'tools': [],
        'tools_count': 0
    }
