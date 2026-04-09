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
from typing import Optional

from langchain_core.messages import HumanMessage

from pylon.core.tools import log
from pylon.core.tools import web

from ..utils.exceptions import InternalSDKError
from ..utils.node_interface import EventTypes

# Import shared components
from .agent_common import (
    execution_error,
    _fetch_pgvector_connstr_with_retry,
    temp_elitea_client,
    fetch_langfuse_config,
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
)
from ..utils.langfuse_callback import flush_langfuse_callback, langfuse_trace_context
from ..utils.image_helpers import resolve_filepath_images, resolve_generated_image_thumbnails

from pydantic import ValidationError
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired
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
                pgvector_connstr = _fetch_pgvector_connstr_with_retry(temp_client)

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
            if not hitl_resume and not should_continue:
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
            log.info(f'exception_handling_enabled "{exception_handling_enabled}"')

            # Prepare context_settings with summarization callbacks
            context_settings = kwargs.get("context_settings", {})
            context_settings['callbacks'] = create_summarization_callbacks(node_interface)

            # Create application agent
            agent_executor = client.application(
                application_id=kwargs.get("application", {})["id"],
                application_version_id=kwargs.get("application", {})["version_id"],
                memory=memory,
                application_variables=kwargs.get("application", {}).get('variables'),
                version_details=deepcopy(version_details),
                mcp_tokens=kwargs.get("mcp_tokens", None),
                conversation_id=conversation_id,
                ignored_mcp_servers=kwargs.get("ignored_mcp_servers", None),
                exception_handling_enabled=exception_handling_enabled,
                context_settings=context_settings,
                auto_approve_sensitive_actions=kwargs.get("auto_approve_sensitive_actions", False),
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

            # HITL resume takes precedence over generic checkpoint continuation.
            if hitl_resume:
                invoke_input['hitl_resume'] = True
                invoke_input['hitl_action'] = hitl_action
                invoke_input['hitl_value'] = hitl_value
                log.info(f'[HITL] Resume action: {invoke_input["hitl_action"]}')
            elif should_continue:
                invoke_input, invoke_config = configure_checkpoint_resume(
                    agent_executor,
                    thread_id,
                    kwargs.get('checkpoint_id'),
                    invoke_input,
                    invoke_config
                )

            # Invoke the agent executor with Langfuse trace context
            with langfuse_trace_context(langfuse_trace_attrs):
                response = agent_executor.invoke(invoke_input, invoke_config)

            # Extract context info (includes summarization details when summarization occurred)
            context_info = response.get('context_info')

            # Extract and normalize response content using unified parsing
            response_content = extract_response_content(response, response_format='output')
            output = build_output_message(response_content)

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
            )

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
        except ValueError:
            return execution_error(
                node_interface, user_input, chat_history,
                f"Seems like your agent is missconfigured on user input: {user_input}",
                thread_id, message_id, tasknode_task.meta,
                human_readable="Seems like your agent is configured incorrectly",
                execution_start_time=execution_start_time
            )
        except McpAuthorizationRequired as e:
            auth_metadata = e.to_dict()
            auth_metadata['chat_project_id'] = tasknode_task.meta.get('chat_project_id')
            node_interface.emit(
                type=EventTypes.mcp_authorization_required,
                content=str(e),
                response_metadata=auth_metadata,
            )
            return {
                'chat_history': chat_history,
                'error': str(e),
            }
        except InternalServerError:
            return execution_error(
                node_interface, user_input, chat_history,
                f"OpenAI Responded with Internal Server Error on User Input: {user_input}",
                thread_id, message_id, tasknode_task.meta,
                human_readable="The AI service is currently experiencing issues",
                execution_start_time=execution_start_time
            )
        except Exception as e:
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
            # Flush Langfuse traces
            flush_langfuse_callback(langfuse_client, langfuse_callback)

            if tasknode_task.multiprocessing_context == "fork":
                local_event_node.stop()

        return build_success_result(chat_history, elitea_callback, total_tokens_in, total_tokens_out, context_info)
