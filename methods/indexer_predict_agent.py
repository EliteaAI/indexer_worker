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

""" Method for Predict Agent """
from typing import Optional

from langchain_core.messages import HumanMessage

from pylon.core.tools import log
from pylon.core.tools import web

from ..utils.exceptions import InternalSDKError
from ..utils.node_interface import EventTypes

# Import shared components
from .agent_common import (
    execution_error,
    EVENTNODE_EVENT_NAME,
    _fetch_pgvector_connstr_with_retry,
    temp_elitea_client,
    fetch_langfuse_config,
    unsecret_mcp_tools,
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

# Collect LLM authentication/authorization error types
_LLM_AUTH_ERRORS = []
try:
    from anthropic import AuthenticationError as AnthropicAuthError
    _LLM_AUTH_ERRORS.append(AnthropicAuthError)
except ImportError:
    pass
try:
    from openai import AuthenticationError as OpenAIAuthError
    _LLM_AUTH_ERRORS.append(OpenAIAuthError)
except ImportError:
    pass
LLM_AUTH_ERRORS = tuple(_LLM_AUTH_ERRORS) if _LLM_AUTH_ERRORS else None

# Collect LLM rate limit error types
_LLM_RATE_LIMIT_ERRORS = []
try:
    from anthropic import RateLimitError as AnthropicRateLimitError
    _LLM_RATE_LIMIT_ERRORS.append(AnthropicRateLimitError)
except ImportError:
    pass
try:
    from openai import RateLimitError as OpenAIRateLimitError
    _LLM_RATE_LIMIT_ERRORS.append(OpenAIRateLimitError)
except ImportError:
    pass
LLM_RATE_LIMIT_ERRORS = tuple(_LLM_RATE_LIMIT_ERRORS) if _LLM_RATE_LIMIT_ERRORS else None


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource for Predict Agent

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def indexer_predict_agent(  # pylint: disable=R0914,W1113
            self,
            stream_id: Optional[str] = None,
            message_id: Optional[str] = None,
            *args,
            **kwargs,
    ):
        """ Run predict agent target """
        self.indexer_enable_logging()
        #
        log.debug(f'indexer_predict_agent start stream_id={stream_id}, message_id={message_id}')
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
                return self._indexer_predict_agent_task(
                    memory,
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
            log.exception("indexer_predict_agent failed to start")
            raise

    @web.method()
    def _indexer_predict_agent_task(  # pylint: disable=R0914,R0915
            self,
            memory,
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
        if traceparent:
            log.debug(f"Received traceparent from pylon_main: {traceparent}")

        # Execute with optional tracing span
        return with_tracing_span(
            traceparent,
            'indexer_predict_agent',
            stream_id,
            message_id,
            str(tasknode_task.meta.get('project_id', '')),
            self._indexer_predict_agent_task_inner,
            memory, client_args, api_token, api_extra_headers,
            stream_id=stream_id, message_id=message_id, **kwargs
        )

    @web.method()
    def _indexer_predict_agent_task_inner(  # pylint: disable=R0914,R0915
            self,
            memory,
            client_args,
            api_token,
            api_extra_headers,
            stream_id: Optional[str] = None,
            message_id: Optional[str] = None,
            **kwargs,
    ):
        """Inner predict agent task execution (separated for tracing)."""
        import tasknode_task  # pylint: disable=E0401,C0415
        from openai import InternalServerError  # pylint: disable=C0415,E0401
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

        # Get tools and internal_tools - SDK handles internal_tools processing now
        adhoc_tools = kwargs.get("tools", [])
        # Resolve {{secret.xxx}} placeholders in MCP tool settings at indexer level
        adhoc_tools = unsecret_mcp_tools(adhoc_tools, client)
        internal_tools = kwargs.get("internal_tools", [])
        lazy_tools_mode = 'lazy_tools_mode' in internal_tools

        log.debug(f'adhoc_tools "{adhoc_tools}", internal_tools "{internal_tools}"')

        try:
            client_args = kwargs.get("llm").get("kwargs", {})
            application_data = kwargs.get("application", {})

            # Create LLM
            llm = client.get_llm(
                model_name=client_args.get("model"),
                model_config={
                    "model_project_id": client_args.get("model_project_id"),
                    "max_tokens": client_args.get("max_tokens"),
                    "reasoning_effort": client_args.get("reasoning_effort"),
                    "temperature": client_args.get("temperature"),
                    "streaming": client_args.get("stream", True)
                }
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

            # Create predict agent
            steps_limit = kwargs.get("steps_limit")
            agent_executor = client.predict_agent(
                llm=llm,
                instructions=application_data.get('instructions', 'You are a helpful assistant.'),
                tools=adhoc_tools,
                chat_history=chat_history,
                memory=memory,
                debug_mode=kwargs.get("debug_mode", True),
                mcp_tokens=kwargs.get("mcp_tokens", None),
                conversation_id=conversation_id,
                ignored_mcp_servers=kwargs.get("ignored_mcp_servers", None),
                persona=kwargs.get("persona", "generic"),
                lazy_tools_mode=lazy_tools_mode,
                internal_tools=internal_tools,
                exception_handling_enabled=exception_handling_enabled,
                context_settings=context_settings,
                step_limit=steps_limit,
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
            application_name = application_data.get("name", "predict-agent")
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

            invoke_input = prepare_invoke_input(
                chat_history,
                user_message,
                conversation_id,
                include_attachment_system_message=any(
                    t.get('name') == 'Attachments' for t in (adhoc_tools or [])
                ),
                model_name=client_args.get('model', ''),
                supports_vision=supports_vision,
            )
            invoke_config = {
                "callbacks": callbacks,
                "configurable": {"thread_id": thread_id},
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

            if elitea_callback.llm_error:
                raise elitea_callback.llm_error

            # Extract context info (includes summarization details when summarization occurred)
            context_info = response.get('context_info')

            # Extract and normalize response content using unified parsing
            response_content = extract_response_content(response, response_format='messages')
            output = build_output_message(response_content)

            # Resolve thumbnails for tool-generated images and merge with user-upload thumbnails
            resolve_generated_image_thumbnails(elitea_custom_callback, image_thumbnails, client)

            # Emit response events
            total_tokens_in, total_tokens_out, thread_id_response = emit_response_events(
                node_interface,
                response,
                output,
                thread_id,
                message_id,
                elitea_callback,
                elitea_custom_callback,
                tasknode_task.meta,
                application_data,
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
                f"Seems like your predict agent is missconfigured on user input: {user_input}",
                thread_id, message_id, tasknode_task.meta,
                human_readable="Seems like your agent is configured incorrectly",
                execution_start_time=execution_start_time
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
            # Check for LLM rate limit errors
            if LLM_RATE_LIMIT_ERRORS and isinstance(e, LLM_RATE_LIMIT_ERRORS):
                return execution_error(
                    node_interface, user_input, chat_history,
                    f"LLM RateLimitError on user input",
                    thread_id, message_id, tasknode_task.meta,
                    human_readable="The AI service rate limit was exceeded. Please try again in a moment.",
                    execution_start_time=execution_start_time
                )
            # Check for LLM authentication/authorization errors (model access denied, invalid keys, etc.)
            if LLM_AUTH_ERRORS and isinstance(e, LLM_AUTH_ERRORS):
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
