#!/usr/bin/python3
# coding=utf-8

#   Copyright 2025 EPAM Systems
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

"""Langfuse tracing callback helper for agent execution"""

import sys
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pylon.core.tools import log


def _set_root_trace_io_from_chain_end(span, parent_run_id, inputs, outputs):
    """Set trace-level I/O on the LangChain root observation before it ends."""
    if span is None or parent_run_id is not None:
        return

    try:
        clean_input, clean_output = _normalize_root_trace_io(parent_run_id, inputs, outputs)
        span.set_trace_io(
            input=clean_input,
            output=clean_output,
        )
    except Exception as e:
        log.warning(f"Failed to update Langfuse trace I/O: {e}")


def _normalize_root_trace_io(parent_run_id, inputs, outputs, run_id=None, input_cache=None):
    if parent_run_id is not None:
        return inputs, outputs

    if inputs is None and input_cache is not None and run_id is not None:
        inputs = input_cache.pop(run_id, None)

    return _extract_clean_trace_input(inputs), _extract_clean_trace_output(outputs)


def _cache_root_trace_input(input_cache, run_id, parent_run_id, inputs):
    if parent_run_id is not None:
        return

    input_cache[run_id] = _extract_clean_trace_input(inputs)


def _extract_clean_trace_input(inputs):
    if isinstance(inputs, dict) and "input" in inputs:
        return _extract_content_text(inputs.get("input"))

    return _extract_content_text(inputs)


def _extract_clean_trace_output(outputs):
    if isinstance(outputs, dict):
        messages = outputs.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                message_type = _get_message_value(message, "type")
                if message_type in ("ai", "assistant") or _is_ai_message(message):
                    return _extract_content_text(_get_message_value(message, "content"))

        for key in ("output", "response", "content"):
            if key in outputs:
                return _extract_content_text(outputs.get(key))

    return _extract_content_text(outputs)


def _get_message_value(message, key):
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _is_ai_message(message):
    message_type = getattr(message, "type", None)
    if message_type in ("ai", "assistant"):
        return True
    return message.__class__.__name__ in ("AIMessage", "AIMessageChunk")


def _extract_content_text(value):
    """
    Extract text content from various response formats.

    Delegates to normalize_response_content() for consistent behavior
    between user-visible output and Langfuse traces.
    """
    from .agent_execution_common import normalize_response_content
    return normalize_response_content(value)


def _get_audit_callback(user_id=None, user_email=None, project_id=None):
    """Try to get an AuditLangChainCallback from the tracing plugin."""
    try:
        from tools import this
        tracing_module = this.for_module("tracing").module
        if not tracing_module.enabled:
            return None
        audit_config = tracing_module.config.get("audit_trail", {})
        if not audit_config.get("enabled", False):
            return None
        from plugins.tracing.utils.audit_langchain_callback import AuditLangChainCallback
        callback = AuditLangChainCallback(
            user_id=user_id,
            user_email=user_email,
            project_id=project_id,
        )
        return callback
    except Exception:
        return None


def _build_langfuse_callback_handler_class(callback_handler_cls):
    """Create the Elitea Langfuse callback subclass without importing Langfuse at module load."""
    class EliteaLangfuseCallbackHandler(callback_handler_cls):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._elitea_root_inputs = {}

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
            _cache_root_trace_input(
                self._elitea_root_inputs,
                run_id,
                parent_run_id,
                inputs,
            )
            try:
                return super().on_chain_start(
                    serialized,
                    inputs,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                log.warning(f"Langfuse on_chain_start failed: {e}")
                return None

        def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
            # Guard against Langfuse SDK changes to private _runs attribute.
            runs = getattr(self, "_runs", {})
            span = runs.get(run_id, None) if isinstance(runs, dict) else None
            clean_inputs, clean_outputs = _normalize_root_trace_io(
                parent_run_id,
                kwargs.get("inputs"),
                outputs,
                run_id=run_id,
                input_cache=self._elitea_root_inputs,
            )
            _set_root_trace_io_from_chain_end(
                span,
                parent_run_id,
                clean_inputs,
                clean_outputs,
            )
            if parent_run_id is None:
                kwargs["inputs"] = clean_inputs
                outputs = clean_outputs
            try:
                return super().on_chain_end(
                    outputs,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                log.warning(f"Langfuse on_chain_end failed: {e}")
                return None

    return EliteaLangfuseCallbackHandler


def create_langfuse_callback(
    langfuse_config: Optional[Dict[str, Any]],
    trace_name: str = "agent-execution",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    environment: Optional[str] = None,
):
    """
    Create a Langfuse CallbackHandler for LangChain integration.

    The Langfuse SDK 3.x requires initializing a Langfuse client first
    (which registers globally), then the CallbackHandler uses that client.

    Args:
        langfuse_config: Dict with base_url, public_key, secret_key
        trace_name: Name for the trace (e.g., agent or application name)
        session_id: Session/thread ID for grouping traces
        user_id: User ID for attribution
        metadata: Additional metadata dict (values should be strings)
        environment: Langfuse environment for trace segregation (e.g., project-{id})

    Returns:
        Tuple of (Langfuse client, CallbackHandler, trace_attrs) or (None, None, None)
        trace_attrs is a dict with session_id, user_id, metadata, trace_name for use with propagate_attributes
    """
    if not langfuse_config:
        log.debug("Langfuse config not provided, trying audit callback fallback")
        # Extract project_id from metadata (it's stored as string for Langfuse)
        project_id = None
        if metadata:
            raw_pid = metadata.get("project_id") or metadata.get("chat_project_id")
            if raw_pid and raw_pid not in ("", "None"):
                project_id = raw_pid
        audit_handler = _get_audit_callback(
            user_id=user_id,
            user_email=metadata.get("user_email") if metadata else None,
            project_id=project_id,
        )
        if audit_handler:
            log.info(f"Using AuditLangChainCallback (user_id={user_id}, project_id={project_id})")
            return None, audit_handler, None
        return None, None, None

    base_url = langfuse_config.get('base_url')
    public_key = langfuse_config.get('public_key')
    secret_key = langfuse_config.get('secret_key')

    if not all([base_url, public_key, secret_key]):
        log.warning("Langfuse config incomplete, skipping tracing")
        return None, None, None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Initialize Langfuse client first - this registers it globally
        # so the CallbackHandler can use it
        langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=environment,
        )

        # Create callback handler - it will use the globally registered client
        EliteaLangfuseCallbackHandler = _build_langfuse_callback_handler_class(CallbackHandler)
        handler = EliteaLangfuseCallbackHandler(
            public_key=public_key,
        )

        # Store trace attributes for use with propagate_attributes
        trace_attrs = {
            "trace_name": trace_name,
            "session_id": session_id,
            "user_id": user_id,
            "metadata": metadata,
        }

        log.info(f"Langfuse callback handler created for tracing: {trace_name}")
        return langfuse_client, handler, trace_attrs
    except ImportError:
        log.warning("langfuse package not installed, skipping tracing")
        return None, None, None
    except Exception as e:
        log.warning(f"Failed to create Langfuse callback: {e}")
        return None, None, None


@contextmanager
def langfuse_trace_context(trace_attrs: Optional[Dict[str, Any]], langfuse_client=None):
    """
    Context manager that propagates Langfuse trace attributes to LangChain callbacks.

    The LangChain CallbackHandler creates the root observation for agent execution.
    Creating an additional current observation here adds a duplicate wrapper span, so
    this context only propagates trace-level attributes.

    Args:
        trace_attrs: Dict with session_id, user_id, metadata, trace_name
        langfuse_client: The Langfuse client instance. Used only as a guard that Langfuse is enabled.

    Yields:
        None. Trace input/output can only be updated if an active observation exists.
    """
    if trace_attrs is None or langfuse_client is None:
        yield None
        return

    # Import and setup outside the yield to avoid masking application exceptions
    try:
        from langfuse import propagate_attributes
    except ImportError:
        log.debug("Langfuse not available, skipping trace context")
        yield None
        return

    try:
        trace_name = trace_attrs.get("trace_name", "agent-execution")
        ctx = propagate_attributes(
            trace_name=trace_name,
            session_id=trace_attrs.get("session_id"),
            user_id=trace_attrs.get("user_id"),
            metadata=trace_attrs.get("metadata"),
        )
    except Exception as e:
        log.warning(f"Failed to create Langfuse trace context: {e}")
        yield None
        return

    # Explicitly enter/exit context to guard against SDK failures
    # while allowing exceptions from the body to propagate normally
    try:
        ctx.__enter__()
    except Exception as e:
        log.warning(f"Failed to enter Langfuse trace context: {e}")
        yield None
        return

    try:
        yield None
    finally:
        # Pass exception info to __exit__ so Langfuse can mark traces as failed
        exc_info = sys.exc_info()
        try:
            ctx.__exit__(*exc_info)
        except Exception as e:
            log.warning(f"Failed to exit Langfuse trace context: {e}")

def _is_langfuse_on_shared_provider(langfuse_client, tracing_module):
    """True only if the Langfuse client emits into the tracing plugin's provider.

    The per-processor flush below only reaches Langfuse when Langfuse's span
    processor lives on the platform's shared TracerProvider — which holds today
    solely because the tracing plugin sets the global provider before any
    Langfuse client is created. If that ever stops being true (SDK or init-order
    change), Langfuse owns its own provider and must be flushed via its native
    ``client.flush()`` instead; flushing the shared provider would then flush
    OTLP/audit but never Langfuse — strictly worse than the old behaviour. Verify
    the assumption explicitly and fail loud rather than silently regress.
    """
    if tracing_module is None or not getattr(tracing_module, "enabled", False):
        return False

    shared_provider = getattr(tracing_module, "tracer_provider", None)
    lf_resources = getattr(langfuse_client, "_resources", None)
    lf_provider = getattr(lf_resources, "tracer_provider", None)
    if shared_provider is not None and lf_provider is shared_provider:
        return True

    log.warning(
        "Langfuse is not attached to the tracing plugin's shared TracerProvider; "
        "flushing via client.flush(). The Langfuse trace-export wiring may have changed."
    )
    return False


def flush_langfuse_callback(langfuse_client, handler):
    """
    Flush any pending traces from Langfuse.

    When the platform ``tracing`` plugin owns the global OTEL TracerProvider,
    Langfuse appends its span processor to that shared provider. Calling
    ``langfuse_client.flush()`` then routes through the provider's
    SynchronousMultiSpanProcessor, whose shared-deadline ``force_flush`` can be
    starved by an earlier, fork-unsafe OTLP processor in the forked task worker —
    dropping Langfuse's final span batch (the last executed node + the root span
    carrying trace-level I/O). See issues #5391 / #5390.

    So when Langfuse is confirmed to share that provider we flush each processor
    independently via the tracing plugin (which guarantees Langfuse's processor
    is reached). Otherwise (tracing disabled, or Langfuse owns its own provider)
    its native ``flush()`` is the correct path — see
    :func:`_is_langfuse_on_shared_provider`.

    Args:
        langfuse_client: The Langfuse client instance to flush
        handler: The CallbackHandler instance (for future use)
    """
    if langfuse_client is None:
        return

    # Resolve the tracing plugin (mirrors _get_audit_callback's `.module` deref).
    try:
        from tools import this
        tracing_module = this.for_module("tracing").module
    except Exception as e:
        tracing_module = None
        log.debug(f"Tracing module unavailable for Langfuse flush: {e}")

    try:
        if _is_langfuse_on_shared_provider(langfuse_client, tracing_module):
            tracing_module.flush_span_processors()
        else:
            langfuse_client.flush()
        log.debug("Langfuse traces flushed")
    except Exception as e:
        log.warning(f"Failed to flush Langfuse: {e}")
