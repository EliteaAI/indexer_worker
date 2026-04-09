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

from typing import Optional, Dict, Any
from contextlib import contextmanager
from pylon.core.tools import log


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


def create_langfuse_callback(
    langfuse_config: Optional[Dict[str, Any]],
    trace_name: str = "agent-execution",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
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
        )

        # Create callback handler - it will use the globally registered client
        handler = CallbackHandler(
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
def langfuse_trace_context(trace_attrs: Optional[Dict[str, Any]]):
    """
    Context manager that wraps agent execution with Langfuse propagate_attributes.

    This sets trace-level attributes (user_id, session_id, metadata, trace_name)
    on all spans created within the context.

    Args:
        trace_attrs: Dict with session_id, user_id, metadata, trace_name
    """
    if trace_attrs is None:
        yield
        return

    use_propagate = False
    try:
        from langfuse import propagate_attributes
        use_propagate = True
    except ImportError:
        pass

    if use_propagate:
        try:
            with propagate_attributes(
                trace_name=trace_attrs.get("trace_name"),
                session_id=trace_attrs.get("session_id"),
                user_id=trace_attrs.get("user_id"),
                metadata=trace_attrs.get("metadata"),
            ):
                yield
        except Exception:
            raise
    else:
        yield


def flush_langfuse_callback(langfuse_client, handler):
    """
    Flush any pending traces from Langfuse.

    Args:
        langfuse_client: The Langfuse client instance to flush
        handler: The CallbackHandler instance (for future use)
    """
    if langfuse_client is None:
        return

    try:
        langfuse_client.flush()
        log.debug("Langfuse client flushed")
    except Exception as e:
        log.warning(f"Failed to flush Langfuse: {e}")
