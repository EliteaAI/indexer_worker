#!/usr/bin/python3
# coding=utf-8

#   Copyright 2026 EPAM Systems
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

"""usage_event rows for tool calls (#6572).

Deliberately separate from EliteACallback (the UI event stream) and from the
Langfuse/audit callbacks: those are None whenever tracing is disabled, and tool
usage rows must be written regardless of tracing.
"""

from time import monotonic
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler  # pylint: disable=E0401

from pylon.core.tools import log

from .usage_tool_events import record_tool_event


class UsageToolCallback(BaseCallbackHandler):
    """Writes one usage_event row per completed or failed tool call."""

    def __init__(self, attribution: dict):
        self._attribution = attribution
        self._open = {}  # lc run_id -> {'name', 'started', 'meta'}

    def on_tool_start(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        try:
            serialized = args[0] if args and isinstance(args[0], dict) else {}
            # The serialized name is the action that actually ran; execution
            # metadata.original_name can be the enclosing Application's name
            # (same trap as EliteACallback.on_tool_start).
            name = serialized.get("name") or kwargs.get("name")
            tool_meta = kwargs.get("metadata") or {}
            # Collision-free channel injected by the SDK's Application._run into
            # nested config: identifies the in-process sub-agent this tool ran
            # inside. Recorded in meta, never in entity_name — entity_* must stay
            # internally consistent with the ids the run payload actually carries.
            parent_agent_name = tool_meta.get("parent_agent_name")
            self._open[str(run_id)] = {
                "name": name,
                "started": monotonic(),
                "meta": {"parent_agent_name": parent_agent_name} if parent_agent_name else None,
            }
        except Exception as exc:  # pylint: disable=W0703
            log.debug("usage: tool start capture failed: %s", exc)

    def on_tool_end(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        self._finish(run_id, is_error=False)

    def on_tool_error(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        self._finish(run_id, is_error=True)

    def _finish(self, run_id, is_error):
        try:
            started = self._open.pop(str(run_id), None)
            if started is None:
                # No matching start (callback attached mid-flight): a row with no
                # duration would be worse than no row.
                return
            record_tool_event(
                self._attribution,
                tool_name=started.get("name"),
                duration_ms=int((monotonic() - started["started"]) * 1000),
                is_error=is_error,
                lc_run_id=run_id,
                meta=started.get("meta"),
            )
        except Exception as exc:  # pylint: disable=W0703
            log.debug("usage: tool end capture failed: %s", exc)
