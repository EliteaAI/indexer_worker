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

import re
from time import monotonic
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler  # pylint: disable=E0401

try:
    from elitea_sdk.runtime.langchain.constants import LOAD_SKILL_UNKNOWN, LOADED_SKILL_PREFIX_RE
except ImportError:
    # Shim for SDKs predating the shared patterns, as in agent_execution_common
    LOADED_SKILL_PREFIX_RE = re.compile(r'^Skill "([^"]+)" is now active')
    LOAD_SKILL_UNKNOWN = 'No skill named "{name}" is attached to this agent.'

from pylon.core.tools import log

from .usage_tool_events import (
    ENTITY_TYPE_APPLICATION,
    SKILL_OUTCOME_UNKNOWN,
    SKILL_SOURCE_LOAD,
    record_skill_event,
    record_tool_event,
)

LOAD_SKILL_TOOL_NAME = "load_skill"
_UNKNOWN_SKILL_PREFIX = LOAD_SKILL_UNKNOWN.split("{name}", 1)[0]
_SKILL_BODY_RE = re.compile(r'<skill name="[^"]*">\n(.*)\n</skill>\s*$', re.DOTALL)


class UsageToolCallback(BaseCallbackHandler):
    """Writes one usage_event row per completed or failed tool call, plus a skill row when
    load_skill returned a skill body (#6926)."""

    def __init__(self, attribution: dict, attached_skills=None, subagent_skills=None):
        self._attribution = attribution
        # The run's own registry (a parallel child carries its own)
        self._skills_by_name = skill_identities(attached_skills)
        # An in-process sub-agent loads from its own registry, never the root's; it is known
        # only for the sub-agents core prefetched, otherwise the load stays name-only
        self._subagent_skills = subagent_skills or {}
        self._open = {}  # lc run_id -> {'name', 'started', 'meta', 'inputs'}

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
                "inputs": kwargs.get("inputs"),
            }
        except Exception as exc:  # pylint: disable=W0703
            log.debug("usage: tool start capture failed: %s", exc)

    def on_tool_end(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        self._finish(run_id, is_error=False, output=args[0] if args else kwargs.get("output"))

    def on_tool_error(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        self._finish(run_id, is_error=True)

    def _finish(self, run_id, is_error, output=None):
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
            if started.get("name") == LOAD_SKILL_TOOL_NAME:
                self._record_skill_load(started, output)
        except Exception as exc:  # pylint: disable=W0703
            log.debug("usage: tool end capture failed: %s", exc)

    def _record_skill_load(self, started, output):
        """A skill row only when the body was returned; "already loaded" answers write none."""
        content = getattr(output, "content", output)
        if not isinstance(content, str):
            return
        tool_meta = started.get("meta") or {}
        in_process_parent = tool_meta.get("parent_agent_name")
        parent_agent_name = in_process_parent or self._child_agent_name()
        loaded = LOADED_SKILL_PREFIX_RE.match(content)
        if loaded:
            name = loaded.group(1)
            registry = (
                self._subagent_skills.get(in_process_parent.strip().lower(), {})
                if in_process_parent else self._skills_by_name
            )
            registered = registry.get(name.strip().lower(), {})
            body = _SKILL_BODY_RE.search(content)
            record_skill_event(
                self._attribution,
                {
                    "skill_id": registered.get("skill_id"),
                    "skill_version_id": registered.get("skill_version_id"),
                    "name": name,
                },
                SKILL_SOURCE_LOAD,
                body_chars=len(body.group(1)) if body else 0,
                parent_agent_name=parent_agent_name,
            )
        elif content.startswith(_UNKNOWN_SKILL_PREFIX):
            record_skill_event(
                self._attribution,
                {"name": _requested_skill(started.get("inputs"), content)},
                SKILL_SOURCE_LOAD,
                outcome=SKILL_OUTCOME_UNKNOWN,
                parent_agent_name=parent_agent_name,
            )

    def _child_agent_name(self):
        """A parallel sub-agent child runs as its own task: its leaf is not the run's root."""
        attribution = self._attribution or {}
        is_child = (
            attribution.get("entity_type") == ENTITY_TYPE_APPLICATION
            and attribution.get("entity_id") != attribution.get("root_entity_id")
        )
        return attribution.get("entity_name") if is_child else None


def skill_identities(skills):
    """{skill name (lowercased): {'skill_id', 'skill_version_id'}} for a runtime skill registry."""
    return {
        s["name"].strip().lower(): {"skill_id": s.get("skill_id"), "skill_version_id": s.get("skill_version_id")}
        for s in skills or []
        if isinstance(s, dict) and s.get("name")
    }


def subagent_skill_registries(application):
    """{sub-agent name (lowercased): skill_identities} from the sub-agents core prefetched.

    The SDK names a sub-agent's tool calls by the sub-agent's name only, so a name shared by
    two prefetched sub-agents with different skills is left out rather than guessed.
    """
    registries = {}
    ambiguous = set()
    prefetched = (application or {}).get("subagent_version_details") if isinstance(application, dict) else None
    for entry in (prefetched or {}).values():
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        agent = entry["name"].strip().lower()
        skills = skill_identities((entry.get("version_details") or {}).get("attached_skills"))
        if agent in registries and registries[agent] != skills:
            ambiguous.add(agent)
        registries[agent] = skills
    return {agent: skills for agent, skills in registries.items() if agent not in ambiguous}


def _requested_skill(inputs, content):
    if isinstance(inputs, dict) and isinstance(inputs.get("skill"), str):
        return inputs["skill"]
    return content[len(_UNKNOWN_SKILL_PREFIX):].split('"', 1)[0] or None
