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

"""usage_event rows for skill activations (#6926)"""

import re
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler  # pylint: disable=E0401

try:
    from elitea_sdk.runtime.langchain.constants import LOAD_SKILL_UNKNOWN, LOADED_SKILL_PREFIX_RE
except ImportError:
    LOADED_SKILL_PREFIX_RE = re.compile(r'^Skill "([^"]+)" is now active')
    LOAD_SKILL_UNKNOWN = 'No skill named "{name}" is attached to this agent.'

from pylon.core.tools import log

from .usage_skill_events import (
    SKILL_OUTCOME_UNKNOWN,
    SKILL_SOURCE_LOAD,
    record_skill_event,
    record_skill_mention,
)
from .usage_tool_events import ENTITY_TYPE_APPLICATION

LOAD_SKILL_TOOL_NAME = "load_skill"
_UNKNOWN_SKILL_PREFIX = LOAD_SKILL_UNKNOWN.split("{name}", 1)[0]
_SKILL_BODY_RE = re.compile(r'<skill name="[^"]*">\n(.*)\n</skill>\s*$', re.DOTALL)


class UsageSkillCallback(BaseCallbackHandler):
    """One skill row when load_skill returned a body, and one per ~mention of the run."""

    def __init__(self, attribution: dict, attached_skills=None, subagent_skills=None, mentioned_skills=None):
        self._attribution = attribution
        self._skills_by_name = skill_identities(attached_skills)
        self._subagent_skills = subagent_skills or {}
        # A ~mention fires no LangChain event of its own
        self._pending_mentions = list(mentioned_skills or [])
        self._open = {}

    def on_chain_start(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        if not self._pending_mentions:
            return
        pending, self._pending_mentions = self._pending_mentions, []
        for skill in pending:
            record_skill_mention(self._attribution, skill)

    def on_tool_start(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        try:
            serialized = args[0] if args and isinstance(args[0], dict) else {}
            if (serialized.get("name") or kwargs.get("name")) != LOAD_SKILL_TOOL_NAME:
                return
            self._open[str(run_id)] = {
                "parent_agent_name": (kwargs.get("metadata") or {}).get("parent_agent_name"),
                "inputs": kwargs.get("inputs"),
            }
        except Exception as exc:  # pylint: disable=W0703
            log.debug("usage: skill load start capture failed: %s", exc)

    def on_tool_end(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        try:
            started = self._open.pop(str(run_id), None)
            if started is not None:
                self._record_skill_load(started, args[0] if args else kwargs.get("output"))
        except Exception as exc:  # pylint: disable=W0703
            log.debug("usage: skill load capture failed: %s", exc)

    def on_tool_error(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        self._open.pop(str(run_id), None)

    def _record_skill_load(self, started, output):
        content = getattr(output, "content", output)
        if not isinstance(content, str):
            return
        in_process_parent = started.get("parent_agent_name")
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
        attribution = self._attribution or {}
        is_child = (
            attribution.get("entity_type") == ENTITY_TYPE_APPLICATION
            and attribution.get("entity_id") != attribution.get("root_entity_id")
        )
        return attribution.get("entity_name") if is_child else None


def skill_identities(skills):
    return {
        s["name"].strip().lower(): {"skill_id": s.get("skill_id"), "skill_version_id": s.get("skill_version_id")}
        for s in skills or []
        if isinstance(s, dict) and s.get("name")
    }


def subagent_skill_registries(application):
    """The SDK names a sub-agent's tool calls by its name only, so a name shared by two
    prefetched sub-agents with different skills is left out rather than guessed.
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
