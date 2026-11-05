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
import time
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler  # pylint: disable=E0401

try:
    from elitea_sdk.runtime.langchain.constants import LOAD_SKILL_UNKNOWN, LOADED_SKILL_PREFIX_RE
except ImportError:
    LOADED_SKILL_PREFIX_RE = re.compile(r'^Skill "([^"]+)" is now active')
    LOAD_SKILL_UNKNOWN = 'No skill named "{name}" is attached to this agent.'

try:
    from elitea_sdk.runtime.langchain.constants import READ_SKILL_FILE_PREFIX_RE, SKILL_FILES_BLOCK_END
except ImportError:
    READ_SKILL_FILE_PREFIX_RE = re.compile(
        '^Skill "([^"]+)" file "([^"]+)"((?: lines [0-9]+-(?:[0-9]+|end))?):\n'
    )
    SKILL_FILES_BLOCK_END = "</skill_files>"

from pylon.core.tools import log

from .usage_skill_events import (
    SKILL_OUTCOME_UNKNOWN,
    SKILL_SOURCE_LOAD,
    SKILL_SOURCE_READ_FILE,
    record_skill_event,
    record_skill_mention,
    record_skill_script_run,
)
from .usage_tool_events import ENTITY_TYPE_APPLICATION

LOAD_SKILL_TOOL_NAME = "load_skill"
READ_SKILL_FILE_TOOL_NAME = "read_skill_file"
LIST_SKILL_FILES_TOOL_NAME = "list_skill_files"
RUN_SKILL_SCRIPT_TOOL_NAME = "run_skill_script"
_UNKNOWN_SKILL_PREFIX = LOAD_SKILL_UNKNOWN.split("{name}", 1)[0]
_SKILL_BODY_RE = re.compile(r'<skill name="[^"]*">\n(.*)\n</skill>', re.DOTALL)
_SKILL_FILE_CONTENT_RE = re.compile(r'<skill_file path="[^"]*">\n(.*)\n</skill_file>\s*$', re.DOTALL)


def is_skill_tool(tool_name):
    """Whether a tool's results can carry a skill body, a skill file or a script run worth a usage row."""
    return tool_name in (LOAD_SKILL_TOOL_NAME, READ_SKILL_FILE_TOOL_NAME, LIST_SKILL_FILES_TOOL_NAME, RUN_SKILL_SCRIPT_TOOL_NAME)


def _split_auto_loaded(content):
    """A file tool's result as the skill body it loaded on the way (empty when none) and the tool's own answer."""
    if not isinstance(content, str) or not LOADED_SKILL_PREFIX_RE.match(content):
        return "", content
    loaded, _separator, answer = content.partition(SKILL_FILES_BLOCK_END)
    return loaded + SKILL_FILES_BLOCK_END, answer.lstrip("\n")


class UsageSkillCallback(BaseCallbackHandler):
    """One skill row when load_skill or a file tool loading on the way returned a body, one when read_skill_file returned
    a file, and one per ~mention of the run."""

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
            tool_name = serialized.get("name") or kwargs.get("name")
            if not is_skill_tool(tool_name):
                return
            self._open[str(run_id)] = {
                "tool_name": tool_name,
                "parent_agent_name": (kwargs.get("metadata") or {}).get("parent_agent_name"),
                "inputs": kwargs.get("inputs"),
                "started": time.monotonic(),
                "run_id": str(run_id),
            }
        except Exception as exc:  # pylint: disable=W0703
            log.debug("usage: skill load start capture failed: %s", exc)

    def on_tool_end(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        try:
            started = self._open.pop(str(run_id), None)
            if started is None:
                return
            output = args[0] if args else kwargs.get("output")
            if started["tool_name"] == LOAD_SKILL_TOOL_NAME:
                self._record_skill_load(started, output)
                return
            loaded, answer = _split_auto_loaded(getattr(output, "content", output))
            if loaded:
                self._record_skill_load(started, loaded)
            if started["tool_name"] == RUN_SKILL_SCRIPT_TOOL_NAME:
                self._record_skill_script_run(started, output)
                return
            self._record_skill_file_read(started, answer)
        except Exception as exc:  # pylint: disable=W0703
            log.debug("usage: skill load capture failed: %s", exc)

    def on_tool_error(self, *args, run_id: UUID, **kwargs):
        """Callback"""
        self._open.pop(str(run_id), None)

    def _record_skill_load(self, started, output):
        content = getattr(output, "content", output)
        if not isinstance(content, str):
            return
        parent_agent_name = started.get("parent_agent_name") or self._child_agent_name()
        loaded = LOADED_SKILL_PREFIX_RE.match(content)
        if loaded:
            body = _SKILL_BODY_RE.search(content)
            record_skill_event(
                self._attribution,
                self._registered_skill(started, loaded.group(1)),
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

    def _record_skill_file_read(self, started, output):
        """One row per file read_skill_file served; refusals and repeats of a read in context write nothing."""
        content = getattr(output, "content", output)
        if not isinstance(content, str):
            return
        read = READ_SKILL_FILE_PREFIX_RE.match(content)
        if not read:
            return
        file_content = _SKILL_FILE_CONTENT_RE.search(content)
        record_skill_event(
            self._attribution,
            self._registered_skill(started, read.group(1)),
            SKILL_SOURCE_READ_FILE,
            parent_agent_name=started.get("parent_agent_name") or self._child_agent_name(),
            file_read={
                "path": read.group(2),
                "lines": read.group(3),
                "bytes": len(file_content.group(1).encode("utf-8")) if file_content else 0,
            },
        )

    def _record_skill_script_run(self, started, output):
        """One row per script the sandbox ran, from the facts the run carries; a refusal before the run carries none
        and writes nothing."""
        facts = getattr(output, "artifact", None)
        inputs = started.get("inputs") if isinstance(started.get("inputs"), dict) else {}
        if not isinstance(facts, dict) or not isinstance(inputs.get("skill"), str):
            return
        record_skill_script_run(
            self._attribution,
            self._registered_skill(started, inputs["skill"]),
            {
                "path": str(inputs.get("path") or ""),
                "exit_code": facts["exit_code"],
                "duration_ms": int((time.monotonic() - started["started"]) * 1000),
                "stdout_bytes": facts["stdout_bytes"],
                "stderr_bytes": facts["stderr_bytes"],
                "output_files": len(facts["shared"]),
                "timed_out": facts["timed_out"],
                "call_id": started["run_id"],
            },
            parent_agent_name=started.get("parent_agent_name") or self._child_agent_name(),
        )

    def _registered_skill(self, started, name):
        """The run's identity of a skill named in a tool result: the sub-agent's registry for its own calls."""
        in_process_parent = started.get("parent_agent_name")
        registry = (
            self._subagent_skills.get(in_process_parent.strip().lower(), {})
            if in_process_parent else self._skills_by_name
        )
        registered = registry.get(name.strip().lower(), {})
        return {
            "skill_id": registered.get("skill_id"),
            "skill_version_id": registered.get("skill_version_id"),
            "skill_project_id": registered.get("skill_project_id"),
            "name": name,
        }

    def _child_agent_name(self):
        attribution = self._attribution or {}
        is_child = (
            attribution.get("entity_type") == ENTITY_TYPE_APPLICATION
            and attribution.get("entity_id") != attribution.get("root_entity_id")
        )
        return attribution.get("entity_name") if is_child else None


def collect_disclosed_skills(kwargs):
    attached = [s for s in kwargs.get("attached_skills") or [] if isinstance(s, dict) and s.get("name")]
    taken = {s["name"].strip().lower() for s in attached}
    project = [
        s for s in kwargs.get("project_skills") or []
        if isinstance(s, dict) and s.get("name") and s["name"].strip().lower() not in taken
    ]
    return attached + project


def skill_identities(skills):
    return {
        s["name"].strip().lower(): {
            "skill_id": s.get("skill_id"),
            "skill_version_id": s.get("skill_version_id"),
            "skill_project_id": s.get("skill_project_id"),
        }
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
