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

"""Skill activation rows in `centry.usage_event` (#6926)"""

import json

from pylon.core.tools import log

from .usage_tool_events import _COLUMNS, _SCHEMA, _VALUES, _correlation, _row_params, _write

#: Zero tokens and cost: the skill body's tokens are already in the agent's own llm rows
EVENT_TYPE_SKILL = "skill"
EVENT_TYPE_SKILL_FILE = "skill_file"
EVENT_TYPE_SKILL_SCRIPT = "skill_script"
#: Twin of elitea_core utils/usage_attribution.ENTITY_TYPE_SKILL
ENTITY_TYPE_SKILL = "skill"
SKILL_SOURCE_LOAD = "load_skill"
SKILL_SOURCE_MENTION = "mention"
SKILL_SOURCE_READ_FILE = "read_skill_file"
SKILL_SOURCE_RUN_SCRIPT = "run_skill_script"
SKILL_OUTCOME_LOADED = "loaded"
SKILL_OUTCOME_UNKNOWN = "unknown_skill"
BODY_CHARS_PER_TOKEN = 4
RESUME_DISPATCH_FLAGS = ("hitl_resume", "mcp_auth_resume", "should_continue", "parallel_reconcile")
MAX_PROJECT_ID = 999_999_999

# The unique key includes ts, so ON CONFLICT cannot stop a later duplicate
_INSERT_ONCE_SQL = f"""
INSERT INTO {_SCHEMA}.usage_event ({_COLUMNS})
SELECT {_VALUES}
WHERE NOT EXISTS (
    SELECT 1 FROM {_SCHEMA}.usage_event WHERE idempotency_key = :idempotency_key
)
ON CONFLICT (idempotency_key, ts) DO NOTHING
"""

def own_project_id(skill):
    value = skill.get("skill_project_id")
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_PROJECT_ID:
        return value
    return None


def build_skill_key_segment(skill, outcome=SKILL_OUTCOME_LOADED):
    """The idempotency-key segment that names the skill: its project-qualified id, else its name, after any non-loaded outcome."""
    name = skill.get("name")
    skill_key = skill.get("skill_id") or (name or "").strip().lower()
    skill_project_id = own_project_id(skill)
    if skill_project_id and skill.get("skill_id"):
        skill_key = f"{skill_project_id}:{skill_key}"
    if outcome != SKILL_OUTCOME_LOADED:
        skill_key = f"{outcome}:{skill_key}"
    return skill_key


def record_skill_event(attribution, skill, source, outcome=SKILL_OUTCOME_LOADED, body_chars=0,
                       parent_agent_name=None, file_read=None):
    """One skill usage row per run and activation; a file read is a skill_file row keyed by its path and line range too."""
    if not attribution or not attribution.get("project_id") or not isinstance(skill, dict):
        return
    try:
        name = skill.get("name")
        skill_key = build_skill_key_segment(skill, outcome)
        skill_project_id = own_project_id(skill)
        meta = {
            "source": source,
            "outcome": outcome,
            "body_chars": body_chars,
            "est_body_tokens": body_chars // BODY_CHARS_PER_TOKEN,
        }
        if parent_agent_name:
            meta["parent_agent_name"] = parent_agent_name
        if skill_project_id:
            meta["skill_project_id"] = skill_project_id
        if file_read:
            meta["path"] = file_read["path"]
            meta["bytes"] = file_read["bytes"]
            skill_key = f"{skill_key}:{file_read['path']}{file_read.get('lines') or ''}"
        params = _row_params(
            attribution, f"skill:{_correlation(attribution)}:{source}:{skill_key}",
            {
                "entity_type": ENTITY_TYPE_SKILL,
                "entity_id": skill.get("skill_id"),
                "entity_version_id": skill.get("skill_version_id"),
                "entity_name": name,
            },
            event_type=EVENT_TYPE_SKILL_FILE if file_read else EVENT_TYPE_SKILL,
            tool_name=None,
            duration_ms=None,
            is_error=False,
            meta=json.dumps(meta),
        )
    except Exception as exc:  # pylint: disable=W0703
        log.warning("usage: failed to build skill usage event: %s", exc)
        return
    _write(_INSERT_ONCE_SQL, params, "skill", lock_key=True)


def record_skill_script_run(attribution, skill, script, parent_agent_name=None):
    """One skill_script row per script run, keyed by the tool call so a rerun of the same script is its own row."""
    if not attribution or not attribution.get("project_id") or not isinstance(skill, dict):
        return
    try:
        skill_project_id = own_project_id(skill)
        meta = {
            "source": SKILL_SOURCE_RUN_SCRIPT,
            "path": script["path"],
            "exit_code": script["exit_code"],
            "stdout_bytes": script["stdout_bytes"],
            "stderr_bytes": script["stderr_bytes"],
            "output_files": script["output_files"],
            "timed_out": script["timed_out"],
        }
        if parent_agent_name:
            meta["parent_agent_name"] = parent_agent_name
        if skill_project_id:
            meta["skill_project_id"] = skill_project_id
        params = _row_params(
            attribution,
            f"skill:{_correlation(attribution)}:{SKILL_SOURCE_RUN_SCRIPT}:{build_skill_key_segment(skill)}:{script['path']}:{script['call_id']}",
            {
                "entity_type": ENTITY_TYPE_SKILL,
                "entity_id": skill.get("skill_id"),
                "entity_version_id": skill.get("skill_version_id"),
                "entity_name": skill.get("name"),
            },
            event_type=EVENT_TYPE_SKILL_SCRIPT,
            tool_name=SKILL_SOURCE_RUN_SCRIPT,
            duration_ms=script["duration_ms"],
            is_error=script["exit_code"] != 0,
            meta=json.dumps(meta),
        )
    except Exception as exc:  # pylint: disable=W0703
        log.warning("usage: failed to build skill script usage event: %s", exc)
        return
    _write(_INSERT_ONCE_SQL, params, "skill", lock_key=True)


def fresh_dispatch_mentions(kwargs):
    """A resume re-dispatches the same run with the same message: only a first dispatch counts."""
    kwargs = kwargs or {}
    if any(kwargs.get(flag) for flag in RESUME_DISPATCH_FLAGS):
        return []
    return [s for s in kwargs.get("invoked_skills") or [] if isinstance(s, dict) and s.get("name")]


def record_skill_mention(attribution, skill):
    record_skill_event(
        attribution, skill, SKILL_SOURCE_MENTION, body_chars=len(skill.get("instructions") or ""),
    )
