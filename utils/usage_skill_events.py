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
#: Twin of elitea_core utils/usage_attribution.ENTITY_TYPE_SKILL
ENTITY_TYPE_SKILL = "skill"
SKILL_SOURCE_LOAD = "load_skill"
SKILL_SOURCE_MENTION = "mention"
SKILL_OUTCOME_LOADED = "loaded"
SKILL_OUTCOME_UNKNOWN = "unknown_skill"
BODY_CHARS_PER_TOKEN = 4
RESUME_DISPATCH_FLAGS = ("hitl_resume", "mcp_auth_resume", "should_continue", "parallel_reconcile")

# The unique key includes ts, so ON CONFLICT cannot stop a later duplicate
_INSERT_ONCE_SQL = f"""
INSERT INTO {_SCHEMA}.usage_event ({_COLUMNS})
SELECT {_VALUES}
WHERE NOT EXISTS (
    SELECT 1 FROM {_SCHEMA}.usage_event WHERE idempotency_key = :idempotency_key
)
ON CONFLICT (idempotency_key, ts) DO NOTHING
"""

def record_skill_event(attribution, skill, source, outcome=SKILL_OUTCOME_LOADED, body_chars=0,
                       parent_agent_name=None):
    if not attribution or not attribution.get("project_id") or not isinstance(skill, dict):
        return
    try:
        name = skill.get("name")
        skill_key = skill.get("skill_id") or (name or "").strip().lower()
        if outcome != SKILL_OUTCOME_LOADED:
            skill_key = f"{outcome}:{skill_key}"
        meta = {
            "source": source,
            "outcome": outcome,
            "body_chars": body_chars,
            "est_body_tokens": body_chars // BODY_CHARS_PER_TOKEN,
        }
        if parent_agent_name:
            meta["parent_agent_name"] = parent_agent_name
        params = _row_params(
            attribution, f"skill:{_correlation(attribution)}:{source}:{skill_key}",
            {
                "entity_type": ENTITY_TYPE_SKILL,
                "entity_id": skill.get("skill_id"),
                "entity_version_id": skill.get("skill_version_id"),
                "entity_name": name,
            },
            event_type=EVENT_TYPE_SKILL,
            tool_name=None,
            duration_ms=None,
            is_error=False,
            meta=json.dumps(meta),
        )
    except Exception as exc:  # pylint: disable=W0703
        log.warning("usage: failed to build skill usage event: %s", exc)
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
