"""In-agent skill activations become usage_event rows (#6926).

A load_skill that returned a body, and a ~mention on a run's first dispatch, each write one
event_type='skill' row under the agent's run: zero cost, tool_name NULL, never an error. The
generic load_skill tool row is kept as it was.

Run: python3 -m pytest test_6926_skill_usage_rows.py
"""

import base64
import importlib
import json
import pathlib
import re
import sys
import types
import uuid

import pytest

import test_6677_evaluation_attribution  # noqa: F401  installs the pylon/elitea_sdk stubs

UTILS_DIR = pathlib.Path(__file__).resolve().parents[1] / "utils"

# Verbatim from elitea_sdk.runtime.langchain.constants: what load_skill actually answers
LOADED_SKILL_PREFIX = 'Skill "{name}" is now active'
LOADED_SKILL_RESULT = LOADED_SKILL_PREFIX + """. Follow these instructions exactly.

<skill name="{name}">
{instructions}
</skill>"""
LOAD_SKILL_ALREADY_ACTIVE = (
    'Skill "{name}" is already loaded — its instructions are already in this conversation.'
)
LOAD_SKILL_UNKNOWN = (
    "No skill named \"{name}\" is attached to this agent. Available skills: {available}. "
    "Call load_skill with one of these exact names, or proceed without a skill if none apply."
)


def _load_package():
    constants = types.ModuleType("elitea_sdk.runtime.langchain.constants")
    constants.LOADED_SKILL_PREFIX_RE = re.compile(
        "^" + re.escape(LOADED_SKILL_PREFIX).replace(re.escape("{name}"), '([^"]+)')
    )
    constants.LOAD_SKILL_UNKNOWN = LOAD_SKILL_UNKNOWN
    sys.modules["elitea_sdk.runtime.langchain"] = types.ModuleType("elitea_sdk.runtime.langchain")
    sys.modules["elitea_sdk.runtime.langchain.constants"] = constants
    package = types.ModuleType("usage_utils_6926")
    package.__path__ = [str(UTILS_DIR)]
    sys.modules["usage_utils_6926"] = package
    return tuple(importlib.import_module(f"usage_utils_6926.{name}") for name in (
        "usage_tool_events", "usage_skill_events", "usage_tool_callback", "usage_skill_callback",
    ))


events, skill_events, tool_callback_module, callback_module = _load_package()


class RunCallbacks:
    """Both usage handlers, attached together the way an agent run attaches them."""

    def __init__(self, attribution, *args, **kwargs):
        self.handlers = [
            tool_callback_module.UsageToolCallback(attribution),
            callback_module.UsageSkillCallback(attribution, *args, **kwargs),
        ]

    def __getattr__(self, event):
        return lambda *args, **kwargs: [getattr(h, event)(*args, **kwargs) for h in self.handlers]

REGISTRY = [
    {"skill_id": 41, "skill_version_id": 410, "name": "pdf-report", "instructions": "Write a PDF."},
    {"skill_id": 42, "skill_version_id": 420, "name": "tone", "instructions": "Be kind."},
]
AGENT_RUN = {
    "project_id": 3, "user_id": 9, "run_id": "run-1", "task_id": "t-1", "conversation_id": "c-1",
    "entity_type": "application", "entity_id": 7, "entity_version_id": 70, "entity_name": "Agent (base)",
    "root_entity_type": "application", "root_entity_id": 7, "root_entity_version_id": 70,
    "root_entity_project_id": 3,
}


@pytest.fixture
def rows(monkeypatch):
    """Every row written, with the statement that wrote it; .statements has every statement."""
    written = Rows()

    class Connection:
        def execute(self, statement, params):
            written.statements.append((str(statement), params))
            if "event_type" in params:
                written.append({**params, "_sql": str(statement)})

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(events, "_get_engine", lambda: types.SimpleNamespace(connect=Connection))
    return written


class Rows(list):
    def __init__(self):
        super().__init__()
        self.statements = []


def skill_rows(written):
    return [r for r in written if r.get("event_type") == "skill"]


def run_tool(callback, output, *, skill="pdf-report", metadata=None, error=False):
    run_id = uuid.uuid4()
    callback.on_tool_start(
        {"name": "load_skill"}, json.dumps({"skill": skill}),
        run_id=run_id, inputs={"skill": skill}, metadata=metadata or {},
    )
    if error:
        callback.on_tool_error(RuntimeError("boom"), run_id=run_id)
    else:
        callback.on_tool_end(output, run_id=run_id)


def loaded(name, instructions):
    return LOADED_SKILL_RESULT.format(name=name, instructions=instructions)


class TestLoadSkill:
    def test_a_returned_body_writes_one_skill_row_and_keeps_the_tool_row(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(callback, loaded("pdf-report", "Write a PDF."))

        tool_rows = [r for r in rows if r.get("event_type") == "tool"]
        assert [r["tool_name"] for r in tool_rows] == ["load_skill"]
        [row] = skill_rows(rows)
        assert (row["entity_type"], row["entity_id"], row["entity_version_id"], row["entity_name"]) == (
            "skill", 41, 410, "pdf-report",
        )
        assert (row["root_entity_type"], row["root_entity_id"]) == ("application", 7)
        assert (row["run_id"], row["conversation_id"], row["project_id"]) == ("run-1", "c-1", 3)
        assert row["tool_name"] is None
        assert row["is_error"] is False
        assert json.loads(row["meta"]) == {
            "source": "load_skill", "outcome": "loaded", "body_chars": 12, "est_body_tokens": 3,
        }

    def test_the_generic_tool_callback_alone_writes_no_skill_row(self, rows):
        callback = tool_callback_module.UsageToolCallback(AGENT_RUN)

        run_tool(callback, loaded("pdf-report", "Write a PDF."))

        assert [(r["event_type"], r["tool_name"]) for r in rows] == [("tool", "load_skill")]

    def test_the_row_carries_no_tokens_or_cost(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(callback, loaded("pdf-report", "Write a PDF."))

        [row] = skill_rows(rows)
        assert not {key for key in row if "token" in key or "cost" in key}
        assert "input_tokens" not in row["_sql"] and "cost" not in row["_sql"]

    def test_a_tool_message_output_is_read_through_its_content(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(callback, types.SimpleNamespace(content=loaded("tone", "Be kind.")), skill="tone")

        assert [r["entity_id"] for r in skill_rows(rows)] == [42]

    def test_an_already_loaded_answer_writes_no_skill_row(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(callback, LOAD_SKILL_ALREADY_ACTIVE.format(name="pdf-report"))

        assert skill_rows(rows) == []
        assert [r["tool_name"] for r in rows] == ["load_skill"]

    def test_a_failed_load_skill_writes_no_skill_row(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(callback, None, error=True)

        assert skill_rows(rows) == []

    def test_another_tool_never_writes_a_skill_row(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)
        run_id = uuid.uuid4()

        callback.on_tool_start({"name": "read_file"}, "{}", run_id=run_id)
        callback.on_tool_end(loaded("pdf-report", "x"), run_id=run_id)

        assert skill_rows(rows) == []

    def test_an_unknown_name_is_recorded_without_an_error(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(callback, LOAD_SKILL_UNKNOWN.format(name="nope", available="pdf-report, tone"), skill="nope")

        [row] = skill_rows(rows)
        assert (row["entity_id"], row["entity_version_id"], row["entity_name"]) == (None, None, "nope")
        assert row["is_error"] is False
        assert json.loads(row["meta"])["outcome"] == "unknown_skill"
        assert [r["is_error"] for r in rows] == [False, False]

    def test_an_unknown_name_falls_back_to_the_tool_message(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)
        run_id = uuid.uuid4()

        callback.on_tool_start({"name": "load_skill"}, "nope", run_id=run_id)
        callback.on_tool_end(LOAD_SKILL_UNKNOWN.format(name="nope", available="tone"), run_id=run_id)

        assert [r["entity_name"] for r in skill_rows(rows)] == ["nope"]

    def test_every_skill_row_is_written_once_per_run_source_and_skill(self, rows):
        first = RunCallbacks(AGENT_RUN, REGISTRY)
        second = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(first, loaded("pdf-report", "x"))
        run_tool(second, loaded("pdf-report", "x"))

        keys = {r["idempotency_key"] for r in skill_rows(rows)}
        assert keys == {"skill:run-1:load_skill:41"}
        assert all("WHERE NOT EXISTS" in r["_sql"] for r in skill_rows(rows))

    def test_writers_of_one_key_take_its_lock_before_inserting(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(callback, loaded("pdf-report", "x"))

        (lock_sql, lock_params), (insert_sql, insert_params) = rows.statements[-2:]
        assert "pg_advisory_xact_lock" in lock_sql
        assert lock_params["idempotency_key"] == insert_params["idempotency_key"] == "skill:run-1:load_skill:41"
        assert "WHERE NOT EXISTS" in insert_sql

    def test_tool_rows_take_no_lock(self, rows):
        events.record_tool_event(AGENT_RUN, "read_file", 5, False, "lc-1")

        assert ["pg_advisory_xact_lock" in sql for sql, _ in rows.statements] == [False]


class TestSubAgents:
    def test_an_in_process_sub_agent_load_is_name_only_with_its_parent(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(callback, loaded("pdf-report", "x"), metadata={"parent_agent_name": "Researcher"})

        [row] = skill_rows(rows)
        assert (row["entity_id"], row["entity_name"]) == (None, "pdf-report")
        assert (row["run_id"], row["root_entity_id"]) == ("run-1", 7)
        assert json.loads(row["meta"])["parent_agent_name"] == "Researcher"

    def test_an_in_process_sub_agent_load_resolves_through_its_prefetched_registry(self, rows):
        subagents = callback_module.subagent_skill_registries({"subagent_version_details": {
            "12:120": {"name": "Researcher", "version_details": {"attached_skills": [
                {"skill_id": 55, "skill_version_id": 551, "name": "pdf-report", "instructions": "x"},
            ]}},
        }})
        callback = RunCallbacks(AGENT_RUN, REGISTRY, subagents)

        run_tool(callback, loaded("pdf-report", "x"), metadata={"parent_agent_name": "Researcher"})

        [row] = skill_rows(rows)
        assert (row["entity_id"], row["entity_version_id"]) == (55, 551)
        assert row["root_entity_id"] == 7
        assert json.loads(row["meta"])["parent_agent_name"] == "Researcher"

    def test_a_sub_agent_name_shared_with_different_skills_stays_name_only(self, rows):
        subagents = callback_module.subagent_skill_registries({"subagent_version_details": {
            "12:120": {"name": "Researcher", "version_details": {"attached_skills": [
                {"skill_id": 55, "skill_version_id": 551, "name": "pdf-report"},
            ]}},
            "13:130": {"name": "researcher", "version_details": {"attached_skills": [
                {"skill_id": 56, "skill_version_id": 561, "name": "pdf-report"},
            ]}},
        }})
        callback = RunCallbacks(AGENT_RUN, REGISTRY, subagents)

        run_tool(callback, loaded("pdf-report", "x"), metadata={"parent_agent_name": "Researcher"})

        assert [(r["entity_id"], r["entity_name"]) for r in skill_rows(rows)] == [(None, "pdf-report")]

    def test_two_versions_of_one_sub_agent_with_the_same_skills_still_resolve(self):
        same = {"attached_skills": [{"skill_id": 55, "skill_version_id": 551, "name": "pdf-report"}]}

        registries = callback_module.subagent_skill_registries({"subagent_version_details": {
            "12:120": {"name": "Researcher", "version_details": same},
            "12:121": {"name": "Researcher", "version_details": same},
        }})

        assert registries == {"researcher": {"pdf-report": {"skill_id": 55, "skill_version_id": 551}}}

    def test_no_prefetch_means_no_sub_agent_registries(self):
        assert callback_module.subagent_skill_registries({"id": 7}) == {}
        assert callback_module.subagent_skill_registries(None) == {}

    def test_a_parallel_child_resolves_its_own_registry_and_names_itself(self, rows):
        child = {**AGENT_RUN, "entity_id": 31, "entity_version_id": 310, "entity_name": "Child (base)"}
        callback = RunCallbacks(child, [
            {"skill_id": 55, "skill_version_id": 550, "name": "pdf-report", "instructions": "x"},
        ])

        run_tool(callback, loaded("pdf-report", "x"))

        [row] = skill_rows(rows)
        assert (row["entity_id"], row["entity_version_id"]) == (55, 550)
        assert (row["run_id"], row["root_entity_id"]) == ("run-1", 7)
        assert json.loads(row["meta"])["parent_agent_name"] == "Child (base)"

    def test_the_root_agent_names_no_parent(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY)

        run_tool(callback, loaded("pdf-report", "x"))

        assert "parent_agent_name" not in json.loads(skill_rows(rows)[0]["meta"])


MENTIONED = [
    {"skill_id": 42, "skill_version_id": 420, "name": "tone", "instructions": "Be kind."},
    {"skill_id": 41, "skill_version_id": 410, "name": "pdf-report", "instructions": "Write a PDF."},
]


class TestMentions:
    @staticmethod
    def start_run(callback, parent_run_id=None):
        callback.on_chain_start({}, {}, run_id=uuid.uuid4(), parent_run_id=parent_run_id)

    def test_mentions_are_written_when_the_run_starts_not_when_built(self, rows):
        callback = RunCallbacks(
            AGENT_RUN, REGISTRY, mentioned_skills=skill_events.fresh_dispatch_mentions({"invoked_skills": MENTIONED}),
        )
        assert rows == []

        self.start_run(callback)

        assert [(r["entity_id"], r["entity_version_id"]) for r in rows] == [(42, 420), (41, 410)]
        assert [json.loads(r["meta"])["source"] for r in rows] == ["mention", "mention"]
        assert {r["tool_name"] for r in rows} == {None}
        assert json.loads(rows[0]["meta"])["body_chars"] == len("Be kind.")

    def test_later_chain_starts_write_nothing_more(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY, mentioned_skills=MENTIONED)
        self.start_run(callback)
        written = len(rows)

        self.start_run(callback, parent_run_id=uuid.uuid4())
        self.start_run(callback)

        assert len(rows) == written == 2

    @pytest.mark.parametrize("flag", ["hitl_resume", "mcp_auth_resume", "should_continue", "parallel_reconcile"])
    def test_a_resume_has_no_mentions_to_write(self, flag):
        assert skill_events.fresh_dispatch_mentions({"invoked_skills": MENTIONED, flag: True}) == []

    def test_a_dispatch_without_mentions_writes_nothing(self, rows):
        callback = RunCallbacks(
            AGENT_RUN, REGISTRY, mentioned_skills=skill_events.fresh_dispatch_mentions({"invoked_skills": []}),
        )

        self.start_run(callback)

        assert rows == []

    def test_a_mention_and_a_load_of_one_skill_are_separate_rows(self, rows):
        callback = RunCallbacks(AGENT_RUN, REGISTRY, mentioned_skills=MENTIONED[:1])
        self.start_run(callback)
        run_tool(callback, loaded("tone", "x"), skill="tone")

        assert sorted(r["idempotency_key"] for r in skill_rows(rows)) == [
            "skill:run-1:load_skill:42", "skill:run-1:mention:42",
        ]


class TestFailSafe:
    def test_a_write_failure_never_raises(self, monkeypatch):
        def broken():
            raise RuntimeError("partition missing")

        monkeypatch.setattr(events, "_get_engine", broken)

        skill_events.record_skill_event(AGENT_RUN, MENTIONED[0], skill_events.SKILL_SOURCE_MENTION)

    def test_no_project_writes_nothing(self, rows):
        callback = RunCallbacks({**AGENT_RUN, "project_id": None}, mentioned_skills=MENTIONED)
        TestMentions.start_run(callback)

        assert rows == []


def _decode(header):
    return json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))


class TestSkillRootAttribution:
    """S1/S4 stamp the skill as leaf and root; the signed header carries it to the llm rows."""

    KWARGS = {
        events.ENTITY_KWARGS_KEY: {"type": "skill", "id": 41, "version_id": 410, "name": "pdf-report"},
        events.ROOT_ENTITY_KWARGS_KEY: {"type": "skill", "id": 41, "version_id": 410, "project_id": 1},
        "application": {"id": None, "name": "pdf-report"},
        "conversation_id": "c-9",
    }

    def test_the_skill_is_leaf_and_root(self):
        out = events.run_attribution(self.KWARGS)

        assert (out["entity_type"], out["entity_id"], out["entity_version_id"], out["entity_name"]) == (
            "skill", 41, 410, "pdf-report",
        )
        assert (out["root_entity_type"], out["root_entity_id"], out["root_entity_project_id"]) == ("skill", 41, 1)

    def test_the_header_signs_the_skill_entity_for_the_billed_project(self):
        decoded = _decode(events.attribution_header(self.KWARGS, project_id=2, key=b"k" * 32))
        signature = decoded.pop("sig")

        assert decoded["entity_type"] == decoded["root_entity_type"] == "skill"
        assert signature == events.sign_attribution(decoded, 2, b"k" * 32)
