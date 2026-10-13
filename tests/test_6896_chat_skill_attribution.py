"""#6896 - a chat skill loaded or mentioned inside an agent run is credited to its own project."""
import json

import pytest

from test_6926_skill_usage_rows import (  # noqa: F401
    AGENT_RUN,
    RunCallbacks,
    loaded,
    rows,
    run_tool,
    skill_rows,
)

CATALOG_PROJECT_ID = 1
REGISTRY = [
    {"skill_id": 10, "skill_version_id": 100, "name": "reviewer", "instructions": "Review."},
    {"skill_id": 10, "skill_version_id": 200, "skill_project_id": CATALOG_PROJECT_ID,
     "name": "ui-review", "instructions": "Review the screenshot."},
]


def test_a_loaded_catalog_chat_skill_records_its_project(rows):  # noqa: F811
    run_tool(RunCallbacks(AGENT_RUN, REGISTRY), loaded("ui-review", "Review the screenshot."), skill="ui-review")

    [row] = skill_rows(rows)
    assert (row["entity_id"], row["entity_version_id"]) == (10, 200)
    assert json.loads(row["meta"])["skill_project_id"] == CATALOG_PROJECT_ID


def test_a_loaded_own_skill_keeps_the_runs_project(rows):  # noqa: F811
    run_tool(RunCallbacks(AGENT_RUN, REGISTRY), loaded("reviewer", "Review."), skill="reviewer")

    [row] = skill_rows(rows)
    assert (row["entity_id"], row["entity_version_id"]) == (10, 100)
    assert "skill_project_id" not in json.loads(row["meta"])


def test_a_mentioned_catalog_chat_skill_records_its_project(rows):  # noqa: F811
    callback = RunCallbacks(AGENT_RUN, REGISTRY, mentioned_skills=[REGISTRY[1]])

    callback.on_chain_start({}, {}, run_id="chain-1")

    [row] = skill_rows(rows)
    assert json.loads(row["meta"])["skill_project_id"] == CATALOG_PROJECT_ID


def test_same_id_skills_from_two_projects_get_separate_rows(rows):  # noqa: F811
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    run_tool(callback, loaded("reviewer", "Review."), skill="reviewer")
    run_tool(callback, loaded("ui-review", "Review the screenshot."), skill="ui-review")

    keys = [row["idempotency_key"] for row in skill_rows(rows)]
    assert len(keys) == 2 and len(set(keys)) == 2



@pytest.mark.parametrize("bad_project_id", ["abc", "1", 1.5, True, 0, -3, 1_000_000_000, 2**31 - 1])
def test_a_project_id_that_is_not_a_project_is_not_recorded(rows, bad_project_id):  # noqa: F811
    registry = [{**REGISTRY[1], "skill_project_id": bad_project_id}]

    run_tool(RunCallbacks(AGENT_RUN, registry), loaded("ui-review", "Review the screenshot."), skill="ui-review")

    [row] = skill_rows(rows)
    assert "skill_project_id" not in json.loads(row["meta"])
    assert row["idempotency_key"].endswith(":load_skill:10")


def test_the_largest_project_id_analytics_can_read_is_recorded(rows):  # noqa: F811
    registry = [{**REGISTRY[1], "skill_project_id": 999_999_999}]

    run_tool(RunCallbacks(AGENT_RUN, registry), loaded("ui-review", "Review the screenshot."), skill="ui-review")

    [row] = skill_rows(rows)
    assert json.loads(row["meta"])["skill_project_id"] == 999_999_999
