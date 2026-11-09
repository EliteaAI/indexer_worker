"""A run_skill_script that ran a script becomes one usage_event skill_script row from the facts the SDK attaches to
its result (#7185).

Run: python3 -m pytest test_7185_skill_script_usage_rows.py
"""

import json
import uuid

from langchain_core.messages import ToolMessage

from test_6926_skill_usage_rows import AGENT_RUN, RunCallbacks, rows, skill_rows  # noqa: F401  rows is a fixture

REGISTRY = [
    {"skill_id": 41, "skill_version_id": 410, "skill_project_id": 1, "name": "pptx", "instructions": "Run add_slide."},
]
LOADED_BODY = 'Skill "pptx" is now active. Follow these instructions exactly.\n\n<skill name="pptx">\nRun add_slide.\n</skill>\n\nSkill root: skill://pptx/\n<skill_files>\nscripts/add_slide.py (14 KB)\n</skill_files>'


def ran(content, exit_code=0, *, stdout_bytes=0, stderr_bytes=0, shared=(), timed_out=False):
    return ToolMessage(content=content, tool_call_id="", artifact={
        "exit_code": exit_code, "timed_out": timed_out, "stdout_bytes": stdout_bytes, "stderr_bytes": stderr_bytes,
        "shared": list(shared),
    })


def script_rows(written):
    return [r for r in written if r.get("event_type") == "skill_script"]


def run_script(callback, output, *, skill="pptx", path="scripts/add_slide.py", metadata=None):
    run_id = uuid.uuid4()
    inputs = {"skill": skill, "path": path, "args": ["deck.pptx"], "files": ["deck.pptx"]}
    callback.on_tool_start({"name": "run_skill_script"}, json.dumps(inputs), run_id=run_id, inputs=inputs, metadata=metadata or {})
    callback.on_tool_end(output, run_id=run_id)
    return str(run_id)


def test_a_finished_script_writes_one_skill_script_row_with_exit_code_and_duration(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    call_id = run_script(callback, ran(
        "Exit code 0\nCreated ppt/slides/slide3.xml\nFiles shared: deck-2.pptx (48 KB), notes, draft.txt (not shared: denied)",
        stdout_bytes=4000, stderr_bytes=12, shared=["deck-2.pptx"],
    ))

    assert [r["tool_name"] for r in rows if r.get("event_type") == "tool"] == ["run_skill_script"]
    [row] = script_rows(rows)
    assert (row["entity_type"], row["entity_id"], row["entity_version_id"], row["entity_name"]) == ("skill", 41, 410, "pptx")
    assert (row["tool_name"], row["is_error"]) == ("run_skill_script", False)
    assert isinstance(row["duration_ms"], int) and row["duration_ms"] >= 0
    assert json.loads(row["meta"]) == {
        "source": "run_skill_script", "path": "scripts/add_slide.py", "exit_code": 0,
        "stdout_bytes": 4000, "stderr_bytes": 12, "output_files": 1, "timed_out": False, "skill_project_id": 1,
    }
    assert row["idempotency_key"] == f"skill:run-1:run_skill_script:1:41:scripts/add_slide.py:{call_id}"
    assert skill_rows(rows) == []


def test_a_failed_script_is_an_error_row_and_a_rerun_is_its_own_row(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    first = run_script(callback, ran("Exit code 3\npartial\n[stderr]\nERR: broken", 3))
    second = run_script(callback, ran("Exit code 3\npartial\n[stderr]\nERR: broken", 3))

    assert [r["is_error"] for r in script_rows(rows)] == [True, True]
    assert [json.loads(r["meta"])["exit_code"] for r in script_rows(rows)] == [3, 3]
    assert {r["idempotency_key"].rsplit(":", 1)[1] for r in script_rows(rows)} == {first, second}


def test_a_timed_out_script_is_recorded_as_such(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    run_script(callback, ran("Exit code -1 (timed out after 55 s)\nNo output was captured; the script was stopped.", -1, timed_out=True))

    [row] = script_rows(rows)
    assert row["is_error"] is True
    assert json.loads(row["meta"])["timed_out"] is True
    assert json.loads(row["meta"])["exit_code"] == -1


def test_a_refusal_before_the_run_writes_no_script_row(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    run_script(callback, '"scripts/run.sh" cannot run here: only Python (.py) scripts run on this platform.', path="scripts/run.sh")
    run_script(callback, "The script could not be run: the sandbox is not available right now.")
    run_script(callback, ToolMessage(content="Exit code 0\nlooks like a run", tool_call_id=""))

    assert script_rows(rows) == []


def test_a_run_that_loaded_the_skill_on_the_way_writes_the_skill_row_and_the_script_row(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    run_script(callback, ran(LOADED_BODY + "\n\nExit code 0\n(no output)"))

    assert [r["entity_name"] for r in skill_rows(rows)] == ["pptx"]
    [row] = script_rows(rows)
    assert json.loads(row["meta"])["exit_code"] == 0
    assert json.loads(row["meta"])["output_files"] == 0
