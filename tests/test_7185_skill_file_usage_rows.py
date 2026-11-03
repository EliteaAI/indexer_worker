"""A read_skill_file that returned a file becomes one usage_event skill_file row with its path and size (#7185).

Run: python3 -m pytest test_7185_skill_file_usage_rows.py
"""

import json
import uuid

from test_6926_skill_usage_rows import AGENT_RUN, RunCallbacks, loaded, rows, skill_rows  # noqa: F401  rows is a fixture

READ_SKILL_FILE_RESULT = 'Skill "{name}" file "{path}"{lines}' + """:
<skill_file path="{path}">
{content}
</skill_file>"""
FILES_BLOCK = """Skill root: skill://pdf/
Read this skill's files only with read_skill_file(skill="pdf", path="forms.md").
<skill_files>
forms.md (12 KB)
</skill_files>"""

REGISTRY = [
    {"skill_id": 41, "skill_version_id": 410, "skill_project_id": 1, "name": "pdf", "instructions": "Read FORMS.md."},
]


def file_rows(written):
    return [r for r in written if r.get("event_type") == "skill_file"]


def read_file(callback, output, *, skill="pdf", path="forms.md", metadata=None, tool="read_skill_file"):
    run_id = uuid.uuid4()
    inputs = {"skill": skill, "path": path}
    callback.on_tool_start({"name": tool}, json.dumps(inputs), run_id=run_id, inputs=inputs, metadata=metadata or {})
    callback.on_tool_end(output, run_id=run_id)


def served(path="forms.md", content="Step 1: open the form.", lines=""):
    return READ_SKILL_FILE_RESULT.format(name="pdf", path=path, lines=lines, content=content)


def test_a_served_file_writes_one_skill_file_row_with_its_path_and_bytes(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    read_file(callback, served(content="Étape 1"))

    assert [r["tool_name"] for r in rows if r.get("event_type") == "tool"] == ["read_skill_file"]
    [row] = file_rows(rows)
    assert (row["entity_type"], row["entity_id"], row["entity_version_id"], row["entity_name"]) == (
        "skill", 41, 410, "pdf",
    )
    assert row["tool_name"] is None and row["is_error"] is False
    assert json.loads(row["meta"]) == {
        "source": "read_skill_file", "outcome": "loaded", "body_chars": 0, "est_body_tokens": 0,
        "skill_project_id": 1, "path": "forms.md", "bytes": len("Étape 1".encode("utf-8")),
    }
    assert row["idempotency_key"] == "skill:run-1:read_skill_file:1:41:forms.md"


def test_a_served_file_is_not_counted_as_a_skill_use(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    read_file(callback, served())

    assert skill_rows(rows) == []
    assert len(file_rows(rows)) == 1


def test_each_path_and_range_is_its_own_row_and_a_repeat_keeps_its_key(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    read_file(callback, served())
    read_file(callback, served(path="scripts/fill.py"), path="scripts/fill.py")
    read_file(callback, served(lines=" lines 1-20"))
    read_file(callback, served())

    assert [r["idempotency_key"] for r in file_rows(rows)] == [
        "skill:run-1:read_skill_file:1:41:forms.md",
        "skill:run-1:read_skill_file:1:41:scripts/fill.py",
        "skill:run-1:read_skill_file:1:41:forms.md lines 1-20",
        "skill:run-1:read_skill_file:1:41:forms.md",
    ]
    assert [json.loads(r["meta"])["path"] for r in file_rows(rows)] == [
        "forms.md", "scripts/fill.py", "forms.md", "forms.md",
    ]


def test_a_refused_or_failed_read_writes_no_row(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    read_file(callback, 'Skill "pdf" is not loaded in this conversation, so its files cannot be read yet.')
    read_file(callback, 'Path "../SKILL.md" is outside skill "pdf".', path="../SKILL.md")
    read_file(callback, 'Skill "pdf" file "forms.md" is already in this conversation. Use the earlier result.')
    read_file(callback, '{"__result_status__": "content_too_large"}')

    assert skill_rows(rows) == [] and file_rows(rows) == []


def test_the_read_file_answer_from_another_tool_writes_no_row(rows):
    read_file(RunCallbacks(AGENT_RUN, REGISTRY), served(), tool="read_file")

    assert skill_rows(rows) == [] and file_rows(rows) == []


def test_a_sub_agent_read_names_its_parent(rows):
    read_file(RunCallbacks(AGENT_RUN, REGISTRY), served(), metadata={"parent_agent_name": "Researcher"})

    [row] = file_rows(rows)
    assert row["entity_id"] is None
    assert json.loads(row["meta"])["parent_agent_name"] == "Researcher"


def test_a_load_with_a_files_block_still_measures_only_the_body(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)
    run_id = uuid.uuid4()
    callback.on_tool_start({"name": "load_skill"}, '{"skill": "pdf"}', run_id=run_id, inputs={"skill": "pdf"})

    callback.on_tool_end(loaded("pdf", "Read FORMS.md.") + "\n\n" + FILES_BLOCK, run_id=run_id)

    [row] = skill_rows(rows)
    assert json.loads(row["meta"])["body_chars"] == len("Read FORMS.md.")


def test_a_read_that_loads_the_skill_on_the_way_writes_the_skill_use_and_the_file_read(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    read_file(callback, loaded("pdf", "Read FORMS.md.") + "\n\n" + FILES_BLOCK + "\n\n" + served(content="Étape 1"))

    [use] = skill_rows(rows)
    assert json.loads(use["meta"])["source"] == "load_skill"
    assert json.loads(use["meta"])["body_chars"] == len("Read FORMS.md.")
    [read] = file_rows(rows)
    assert (json.loads(read["meta"])["path"], json.loads(read["meta"])["bytes"]) == ("forms.md", len("Étape 1".encode("utf-8")))


def test_a_listing_that_loads_the_skill_on_the_way_writes_one_skill_use_and_no_file_read(rows):
    callback = RunCallbacks(AGENT_RUN, REGISTRY)

    listing = 'Files of skill "pdf" under skill://pdf/:\nforms.md (12 KB)'

    read_file(callback, loaded("pdf", "Read FORMS.md.") + "\n\n" + FILES_BLOCK + "\n\n" + listing, tool="list_skill_files")
    read_file(callback, listing, tool="list_skill_files")

    assert [r["entity_name"] for r in skill_rows(rows)] == ["pdf"] and file_rows(rows) == []
