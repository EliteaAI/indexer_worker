"""#7553 — tool usage rows record the toolkit type so complexity weights can tell
`github:read_file` from `sharepoint:read_file`; the SDK puts it in tool.metadata."""
import json
import uuid

from test_6926_skill_usage_rows import AGENT_RUN, rows, tool_callback_module  # noqa: F401


def run(callback, name, metadata):
    run_id = uuid.uuid4()
    callback.on_tool_start({"name": name}, "{}", run_id=run_id, metadata=metadata)
    callback.on_tool_end("ok", run_id=run_id)


def meta_of(written):
    (row,) = written
    return json.loads(row["meta"]) if row["meta"] else None


class TestToolkitType:
    def test_the_toolkit_type_is_recorded_in_meta(self, rows):  # noqa: F811
        run(tool_callback_module.UsageToolCallback(AGENT_RUN), "read_file", {"toolkit_type": "github"})

        assert meta_of(rows) == {"toolkit_type": "github"}

    def test_it_sits_next_to_the_parent_agent_name(self, rows):  # noqa: F811
        run(
            tool_callback_module.UsageToolCallback(AGENT_RUN), "read_file",
            {"toolkit_type": "github", "parent_agent_name": "Researcher"},
        )

        assert meta_of(rows) == {"toolkit_type": "github", "parent_agent_name": "Researcher"}

    def test_no_metadata_still_writes_no_meta(self, rows):  # noqa: F811
        run(tool_callback_module.UsageToolCallback(AGENT_RUN), "read_file", {})

        assert meta_of(rows) is None

    def test_a_non_string_toolkit_type_is_dropped(self, rows):  # noqa: F811
        # Only flat strings go into meta: the lookup key is "<type>:<tool>"
        run(tool_callback_module.UsageToolCallback(AGENT_RUN), "read_file", {"toolkit_type": {"x": 1}})

        assert meta_of(rows) is None
