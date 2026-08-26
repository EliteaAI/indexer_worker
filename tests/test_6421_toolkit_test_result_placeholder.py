"""Regression tests for #6421 (toolkit-test results collapsed to a placeholder).

`clean_for_json_serialization` (used by `_indexer_test_toolkit_tool_task` before formatting
the toolkit-test UI response) stripped any dict key whose *name* merely contained a
substring like "client" or "instance" -- regardless of whether the value was itself a
plain, JSON-safe type. Real tool results frequently contain legitimately serializable
fields with such names (e.g. "client_id", "instances", "clientMutationId"), so genuinely
successful, non-empty results were silently emptied out, and the caller's
`if not formatted_content: formatted_content = "Tool executed successfully"` fallback then
overwrote the (now-empty) real data with a misleading generic success message.

The fix only drops an entry by key name when its value is not already a plain, JSON-safe
type (str/int/float/bool/None/dict/list); actual non-serializable objects (real client/
callback instances) are still stripped as before.

Loaded by source so the suite runs without the pylon runtime.

Run from this directory (`cd tests && python3 -m pytest test_6421_toolkit_test_result_placeholder.py`):
invoking pytest from the plugin root makes it import the plugin's module.py, which needs
pylon's `tools` and fails collection -- the pre-existing test files here behave the same way.
"""

import pathlib
import typing


def _load_clean_for_json_serialization():
    """Exec just clean_for_json_serialization, avoiding pylon imports."""
    source = (pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'indexer_test_toolkit.py').read_text()
    start = source.index("def clean_for_json_serialization")
    end = source.index("\ndef test_error")
    namespace = {'Any': typing.Any}
    exec(compile("from typing import Any\n" + source[start:end], '<clean_for_json_serialization>', 'exec'),
         namespace)  # pylint: disable=W0122
    return namespace['clean_for_json_serialization']


clean_for_json_serialization = _load_clean_for_json_serialization()


class FakeClient:
    """Stands in for an actual, non-serializable SDK/API client instance."""


def test_legitimate_fields_named_like_client_or_instance_are_preserved():
    # Regression case: real tool result with keys that merely *contain* the suspect
    # substrings but hold plain, JSON-safe values (this is what genuinely happened, e.g.
    # for GitHub/Jira comment-list results).
    result = {
        "issue_id": 42,
        "client_id": "abc123",
        "instances": ["comment1", "comment2"],
        "comments": [{"author": "bob", "text": "hello", "clientMutationId": "xyz"}],
    }

    cleaned = clean_for_json_serialization(result, "fallback")

    assert cleaned["client_id"] == "abc123"
    assert cleaned["instances"] == ["comment1", "comment2"]
    assert cleaned["comments"][0]["text"] == "hello"
    assert cleaned["comments"][0]["clientMutationId"] == "xyz"
    # Not empty -- must not trigger the "Tool executed successfully" placeholder downstream.
    assert cleaned


def test_actual_non_serializable_client_object_is_still_stripped():
    result = {"data": "ok", "some_client": FakeClient(), "events_dispatched": [1, 2, 3]}

    cleaned = clean_for_json_serialization(result, "fallback")

    assert "some_client" not in cleaned
    assert cleaned["data"] == "ok"
    assert cleaned["events_dispatched"] == "<3 events (cleaned for serialization)>"


def test_nested_legitimate_fields_are_preserved():
    result = {"outer": {"instance_count": 3, "callback_url": "https://example.com/hook"}}

    cleaned = clean_for_json_serialization(result, "fallback")

    assert cleaned["outer"]["instance_count"] == 3
    assert cleaned["outer"]["callback_url"] == "https://example.com/hook"
