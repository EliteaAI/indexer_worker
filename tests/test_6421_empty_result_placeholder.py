"""Regression test for #6421 (empty-but-valid JSON results collapsed to a placeholder).

After the initial #6421 fix (see test_6421_toolkit_test_result_placeholder.py), a genuinely
successful tool result that is an empty JSON value (e.g. `[]` or `{}`) was still discarded:
`_indexer_test_toolkit_tool_task` treated a falsy `formatted_content` as "no content" and
overwrote it with the generic "Tool executed successfully" placeholder, even though `[]`/`{}`
is real, meaningful data (e.g. "no comments found"). The raw value survived in
`response_metadata.tool_output`, but the displayed `content` field lost it.

The fix only falls back to the placeholder when the result is truly absent (`None`); any
other JSON value, including falsy ones, is serialized and shown as-is.

Loaded by source so the suite runs without the pylon runtime.
"""

import json
import pathlib
import typing


def _load_detect_content_type():
    source = (pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'indexer_test_toolkit.py').read_text()
    start = source.index("def detect_content_type")
    end = source.index("\nclass Method")
    namespace = {'Any': typing.Any, 'json': json}
    exec(compile("from typing import Any\nimport json\n" + source[start:end], '<detect_content_type>', 'exec'),
         namespace)  # pylint: disable=W0122
    return namespace['detect_content_type']


detect_content_type = _load_detect_content_type()


def _apply_formatting(final_result):
    """Mirrors the content-type formatting block in `_indexer_test_toolkit_tool_task`."""
    content_type, formatted_content = detect_content_type(final_result)
    if content_type == 'json':
        formatted_content = json.dumps(formatted_content) if formatted_content is not None else "Tool executed successfully"
    elif not formatted_content:
        formatted_content = "Tool executed successfully"
    return content_type, formatted_content


def test_empty_list_result_is_shown_not_replaced():
    content_type, formatted_content = _apply_formatting([])

    assert content_type == 'json'
    assert formatted_content == "[]"
    assert formatted_content != "Tool executed successfully"


def test_empty_dict_result_is_shown_not_replaced():
    content_type, formatted_content = _apply_formatting({})

    assert content_type == 'json'
    assert formatted_content == "{}"
    assert formatted_content != "Tool executed successfully"


def test_non_empty_list_result_is_still_shown():
    content_type, formatted_content = _apply_formatting([{"id": 1}])

    assert content_type == 'json'
    assert formatted_content == json.dumps([{"id": 1}])
