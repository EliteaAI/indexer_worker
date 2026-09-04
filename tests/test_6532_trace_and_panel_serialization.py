"""Regression tests for the #6532 trace and toolkit-panel changes.

Three behaviour changes shipped here without tests, and each had a defect on the
line it changed:

* the panel emitted bare NaN/Infinity, which JSON.parse rejects — and the UI
  parses before rendering a result as JSON, so one missing value in a
  pandas-backed result blanked the whole panel;
* `clean_for_json_serialization` had no cycle guard, so a self-referential result
  recursed ~1000 deep and then degraded the payload to a repr;
* `indent_for_trace` caught RecursionError around the parse but not the dump, and
  the uncovered band was the shallower one.

Loaded by source, like the #6421 tests beside this file, so the suite runs
without the pylon runtime. Run from this directory.
"""

import ast
import json
import logging
import pathlib
import sys
import typing

try:
    from elitea_sdk.tools.utils.serialization import NON_FINITE, make_json_safe, to_json_primitive
except ImportError:  # pragma: no cover - checkout layout, not an installed SDK
    _sdk = pathlib.Path(__file__).resolve().parents[5] / 'elitea-sdk'
    if not _sdk.exists():
        import pytest

        pytest.skip("elitea_sdk is neither installed nor beside this checkout", allow_module_level=True)
    sys.path.insert(0, str(_sdk))
    from elitea_sdk.tools.utils.serialization import NON_FINITE, make_json_safe, to_json_primitive

METHODS = pathlib.Path(__file__).resolve().parents[1] / 'methods'


def _load(module_name, wanted):
    source = (METHODS / module_name).read_text()
    body = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {
        'json': json, 'log': logging.getLogger('test'), 'Any': typing.Any,
        'to_json_primitive': to_json_primitive, 'make_json_safe': make_json_safe,
        'NON_FINITE': NON_FINITE, 'TRACE_STEP_FIELD_MAX_CHARS': 200_000,
        'frozenset': frozenset,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), module_name, 'exec'), namespace)
    return namespace


_panel = _load('indexer_test_toolkit.py', {'clean_for_json_serialization', 'safe_json_dumps'})
_trace = _load('agent_common.py', {'indent_for_trace'})
clean_for_json_serialization = _panel['clean_for_json_serialization']
safe_json_dumps = _panel['safe_json_dumps']
indent_for_trace = _trace['indent_for_trace']


class SelfReferencing:
    def __init__(self):
        self.me = self


def test_a_missing_value_does_not_blank_the_panel():
    rendered = safe_json_dumps(clean_for_json_serialization([{"id": 1, "score": float("nan")}]))

    assert json.loads(rendered) == [{"id": 1, "score": None}]


def test_infinity_is_also_rendered_as_null():
    rendered = safe_json_dumps(clean_for_json_serialization({"hi": float("inf")}))

    assert json.loads(rendered) == {"hi": None}


def test_a_cycle_is_marked_rather_than_recursed():
    payload = {"name": "root"}
    payload["self"] = payload

    rendered = safe_json_dumps(clean_for_json_serialization(payload))

    assert json.loads(rendered) == {"name": "root", "self": "<circular reference>"}


def test_a_shared_value_is_not_mistaken_for_a_cycle():
    shared = {"a": 1}

    rendered = safe_json_dumps(clean_for_json_serialization({"x": shared, "y": shared}))

    assert json.loads(rendered) == {"x": {"a": 1}, "y": {"a": 1}}


def test_a_self_referential_object_still_renders():
    rendered = safe_json_dumps(clean_for_json_serialization({"x": SelfReferencing()}))

    assert json.loads(rendered) == {"x": "<SelfReferencing>"}


def test_null_fields_survive_because_they_are_data():
    rendered = safe_json_dumps(clean_for_json_serialization([{"author": None, "id": 1}]))

    assert json.loads(rendered) == [{"author": None, "id": 1}]


def test_non_ascii_is_not_escaped():
    rendered = safe_json_dumps(clean_for_json_serialization({"t": "Ошибка 日本語"}))

    assert "Ошибка 日本語" in rendered


def test_trace_indents_a_json_string():
    assert indent_for_trace('[{"n": 1}]') == '[\n  {\n    "n": 1\n  }\n]'


def test_trace_leaves_prose_alone():
    assert indent_for_trace("Tool executed successfully") == "Tool executed successfully"


def test_trace_survives_nesting_that_breaks_the_encoder():
    # Deep enough to blow the encoder's stack but shallow enough to parse: the
    # earlier guard covered only the parse, so this band still escaped.
    deep = "[" * 8000 + "]" * 8000

    assert isinstance(indent_for_trace(deep), str)


def test_trace_survives_nesting_that_breaks_the_parser():
    deep = "[" * 12000 + "]" * 12000

    assert isinstance(indent_for_trace(deep), str)


def test_trace_hands_back_a_payload_it_cannot_encode():
    # An ESCAPED lone surrogate is inert; parsing makes it live, and the result
    # then raises at the first strict-UTF-8 boundary.
    payload = '[{"t": "\\ud800"}]'

    result = indent_for_trace(payload)

    assert result == payload
    result.encode('utf-8')
