"""Evaluation/judge runs must carry real attribution instead of NULL columns (#6677).

run_attribution() gained a leaf override (ENTITY_KWARGS_KEY) symmetric to the existing
root override (ROOT_ENTITY_KWARGS_KEY), so an eval payload can mark entity_type='evaluation'
while root_entity_type stays the real application/pipeline being evaluated. These tests pin
that override plus its fallback, and confirm ordinary chat/agent payloads are unaffected.

Run: python3 -m pytest test_6677_evaluation_attribution.py
"""

import importlib.util
import pathlib
import sys
import types
from unittest.mock import MagicMock

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "utils" / "usage_tool_events.py"
)


def _load_module():
    """Same stubbing as test_6647: pylon/elitea_sdk aren't installed outside the container."""
    pylon_core_tools = types.ModuleType("pylon.core.tools")
    pylon_core_tools.log = MagicMock()
    sys.modules["pylon"] = types.ModuleType("pylon")
    sys.modules["pylon.core"] = types.ModuleType("pylon.core")
    sys.modules["pylon.core.tools"] = pylon_core_tools

    sdk_utils = types.ModuleType("elitea_sdk.runtime.utils.utils")
    sdk_utils.PREDICT_RUN_ID_KWARGS_KEY = "predict_run_id"
    sys.modules["elitea_sdk"] = types.ModuleType("elitea_sdk")
    sys.modules["elitea_sdk.runtime"] = types.ModuleType("elitea_sdk.runtime")
    sys.modules["elitea_sdk.runtime.utils"] = types.ModuleType("elitea_sdk.runtime.utils")
    sys.modules["elitea_sdk.runtime.utils.utils"] = sdk_utils

    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        sqlalchemy_stub = types.ModuleType("sqlalchemy")
        sqlalchemy_stub.text = lambda statement: statement
        sys.modules["sqlalchemy"] = sqlalchemy_stub

    spec = importlib.util.spec_from_file_location("usage_tool_events_6677", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_leaf_override_marks_evaluation_root_stays_application():
    kwargs = {
        module.ENTITY_KWARGS_KEY: {"type": "evaluation", "id": 42},
        module.ROOT_ENTITY_KWARGS_KEY: {"type": "application", "id": 7, "version_id": 3},
    }
    out = module.run_attribution(kwargs)
    assert out["entity_type"] == "evaluation"
    assert out["entity_id"] == 42
    assert out["root_entity_type"] == "application"
    assert out["root_entity_id"] == 7
    assert out["root_entity_version_id"] == 3


def test_no_override_is_byte_identical_to_today():
    # Regression pin: an ordinary chat/agent payload (no leaf/root override keys) must derive
    # everything from `application` exactly as before this fix, entity == root.
    kwargs = {"application": {"id": 7, "version_id": 3, "name": "My Agent"},
              "conversation_id": "conv-1"}
    out = module.run_attribution(kwargs)
    assert out == {
        "conversation_id": "conv-1",
        "entity_type": "application", "entity_id": 7,
        "entity_version_id": 3, "entity_name": "My Agent",
        "root_entity_type": "application", "root_entity_id": 7,
        "root_entity_version_id": 3,
    }


def test_leaf_override_without_id_falls_back_to_application():
    kwargs = {
        module.ENTITY_KWARGS_KEY: {"type": "evaluation"},  # no id -> ignored
        "application": {"id": 7, "name": "My Agent"},
    }
    out = module.run_attribution(kwargs)
    assert out["entity_type"] == "application"
    assert out["entity_id"] == 7


def test_leaf_override_without_application_has_no_root_to_fall_back_to():
    # No `application` and no root override: root falls back to the leaf override itself.
    kwargs = {module.ENTITY_KWARGS_KEY: {"type": "evaluation", "id": 42}}
    out = module.run_attribution(kwargs)
    assert out["entity_type"] == "evaluation" and out["entity_id"] == 42
    assert out["root_entity_type"] == "evaluation" and out["root_entity_id"] == 42


def test_attribution_header_round_trips_differing_types():
    import base64
    import json

    kwargs = {
        module.ENTITY_KWARGS_KEY: {"type": "evaluation", "id": 42},
        module.ROOT_ENTITY_KWARGS_KEY: {"type": "application", "id": 7},
    }
    header = module.attribution_header(kwargs)
    assert header is not None
    padded = header + "=" * (-len(header) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded))
    assert decoded["entity_type"] == "evaluation" and decoded["entity_id"] == 42
    assert decoded["root_entity_type"] == "application" and decoded["root_entity_id"] == 7


def test_record_tool_event_writes_differing_entity_and_root_type(monkeypatch):
    captured = {}

    class _FakeConnection:
        def execute(self, _statement, params):
            captured.update(params)

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    class _FakeEngine:
        def connect(self):
            return _FakeConnection()

    monkeypatch.setattr(module, "_get_engine", lambda: _FakeEngine())

    module.record_tool_event(
        attribution={
            "project_id": 3, "run_id": "run-1",
            "entity_type": "evaluation", "entity_id": 42,
            "root_entity_type": "application", "root_entity_id": 7,
        },
        tool_name="read_file", duration_ms=12, is_error=False, lc_run_id="lc-1",
    )

    assert captured["entity_type"] == "evaluation" and captured["entity_id"] == 42
    assert captured["root_entity_type"] == "application" and captured["root_entity_id"] == 7
