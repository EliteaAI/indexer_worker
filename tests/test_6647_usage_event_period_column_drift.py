"""usage_tool_events must not write a `period` column: it was dropped from usage_event (#6647).

Regression guard for the exact drift that caused every tool-usage row write to raise
UndefinedColumn in production: the indexer's hand-written INSERT statement and params
dict named a column the owning model (usage plugin, different repo) had already removed.

Run: python3 -m pytest test_6647_usage_event_period_column_drift.py
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
    """Import usage_tool_events.py with its external deps stubbed where missing.

    Neither `pylon` nor `elitea_sdk` is installed outside the container, and this
    module's only use of either is `log` (unused by the assertions below) and a
    string constant, so a bare stub is enough. `sqlalchemy` isn't in this repo's
    tests/requirements-dev.txt (CI installs only that file), and record_tool_event
    imports `text` from it lazily inside a try/except that swallows the resulting
    ImportError — so without a stub the CI run "passes" the import and silently
    never populates `params` at all, masking the very column it should be checking.
    """
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

    spec = importlib.util.spec_from_file_location("usage_tool_events", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_insert_sql_does_not_name_a_period_column():
    assert "period" not in module._INSERT_SQL


def test_record_tool_event_params_do_not_stamp_a_period_key(monkeypatch):
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
        attribution={"project_id": 3, "run_id": "run-1"},
        tool_name="read_file",
        duration_ms=12,
        is_error=False,
        lc_run_id="lc-1",
    )

    assert "period" not in captured
    assert captured["project_id"] == 3
    assert captured["tool_name"] == "read_file"


def test_every_named_column_has_a_matching_bind_param():
    import re

    before_values, _, after_values = module._INSERT_SQL.partition("VALUES")
    columns_part = before_values.split("(", 1)[1].rsplit(")", 1)[0]
    columns = [c.strip() for c in columns_part.split(",")]
    binds = set(re.findall(r":(\w+)", after_values))

    for column in columns:
        assert column in binds, f"column {column!r} has no matching :{column} bind param"
