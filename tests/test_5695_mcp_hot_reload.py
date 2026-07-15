"""EL-5695 MCP configuration hot-reload contract."""

import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock

from jsonschema import Draft202012Validator


PLUGIN_ROOT = Path(__file__).parents[1]


def _load_worker_module(monkeypatch):
    pylon = types.ModuleType("pylon")
    pylon_core = types.ModuleType("pylon.core")
    pylon_tools = types.ModuleType("pylon.core.tools")
    pylon_tools.log = types.SimpleNamespace(
        info=Mock(), warning=Mock(), error=Mock(), exception=Mock(), debug=Mock(),
    )
    pylon_tools.module = types.SimpleNamespace(ModuleModel=object)
    monkeypatch.setitem(sys.modules, "pylon", pylon)
    monkeypatch.setitem(sys.modules, "pylon.core", pylon_core)
    monkeypatch.setitem(sys.modules, "pylon.core.tools", pylon_tools)
    monkeypatch.setitem(sys.modules, "arbiter", types.ModuleType("arbiter"))

    tools = types.ModuleType("tools")
    tools.worker_core = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "tools", tools)

    spec = importlib.util.spec_from_file_location(
        "indexer_worker_module_el5695",
        PLUGIN_ROOT / "module.py",
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_admin_schema_marks_mcp_config_live_and_validates_transports():
    schema = json.loads((PLUGIN_ROOT / "admin_schema.json").read_text())
    field = schema["properties"]["mcp_servers"]

    assert field["requires_restart"] is False
    validator = Draft202012Validator(field["value_schema"])
    assert not list(validator.iter_errors({
        "Remote": {"type": "http", "url": "https://mcp.example.test/api"},
        "Local": {"type": "stdio", "command": "npx"},
    }))
    assert list(validator.iter_errors({"Remote": {"type": "http"}}))
    assert list(validator.iter_errors({"Local": {"type": "stdio"}}))


def test_changed_mcp_config_refreshes_sdk_and_broadcasts_both_views(monkeypatch):
    worker_module = _load_worker_module(monkeypatch)
    refresh = Mock()
    sdk_mcp = types.ModuleType("elitea_sdk.runtime.toolkits.mcp_config")
    sdk_mcp.refresh_mcp_server_configs = refresh
    monkeypatch.setitem(sys.modules, "elitea_sdk", types.ModuleType("elitea_sdk"))
    monkeypatch.setitem(sys.modules, "elitea_sdk.runtime", types.ModuleType("elitea_sdk.runtime"))
    monkeypatch.setitem(sys.modules, "elitea_sdk.runtime.toolkits", types.ModuleType("elitea_sdk.runtime.toolkits"))
    monkeypatch.setitem(sys.modules, "elitea_sdk.runtime.toolkits.mcp_config", sdk_mcp)

    old = {"Old": {"type": "http", "url": "https://old.example.test/mcp"}}
    new = {"New": {"type": "http", "url": "https://new.example.test/mcp"}}
    instance = worker_module.Module.__new__(worker_module.Module)
    instance.descriptor = types.SimpleNamespace(config={"mcp_servers": new})
    instance.agent_event_node = object()
    instance._mcp_servers_snapshot = old
    instance.toolkit_configurations_request = Mock()
    instance.mcp_prebuilt_config_request = Mock()

    instance._reload_mcp_servers()

    refresh.assert_called_once_with(new)
    instance.toolkit_configurations_request.assert_called_once_with(None, None)
    instance.mcp_prebuilt_config_request.assert_called_once_with(None, None)
    assert instance._mcp_servers_snapshot == new


def test_unchanged_mcp_config_avoids_expensive_schema_rebuild(monkeypatch):
    worker_module = _load_worker_module(monkeypatch)
    current = {"Remote": {"type": "http", "url": "https://mcp.example.test/api"}}
    instance = worker_module.Module.__new__(worker_module.Module)
    instance.descriptor = types.SimpleNamespace(config={"mcp_servers": current})
    instance.agent_event_node = object()
    instance._mcp_servers_snapshot = current.copy()
    instance.toolkit_configurations_request = Mock()
    instance.mcp_prebuilt_config_request = Mock()

    instance._reload_mcp_servers()

    instance.toolkit_configurations_request.assert_not_called()
    instance.mcp_prebuilt_config_request.assert_not_called()
