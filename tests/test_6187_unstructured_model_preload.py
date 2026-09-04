"""Startup contract for the Unstructured NLP model security upgrade."""

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock


PLUGIN_ROOT = Path(__file__).parents[1]


def _load_worker_module(monkeypatch):
    pylon = types.ModuleType("pylon")
    pylon_core = types.ModuleType("pylon.core")
    pylon_tools = types.ModuleType("pylon.core.tools")
    pylon_tools.log = types.SimpleNamespace(
        info=Mock(),
        warning=Mock(),
        error=Mock(),
        exception=Mock(),
        debug=Mock(),
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
        "indexer_worker_module_el6187",
        PLUGIN_ROOT / "module.py",
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded, pylon_tools.log


def test_startup_preloads_unstructured_model(monkeypatch):
    worker_module, log = _load_worker_module(monkeypatch)
    preload = Mock()
    sdk_utils = types.ModuleType("elitea_sdk.runtime.langchain.tools.utils")
    sdk_utils.preload_unstructured_nlp_model = preload
    monkeypatch.setitem(
        sys.modules,
        "elitea_sdk.runtime.langchain.tools.utils",
        sdk_utils,
    )

    worker_module._preload_unstructured_nlp_model()

    preload.assert_called_once_with()
    log.info.assert_any_call("Unstructured NLP model is ready")


def test_startup_remains_available_when_model_download_fails(monkeypatch):
    worker_module, log = _load_worker_module(monkeypatch)
    sdk_utils = types.ModuleType("elitea_sdk.runtime.langchain.tools.utils")
    sdk_utils.preload_unstructured_nlp_model = Mock(
        side_effect=OSError("offline")
    )
    monkeypatch.setitem(
        sys.modules,
        "elitea_sdk.runtime.langchain.tools.utils",
        sdk_utils,
    )

    worker_module._preload_unstructured_nlp_model()

    log.exception.assert_called_once_with(
        "Failed to preload Unstructured NLP model; "
        "indexing will retry on demand"
    )
