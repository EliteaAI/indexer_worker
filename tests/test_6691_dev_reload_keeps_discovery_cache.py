"""With dev reload on, the SDK's discovery-cache module must survive every reload (#6691
review): its backend registry is process state pushed in by the parent at init, so a fresh
import would start without one and Load Tools would silently stop invalidating. Re-breaks
if `clear_sdk_modules` stops keeping `KEPT_ACROSS_DEV_RELOAD` resident.
"""

import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
CACHE_MODULE = "elitea_sdk.runtime.utils.mcp_discovery_cache"
SIBLING_MODULE = "elitea_sdk.runtime.utils.mcp_oauth"
RELOAD_HELPERS = {"is_dev_reload_enabled", "set_dev_reload_enabled", "clear_sdk_modules", "dev_reload_sdk"}
RELOAD_GLOBALS = {"_DEV_MODE_RELOAD", "KEPT_ACROSS_DEV_RELOAD"}


def _install_sdk_stubs(monkeypatch):
    for name in ("elitea_sdk", "elitea_sdk.runtime", "elitea_sdk.runtime.utils", SIBLING_MODULE, CACHE_MODULE):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    return sys.modules[CACHE_MODULE]


def _load_reload_helpers():
    tree = ast.parse((PLUGIN_ROOT / "utils" / "funcs.py").read_text())
    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in RELOAD_HELPERS)
        or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in RELOAD_GLOBALS for t in node.targets))
    ]
    assert {node.name for node in body if isinstance(node, ast.FunctionDef)} == RELOAD_HELPERS
    namespace = {"os": os, "sys": sys, "Optional": Optional, "log": SimpleNamespace(debug=Mock())}
    exec(compile(ast.Module(body=body, type_ignores=[]), "funcs-6691", "exec"), namespace)  # pylint: disable=exec-used
    namespace["set_dev_reload_enabled"](True)
    return namespace


@pytest.mark.parametrize("target", [None, "elitea_sdk", "elitea_sdk.runtime", "elitea_sdk.runtime.utils", CACHE_MODULE])
def test_the_cache_module_stays_resident_through_every_reload_that_covers_it(monkeypatch, target):
    cache_module = _install_sdk_stubs(monkeypatch)
    helpers = _load_reload_helpers()

    helpers["dev_reload_sdk"](target)

    assert sys.modules[CACHE_MODULE] is cache_module


def test_the_rest_of_the_target_tree_is_still_reloaded(monkeypatch):
    _install_sdk_stubs(monkeypatch)
    helpers = _load_reload_helpers()

    cleared = helpers["clear_sdk_modules"]("elitea_sdk.runtime.utils")

    assert cleared == 2
    assert SIBLING_MODULE not in sys.modules
    assert "elitea_sdk.runtime.utils" not in sys.modules


def test_nothing_is_cleared_when_dev_reload_is_off(monkeypatch):
    _install_sdk_stubs(monkeypatch)
    helpers = _load_reload_helpers()
    helpers["set_dev_reload_enabled"](False)

    assert helpers["clear_sdk_modules"]() == 0
    assert SIBLING_MODULE in sys.modules
