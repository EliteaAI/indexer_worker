"""The worker imports the first ChatOpenAI's lazy modules before agent tasks fork.

Agent tasks run in a process forked per task, so anything the parent has not imported is
imported again by every run. `openai.resources` (reached through the OpenAI client's lazy
`.chat`) was re-imported by every Test toolkit call. Re-breaks if the preload is dropped,
moves after the agent task node is created, or stops tolerating an import failure.
"""

import ast
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
        "indexer_worker_module_preload_llm",
        PLUGIN_ROOT / "module.py",
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded, pylon_tools.log


def _stub_modules(monkeypatch):
    openai = types.ModuleType("openai")
    resources = types.ModuleType("openai.resources")
    openai.resources = resources
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "openai.resources", resources)
    monkeypatch.setitem(sys.modules, "httpcore", types.ModuleType("httpcore"))


def test_preload_imports_the_lazy_llm_client_modules(monkeypatch):
    worker_module, log = _load_worker_module(monkeypatch)
    _stub_modules(monkeypatch)
    imported = []
    real_import = __import__

    def tracking_import(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)
    worker_module._preload_llm_client_modules()

    assert "openai.resources" in imported
    assert "httpcore" in imported
    log.exception.assert_not_called()


def test_preload_failure_does_not_stop_the_worker(monkeypatch):
    worker_module, log = _load_worker_module(monkeypatch)
    _stub_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "httpcore", None)  # import raises ImportError

    worker_module._preload_llm_client_modules()

    log.exception.assert_called_once()


def test_preload_runs_before_the_agent_task_node_forks():
    tree = ast.parse((PLUGIN_ROOT / "module.py").read_text())
    init = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "init"
    )
    preload_lines = [
        node.lineno for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_preload_llm_client_modules"
    ]
    agent_node_lines = [
        node.lineno for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "TaskNode"
        and any(kw.arg == "pool" and isinstance(kw.value, ast.Constant) and kw.value.value == "agents"
                for kw in node.keywords)
    ]
    assert len(preload_lines) == 1 and len(agent_node_lines) == 1
    assert preload_lines[0] < agent_node_lines[0]
