"""elitea_issues#7660: Main now resolves hidden pgvector connstrs too and sends them in the cleanup event; the
handler's start line logged the whole payload, so the project DB password reached the log."""

import pathlib
import types

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONNSTR = "postgresql://user:hidden-password@db/project_8"


def test_the_cleanup_start_line_never_logs_the_connstr():
    logged = []
    source = (PLUGIN_ROOT / "methods" / "indexer_empty_agent_state.py").read_text()
    source = source.replace("from pylon.core.tools import log  # pylint: disable=E0611,E0401", "")
    source = source.replace("from pylon.core.tools import web  # pylint: disable=E0611,E0401", "")
    source = source.replace(
        "from ..utils.checkpoint_utils import resolve_memory_config, delete_checkpoints_by_thread_ids", "")
    record = lambda message, *args, **kwargs: logged.append(message % args if args else message)  # noqa: E731
    namespace = {
        "log": types.SimpleNamespace(debug=record, info=record, warning=record, error=record, exception=record),
        "web": types.SimpleNamespace(method=lambda *a, **k: (lambda f: f)),
        "resolve_memory_config": lambda config, connstr: {"type": "memory"},
        "delete_checkpoints_by_thread_ids": lambda *a, **k: None,
    }
    exec(compile(source, "indexer_empty_agent_state.py", "exec"), namespace)  # pylint: disable=W0122
    module = types.SimpleNamespace(indexer_enable_logging=lambda: None, descriptor=types.SimpleNamespace(config={}))

    namespace["Method"].empty_agent_state(module, None, {"days_to_retain": 3, "pgvector_connstr": CONNSTR})

    assert logged and not any("hidden-password" in line for line in logged)
