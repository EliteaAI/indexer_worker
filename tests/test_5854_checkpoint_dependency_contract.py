"""LangGraph checkpoint compatibility contract for issue #5854.

The worker owns saver construction and direct checkpoint-table cleanup while the
SDK plugin owns the installed LangGraph packages. These tests exercise those
worker functions without importing the Pylon plugin package.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import Mock

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import empty_checkpoint

PLUGIN_ROOT = Path(__file__).parents[1]


def _load_function(relative_path: str, function_name: str, namespace: dict):
    source = (PLUGIN_ROOT / relative_path).read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    exec(  # pylint: disable=exec-used
        compile(ast.Module([function], type_ignores=[]), relative_path, "exec"),
        namespace,
    )
    return namespace[function_name]


def test_sqlite_saver_round_trip_and_worker_cleanup(tmp_path):
    """New saver versions must retain the worker's SQLite schema contract."""

    create_memory_saver = _load_function(
        "utils/agent_execution_common.py",
        "create_memory_saver",
        {"Any": Any, "Dict": Dict},
    )
    log = SimpleNamespace(
        debug=Mock(),
        error=Mock(),
    )
    delete_checkpoints = _load_function(
        "utils/checkpoint_utils.py",
        "delete_checkpoints_by_thread_ids",
        {"log": log},
    )

    database = tmp_path / "checkpoints.db"
    memory, close_memory = create_memory_saver(
        "sqlite",
        {"path": str(database)},
    )
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {
        "messages": [
            AIMessage(
                content="working",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"query": "elitea"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }
    config = {
        "configurable": {
            "thread_id": "compatibility-thread",
            "checkpoint_ns": "",
        }
    }

    memory_closed = False
    try:
        saved_config = memory.put(
            config,
            checkpoint,
            {"source": "issue-5854"},
            {},
        )
        restored = memory.get_tuple(saved_config)

        assert restored is not None
        restored_message = restored.checkpoint["channel_values"]["messages"][0]
        assert isinstance(restored_message, AIMessage)
        assert restored_message.tool_calls[0]["id"] == "call-1"
        assert restored.metadata == {"source": "issue-5854"}

        close_memory()
        memory_closed = True
        delete_checkpoints(
            {"type": "sqlite", "path": str(database)},
            ["compatibility-thread"],
        )
        log.error.assert_not_called()

        reopened_memory, close_reopened_memory = create_memory_saver(
            "sqlite",
            {"path": str(database)},
        )
        try:
            assert reopened_memory.get_tuple(saved_config) is None
        finally:
            close_reopened_memory()
    finally:
        if not memory_closed:
            close_memory()
