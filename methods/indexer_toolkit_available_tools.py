"""Indexer task: compute available tools + per-tool schemas for a toolkit instance.

This task is intentionally toolkit-agnostic. Any toolkit-specific logic (e.g. OpenAPI
spec parsing) must live in the SDK, not in indexer_worker.
"""

from __future__ import annotations

from pylon.core.tools import log, web  # pylint: disable=E0611,E0401


class Method:  # pylint: disable=E1101,R0903,W0201
    """Task methods mixed into the indexer_worker Module instance."""

    @web.method()
    def indexer_toolkit_available_tools(self, *args, toolkit_type: str, settings: dict, **kwargs) -> dict:
        """Return available tool names and JSON Schemas for their inputs.

        Args:
            toolkit_type: toolkit type string (e.g., 'openapi')
            settings: persisted toolkit settings

        Returns:
            {
              "tools": [{"name": str, "description": str}],
              "args_schemas": {"tool_name": <json schema dict>}
            }
        """
        _ = args
        _ = kwargs

        try:
            from elitea_sdk.tools import get_toolkit_available_tools  # pylint: disable=E0401,C0415

            return get_toolkit_available_tools(toolkit_type=toolkit_type, settings=settings)

        except Exception as e:  # pylint: disable=W0718
            log.exception("Failed to compute available tools")
            return {"tools": [], "args_schemas": {}, "error": str(e)}
