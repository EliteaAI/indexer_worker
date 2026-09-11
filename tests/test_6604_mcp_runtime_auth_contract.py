"""Worker contract for runtime MCP family authorization."""

import ast
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_application_chat_lets_sdk_expose_auth_control_only_after_a_real_challenge():
    source = (ROOT / "methods" / "indexer_agent.py").read_text()
    tree = ast.parse(source)
    application_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "application"
    ]

    assert len(application_calls) == 1
    keyword_names = {keyword.arg for keyword in application_calls[0].keywords}
    assert "mcp_tokens" in keyword_names
    assert "user_declined_mcp_servers" in keyword_names
    assert "tools" not in keyword_names
    assert "_make_mcp_auth_tools" not in source
