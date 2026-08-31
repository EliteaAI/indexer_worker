"""Worker pass-through contracts for on-demand Project Context."""

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_both_chat_entrypoints_pass_project_context_to_sdk():
    for relative_path in ("methods/indexer_agent.py", "methods/indexer_predict_agent.py"):
        source = (ROOT / relative_path).read_text()
        assert 'project_context=kwargs.get("project_context")' in source


def test_parent_reconcile_preserves_project_context():
    source = (ROOT / "utils/agent_execution_common.py").read_text()
    carry_keys = source[source.index("carry_keys = ("):source.index("payload = {k:")]

    assert "'project_context'" in carry_keys
