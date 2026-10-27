"""C-1009 (#7229 item 2): the toolkit Test run and on-demand indexing are admitted runs. The runner takes the run token
out of the kwargs first, validates its context with Main before any tool work (a readable refusal otherwise), hangs the
token and the trusted resolver on its client so the SDK releases the credential slots, and closes both. A governed
dispatch Main did not admit (the platform's attachment read, no context, no credential) runs as before."""

import pathlib

from utils import context_guard

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (PLUGIN_ROOT / "methods" / "indexer_test_toolkit.py").read_text()
WRAPPER = SOURCE[SOURCE.index("def indexer_test_toolkit_tool("):SOURCE.index("def _indexer_test_toolkit_tool_task(")]
TASK = SOURCE[SOURCE.index("def _indexer_test_toolkit_tool_task("):SOURCE.index("def indexer_test_mcp_connection(")]


def test_the_run_token_is_taken_before_the_envelope_and_given_back_when_the_task_ends():
    assert WRAPPER.index("begin_run_token(kwargs)") < WRAPPER.index("enforce_envelope(kwargs)")
    assert "finally:\n            end_run_token(run_token)" in WRAPPER


def test_an_admitted_run_is_validated_before_any_tool_work_and_refused_readably():
    validate = TASK.index("enforce_execution_context(")
    assert TASK.index("if read_envelope(kwargs).context_ref:") < validate < TASK.index("client.test_toolkit_tool(")
    assert 'task_name="indexer_test_toolkit_tool"' in TASK[validate:validate + 200]
    refused = TASK[TASK.index("except ContextRejected as rejected:"):]
    assert "rejection_message(rejected.reason_code)" in refused[:600]
    assert 'result["reason_code"] = rejected.reason_code' in refused[:900]


def test_the_client_gets_the_run_token_and_the_trusted_resolver_and_both_are_closed():
    client = TASK.index("client = EliteAClient(")
    after = TASK[client:]
    assert after.index("attach_run_token(client)") < after.index("attach_credential_resolver(") \
        < after.index("client.test_toolkit_tool(")
    assert 'task_name="indexer_test_toolkit_tool"' in after[after.index("attach_credential_resolver("):][:300]
    tail = TASK[TASK.rindex("finally:"):]
    assert "credential_resolver.close()" in tail and "run_rpc.close()" in tail


def test_a_governed_credential_refusal_reads_as_its_reason():
    assert "error_msg = rejection_message(governed_code)" in TASK
    assert context_guard.rejection_message("NO_SUCH_CODE") == context_guard.DEFAULT_REJECTION_MESSAGE
