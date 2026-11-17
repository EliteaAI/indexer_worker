"""Static contract for the sandbox --bridge protocol (#7578).

CI here has no Deno/Pyodide, so this pins the parts the worker side (elitea-sdk
PyodideSandbox.execute_with_bridge) depends on; behaviour is covered by the SDK
integration tests, which can run against this file via BRIDGE_MAIN_JS.
"""
import pathlib

MAIN_JS = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'sandbox' / 'main.js'
SOURCE = MAIN_JS.read_text()


def test_flags_are_parsed():
    assert 'case "--bridge":' in SOURCE
    assert 'case "--max-line-bytes":' in SOURCE
    assert 'if (flags.bridge) {' in SOURCE


def test_module_name_seen_by_python_programs():
    assert 'registerJsModule("elitea_bridge"' in SOURCE


def test_message_types_and_fields():
    assert '{ type: "call", id, name:' in SOURCE
    assert 'type: "final", success: false' in SOURCE
    assert 'writeBridgeText("stdout"' in SOURCE and 'writeBridgeText("stderr"' in SOURCE
    # The worker reads these keys from incoming messages.
    assert 'message?.type === "result"' in SOURCE
    assert 'message.ok' in SOURCE and 'message.value' in SOURCE and 'message.error' in SOURCE
    assert 'codeMessage.type !== "code"' in SOURCE


def test_error_types_match_the_worker_classification():
    for error_type in ('out_of_memory', 'python_exception', 'result_too_large', 'sandbox_crashed'):
        assert f'"{error_type}"' in SOURCE


def test_stdin_mode_is_still_the_default_path():
    # Without --bridge the original single-JSON output must stay as before.
    assert 'console.log(JSON.stringify(outputJson));' in SOURCE
