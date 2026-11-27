"""EL-7679 (review #1): a structured run that recovered from a rejected forced-tool call must not fail.

The rejected call fires ``on_llm_error``, and ``EliteACallback.llm_error`` never clears; the SDK
marks the response ``structured_ok`` and the worker must not raise the stale error then.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_worker_does_not_raise_llm_error_for_a_successful_structured_run():
    source = (ROOT / 'methods' / 'indexer_predict_agent.py').read_text()
    guard = 'if elitea_callback.llm_error and not response.get("structured_ok"):'

    assert guard in source
    assert 'if elitea_callback.llm_error:\n                raise elitea_callback.llm_error' not in source


def test_output_schema_is_forwarded_to_predict_agent():
    source = (ROOT / 'methods' / 'indexer_predict_agent.py').read_text()

    assert 'output_schema=kwargs.get("output_schema"),' in source
