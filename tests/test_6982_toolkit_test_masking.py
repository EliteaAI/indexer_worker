"""#6982: toolkit-test events must not carry expanded credentials.

A toolkit test emits its toolkit_config to the browser (exception event, callbacks, final message).
That config is expanded with unsecret=True, so it used to carry raw keys. The emitted copy now
collapses every expanded credential to its reference and masks secret-named fields.
Loaded by source (same slice as test_6421) so the suite runs without the pylon runtime.
"""
import json
import pathlib
import typing


def _load():
    source = (pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'indexer_test_toolkit.py').read_text()
    start = source.index("def clean_for_json_serialization")
    end = source.index("\ndef test_error")
    namespace = {'Any': typing.Any, 'to_json_primitive': lambda v: str(v), 'log': None}
    exec(compile("from typing import Any\n" + source[start:end], '<masking>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace


NS = _load()
clean = NS['clean_toolkit_config_for_events']

SECRET = "tvly-THIS-IS-THE-PLATFORM-KEY"


def _expanded_tavily():
    return {
        "type": "tavily", "id": 16122, "toolkit_name": "Tavily",
        "settings": {
            "tavily_configuration": {
                "api_key": SECRET, "elitea_title": "tavily_platform",
                "configuration_uuid": "u-1", "configuration_project_id": 1, "configuration_type": "tavily",
            },
            "pgvector_configuration": {
                "connection_string": "postgresql://u:pw@db/x", "configuration_type": "pgvector",
            },
            "selected_tools": ["web_search"],
            "max_results": 5,
        },
    }


def test_expanded_credentials_collapse_to_their_reference():
    out = clean(_expanded_tavily(), "fallback")
    assert out["settings"]["tavily_configuration"] == {
        "elitea_title": "tavily_platform", "configuration_type": "tavily", "configuration_uuid": "u-1",
    }
    assert out["settings"]["pgvector_configuration"] == {"configuration_type": "pgvector"}
    dumped = json.dumps(out)
    assert SECRET not in dumped and "pw@db" not in dumped


def test_non_secret_settings_survive():
    out = clean(_expanded_tavily(), "fallback")
    assert out["settings"]["selected_tools"] == ["web_search"]
    assert out["settings"]["max_results"] == 5
    assert out["type"] == "tavily" and out["id"] == 16122


def test_flat_secret_named_fields_are_masked():
    # Legacy toolkits keep secrets inline in settings, outside any configuration block
    cfg = {"settings": {"token": "ghp_x", "client_secret": "s", "api_key": "k", "base_url": "https://x",
                        "use_token": True, "headers": {"Authorization": "Bearer z"}}}
    out = clean(cfg, "fallback")["settings"]
    assert out["token"] == out["client_secret"] == out["api_key"] == "***"
    assert out["headers"]["Authorization"] == "***"
    assert out["base_url"] == "https://x" and out["use_token"] is True


def test_empty_secret_values_are_left_alone():
    assert clean({"settings": {"token": ""}}, "f")["settings"]["token"] == ""
