"""The indexer signs tool spend records for the SDK (#6980) so pylon_main knows they came from a worker.

The SDK gets only a callable; the key stays here, derived exactly like the attribution key.
Run: python3 -m pytest test_6980_tool_spend_signer.py
"""

import hashlib
import hmac
import sys
import types

from test_6677_evaluation_attribution import module

# Twinned with elitea-sdk and elitea_core RECORD_VECTOR — all three must agree
RECORD_VECTOR_KEY = b"k" * 32
RECORD_VECTOR_MESSAGE = b"tool-spend-record-v1\n3\nweb_search\ntavily\nsearch\nu-1\nP1\n2.0\n1\n15"
RECORD_VECTOR_SIG = "43646bedcb2b154ed310f0a96213b099788909601e653e4e6879579f14fb0551"


def test_the_signer_matches_the_shared_vector():
    assert module.tool_spend_signer(RECORD_VECTOR_KEY)(RECORD_VECTOR_MESSAGE) == RECORD_VECTOR_SIG


def test_the_default_key_is_the_attribution_key(monkeypatch):
    tools = types.ModuleType("tools")
    tools.worker_core = types.SimpleNamespace(
        descriptor=types.SimpleNamespace(config={"event_node": {"hmac_key": "base"}}),
    )
    monkeypatch.setitem(sys.modules, "tools", tools)
    key = hmac.new(b"base", b"usage-attribution-v1", hashlib.sha256).digest()
    assert module.tool_spend_signer()(b"m") == hmac.new(key, b"m", hashlib.sha256).hexdigest()


def test_no_key_means_no_signer(monkeypatch):
    tools = types.ModuleType("tools")
    tools.worker_core = types.SimpleNamespace(
        descriptor=types.SimpleNamespace(config={"event_node": {"hmac_key": ""}}),
    )
    monkeypatch.setitem(sys.modules, "tools", tools)
    assert module.tool_spend_signer() is None
