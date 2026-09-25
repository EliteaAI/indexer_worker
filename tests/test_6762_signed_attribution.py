"""The attribution header is signed so pylon_main can tell it from a hand-built one (#6762).

A user calling /llm/v1 directly with their own token could put any agent id or name into
X-Elitea-Attribution. pylon_main now keeps the labels only when the signature verifies, and
the key is derived from the indexer event-node key, which no project member can read.

Run: python3 -m pytest test_6762_signed_attribution.py
"""

import base64
import hashlib
import hmac
import json
import sys
import types

from test_6677_evaluation_attribution import module

KEY = b"k" * 32
# Twinned with pylon_main usage tests/unit/test_hooks.py VECTOR_SIG — both sides must agree
VECTOR_SIG = "7951aabd37cfed494032f17d7facdb8cf1ecfb5d4d35f91c312f36c719f85659"

KWARGS = {
    module.ENTITY_KWARGS_KEY: {"type": "application", "id": 5, "name": "Agent"},
    module.ROOT_ENTITY_KWARGS_KEY: {"type": "application", "id": 5},
}


def _decode(header):
    return json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))


def test_the_signature_matches_the_shared_vector():
    assert module.sign_attribution({"entity_id": 1, "conversation_id": "c"}, 7, KEY) == VECTOR_SIG


def test_the_header_carries_a_signature_over_its_columns_and_project():
    decoded = _decode(module.attribution_header(KWARGS, project_id="3", key=KEY))
    signature = decoded.pop("sig")
    assert signature == module.sign_attribution(decoded, "3", KEY)
    # Bound to the project: the same columns for another project sign differently
    assert signature != module.sign_attribution(decoded, "4", KEY)


def test_an_int_and_a_str_project_id_sign_the_same():
    # The call site may take it from X-Project-Id (str) or the client kwargs (int)
    columns = {"entity_id": 1}
    assert module.sign_attribution(columns, 3, KEY) == module.sign_attribution(columns, "3", KEY)


def test_without_a_key_the_header_is_sent_unsigned():
    # pylon_main then drops the labels but still records the call
    assert "sig" not in _decode(module.attribution_header(KWARGS, project_id="3", key=None))
    assert "sig" not in _decode(module.attribution_header(KWARGS))


def test_nothing_to_attribute_still_sends_nothing():
    assert module.attribution_header({}, project_id="3", key=KEY) is None


def test_the_key_is_derived_from_the_event_node_key(monkeypatch):
    tools = types.ModuleType("tools")
    tools.worker_core = types.SimpleNamespace(
        descriptor=types.SimpleNamespace(config={"event_node": {"hmac_key": "base"}}),
    )
    monkeypatch.setitem(sys.modules, "tools", tools)
    derived = module.attribution_signing_key()
    assert derived == hmac.new(b"base", b"usage-attribution-v1", hashlib.sha256).digest()
    assert derived != b"base"


def test_a_missing_event_node_key_means_no_signing_key(monkeypatch):
    tools = types.ModuleType("tools")
    tools.worker_core = types.SimpleNamespace(
        descriptor=types.SimpleNamespace(config={"event_node": {"hmac_key": ""}}),
    )
    monkeypatch.setitem(sys.modules, "tools", tools)
    assert module.attribution_signing_key() is None
