"""trigger_source rides the run attribution so analytics can separate automated runs (#6881).

Scheduled, webhook and index runs bill the configuring user, but are not that user's activity.
Only automated sources are stamped: manual (UI/API) and anything unknown stay absent/NULL,
so a manual run's header and tool row are unchanged from before.

Run: python3 -m pytest test_6881_trigger_source.py
"""

import base64
import json

import pytest

from test_6677_evaluation_attribution import module

KEY = b"k" * 32
APP = {"application": {"id": 7, "version_id": 3, "name": "Pipe"}, "conversation_id": "c-1"}


def _decode(header):
    return json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))


@pytest.mark.parametrize("source", ["scheduled", "webhook", "index"])
def test_automated_source_is_signed_into_the_header(source):
    decoded = _decode(module.attribution_header({**APP, "trigger_source": source}, project_id=3, key=KEY))
    signature = decoded.pop("sig")
    assert decoded["trigger_source"] == source
    # Covered by the signature, so a caller cannot relabel its own run as automated
    assert signature == module.sign_attribution(decoded, 3, KEY)


@pytest.mark.parametrize("source", [None, "", "manual", "cron", 5])
def test_manual_or_unknown_source_is_not_stamped(source):
    assert module.run_attribution({**APP, "trigger_source": source})["trigger_source"] is None
    decoded = _decode(module.attribution_header({**APP, "trigger_source": source}, project_id=3, key=KEY))
    assert "trigger_source" not in decoded


def test_index_run_with_no_entity_still_sends_a_header():
    # An index task names no application, but the run is still worth labelling as automated
    decoded = _decode(module.attribution_header({"trigger_source": "index"}, project_id=3, key=KEY))
    assert decoded["trigger_source"] == "index"


def test_tool_row_and_attribution_carry_the_same_source(monkeypatch):
    captured = {}

    class _Conn:
        def execute(self, _statement, params):
            captured.update(params)

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(module, "_get_engine", lambda: type("E", (), {"connect": lambda self: _Conn()})())

    attribution = module.build_attribution(
        {**APP, "trigger_source": "scheduled"}, {"project_id": 3, "user_context": {"user_id": 9}}, "t-1",
    )
    module.record_tool_event(attribution, "read_file", 5, False, "lc-1")

    assert captured["trigger_source"] == "scheduled"
    assert captured["user_id"] == 9
