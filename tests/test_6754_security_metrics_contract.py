"""Contract of the security metrics accessor copy in this plugin (C-1009, #6754).

``utils/security_metrics_client.py`` duplicates the accessor of the tracing plugin. The values below are the
contract: the same vectors are pinned in tracing, auth, indexer_worker and elitea_core, so a change to the
line format, the token rules, the label allow-list or the rate cap has to be made in all four places.
"""

import sys
import types

import pytest

from utils import envelope_guard, security_metrics_client as client

# What this plugin declares: metric name -> label keys.
DECLARED = {"worker_capability_reject_total": ("capability",)}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def local(monkeypatch):
    """The accessor in its local-fallback mode: no tracing tool, lines captured from pylon's log."""
    lines = []
    clock = Clock()
    monkeypatch.setitem(sys.modules, "pylon", types.ModuleType("pylon"))
    monkeypatch.setitem(sys.modules, "pylon.core", types.ModuleType("pylon.core"))
    monkeypatch.setitem(sys.modules, "pylon.core.tools",
                        types.SimpleNamespace(log=types.SimpleNamespace(info=lambda msg, *a: lines.append(msg))))
    monkeypatch.setitem(sys.modules, "tools", types.ModuleType("tools"))
    monkeypatch.setattr(client, "_clock", clock)
    return types.SimpleNamespace(lines=lines, clock=clock)


def test_pinned_constants():
    assert (client.ALLOWED_LABEL_KEYS, client.MAX_LABEL_VALUE_LEN, client.LOG_INTERVAL_S) == (
        frozenset({"capability"}), 64, 10.0)


def test_line_format_token_rules_and_rate_cap(local):
    plain = client.SecurityCounter("auth_epoch_mismatch_total", "d")
    labelled = client.SecurityCounter("worker_capability_reject_total", "d", ("capability",))

    plain.inc()
    labelled.inc(capability="task_envelope.v1")
    labelled.inc(capability="x y=1\nz")
    plain.inc()
    plain.inc()
    local.clock.t += 10
    plain.inc()
    labelled.inc(capability="")
    labelled.inc(capability="é" * 100)

    assert local.lines == [
        "security_metric name=auth_epoch_mismatch_total value=1",
        "security_metric name=worker_capability_reject_total value=1 capability=task_envelope.v1",
        "security_metric name=worker_capability_reject_total value=1 capability=x_y_1_z",
        "security_metric name=auth_epoch_mismatch_total value=3",
        "security_metric name=worker_capability_reject_total value=1 capability=-",
        "security_metric name=worker_capability_reject_total value=1 capability=" + "_" * 64,
    ]
    assert plain.total() == 4 and labelled.total() == 4


@pytest.mark.parametrize("key", ["user_id", "project_id", "token", "reason"])
def test_label_keys_outside_the_allow_list_are_refused(key):
    with pytest.raises(ValueError):
        client.SecurityCounter("m_total", "d", (key,))


def test_wrong_labels_are_a_caller_error(local):
    counter = client.SecurityCounter("m_total", "d", ("capability",))

    with pytest.raises(ValueError):
        counter.inc()
    with pytest.raises(ValueError):
        counter.inc(capability="a", extra="b")
    assert counter.total() == 0 and local.lines == []


def test_declared_counters_match_the_contract():
    declared = {envelope_guard.WORKER_CAPABILITY_REJECT.name: envelope_guard.WORKER_CAPABILITY_REJECT.labels}

    assert declared == DECLARED
