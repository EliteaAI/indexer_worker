"""C-1009 (#6754): the capability-reject counter uses the shared tool, or a local fallback."""

import pathlib
import sys
import time
import types

import pytest

from utils import envelope_guard, security_metrics_client
from utils.task_envelope import CAPABILITY_ENVELOPE_V1, EnvelopeRejected, attach_envelope

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def lines(monkeypatch):
    captured = []
    log = types.SimpleNamespace(info=lambda msg, *a: captured.append(msg))
    monkeypatch.setitem(sys.modules, "pylon", types.ModuleType("pylon"))
    monkeypatch.setitem(sys.modules, "pylon.core", types.ModuleType("pylon.core"))
    monkeypatch.setitem(sys.modules, "pylon.core.tools", types.SimpleNamespace(log=log))
    monkeypatch.setitem(sys.modules, "tools", types.ModuleType("tools"))
    return captured


class SharedCounter:
    def __init__(self):
        self.incs = []

    def inc(self, value=1, **labels):
        self.incs.append((value, labels))

    def total(self, **labels):
        return sum(v for v, got in self.incs if all(got.get(k) == x for k, x in labels.items()))


class SharedStub:
    def __init__(self):
        self.calls, self.counters = [], {}

    def counter(self, name, description, labels=()):
        self.calls.append((name, description, tuple(labels)))
        return self.counters.setdefault(name, SharedCounter())


def test_without_the_tool_it_counts_per_label_and_logs_the_structured_line(lines, monkeypatch):
    counter = security_metrics_client.SecurityCounter("c_total", "d", ("capability",))

    counter.inc(capability="future.v9")
    monkeypatch.setattr(security_metrics_client, "_clock", lambda: time.monotonic() + 3600)
    counter.inc(capability="future.v9")
    counter.inc(capability="-")

    assert (counter.total(), counter.total(capability="future.v9"), counter.total(capability="-")) == (3, 2, 1)
    assert lines == ["security_metric name=c_total value=1 capability=future.v9"] * 2 \
        + ["security_metric name=c_total value=1 capability=-"]


def test_with_the_tool_it_delegates_and_leaves_logging_to_it(lines):
    shared = SharedStub()
    sys.modules["tools"].security_metrics = shared
    counter = security_metrics_client.SecurityCounter("c_total", "d", ("capability",))

    counter.inc(capability="future.v9")

    assert shared.calls == [("c_total", "d", ("capability",))]
    assert shared.counters["c_total"].incs == [(1, {"capability": "future.v9"})]
    assert lines == []
    assert counter.total(capability="future.v9") == 1


def test_unsafe_label_values_are_reduced_to_one_token(lines):
    counter = security_metrics_client.SecurityCounter("c_total", "d", ("capability",))

    counter.inc(capability="a b\nsecurity_metric name=x")

    assert lines == ["security_metric name=c_total value=1 capability=a_b_security_metric_name_x"]


def test_guard_reaches_the_shared_tool_with_the_capability_label(lines):
    shared = SharedStub()
    sys.modules["tools"].security_metrics = shared
    kwargs = attach_envelope({}, "governed")
    kwargs["envelope"]["required_capabilities"] = [CAPABILITY_ENVELOPE_V1, "future.v9"]

    with pytest.raises(EnvelopeRejected):
        envelope_guard.enforce_envelope(kwargs)

    assert shared.calls == [("worker_capability_reject_total", envelope_guard.WORKER_CAPABILITY_REJECT.description,
                             ("capability",))]
    assert shared.counters["worker_capability_reject_total"].incs == [(1, {"capability": "unknown"})]


def test_guard_falls_back_when_the_tool_is_absent(lines):
    kwargs = attach_envelope({}, "governed")
    kwargs["envelope"]["required_capabilities"] = [CAPABILITY_ENVELOPE_V1, "future.v9"]
    before = envelope_guard.WORKER_CAPABILITY_REJECT.total(capability="unknown")

    with pytest.raises(EnvelopeRejected):
        envelope_guard.enforce_envelope(kwargs)

    assert envelope_guard.WORKER_CAPABILITY_REJECT.total(capability="unknown") == before + 1
    assert lines == ["security_metric name=worker_capability_reject_total value=1 capability=unknown"]


def test_counter_name_and_label_are_unchanged():
    assert envelope_guard.WORKER_CAPABILITY_REJECT.name == "worker_capability_reject_total"
    assert envelope_guard.WORKER_CAPABILITY_REJECT.labels == ("capability",)
    assert "prometheus_client" not in (PLUGIN_ROOT / "utils" / "envelope_guard.py").read_text()


def _refuse(*required):
    kwargs = attach_envelope({}, "governed")
    kwargs["envelope"]["required_capabilities"] = [CAPABILITY_ENVELOPE_V1, *required]
    with pytest.raises(EnvelopeRejected):
        envelope_guard.enforce_envelope(kwargs)


def test_an_unknown_or_long_capability_is_bucketed_as_unknown(lines):
    shared = SharedStub()
    sys.modules["tools"].security_metrics = shared

    _refuse("x" * 5000, "future.v9\nsecurity_metric name=forged")

    assert shared.counters["worker_capability_reject_total"].incs == [(1, {"capability": "unknown"})] * 2


def test_a_known_capability_keeps_its_own_label(lines, monkeypatch):
    shared = SharedStub()
    sys.modules["tools"].security_metrics = shared
    monkeypatch.setattr(envelope_guard, "slots_supported", lambda: False)

    _refuse("credential_slots.v1")

    assert shared.counters["worker_capability_reject_total"].incs == [(1, {"capability": "credential_slots.v1"})]


def test_twenty_missing_capabilities_count_at_most_eight_times(lines):
    shared = SharedStub()
    sys.modules["tools"].security_metrics = shared

    _refuse(*[f"future.v{n:02d}" for n in range(20)])

    assert len(shared.counters["worker_capability_reject_total"].incs) == envelope_guard.MAX_REJECT_LABELS == 8


def test_the_full_detail_goes_to_the_warning_not_the_metric(lines, monkeypatch):
    warnings = []
    sys.modules["pylon.core.tools"].log.warning = lambda msg, *a: warnings.append(msg % a)

    _refuse(*[f"future.v{n:02d}" for n in range(30)])

    assert len(warnings) == 1 and "future.v19" in warnings[0] and "future.v20" not in warnings[0]
    assert "CAPABILITY_MISSING" in warnings[0]
