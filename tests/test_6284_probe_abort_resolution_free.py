"""Abort path of fork_dns_probe must never resolve a hostname (#6284).

The probe detected the poisoned-resolver condition correctly but announced it with
`log.error`, and centry_logging's eventnode handler turns any log record into a
`redis.publish` -> `getaddrinfo` -- the exact lookup the probe just proved impossible.
The abort therefore wedged on the same dead mutex it was reporting, with the log line
never arriving. These tests pin the invariant: between "probe failed" and process exit,
nothing may touch name resolution, and the diagnostic must still land on stderr.

A poisoned resolver is simulated by a `getaddrinfo` that blocks forever; at the C level
the real inherited-NSS-mutex hang is indistinguishable from that.

Run from this directory (`cd tests && python3 -m pytest test_6284_probe_abort_resolution_free.py`):
invoking pytest from the plugin root imports the plugin's module.py, which needs the
pylon runtime and fails collection -- the other test files here behave the same way.
"""

import importlib.util
import logging
import os
import pathlib
import socket
import subprocess
import sys
import threading

import pytest

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_probe():
    """Load the probe by path: importing the package pulls in the pylon runtime."""
    spec = importlib.util.spec_from_file_location(
        "fork_dns_probe_under_test", PLUGIN_ROOT / "utils" / "fork_dns_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fork_dns_probe = _load_probe()


class _ResolvingHandler(logging.Handler):
    """Stands in for EventNodeLogHandler: emitting resolves a hostname."""

    def __init__(self):
        super().__init__()
        self.emitted = 0

    def emit(self, record):
        socket.getaddrinfo("redis.example.invalid", 6379)
        self.emitted += 1


@pytest.fixture(name="poisoned_resolver")
def _poisoned_resolver(monkeypatch):
    """getaddrinfo that never returns, like one holding an orphaned NSS mutex."""
    never = threading.Event()
    calls = []
    #
    def _wedged(*args, **kwargs):
        calls.append(args)
        never.wait()  # never set: the mutex owner does not exist in this process
    #
    monkeypatch.setattr(socket, "getaddrinfo", _wedged)
    return calls


@pytest.fixture(name="fork_child")
def _fork_child(monkeypatch):
    """Make running_in_fork_child() true without an actual fork."""
    monkeypatch.setattr(fork_dns_probe, "running_in_fork_child", lambda: True)


def test_abort_returns_instead_of_wedging(poisoned_resolver, fork_child, capfd):
    # The regression: this call used to block forever inside its own log.error
    handler = _ResolvingHandler()
    logging.root.addHandler(handler)
    try:
        result = fork_dns_probe.check_fork_dns(
            {"fork_dns_probe": {"timeout_seconds": 0.2}}, "task-6284"
        )
    finally:
        logging.root.removeHandler(handler)
    #
    assert result is False
    assert handler.emitted == 0, "the abort path emitted through a resolving handler"


def test_abort_diagnostic_reaches_stderr(poisoned_resolver, fork_child, capfd):
    # Current failure mode is silent, so the message landing is itself the fix
    fork_dns_probe.check_fork_dns({"fork_dns_probe": {"timeout_seconds": 0.2}}, "task-6284")
    #
    stderr = capfd.readouterr().err
    assert "fork_dns_probe" in stderr
    assert "task-6284" in stderr


def test_resolving_handlers_are_detached_before_the_diagnostic(
        poisoned_resolver, fork_child,
):
    # Any log.* between the failed probe and exit has the same hazard, including calls
    # from library code, so the handler is removed rather than merely bypassed
    handler = _ResolvingHandler()
    stream_handler = logging.StreamHandler()
    logging.root.addHandler(handler)
    logging.root.addHandler(stream_handler)
    try:
        fork_dns_probe.check_fork_dns({"fork_dns_probe": {"timeout_seconds": 0.2}}, "task-6284")
        #
        assert handler not in logging.root.handlers
        assert stream_handler in logging.root.handlers, "stderr logging must survive"
    finally:
        for leftover in (handler, stream_handler):
            if leftover in logging.root.handlers:
                logging.root.removeHandler(leftover)


def test_probe_source_has_no_logging_calls_on_the_abort_path():
    # Guard against a future edit reintroducing log.* here: nothing in this module may
    # go through the logging pipeline, since the eventnode handler is always attached
    source = (PLUGIN_ROOT / "utils" / "fork_dns_probe.py").read_text()
    assert "log.error" not in source
    assert "log.warning" not in source
    assert "log.info" not in source
    assert "from pylon.core.tools import log" not in source


def test_exit_branch_does_not_log_before_exiting(poisoned_resolver, fork_child):
    # os._exit(1) does not rescue a wedge that happens before it is reached, so this
    # branch must be resolution-free too. Run in a subprocess: os._exit kills pytest.
    script = f'''
import importlib.util, logging, os, socket, sys, threading
spec = importlib.util.spec_from_file_location(
    "probe", {str(PLUGIN_ROOT / "utils" / "fork_dns_probe.py")!r})
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.running_in_fork_child = lambda: True
never = threading.Event()
socket.getaddrinfo = lambda *a, **k: never.wait()


class Resolving(logging.Handler):
    def emit(self, record):
        socket.getaddrinfo("redis.example.invalid", 6379)


logging.root.addHandler(Resolving())
probe.check_fork_dns(
    {{"fork_dns_probe": {{"timeout_seconds": 0.2}}, "agents_result_transport": "events"}},
    "task-6284",
)
os.write(1, b"NOT_REACHED")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False,
    )
    #
    assert completed.returncode == 1, f"expected os._exit(1), got {completed.returncode}"
    assert "NOT_REACHED" not in completed.stdout
    assert "without a result" in completed.stderr


def test_healthy_process_is_untouched():
    # The guard must stay invisible on the happy path: no handler surgery, no stderr
    handler = _ResolvingHandler()
    logging.root.addHandler(handler)
    try:
        result = fork_dns_probe.check_fork_dns({"fork_dns_probe": {"enabled": True}}, "task-ok")
        assert result is True
        assert handler in logging.root.handlers
    finally:
        logging.root.removeHandler(handler)


def test_negative_probe_target_is_fully_qualified():
    # A bare NXDOMAIN name is retried once per resolv.conf search domain. On a 9-domain
    # search list that walk took 8.7s against a 2.0s probe timeout, so every healthy fork
    # child was aborted; the trailing dot stops the search-list expansion.
    negative_targets = [
        host for host, _ in fork_dns_probe._PROBE_TARGETS if host.endswith(".invalid.")
    ]
    assert negative_targets, "the NXDOMAIN probe target must be fully qualified"


def test_probe_reports_usable_on_a_healthy_resolver():
    # The negative target must resolve (as NXDOMAIN) well inside the timeout, or the guard
    # becomes a self-inflicted outage rather than a protection
    assert fork_dns_probe.probe_dns_usable(timeout=2.0) is True
