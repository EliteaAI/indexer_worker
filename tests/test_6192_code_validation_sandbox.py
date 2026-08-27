"""Indexer-side code-validation execution for #6192 / EVAL-E2E-14 (design §19.7).

This is the only place a code validation's untrusted script actually runs, so these
tests pin the degradation contract that keeps one bad validation from taking down the
whole eval run:
  * Deno absent            -> status='unavailable' (never an unsandboxed exec fallback),
  * sandbox blows up       -> status='error' (construction/dispatch failure is degraded),
  * successful execution   -> the CodeExecutionResult is flattened to the plain dict shape
                              pylon_main's map_execution_result consumes,
and the security posture the default factory must construct: the release-blocking
denials stay hard-off (allow_net=False, allow_run/ffi=False, no elitea_client, base
SyncPyodideSandbox engine), while env/read/write are scoped ONLY to the offline Pyodide
cache dirs the deployed main.js needs to boot (allow_env=['SANDBOX_BASE'],
allow_write=[tmp, deno_cache]) — never True, never the filesystem root.

The module keeps every SDK/Deno import lazy inside its functions, so it loads by source
here and runs with injected stub factory/probe/limits — no SDK, no Deno binary needed.

Run from this directory:
    cd tests && python3 -m pytest test_6192_code_validation_sandbox.py
(pytest from the plugin root imports module.py, which needs pylon's `tools`.)
"""

import importlib.util
import pathlib

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'code_validation_sandbox.py'
)


def _load():
    spec = importlib.util.spec_from_file_location('code_validation_sandbox', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sandbox = _load()


class _FakeExecResult:
    """Stands in for the SDK's CodeExecutionResult dataclass."""

    def __init__(self, *, result, stdout, stderr, status, execution_time):
        self.result = result
        self.stdout = stdout
        self.stderr = stderr
        self.status = status
        self.execution_time = execution_time


class _FakeSandbox:
    def __init__(self, exec_result=None, raise_exc=None):
        self._exec_result = exec_result
        self._raise = raise_exc
        self.execute_calls = []

    def execute(self, code, **kwargs):
        self.execute_calls.append((code, kwargs))
        if self._raise is not None:
            raise self._raise
        return self._exec_result


_LIMITS = {'timeout_seconds': 55.0, 'wasm_max_mem_mb': 512,
           'root_ca_path': None, 'allowed_pyodide_domains': None}


# ---------------------------------------------------------------------------
# Degradation contract
# ---------------------------------------------------------------------------

def test_deno_absent_is_unavailable():
    out = sandbox.run_code_in_sandbox(
        'result = True',
        deno_probe=lambda: False,
        sandbox_factory=_must_not_be_called,
        limits=_LIMITS,
    )
    assert out['status'] == sandbox.STATUS_UNAVAILABLE
    assert out['result'] is None
    assert 'not available' in out['stderr']


def _must_not_be_called():  # pragma: no cover - guard
    raise AssertionError('sandbox_factory must not run when Deno is unavailable')


def test_success_is_flattened_to_plain_dict():
    fake = _FakeSandbox(_FakeExecResult(
        result=True, stdout='ok', stderr=None,
        status='success', execution_time=0.4,
    ))
    out = sandbox.run_code_in_sandbox(
        'result = output == "x"',
        deno_probe=lambda: True,
        sandbox_factory=lambda: fake,
        limits=_LIMITS,
    )
    assert out == {'result': True, 'stdout': 'ok', 'stderr': None,
                   'status': 'success', 'execution_time': 0.4}
    # limits were threaded through to execute()
    _, kwargs = fake.execute_calls[0]
    assert kwargs['timeout_seconds'] == 55.0
    assert kwargs['memory_limit_mb'] == 512


def test_sandbox_timeout_status_error_is_passed_through():
    # The SDK maps TimeoutExpired -> status='error' with a timed-out stderr.
    fake = _FakeSandbox(_FakeExecResult(
        result=None, stdout=None, stderr='Execution timed out after 55 seconds',
        status='error', execution_time=55.0,
    ))
    out = sandbox.run_code_in_sandbox(
        'while True: pass',
        deno_probe=lambda: True,
        sandbox_factory=lambda: fake,
        limits=_LIMITS,
    )
    assert out['status'] == 'error'
    assert 'timed out' in out['stderr']


def test_sandbox_construction_failure_degrades_to_error():
    def boom():
        raise RuntimeError('missing entrypoint')

    out = sandbox.run_code_in_sandbox(
        'result = True',
        deno_probe=lambda: True,
        sandbox_factory=boom,
        limits=_LIMITS,
    )
    assert out['status'] == sandbox.STATUS_ERROR
    assert 'failed to start' in out['stderr']
    assert 'missing entrypoint' in out['stderr']


def test_execute_raising_degrades_to_error():
    fake = _FakeSandbox(raise_exc=ValueError('dispatch blew up'))
    out = sandbox.run_code_in_sandbox(
        'result = True',
        deno_probe=lambda: True,
        sandbox_factory=lambda: fake,
        limits=_LIMITS,
    )
    assert out['status'] == sandbox.STATUS_ERROR
    assert 'dispatch blew up' in out['stderr']


def test_missing_result_attr_defaults_to_none():
    class _Bare:
        status = 'success'  # nothing else

    out = sandbox.run_code_in_sandbox(
        'x = 1',
        deno_probe=lambda: True,
        sandbox_factory=lambda: _FakeSandbox(_Bare()),
        limits=_LIMITS,
    )
    assert out['status'] == 'success'
    assert out['result'] is None
    assert out['stdout'] is None
    assert out['execution_time'] is None


# ---------------------------------------------------------------------------
# Admission gate — task_node_light has no task_limit by default, so this is the
# only thing stopping an eval run from spawning unbounded deno subprocesses on
# the pool that also serves invoke_model/ASR (see sandbox.py's own tool-path check).
# ---------------------------------------------------------------------------

def test_at_concurrency_limit_is_rejected_without_running():
    limits = dict(_LIMITS, max_concurrent=4)
    out = sandbox.run_code_in_sandbox(
        'result = True',
        deno_probe=lambda: True,
        sandbox_factory=_must_not_be_called,
        deno_process_count=lambda: 4,
        limits=limits,
    )
    assert out['status'] == sandbox.STATUS_ERROR
    assert 'concurrency limit' in out['stderr']
    assert out['result'] is None


def test_below_concurrency_limit_runs_normally():
    limits = dict(_LIMITS, max_concurrent=4)
    fake = _FakeSandbox(_FakeExecResult(
        result=True, stdout=None, stderr=None, status='success', execution_time=0.1,
    ))
    out = sandbox.run_code_in_sandbox(
        'result = True',
        deno_probe=lambda: True,
        sandbox_factory=lambda: fake,
        deno_process_count=lambda: 3,
        limits=limits,
    )
    assert out['status'] == 'success'
    assert len(fake.execute_calls) == 1


def test_concurrency_gate_disabled_when_max_concurrent_is_zero():
    limits = dict(_LIMITS, max_concurrent=0)
    fake = _FakeSandbox(_FakeExecResult(
        result=True, stdout=None, stderr=None, status='success', execution_time=0.1,
    ))
    out = sandbox.run_code_in_sandbox(
        'result = True',
        deno_probe=lambda: True,
        sandbox_factory=lambda: fake,
        deno_process_count=_must_not_be_called,
        limits=limits,
    )
    assert out['status'] == 'success'


def test_process_count_probe_failure_fails_open():
    limits = dict(_LIMITS, max_concurrent=4)
    fake = _FakeSandbox(_FakeExecResult(
        result=True, stdout=None, stderr=None, status='success', execution_time=0.1,
    ))

    def boom():
        raise OSError('cannot list processes')

    out = sandbox.run_code_in_sandbox(
        'result = True',
        deno_probe=lambda: True,
        sandbox_factory=lambda: fake,
        deno_process_count=boom,
        limits=limits,
    )
    assert out['status'] == 'success'


# ---------------------------------------------------------------------------
# Security posture the default factory must construct (EVAL-E2E-14)
# ---------------------------------------------------------------------------

def test_default_factory_is_deny_by_default_no_client(monkeypatch):
    """The default factory must build a base SyncPyodideSandbox with every allow_*
    denied and NO elitea_client. We stub the SDK class to capture the kwargs rather
    than boot a real WASM image."""
    captured = {}

    class _StubSyncPyodideSandbox:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys
    import types

    pkg = types.ModuleType('elitea_sdk')
    runtime = types.ModuleType('elitea_sdk.runtime')
    langchain = types.ModuleType('elitea_sdk.runtime.langchain')
    pyo = types.ModuleType('elitea_sdk.runtime.langchain.pyodide_sandbox')
    pyo.SyncPyodideSandbox = _StubSyncPyodideSandbox
    monkeypatch.setitem(sys.modules, 'elitea_sdk', pkg)
    monkeypatch.setitem(sys.modules, 'elitea_sdk.runtime', runtime)
    monkeypatch.setitem(sys.modules, 'elitea_sdk.runtime.langchain', langchain)
    monkeypatch.setitem(sys.modules, 'elitea_sdk.runtime.langchain.pyodide_sandbox', pyo)
    monkeypatch.setenv('SANDBOX_BASE', '/tmp/sbx')
    monkeypatch.setenv('DENO_DIR', '/tmp/deno')

    box = sandbox._default_sandbox_factory()
    assert isinstance(box, _StubSyncPyodideSandbox)

    # release-blocking denials: network, run, ffi hard-off (never relaxed)
    assert captured['allow_net'] is False
    assert captured['allow_run'] is False
    assert captured['allow_ffi'] is False
    assert captured['stateful'] is False
    # env scoped to the single cache-locator var main.js reads — NOT True/unbounded
    assert captured['allow_env'] == ['SANDBOX_BASE']
    # read/write granted ONLY over the offline Pyodide cache dirs (base, its tmp, deno
    # cache) — data dirs, nothing user-controlled, never the filesystem root
    assert captured['allow_read'] == ['/tmp/sbx', '/tmp/sbx/tmp', '/tmp/deno']
    assert captured['allow_write'] == ['/tmp/sbx/tmp', '/tmp/deno']
    # no elitea_client anywhere in the constructor kwargs
    assert not any('client' in k for k in captured)


def test_default_factory_read_paths_false_when_env_unset(monkeypatch):
    captured = {}

    class _StubSyncPyodideSandbox:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys
    import types

    for name in ('elitea_sdk', 'elitea_sdk.runtime',
                 'elitea_sdk.runtime.langchain',
                 'elitea_sdk.runtime.langchain.pyodide_sandbox'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules['elitea_sdk.runtime.langchain.pyodide_sandbox'].SyncPyodideSandbox = \
        _StubSyncPyodideSandbox
    monkeypatch.delenv('SANDBOX_BASE', raising=False)
    monkeypatch.delenv('DENO_DIR', raising=False)

    sandbox._default_sandbox_factory()
    # No env dirs -> read/write collapse to False (not an empty list granting nothing weird)
    assert captured['allow_read'] is False
    assert captured['allow_write'] is False
    # allow_env stays scoped to the single var regardless (harmless if the var is unset)
    assert captured['allow_env'] == ['SANDBOX_BASE']
