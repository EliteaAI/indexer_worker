"""Indexer-side code-validation execution — EVAL-H2 (design §19.7, owns EVAL-E2E-14).

This is the **only** place a code validation's untrusted script actually runs. pylon_main
assembles the trusted prelude (evidence as plain literals) and dispatches the full
``prelude + user script`` here; we execute it inside a **locked-down Deno/Pyodide WASM
sandbox** and return a plain ``{result, stdout, stderr, status, execution_time}`` dict.

Security contract (the release-blocking half of EVAL-E2E-14):
  * Uses ``SyncPyodideSandbox`` (the base engine), **never** ``PyodideSandboxTool`` /
    ``create_sandbox_tool`` / ``SandboxToolkit`` — those default ``allow_net=True`` AND
    inject an ``elitea_client`` with the caller's auth token (sandbox.py:234). A code
    validation is a pure verdict function; it must reach neither the network nor the client.
  * All Deno permission flags are deny-by-default: ``allow_net=False`` (network-denied),
    ``allow_run/ffi/write/env=False``. No ``elitea_client`` is constructed or injected, so
    the name is **undefined** in the prelude — a script that references it gets a NameError
    surfaced as an error verdict, not a silent capability.
  * Resource limits come from ``_read_sandbox_limits_from_env`` (sandbox.py) which returns a
    SAFE default for any missing/malformed var — **never** "unlimited". A timeout/OOM breach
    becomes ``status='error'`` (the SDK maps ``TimeoutExpired`` → ``stderr='...timed out...'``)
    so the run survives and sibling cases still complete.
  * If Deno is absent we return ``status='unavailable'`` — we NEVER fall back to an
    unsandboxed ``exec`` (§19.7).
  * Admission gate: this task runs on the shared ``task_node_light`` pool alongside
    ``invoke_model``/ASR, which has no ``task_limit`` by default — so before spawning we
    check ``max_concurrent`` against the live Deno process count (mirroring the check the
    SDK's own ``PyodideSandboxTool`` path makes) and degrade to ``status='error'`` rather
    than let an eval run spawn unbounded ``deno`` subprocesses on that pool.

The heavy SDK/Deno imports are done **lazily inside the default factory** so this module is
importable — and the mapping / degradation / limit logic is unit-testable — with a stub
``sandbox_factory`` and ``deno_probe``, without the SDK or a Deno binary present.
"""
import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Mirrors utils/code_validation.py on the pylon_main side; kept as bare strings here
# because pylon_indexer must not import elitea_core.
STATUS_SUCCESS = 'success'
STATUS_ERROR = 'error'
STATUS_UNAVAILABLE = 'unavailable'

#: Safe fallback when a caller passes a ``limits`` dict that omits the key entirely
#: (only ever happens in tests — ``_read_sandbox_limits_from_env`` always sets it) so a
#: partial dict degrades safe instead of silently disabling the gate.
_DEFAULT_MAX_CONCURRENT = 16


def _default_deno_probe() -> bool:
    from elitea_sdk.runtime.tools.sandbox import _is_deno_available  # pylint: disable=C0415,E0401
    return _is_deno_available()


def _default_limits() -> dict:
    from elitea_sdk.runtime.tools.sandbox import _read_sandbox_limits_from_env  # pylint: disable=C0415,E0401
    return _read_sandbox_limits_from_env()


def _default_deno_process_count() -> int:
    from elitea_sdk.runtime.tools.sandbox import _count_deno_processes  # pylint: disable=C0415,E0401
    return _count_deno_processes()


def _default_memory_pressure_pct() -> Optional[float]:
    from elitea_sdk.runtime.tools.sandbox import _cgroup_memory_pressure_pct  # pylint: disable=C0415,E0401
    return _cgroup_memory_pressure_pct()


def _default_sandbox_factory():
    """Construct the locked-down sandbox. Deny-by-default, no client, base engine.

    The security-critical denials stay hard-off: ``allow_net=False`` (network-denied),
    ``allow_run/ffi=False``, and no ``elitea_client`` is ever constructed, so the name is
    undefined in the prelude. The only capabilities granted are the minimal ones the
    deployed Pyodide entrypoint (``main.js``) needs to *boot the WASM image offline* — the
    same scoped grants the SDK's own air-gapped tool path uses (sandbox.py):

      * ``allow_env=['SANDBOX_BASE']`` — ``main.js`` reads ``Deno.env.get("SANDBOX_BASE")``
        to locate its warm module/package cache. Scoped to that single var, NOT ``True``.
      * ``allow_read`` — the package cache + deno dir it loads the image from (data dirs,
        nothing user-controlled).
      * ``allow_write=[sandbox_tmp, deno_cache]`` — Pyodide's worker runner needs a writable
        scratch dir under the cache to unpack; scoped to those dirs only, never the root.

    Granting env/read/write ONLY over the non-network cache dirs does not let untrusted code
    reach the network or the client — the release-blocking half of EVAL-E2E-14 (14a/14b) is
    preserved by ``allow_net=False`` + the absent client, independent of these boot grants.
    """
    from elitea_sdk.runtime.langchain.pyodide_sandbox import SyncPyodideSandbox  # pylint: disable=C0415,E0401

    # Mirror the SDK's own defaults (sandbox.py) rather than collapsing to no-access when
    # unset: an unset SANDBOX_BASE/DENO_DIR does not mean "no cache dir exists", it means
    # "use the same default the SDK's air-gapped tool path boots from". Collapsing to
    # allow_read=False here made every validation fail with an opaque Deno permission
    # error in any deployment that never set these vars (this repo's centry/ included).
    sandbox_base = os.environ.get('SANDBOX_BASE') or os.path.expanduser('~/.cache/pyodide')
    sandbox_tmp = os.path.join(sandbox_base, 'tmp')
    deno_cache = os.environ.get('DENO_DIR') or os.path.expanduser('~/.cache/deno')

    return SyncPyodideSandbox(
        stateful=False,
        allow_env=['SANDBOX_BASE'],     # scoped: only the cache-locator var main.js reads
        allow_read=[sandbox_base, sandbox_tmp, deno_cache],
        allow_write=[sandbox_tmp, deno_cache],  # scoped: only the Pyodide scratch/cache dirs
        allow_net=False,   # EVAL-E2E-14 (14a): network-denied
        allow_run=False,
        allow_ffi=False,
    )


def run_code_in_sandbox(
    code: str,
    *,
    sandbox_factory: Optional[Callable[[], Any]] = None,
    deno_probe: Optional[Callable[[], bool]] = None,
    limits: Optional[dict] = None,
    deno_process_count: Optional[Callable[[], int]] = None,
    memory_pressure_pct: Optional[Callable[[], Optional[float]]] = None,
) -> dict:
    """Execute ``code`` (prelude + untrusted script) in the locked-down sandbox.

    Returns a plain dict ``{result, stdout, stderr, status, execution_time}`` — the shape
    pylon_main's ``map_execution_result`` consumes. ``status`` is one of ``'success'`` /
    ``'error'`` / ``'unavailable'``. This function never raises for an execution failure:
    a sandbox timeout/OOM/exception is reported as ``status='error'`` so the caller can turn
    it into a per-case error verdict and keep running sibling cases.

    ``sandbox_factory`` / ``deno_probe`` / ``limits`` / ``deno_process_count`` /
    ``memory_pressure_pct`` are injectable for unit testing; the defaults pull the real SDK
    sandbox + env-derived limits (never "unlimited").

    Admission gate: mirrors the *two* checks the SDK's own ``PyodideSandboxTool`` path makes
    right before spawning (sandbox.py ``max_concurrent``/``_count_deno_processes`` and
    ``memory_pressure_pct``/``_cgroup_memory_pressure_pct``) — this task runs on the shared
    ``task_node_light`` pool alongside ``invoke_model``/ASR, which is unbounded by default
    (``task_limit_light: null``), so nothing else stops an eval run from spawning unbounded
    ``deno`` subprocesses without these checks. Runs *before* the Deno-availability probe:
    a rejected call should not also pay a ``deno --version`` subprocess spawn on the pool the
    gate exists to protect. Both checks are OS/cgroup-global (shared with any other Deno
    consumer on the container, e.g. pipeline Code nodes) and fail OPEN on probe error, same
    posture as the SDK — this is a burst-protection soft reject, not a hard resource lock, so
    a rejected validation is dropped from the run's aggregate rather than retried (no retry
    path exists yet in the caller).
    """
    resolved_limits = limits if limits is not None else _default_limits()

    max_concurrent = resolved_limits.get('max_concurrent', _DEFAULT_MAX_CONCURRENT)
    if max_concurrent and max_concurrent > 0:
        count_deno = deno_process_count or _default_deno_process_count
        try:
            n_deno = count_deno()
        except Exception:  # pylint: disable=W0718
            logger.debug('Sandbox concurrency probe failed; failing open', exc_info=True)
            n_deno = 0  # fail-open on the probe itself, matching the SDK's own posture
        if n_deno >= max_concurrent:
            # Rejections are logged at warning (not just probe failures) so the coverage
            # loss this causes in an eval run's aggregate score is observable rather than
            # silent — this pool is shared with invoke_model/ASR traffic (see module docstring).
            logger.warning(
                'Sandbox validation rejected: %d concurrent Deno processes at limit %d',
                n_deno, max_concurrent,
            )
            return {
                'result': None, 'stdout': None,
                'stderr': f'Sandbox busy: {n_deno} concurrent executions at limit '
                          f'{max_concurrent}. Retry shortly.',
                'status': STATUS_ERROR, 'execution_time': None,
            }

    pressure_pct = resolved_limits.get('memory_pressure_pct')
    if pressure_pct and pressure_pct > 0:
        probe_pressure = memory_pressure_pct or _default_memory_pressure_pct
        try:
            current_pressure = probe_pressure()
        except Exception:  # pylint: disable=W0718
            logger.debug('Sandbox memory-pressure probe failed; failing open', exc_info=True)
            current_pressure = None
        if current_pressure is not None and current_pressure >= pressure_pct:
            logger.warning(
                'Sandbox validation rejected: host memory pressure %.1f%% exceeds threshold %s%%',
                current_pressure, pressure_pct,
            )
            return {
                'result': None, 'stdout': None,
                'stderr': f'Host memory pressure {current_pressure:.1f}% exceeds threshold '
                          f'{pressure_pct}%. Retry shortly.',
                'status': STATUS_ERROR, 'execution_time': None,
            }

    probe = deno_probe or _default_deno_probe
    if not probe():
        return {
            'result': None, 'stdout': None,
            'stderr': 'Deno/Pyodide sandbox runtime is not available.',
            'status': STATUS_UNAVAILABLE, 'execution_time': None,
        }

    factory = sandbox_factory or _default_sandbox_factory

    try:
        sandbox = factory()
        exec_result = sandbox.execute(
            code,
            timeout_seconds=resolved_limits.get('timeout_seconds'),
            memory_limit_mb=resolved_limits.get('wasm_max_mem_mb'),
            root_ca_path=resolved_limits.get('root_ca_path'),
            insecure_tls_domains=resolved_limits.get('allowed_pyodide_domains') or None,
        )
    except Exception as exc:  # pylint: disable=W0718
        # Sandbox construction/dispatch blew up (bad config, missing entrypoint, etc.).
        # Degrade to an error result rather than crashing the whole eval run.
        return {
            'result': None, 'stdout': None,
            'stderr': f'Sandbox execution failed to start: {exc}',
            'status': STATUS_ERROR, 'execution_time': None,
        }

    return {
        'result': getattr(exec_result, 'result', None),
        'stdout': getattr(exec_result, 'stdout', None),
        'stderr': getattr(exec_result, 'stderr', None),
        'status': getattr(exec_result, 'status', STATUS_ERROR),
        'execution_time': getattr(exec_result, 'execution_time', None),
    }
