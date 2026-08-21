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

The heavy SDK/Deno imports are done **lazily inside the default factory** so this module is
importable — and the mapping / degradation / limit logic is unit-testable — with a stub
``sandbox_factory`` and ``deno_probe``, without the SDK or a Deno binary present.
"""
from typing import Any, Callable, Optional

# Mirrors utils/code_validation.py on the pylon_main side; kept as bare strings here
# because pylon_indexer must not import elitea_core.
STATUS_SUCCESS = 'success'
STATUS_ERROR = 'error'
STATUS_UNAVAILABLE = 'unavailable'


def _default_deno_probe() -> bool:
    from elitea_sdk.runtime.tools.sandbox import _is_deno_available  # pylint: disable=C0415,E0401
    return _is_deno_available()


def _default_limits() -> dict:
    from elitea_sdk.runtime.tools.sandbox import _read_sandbox_limits_from_env  # pylint: disable=C0415,E0401
    return _read_sandbox_limits_from_env()


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
    import os  # pylint: disable=C0415
    from elitea_sdk.runtime.langchain.pyodide_sandbox import SyncPyodideSandbox  # pylint: disable=C0415,E0401

    sandbox_base = os.environ.get('SANDBOX_BASE')
    sandbox_tmp = os.path.join(sandbox_base, 'tmp') if sandbox_base else None
    # DENO_DIR if set, else this deployment's warm cache under SANDBOX_BASE, else the
    # SDK's own default. Whichever resolves is the dir the offline module cache lives in.
    deno_cache = os.environ.get('DENO_DIR')
    if not deno_cache and sandbox_base:
        deno_cache = os.path.join(sandbox_base, '.deno_dir')

    read_paths = [p for p in (sandbox_base, sandbox_tmp, deno_cache) if p]
    write_paths = [p for p in (sandbox_tmp, deno_cache) if p]

    return SyncPyodideSandbox(
        stateful=False,
        allow_env=['SANDBOX_BASE'],     # scoped: only the cache-locator var main.js reads
        allow_read=read_paths or False,
        allow_write=write_paths or False,  # scoped: only the Pyodide scratch/cache dirs
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
) -> dict:
    """Execute ``code`` (prelude + untrusted script) in the locked-down sandbox.

    Returns a plain dict ``{result, stdout, stderr, status, execution_time}`` — the shape
    pylon_main's ``map_execution_result`` consumes. ``status`` is one of ``'success'`` /
    ``'error'`` / ``'unavailable'``. This function never raises for an execution failure:
    a sandbox timeout/OOM/exception is reported as ``status='error'`` so the caller can turn
    it into a per-case error verdict and keep running sibling cases.

    ``sandbox_factory`` / ``deno_probe`` / ``limits`` are injectable for unit testing; the
    defaults pull the real SDK sandbox + env-derived limits (never "unlimited").
    """
    probe = deno_probe or _default_deno_probe
    if not probe():
        return {
            'result': None, 'stdout': None,
            'stderr': 'Deno/Pyodide sandbox runtime is not available.',
            'status': STATUS_UNAVAILABLE, 'execution_time': None,
        }

    resolved_limits = limits if limits is not None else _default_limits()
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
