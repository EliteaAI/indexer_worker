"""Pyodide memory snapshot for faster sandbox start (#6718), built once per worker start."""
import glob
import hashlib
import os
import subprocess
import threading
import time
import uuid

from pylon.core.tools import log  # pylint: disable=E0611,E0401

SNAPSHOT_ENV = "SANDBOX_SNAPSHOT_PATH"
_BUILD_TIMEOUT_SECONDS = 120
_STALE_AFTER_SECONDS = 24 * 3600
_lock = threading.Lock()
_state = {"enabled": False}


def _snapshot_dir(sandbox_base):
    return os.path.join(sandbox_base, "snapshots")


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _remove_stale(sandbox_base, keep):
    # The directory may be shared by other workers, so only long-unused files are removed.
    cutoff = time.time() - _STALE_AFTER_SECONDS
    for path in glob.glob(os.path.join(_snapshot_dir(sandbox_base), "*")):
        try:
            if path != keep and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def _build(sandbox_base, entrypoint):
    # A random name per start: a fresh hash seed, and no clash with another worker sharing the directory.
    target = os.path.join(_snapshot_dir(sandbox_base), f"{uuid.uuid4().hex}.bin")
    os.makedirs(_snapshot_dir(sandbox_base), mode=0o700, exist_ok=True)
    pyodide_dir = os.path.join(sandbox_base, "pyodide")
    cmd = [
        os.path.join(sandbox_base, "bin", "deno"), "run", "--node-modules-dir=auto", "--allow-sys=cpus",
        "--allow-env=SANDBOX_BASE", f"--allow-read={pyodide_dir},{_snapshot_dir(sandbox_base)}",
        f"--allow-write={_snapshot_dir(sandbox_base)}", "--no-remote", "--no-npm",
        entrypoint, "--build-snapshot", target,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=_BUILD_TIMEOUT_SECONDS, check=False)
        if result.returncode != 0 or not os.path.isfile(target):
            log.warning("Pyodide snapshot build failed (rc=%s): %s", result.returncode, result.stderr[-500:])
            return
        with open(target, "rb") as handle:
            checksum = hashlib.sha256(handle.read()).hexdigest()
        # main.js reads the expected checksum from the name and cold-loads on a mismatch.
        final = target[:-len(".bin")] + f"-{checksum}.bin"
        os.replace(target, final)
        os.chmod(final, 0o600)
        if not _state["enabled"]:
            _remove(final)
            return
        _remove_stale(sandbox_base, final)
        os.environ[SNAPSHOT_ENV] = final
        log.info("Pyodide snapshot ready: %s (%d bytes)", final, os.path.getsize(final))
    finally:
        _remove(target)
        _remove(target + ".tmp")


def _build_guarded(sandbox_base, entrypoint):
    if not _lock.acquire(blocking=False):
        return
    try:
        _build(sandbox_base, entrypoint)
    except Exception:  # pylint: disable=W0718
        log.exception("Pyodide snapshot build failed - sandboxes keep the cold load")
    finally:
        _lock.release()


def apply_pyodide_snapshot(enabled, sandbox_base, entrypoint):
    """Off: sandboxes cold-load and this worker's snapshot is removed. On: build in the background."""
    _state["enabled"] = bool(enabled)
    current = os.environ.get(SNAPSHOT_ENV, "")
    if not enabled:
        os.environ.pop(SNAPSHOT_ENV, None)
        if current:
            _remove(current)
        return
    if os.path.isfile(current):
        return
    threading.Thread(target=_build_guarded, args=(sandbox_base, entrypoint),
                     name="pyodide-snapshot", daemon=True).start()
