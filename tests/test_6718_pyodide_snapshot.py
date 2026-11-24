"""#6718: the worker-side snapshot lifecycle behind the `sandbox_snapshot` flag.

Loaded by source with a stub pylon logger and a fake `deno` (a small shell script), so it
needs neither pylon nor a real Deno. Pins:
  * flag off -> SANDBOX_SNAPSHOT_PATH is removed and this worker's file deleted, nothing is built;
  * flag on  -> atomic file with owner-only mode and a checksummed name; only long-unused files removed;
  * a failed or late build leaves the env unset and no partial file, so runs keep the cold load;
  * every worker start gets its own random file name (fresh hash seed, no clash with other workers).

Run: cd tests && python3 -m pytest test_6718_pyodide_snapshot.py
"""
import hashlib
import importlib.util
import os
import pathlib
import stat
import sys
import time
import types

import pytest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'pyodide_snapshot.py'


def _load():
    pylon = types.ModuleType('pylon')
    core = types.ModuleType('pylon.core')
    tools = types.ModuleType('pylon.core.tools')
    tools.log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                                      exception=lambda *a, **k: None)
    sys.modules.update({'pylon': pylon, 'pylon.core': core, 'pylon.core.tools': tools})
    spec = importlib.util.spec_from_file_location('pyodide_snapshot', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


snap = _load()


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.delenv(snap.SNAPSHOT_ENV, raising=False)
    (tmp_path / 'pyodide').mkdir()
    (tmp_path / 'pyodide' / 'pyodide.asm.wasm').write_bytes(b'wasm')
    (tmp_path / 'bin').mkdir()
    entry = tmp_path / 'main.js'
    entry.write_text('// main')
    return tmp_path, str(entry)


def _fake_deno(base_dir, body):
    deno = base_dir / 'bin' / 'deno'
    deno.write_text('#!/bin/sh\n' + body)
    deno.chmod(0o755)


def _run_sync(monkeypatch):
    # Run the background build inline so the test can assert on its result.
    monkeypatch.setattr(snap.threading, 'Thread',
                        lambda target, args, **_: types.SimpleNamespace(start=lambda: target(*args)))


def test_flag_off_clears_the_path_and_removes_this_workers_snapshot_only(base, monkeypatch):
    base_dir, entry = base
    _fake_deno(base_dir, 'exit 99\n')
    (base_dir / 'snapshots').mkdir()
    mine = base_dir / 'snapshots' / 'mine.bin'
    other = base_dir / 'snapshots' / 'other-worker.bin'
    mine.write_bytes(b'x'), other.write_bytes(b'y')
    monkeypatch.setenv(snap.SNAPSHOT_ENV, str(mine))
    snap.apply_pyodide_snapshot(False, str(base_dir), entry)
    assert snap.SNAPSHOT_ENV not in os.environ
    assert [p.name for p in (base_dir / 'snapshots').iterdir()] == ['other-worker.bin']


def _good_deno(base_dir):
    # The fake writes whatever path follows --build-snapshot, like main.js does.
    _fake_deno(base_dir, 'for a; do [ "$prev" = --build-snapshot ] && echo snap > "$a"; prev="$a"; done\n')


def test_flag_on_builds_once_with_owner_only_mode_and_a_checksummed_name(base, monkeypatch):
    base_dir, entry = base
    _run_sync(monkeypatch)
    _good_deno(base_dir)

    snap.apply_pyodide_snapshot(True, str(base_dir), entry)

    path = pathlib.Path(os.environ[snap.SNAPSHOT_ENV])
    assert path.read_text() == 'snap\n'
    assert path.name.endswith('-' + hashlib.sha256(b'snap\n').hexdigest() + '.bin')
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in (base_dir / 'snapshots').iterdir()] == [path.name]


def test_only_long_unused_files_of_other_workers_are_removed(base, monkeypatch):
    base_dir, entry = base
    _run_sync(monkeypatch)
    _good_deno(base_dir)
    (base_dir / 'snapshots').mkdir()
    fresh = base_dir / 'snapshots' / 'other-fresh.bin'
    old = base_dir / 'snapshots' / 'other-old.bin'
    fresh.write_bytes(b'a'), old.write_bytes(b'b')
    long_ago = time.time() - 3 * 24 * 3600
    os.utime(old, (long_ago, long_ago))

    snap.apply_pyodide_snapshot(True, str(base_dir), entry)

    names = {p.name for p in (base_dir / 'snapshots').iterdir()}
    assert 'other-fresh.bin' in names and 'other-old.bin' not in names


def test_a_failed_build_keeps_the_cold_load_and_leaves_no_partial_file(base, monkeypatch):
    base_dir, entry = base
    _run_sync(monkeypatch)
    _fake_deno(base_dir, 'for a; do [ "$prev" = --build-snapshot ] && echo part > "$a.tmp"; prev="$a"; done; exit 1\n')
    snap.apply_pyodide_snapshot(True, str(base_dir), entry)
    assert snap.SNAPSHOT_ENV not in os.environ
    assert list((base_dir / 'snapshots').iterdir()) == []


def test_a_build_finishing_after_the_flag_was_turned_off_is_discarded(base, monkeypatch):
    base_dir, entry = base
    _fake_deno(base_dir, 'for a; do [ "$prev" = --build-snapshot ] && echo snap > "$a"; prev="$a"; done\n')
    snap._state['enabled'] = False
    snap._build(str(base_dir), entry)
    assert snap.SNAPSHOT_ENV not in os.environ
    assert list((base_dir / 'snapshots').iterdir()) == []


def test_each_worker_start_builds_under_its_own_name(base, monkeypatch):
    base_dir, entry = base
    _run_sync(monkeypatch)
    _good_deno(base_dir)
    snap.apply_pyodide_snapshot(True, str(base_dir), entry)
    first = os.environ.pop(snap.SNAPSHOT_ENV)
    snap.apply_pyodide_snapshot(True, str(base_dir), entry)
    assert os.environ[snap.SNAPSHOT_ENV] != first
