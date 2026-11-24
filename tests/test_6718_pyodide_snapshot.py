"""#6718: the worker-side snapshot lifecycle behind the `sandbox_snapshot` flag.

Loaded by source with a stub pylon logger and a fake `deno` (a small shell script), so it
needs neither pylon nor a real Deno. Pins:
  * flag off -> SANDBOX_SNAPSHOT_PATH is removed and this worker's file deleted, nothing is built;
  * flag on  -> atomic file with owner-only mode and a checksummed name, recorded in the owner marker
    (`owner-default.txt`, or SANDBOX_SNAPSHOT_OWNER per worker on a shared volume);
  * a file some marker names is never removed (the review case: an old file still in use survives);
  * a failed or late build leaves the env unset and no partial file, so runs keep the cold load;
  * a restart replaces this host's own previous file; unreferenced files go after a 1h grace period.

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
    monkeypatch.delenv(snap.OWNER_ENV, raising=False)
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
    assert not pathlib.Path(snap._marker(str(base_dir))).exists()


def _good_deno(base_dir):
    # The fake writes whatever path follows --build-snapshot, like main.js does.
    _fake_deno(base_dir, 'for a; do [ "$prev" = --build-snapshot ] && echo snap > "$a"; prev="$a"; done\n')


def _bins(base_dir):
    return sorted(p.name for p in (base_dir / 'snapshots').glob('*.bin'))


def _age(path, days):
    then = time.time() - days * 24 * 3600
    os.utime(path, (then, then))


def test_flag_on_builds_once_with_owner_only_mode_and_a_checksummed_name(base, monkeypatch):
    base_dir, entry = base
    _run_sync(monkeypatch)
    _good_deno(base_dir)

    snap.apply_pyodide_snapshot(True, str(base_dir), entry)

    path = pathlib.Path(os.environ[snap.SNAPSHOT_ENV])
    assert path.read_text() == 'snap\n'
    assert path.name.endswith('-' + hashlib.sha256(b'snap\n').hexdigest() + '.bin')
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _bins(base_dir) == [path.name]
    assert pathlib.Path(snap._marker(str(base_dir))).read_text() == path.name


def test_an_old_snapshot_still_used_by_another_worker_survives(base, monkeypatch):
    # The review case: worker A built days ago and is still running; worker B starts now.
    base_dir, entry = base
    _run_sync(monkeypatch)
    _good_deno(base_dir)
    (base_dir / 'snapshots').mkdir()
    live = base_dir / 'snapshots' / 'worker-a.bin'
    live.write_bytes(b'a')
    _age(live, 5)
    (base_dir / 'snapshots' / 'owner-worker-a.txt').write_text('worker-a.bin')
    monkeypatch.setenv(snap.OWNER_ENV, 'worker-b')

    snap.apply_pyodide_snapshot(True, str(base_dir), entry)

    assert 'worker-a.bin' in _bins(base_dir)


def test_a_restart_replaces_this_workers_own_previous_snapshot(base, monkeypatch):
    base_dir, entry = base
    _run_sync(monkeypatch)
    _good_deno(base_dir)
    snap.apply_pyodide_snapshot(True, str(base_dir), entry)
    first = pathlib.Path(os.environ.pop(snap.SNAPSHOT_ENV)).name

    snap.apply_pyodide_snapshot(True, str(base_dir), entry)

    second = pathlib.Path(os.environ[snap.SNAPSHOT_ENV]).name
    assert second != first
    assert _bins(base_dir) == [second]


def test_only_unreferenced_files_past_the_grace_period_are_removed(base, monkeypatch):
    base_dir, entry = base
    _run_sync(monkeypatch)
    _good_deno(base_dir)
    (base_dir / 'snapshots').mkdir()
    orphan = base_dir / 'snapshots' / 'crashed-build.bin'
    orphan.write_bytes(b'o')
    _age(orphan, 1)
    just_published = base_dir / 'snapshots' / 'other-worker-before-its-marker.bin'
    just_published.write_bytes(b'p')

    snap.apply_pyodide_snapshot(True, str(base_dir), entry)

    names = _bins(base_dir)
    assert 'crashed-build.bin' not in names
    assert 'other-worker-before-its-marker.bin' in names


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


def test_a_recreated_container_replaces_the_previous_snapshot_instead_of_leaking_it(base, monkeypatch):
    # The hostname changes when a container is recreated; the default owner does not, so the old file goes.
    base_dir, entry = base
    _run_sync(monkeypatch)
    _good_deno(base_dir)
    snap.apply_pyodide_snapshot(True, str(base_dir), entry)
    first = pathlib.Path(os.environ.pop(snap.SNAPSHOT_ENV)).name
    monkeypatch.setattr(snap.os, 'getpid', lambda: 99999)

    snap.apply_pyodide_snapshot(True, str(base_dir), entry)

    assert _bins(base_dir) == [pathlib.Path(os.environ[snap.SNAPSHOT_ENV]).name]
    assert first not in _bins(base_dir)
    assert [p.name for p in (base_dir / 'snapshots').glob('owner-*.txt')] == ['owner-default.txt']
