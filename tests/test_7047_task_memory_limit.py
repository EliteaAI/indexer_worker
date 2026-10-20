"""A forked agent/index task caps its own data memory so one runaway task cannot OOM the container (#7047).

Run from this directory:
    cd tests && python3 -m pytest test_7047_task_memory_limit.py
"""

import importlib.util
import multiprocessing
import pathlib
import resource
import sys

import pytest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "utils" / "task_memory_limit.py"


def _load():
    spec = importlib.util.spec_from_file_location("task_memory_limit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task_memory = _load()
MIB = task_memory.MIB
GIB = 1024 * MIB


class RecordedLimit:
    def __init__(self):
        self.calls = []

    def __call__(self, kind, limits):
        self.calls.append((kind, limits))


def apply(configured_mb, context="fork", container_limit=8 * GIB, data_bytes=2 * GIB):
    recorded = RecordedLimit()
    limit = task_memory.limit_task_memory(
        configured_mb, context,
        container_limit_reader=lambda: container_limit,
        data_bytes_reader=lambda: data_bytes,
        set_limit=recorded,
    )
    return limit, recorded.calls


def test_a_forked_task_gets_half_the_container_on_top_of_what_it_inherits():
    limit, calls = apply(None)

    assert limit == 2 * GIB + 4 * GIB
    assert calls == [(resource.RLIMIT_DATA, (limit, limit))]


def test_a_configured_budget_overrides_the_container_share():
    limit, calls = apply(1024)

    assert limit == 2 * GIB + 1 * GIB
    assert calls == [(resource.RLIMIT_DATA, (limit, limit))]


def test_zero_disables_the_limit():
    assert apply(0) == (None, [])


def test_no_container_limit_and_no_configuration_leaves_the_task_unlimited():
    assert apply(None, container_limit=None) == (None, [])


def test_a_task_running_inside_the_worker_process_is_never_limited():
    assert apply(1024, context="threading") == (None, [])


def test_unknown_process_data_size_leaves_the_task_unlimited():
    assert apply(1024, data_bytes=None) == (None, [])


@pytest.mark.parametrize("raw, expected", [
    ("max\n", None),
    ("4294967296\n", 4 * GIB),
    ("0\n", None),
])
def test_the_container_limit_is_read_from_the_unified_cgroup(tmp_path, raw, expected):
    (tmp_path / "proc_cgroup").write_text("0::/\n")
    (tmp_path / "memory.max").write_text(raw)

    assert task_memory.read_container_memory_limit(
        str(tmp_path / "proc_cgroup"), str(tmp_path)) == expected


def test_cgroup_v1_has_no_unified_limit(tmp_path):
    (tmp_path / "proc_cgroup").write_text("12:memory:/docker/abc\n")

    assert task_memory.read_container_memory_limit(str(tmp_path / "proc_cgroup"), str(tmp_path)) is None


def test_the_process_data_size_is_read_from_vmdata(tmp_path):
    (tmp_path / "status").write_text("Name:\tpython\nVmData:\t  2048 kB\nVmStk:\t 132 kB\n")

    assert task_memory.read_process_data_bytes(str(tmp_path / "status")) == 2048 * 1024


def _allocate_past_the_limit(results):
    task_memory.limit_task_memory(64, "fork")
    try:
        bytearray(256 * MIB)
        results.put("allocated")
    except MemoryError:
        results.put("MemoryError")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_DATA is enforced on Linux only")
def test_a_limited_task_gets_memory_error_while_its_parent_keeps_running():
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    child = context.Process(target=_allocate_past_the_limit, args=(results,))
    child.start()
    child.join(30)

    assert results.get(timeout=5) == "MemoryError"
    assert child.exitcode == 0
    assert bytearray(256 * MIB)
