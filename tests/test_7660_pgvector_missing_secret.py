"""elitea_issues#7660: a project without ``pgvector_project_connstr`` aborted its agent tasks in the fork DNS probe.

Main resolves the connstr before dispatch and sends ``None`` when the project has none. The worker took that ``None``
for "not sent", so every task re-ran the parent-side warm-up, whose lookups were in flight when arbiter forked, and the
child inherited a held resolver lock. Main's answer, ``None`` included, must now end the lookup on both sides.
"""

import ast
import pathlib
import types
import typing

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
PREFETCHED = "postgresql://prefetched"
POSTGRES = {"agent_memory_config": {"type": "postgres"}}


def _resolver(vault_calls):
    source = (PLUGIN_ROOT / "methods" / "agent_common.py").read_text()
    namespace = {"Optional": typing.Optional, "log": types.SimpleNamespace(warning=lambda *a, **k: None)}
    wanted = {"PGVECTOR_CONNSTR_KWARG", "_pgvector_connstr_cache", "pgvector_connstr_needed", "resolve_pgvector_connstr"}
    for node in ast.parse(source).body:
        names = {getattr(node, "name", None)} | {getattr(t, "id", None) for t in getattr(node, "targets", [])} \
            | {getattr(getattr(node, "target", None), "id", None)}
        if names & wanted:
            exec(compile(ast.Module([node], []), "agent_common.py", "exec"), namespace)  # pylint: disable=W0122
    namespace["DEFAULT_MEMORY_CONFIG"] = {"type": "sqlite"}
    namespace["temp_elitea_client"] = lambda *a: _NullContext()
    namespace["_fetch_pgvector_connstr_with_retry"] = lambda client, project_id=None: vault_calls.append(project_id)
    return namespace["resolve_pgvector_connstr"]


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _approver(asked):
    source = (PLUGIN_ROOT / "utils" / "pgvector_warm.py").read_text()
    source = source.replace("from pylon.core.tools import log  # pylint: disable=E0611,E0401", "")
    source = source.replace(
        "from ..methods.agent_common import (  # pylint: disable=C0415\n"
        "        PGVECTOR_CONNSTR_KWARG, _pgvector_connstr_cache, pgvector_connstr_needed,\n"
        "    )",
        "PGVECTOR_CONNSTR_KWARG = 'pgvector_connstr'; _pgvector_connstr_cache = PARENT_CACHE; "
        "pgvector_connstr_needed = lambda config: True")
    namespace = {"log": types.SimpleNamespace(warning=lambda *a, **k: None), "PARENT_CACHE": {}}
    exec(compile(source, "pgvector_warm.py", "exec"), namespace)  # pylint: disable=W0122
    namespace["request_warm"] = lambda *args: asked.append(args)
    return namespace["agent_task_approver"](POSTGRES), namespace["PARENT_CACHE"]


def _start_event(**kwargs):
    return {"kwargs": {"llm": {"kwargs": {"project_id": 64, "api_key": "pat"}}, **kwargs}}


def test_mains_none_ends_the_lookup_in_the_child():
    vault_calls = []
    resolve = _resolver(vault_calls)
    assert resolve(POSTGRES, {"project_id": 64}, "pat", {}, task_kwargs={"pgvector_connstr": None}) is None
    assert vault_calls == []


def test_mains_value_is_used_as_is():
    vault_calls = []
    resolve = _resolver(vault_calls)
    assert resolve(POSTGRES, {"project_id": 64}, "pat", {}, task_kwargs={"pgvector_connstr": PREFETCHED}) == PREFETCHED
    assert vault_calls == []


def test_a_task_main_did_not_prefetch_still_falls_back_to_the_vault():
    vault_calls = []
    _resolver(vault_calls)(POSTGRES, {"project_id": 64}, "pat", {}, task_kwargs={})
    assert vault_calls == [64]


def test_mains_none_is_kept_for_later_forks_without_a_warm_up():
    asked = []
    approver, cache = _approver(asked)
    assert approver("start", _start_event(pgvector_connstr=None)) is True
    assert asked == [] and cache == {64: None}


def test_mains_value_is_kept_for_later_forks_without_a_warm_up():
    asked = []
    approver, cache = _approver(asked)
    approver("start", _start_event(pgvector_connstr=PREFETCHED))
    assert asked == [] and cache == {64: PREFETCHED}


def test_a_newer_answer_from_main_replaces_the_kept_one():
    approver, cache = _approver([])
    approver("start", _start_event(pgvector_connstr=None))
    approver("start", _start_event(pgvector_connstr=PREFETCHED))
    assert cache == {64: PREFETCHED}


def test_warm_up_still_runs_for_a_task_main_did_not_prefetch():
    asked = []
    approver, cache = _approver(asked)
    approver("start", _start_event())
    assert [args[0] for args in asked] == [64] and cache == {}
