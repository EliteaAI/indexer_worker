"""Retry-on-Timeout contract for #6246.

_fetch_pgvector_connstr_with_retry (methods/agent_common.py) wraps client.unsecret()
in a bare `except Exception` + exponential-backoff loop. Before #6246, unsecret()
could hang forever instead of raising, so this retry path was reachable only for
genuine exceptions (e.g. malformed vault response), never for a stalled connection.
Now that EliteAClient/SandboxClient time out and raise requests.exceptions.Timeout,
this loop must actually catch it, retry, and either recover or give up cleanly.

Loaded by source so the suite runs without the pylon runtime (see #6024 test for
the same pattern in this directory: pylon imports break collection from the plugin
root, so run from this directory with `python3 -m pytest test_6246_pgvector_timeout_retry.py`).
"""

import ast
import pathlib
import time
import typing
from unittest.mock import MagicMock

import pytest
import requests


class _StubLog:
    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


def _load_retry_fn():
    """Exec just _fetch_pgvector_connstr_with_retry and its module-level constants,
    avoiding the pylon-coupled imports at the top of agent_common.py."""
    source = (
        pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'agent_common.py'
    ).read_text()
    namespace = {
        'Optional': typing.Optional,
        'time': time,
        'log': _StubLog(),
    }
    for node in ast.parse(source).body:
        is_target_func = isinstance(node, ast.FunctionDef) and \
            node.name == '_fetch_pgvector_connstr_with_retry'
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            targets = []
        is_target_const = any(
            getattr(target, 'id', '') in ('PGVECTOR_PROJECT_CONNSTR_SECRET', '_pgvector_connstr_cache')
            for target in targets
        )
        if is_target_func or is_target_const:
            exec(compile(ast.Module([node], []), '<retry_fn>', 'exec'), namespace)  # pylint: disable=W0122

    return namespace['_fetch_pgvector_connstr_with_retry'], namespace['_pgvector_connstr_cache']


@pytest.fixture
def retry_fn():
    fn, cache = _load_retry_fn()
    cache.clear()
    return fn


def test_recovers_after_two_timeouts_then_succeeds(retry_fn, monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda *_: None)
    client = MagicMock()
    client.unsecret.side_effect = [
        requests.exceptions.ConnectTimeout("connect timed out"),
        requests.exceptions.ReadTimeout("read timed out"),
        "postgresql://conn",
    ]

    result = retry_fn(client, project_id=1, max_retries=3, base_delay=0.01)

    assert result == "postgresql://conn"
    assert client.unsecret.call_count == 3


def test_gives_up_after_max_retries_all_timing_out(retry_fn, monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda *_: None)
    client = MagicMock()
    client.unsecret.side_effect = requests.exceptions.ReadTimeout("read timed out")

    result = retry_fn(client, project_id=2, max_retries=3, base_delay=0.01)

    assert result is None
    assert client.unsecret.call_count == 3


def test_successful_result_is_cached_across_calls(retry_fn, monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda *_: None)
    client = MagicMock()
    client.unsecret.return_value = "postgresql://conn"

    first = retry_fn(client, project_id=3, max_retries=3, base_delay=0.01)
    second = retry_fn(client, project_id=3, max_retries=3, base_delay=0.01)

    assert first == "postgresql://conn"
    assert second == "postgresql://conn"
    client.unsecret.assert_called_once()  # second call hit the cache, no new HTTP call


def test_timeout_after_max_retries_does_not_poison_cache(retry_fn, monkeypatch):
    """A worker's cache must stay empty after retries exhaust, so a later,
    healthy call still gets a real vault round-trip instead of a stuck None."""
    monkeypatch.setattr(time, 'sleep', lambda *_: None)
    client = MagicMock()
    client.unsecret.side_effect = requests.exceptions.ConnectTimeout("connect timed out")

    result = retry_fn(client, project_id=4, max_retries=2, base_delay=0.01)
    _, cache = _load_retry_fn()

    assert result is None
    assert 4 not in cache
