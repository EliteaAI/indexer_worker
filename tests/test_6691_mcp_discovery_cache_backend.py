"""Pins the indexer_worker Redis backend for the SDK's remote-MCP discovery cache
(#6691) and the wiring that registers it before the first forked agent run.

Re-breaks if the backend stops namespacing keys, rebuilds/reuses the Redis client
on the wrong side of a fork, `build_redis_client` stops passing the pool timeouts
through, `configure_mcp_discovery_cache` stops registering with the SDK, or
`module.py` starts creating the agent TaskNode before the cache backend is
registered (forked children would then inherit no backend).
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _install_pylon_log_stub(monkeypatch):
    pylon = types.ModuleType("pylon")
    pylon_core = types.ModuleType("pylon.core")
    pylon_tools = types.ModuleType("pylon.core.tools")
    pylon_tools.log = types.SimpleNamespace(
        info=Mock(), warning=Mock(), error=Mock(), exception=Mock(), debug=Mock(),
    )
    monkeypatch.setitem(sys.modules, "pylon", pylon)
    monkeypatch.setitem(sys.modules, "pylon.core", pylon_core)
    monkeypatch.setitem(sys.modules, "pylon.core.tools", pylon_tools)
    return pylon_tools.log


def _load_cache_module(monkeypatch):
    _install_pylon_log_stub(monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "indexer_worker_mcp_discovery_cache_6691",
        PLUGIN_ROOT / "utils" / "mcp_discovery_cache.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache_module(monkeypatch):
    return _load_cache_module(monkeypatch)


class _FakeRedisClient:
    def __init__(self):
        self.store = {}
        self.calls = []

    def setex(self, key, ttl, value):
        self.calls.append(("setex", key, ttl, value))
        self.store[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        self.calls.append(("get", key))
        return self.store.get(key)

    def delete(self, key):
        self.calls.append(("delete", key))
        self.store.pop(key, None)


def test_backend_prefixes_keys_and_round_trips_bytes_to_str(cache_module):
    client = _FakeRedisClient()
    backend = cache_module.RedisMcpDiscoveryCacheBackend({}, client_factory=lambda cfg: client)

    backend.set("abc", "payload", 300)
    assert client.store == {"mcp:discovery:abc": b"payload"}
    assert client.calls[-1] == ("setex", "mcp:discovery:abc", 300, "payload")

    assert backend.get("abc") == "payload"
    assert isinstance(client.store["mcp:discovery:abc"], bytes)


def test_get_on_a_missing_key_returns_none(cache_module):
    client = _FakeRedisClient()
    backend = cache_module.RedisMcpDiscoveryCacheBackend({}, client_factory=lambda cfg: client)

    assert backend.get("missing") is None


def test_client_is_rebuilt_only_when_the_pid_changes(cache_module, monkeypatch):
    made = []

    def factory(_config):
        client = _FakeRedisClient()
        made.append(client)
        return client

    backend = cache_module.RedisMcpDiscoveryCacheBackend({}, client_factory=factory)

    monkeypatch.setattr(cache_module.os, "getpid", lambda: 111)
    backend.get("k1")
    backend.get("k1")
    assert len(made) == 1

    monkeypatch.setattr(cache_module.os, "getpid", lambda: 222)
    backend.get("k1")
    assert len(made) == 2

    monkeypatch.setattr(cache_module.os, "getpid", lambda: 222)
    backend.get("k1")
    assert len(made) == 2


def test_build_redis_client_passes_config_through_and_sets_pool_timeouts(cache_module, monkeypatch):
    redis_module = types.ModuleType("redis")
    connection_module = types.ModuleType("redis.connection")

    captured_pool_kwargs = {}

    class _FakePool:
        def __init__(self, **kwargs):
            captured_pool_kwargs.update(kwargs)

    class _FakeRedis:
        def __init__(self, connection_pool):
            self.connection_pool = connection_pool

    redis_module.Redis = _FakeRedis
    connection_module.BlockingConnectionPool = _FakePool
    redis_module.connection = connection_module
    monkeypatch.setitem(sys.modules, "redis", redis_module)
    monkeypatch.setitem(sys.modules, "redis.connection", connection_module)

    redis_config = {"host": "redis", "password": "s3cret", "connection_class": "SSLConnection"}
    client = cache_module.build_redis_client(redis_config)

    assert captured_pool_kwargs["host"] == "redis"
    assert captured_pool_kwargs["password"] == "s3cret"
    assert captured_pool_kwargs["connection_class"] == "SSLConnection"
    assert captured_pool_kwargs["max_connections"] == cache_module.MAX_CONNECTIONS
    assert captured_pool_kwargs["timeout"] == cache_module.SOCKET_TIMEOUT
    assert captured_pool_kwargs["socket_timeout"] == cache_module.SOCKET_TIMEOUT
    assert captured_pool_kwargs["socket_connect_timeout"] == cache_module.SOCKET_TIMEOUT
    assert isinstance(client, _FakeRedis)


def test_build_backend_from_event_node_requires_redis_config(cache_module):
    assert cache_module.build_backend_from_event_node(types.SimpleNamespace()) is None

    backend = cache_module.build_backend_from_event_node(types.SimpleNamespace(redis_config={"host": "redis"}))
    assert isinstance(backend, cache_module.RedisMcpDiscoveryCacheBackend)


def _stub_sdk_registry(monkeypatch):
    register_mock = Mock()
    sdk_pkg = types.ModuleType("elitea_sdk")
    sdk_runtime = types.ModuleType("elitea_sdk.runtime")
    sdk_utils = types.ModuleType("elitea_sdk.runtime.utils")
    sdk_cache = types.ModuleType("elitea_sdk.runtime.utils.mcp_discovery_cache")
    sdk_cache.register_discovery_cache_backend = register_mock
    monkeypatch.setitem(sys.modules, "elitea_sdk", sdk_pkg)
    monkeypatch.setitem(sys.modules, "elitea_sdk.runtime", sdk_runtime)
    monkeypatch.setitem(sys.modules, "elitea_sdk.runtime.utils", sdk_utils)
    monkeypatch.setitem(sys.modules, "elitea_sdk.runtime.utils.mcp_discovery_cache", sdk_cache)
    return register_mock


def test_configure_mcp_discovery_cache_registers_a_redis_backend_with_the_sdk(cache_module, monkeypatch):
    register_mock = _stub_sdk_registry(monkeypatch)

    backend = cache_module.configure_mcp_discovery_cache(types.SimpleNamespace(redis_config={"host": "redis"}))

    assert isinstance(backend, cache_module.RedisMcpDiscoveryCacheBackend)
    register_mock.assert_called_once_with(backend)


def test_configure_mcp_discovery_cache_registers_none_without_a_redis_event_node(cache_module, monkeypatch):
    register_mock = _stub_sdk_registry(monkeypatch)

    backend = cache_module.configure_mcp_discovery_cache(types.SimpleNamespace())

    assert backend is None
    register_mock.assert_called_once_with(None)


def _function_named(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _self_call_lineno(fn_node, method_name):
    for node in ast.walk(fn_node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method_name
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            return node.lineno
    return None


def _self_attribute_assign_lineno(fn_node, attr_name, call_attr):
    for node in ast.walk(fn_node):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and target.attr == attr_name
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == call_attr
        ):
            return node.lineno
    return None


def test_init_configures_the_mcp_cache_before_creating_the_agent_task_node():
    tree = ast.parse((PLUGIN_ROOT / "module.py").read_text())
    init_fn = _function_named(tree, "init")

    configure_lineno = _self_call_lineno(init_fn, "_configure_mcp_discovery_cache")
    task_node_lineno = _self_attribute_assign_lineno(init_fn, "agent_task_node", "TaskNode")

    assert configure_lineno is not None, "init() no longer calls self._configure_mcp_discovery_cache()"
    assert task_node_lineno is not None, "init() no longer assigns self.agent_task_node = arbiter.TaskNode(...)"
    assert configure_lineno < task_node_lineno, (
        "the cache backend must be registered before the agent TaskNode is created, "
        "so every forked agent run inherits it"
    )
