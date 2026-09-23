"""The SDK side of the discovery-cache and Load Tools call contract (#6691).

`test_6691_sync_tools_configured_pat.py` and the backend tests pin what the indexer passes
across the SDK boundary; this file pins that the real SDK still accepts those calls,
returns what the indexer unpacks, and that the indexer's Redis backend satisfies the SDK's
backend protocol, so the two repos cannot drift apart with both suites green. It binds the
sibling checkout when one exists (evicting any other resident copy for the duration of this
module), falls back to an installed SDK, and skips when there is neither or when the
installed SDK predates the modules this contract binds.
"""

import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _sibling_sdk_root():
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "elitea-sdk"
        if (candidate / "elitea_sdk" / "__init__.py").is_file():
            return candidate
    return None


def _is_sdk_entry(name):
    return name == "elitea_sdk" or name.startswith("elitea_sdk.")


@pytest.fixture(scope="module")
def sdk():
    """Import the SDK to bind against, and put `sys.path`/`sys.modules` back afterwards."""
    saved_path = list(sys.path)
    saved_modules = {name: module for name, module in sys.modules.items() if _is_sdk_entry(name)}
    root = _sibling_sdk_root()
    for name in list(saved_modules):
        del sys.modules[name]
    if root:
        sys.path[:] = [entry for entry in sys.path if entry != str(root)]
        sys.path.insert(0, str(root))
    try:
        package = pytest.importorskip("elitea_sdk")
        if root:
            assert Path(package.__file__).resolve().is_relative_to(root.resolve()), (
                f"imported elitea_sdk from {package.__file__}, expected the checkout at {root}"
            )
        yield types.SimpleNamespace(
            oauth=pytest.importorskip("elitea_sdk.runtime.utils.mcp_oauth", reason="installed SDK predates #6691"),
            cache=pytest.importorskip("elitea_sdk.runtime.utils.mcp_discovery_cache", reason="installed SDK predates #6691"),
            discovery=pytest.importorskip("elitea_sdk.runtime.utils.mcp_tools_discovery"),
        )
    finally:
        sys.path[:] = saved_path
        for name in [name for name in sys.modules if _is_sdk_entry(name)]:
            del sys.modules[name]
        sys.modules.update(saved_modules)


@pytest.fixture(scope="module")
def indexer_backend():
    pylon_tools = types.ModuleType("pylon.core.tools")
    pylon_tools.log = types.SimpleNamespace(info=Mock(), warning=Mock(), error=Mock(), exception=Mock(), debug=Mock())
    saved = {name: sys.modules.get(name) for name in ("pylon", "pylon.core", "pylon.core.tools")}
    sys.modules["pylon"] = types.ModuleType("pylon")
    sys.modules["pylon.core"] = types.ModuleType("pylon.core")
    sys.modules["pylon.core.tools"] = pylon_tools
    try:
        spec = importlib.util.spec_from_file_location("indexer_cache_6691_contract", PLUGIN_ROOT / "utils" / "mcp_discovery_cache.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_has_configured_authorization_takes_the_headers_mapping(sdk):
    assert sdk.oauth.has_configured_authorization({"Authorization": "Bearer pat"}) is True


def test_merge_oauth_authorization_returns_headers_and_the_injected_flag(sdk):
    merged, injected = sdk.oauth.merge_oauth_authorization({"X-Trace": "1"}, "token")

    assert isinstance(merged, dict) and isinstance(injected, bool)


def test_has_authorization_on_the_wire_takes_headers_and_the_injected_flag(sdk):
    assert sdk.oauth.has_authorization_on_the_wire({"Authorization": "Bearer "}, False) is True
    assert sdk.oauth.has_authorization_on_the_wire({"Authorization": "Bearer t"}, True) is False


def test_drop_unusable_authorization_returns_a_headers_mapping(sdk):
    assert sdk.oauth.drop_unusable_authorization({"Authorization": "Bearer {x}", "X-A": "1"}) == {"X-A": "1"}
    assert sdk.oauth.drop_unusable_authorization({"Authorization": "Bearer ", "X-A": "1"}) == {"X-A": "1"}


def test_extract_user_friendly_mcp_error_takes_the_exception_and_headers(sdk):
    assert isinstance(sdk.oauth.extract_user_friendly_mcp_error(ValueError("x"), {"Authorization": "Bearer pat"}), str)


def test_invalidate_server_discovery_takes_only_the_url_and_reports_a_bool(sdk):
    inspect.signature(sdk.cache.invalidate_server_discovery).bind("https://mcp.example.test/mcp")
    assert sdk.cache.invalidate_server_discovery("https://mcp.example.test/mcp") is True


def test_discover_mcp_tools_accepts_the_load_tools_keywords(sdk):
    inspect.signature(sdk.discovery.discover_mcp_tools).bind(
        url="https://mcp.example.test/mcp",
        headers={},
        timeout=60,
        session_id=None,
        ssl_verify=True,
        configured_auth=True,
    )


def test_the_registry_accepts_the_indexer_backend_and_hands_it_back(sdk, indexer_backend):
    backend = indexer_backend.RedisMcpDiscoveryCacheBackend({"host": "redis"}, client_factory=lambda _config: Mock())
    sdk.cache.register_discovery_cache_backend(backend)
    try:
        assert sdk.cache.get_discovery_cache_backend() is backend
    finally:
        sdk.cache.register_discovery_cache_backend(None)


def test_the_indexer_backend_implements_every_method_of_the_sdk_backend_protocol(sdk, indexer_backend):
    protocol = sdk.cache.McpDiscoveryCacheBackend
    required = {name for name in vars(protocol) if not name.startswith("_") and callable(getattr(protocol, name))}
    backend = indexer_backend.RedisMcpDiscoveryCacheBackend({"host": "redis"}, client_factory=lambda _config: Mock())

    assert required and required <= {name for name in dir(backend) if callable(getattr(backend, name))}
    for name in required:
        protocol_parameters = list(inspect.signature(getattr(protocol, name)).parameters)[1:]
        inspect.signature(getattr(backend, name)).bind(*protocol_parameters)
