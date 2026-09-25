"""Load Tools wiring for #6691 (`methods/indexer_mcp_sync_tools.py`), with the SDK boundary
stubbed: the method must decide precedence through the SDK helpers, skip the OAuth token
lookup once a credential is configured, hand the merged headers and the configured-auth
flag to `discover_mcp_tools`, leave the caller's headers untouched, and retire the
server's cached tool lists after a successful discovery. The helpers' own behaviour is
pinned in the SDK suite. Re-breaks if the method assigns the Authorization header itself,
looks tokens up regardless of a configured credential, sends an unresolved template when no
token exists, aliases the caller's dict, stops invalidating the discovery cache, reports a
refresh whose cache retirement failed as a plain success, or lets a malformed payload or an
SDK import failure escape the error handler.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock, call

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PKG_NAME = "indexer_worker_pkg_6691"
URL = "https://mcp.example.test/mcp"
MERGED_HEADERS = {"Authorization": "Bearer merged-by-sdk"}


def _install_pylon_stubs(monkeypatch):
    monkeypatch.delenv("ELITEA_SDK_DEV_RELOAD", raising=False)
    pylon_tools = types.ModuleType("pylon.core.tools")
    pylon_tools.log = types.SimpleNamespace(
        info=Mock(), warning=Mock(), error=Mock(), exception=Mock(), debug=Mock(),
    )

    def _method(_name):
        def _decorator(func):
            return func
        return _decorator

    pylon_tools.web = types.SimpleNamespace(method=_method)
    monkeypatch.setitem(sys.modules, "pylon", types.ModuleType("pylon"))
    monkeypatch.setitem(sys.modules, "pylon.core", types.ModuleType("pylon.core"))
    monkeypatch.setitem(sys.modules, "pylon.core.tools", pylon_tools)

    tools_module = types.ModuleType("tools")
    tools_module.worker_core = types.SimpleNamespace(event_node=Mock())
    monkeypatch.setitem(sys.modules, "tools", tools_module)

    tasknode_task = types.ModuleType("tasknode_task")
    tasknode_task.multiprocessing_context = "thread"
    tasknode_task.meta = {}
    monkeypatch.setitem(sys.modules, "tasknode_task", tasknode_task)


class McpAuthorizationRequired(Exception):
    """Stands in for the SDK class; the method recognises it by name, as dev reload re-creates the class."""

    def to_dict(self):
        return {"server_url": URL, "resource_metadata": {}}


class _AnyNameModule(types.ModuleType):
    """A stub module whose unknown names resolve to Mocks, for imports the tests never exercise."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        mock = Mock(name=f"{self.__name__}.{name}")
        setattr(self, name, mock)
        return mock


class SdkBoundary:
    """The SDK functions the Load Tools method calls, as Mocks with sentinel results."""

    def __init__(self):
        self.has_configured_authorization = Mock(return_value=False)
        self.has_authorization_on_the_wire = Mock(side_effect=lambda headers, injected: "Authorization" in headers and not injected)
        self.drop_unusable_authorization = Mock(side_effect=dict)
        self.merge_oauth_authorization = Mock(return_value=(dict(MERGED_HEADERS), True))
        self.invalidate_server_discovery = Mock(return_value=True)
        self.discover_mcp_tools = Mock(return_value=[{"name": "echo"}])
        self.canonical_resource = lambda url: url.rstrip("/").lower()


def _install_sdk_stubs(monkeypatch, boundary):
    for name in ("elitea_sdk", "elitea_sdk.runtime", "elitea_sdk.runtime.utils"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    oauth = _AnyNameModule("elitea_sdk.runtime.utils.mcp_oauth")
    oauth.canonical_resource = boundary.canonical_resource
    oauth.has_configured_authorization = boundary.has_configured_authorization
    oauth.has_authorization_on_the_wire = boundary.has_authorization_on_the_wire
    oauth.drop_unusable_authorization = boundary.drop_unusable_authorization
    oauth.merge_oauth_authorization = boundary.merge_oauth_authorization
    oauth.McpAuthorizationRequired = McpAuthorizationRequired
    oauth.extract_user_friendly_mcp_error = lambda exc, headers=None: str(exc)
    monkeypatch.setitem(sys.modules, oauth.__name__, oauth)

    cache = types.ModuleType("elitea_sdk.runtime.utils.mcp_discovery_cache")
    cache.invalidate_server_discovery = boundary.invalidate_server_discovery
    monkeypatch.setitem(sys.modules, cache.__name__, cache)

    discovery = types.ModuleType("elitea_sdk.runtime.utils.mcp_tools_discovery")
    discovery.discover_mcp_tools = boundary.discover_mcp_tools
    monkeypatch.setitem(sys.modules, discovery.__name__, discovery)


def _load_real_submodule(monkeypatch, modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, modname, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def boundary(monkeypatch):
    sdk = SdkBoundary()
    _install_pylon_stubs(monkeypatch)
    _install_sdk_stubs(monkeypatch, sdk)
    return sdk


@pytest.fixture
def sync_tools(monkeypatch, boundary):  # pylint: disable=unused-argument
    """Load the real `indexer_mcp_sync_tools.py` (plus its real `utils` siblings) under a
    synthetic package so its relative imports resolve, with the pylon runtime and the SDK
    boundary stubbed."""
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PLUGIN_ROOT)]
    monkeypatch.setitem(sys.modules, PKG_NAME, pkg)

    utils_pkg = types.ModuleType(f"{PKG_NAME}.utils")
    utils_pkg.__path__ = [str(PLUGIN_ROOT / "utils")]
    monkeypatch.setitem(sys.modules, f"{PKG_NAME}.utils", utils_pkg)

    methods_pkg = types.ModuleType(f"{PKG_NAME}.methods")
    methods_pkg.__path__ = [str(PLUGIN_ROOT / "methods")]
    monkeypatch.setitem(sys.modules, f"{PKG_NAME}.methods", methods_pkg)

    _load_real_submodule(monkeypatch, f"{PKG_NAME}.utils.constants", PLUGIN_ROOT / "utils" / "constants.py")
    _load_real_submodule(
        monkeypatch, f"{PKG_NAME}.utils.mcp_discovery_cache", PLUGIN_ROOT / "utils" / "mcp_discovery_cache.py"
    )
    _load_real_submodule(monkeypatch, f"{PKG_NAME}.utils.funcs", PLUGIN_ROOT / "utils" / "funcs.py")
    _load_real_submodule(
        monkeypatch, f"{PKG_NAME}.utils.node_interface", PLUGIN_ROOT / "utils" / "node_interface.py"
    )

    # agent_common.py itself cascades into several more relative-import modules
    # unrelated to this method; the method only needs these two constants from it.
    agent_common_stub = types.ModuleType(f"{PKG_NAME}.methods.agent_common")
    agent_common_stub.EVENTNODE_EVENT_NAME = "application_stream_response"
    agent_common_stub.EVENTNODE_FULL_RESPONSE_NAME = "application_full_response"
    monkeypatch.setitem(sys.modules, f"{PKG_NAME}.methods.agent_common", agent_common_stub)

    return _load_real_submodule(
        monkeypatch,
        f"{PKG_NAME}.methods.indexer_mcp_sync_tools",
        PLUGIN_ROOT / "methods" / "indexer_mcp_sync_tools.py",
    )


class PylonModule:
    pass


def _call(sync_tools_module, **overrides):
    kwargs = {
        "stream_id": "stream-1",
        "message_id": "msg-1",
        "url": URL,
        "project_id": 1,
        "headers": {},
        "timeout": 60,
        "mcp_tokens": None,
        "ssl_verify": True,
    }
    kwargs.update(overrides)
    return sync_tools_module.Method.indexer_mcp_sync_tools(PylonModule(), **kwargs)


def test_a_configured_credential_skips_the_token_lookup_and_is_sent_as_configured(sync_tools, boundary):
    boundary.has_configured_authorization.return_value = True
    boundary.merge_oauth_authorization.return_value = ({"Authorization": "Bearer configured-pat"}, False)

    result = _call(
        sync_tools,
        headers={"Authorization": "Bearer configured-pat"},
        mcp_tokens={URL: {"access_token": "oauth-token", "session_id": "oauth-session"}},
    )

    assert result["success"] is True
    boundary.merge_oauth_authorization.assert_called_once_with({"Authorization": "Bearer configured-pat"}, None)
    kwargs = boundary.discover_mcp_tools.call_args.kwargs
    assert kwargs["headers"] == {"Authorization": "Bearer configured-pat"}
    assert kwargs["session_id"] is None
    assert kwargs["configured_auth"] is True


def test_without_a_configured_credential_the_oauth_token_and_its_session_are_used(sync_tools, boundary):
    _call(
        sync_tools,
        headers={"X-Trace": "1"},
        mcp_tokens={URL: {"access_token": "oauth-token", "session_id": "oauth-session"}},
    )

    boundary.merge_oauth_authorization.assert_called_once_with({"X-Trace": "1"}, "oauth-token")
    kwargs = boundary.discover_mcp_tools.call_args.kwargs
    assert kwargs["headers"] == MERGED_HEADERS
    assert kwargs["session_id"] == "oauth-session"
    assert kwargs["configured_auth"] is False


def test_the_merged_headers_are_what_reaches_discovery_not_a_local_assignment(sync_tools, boundary):
    boundary.merge_oauth_authorization.return_value = ({"authorization": "Bearer configured-pat"}, False)
    boundary.has_configured_authorization.return_value = True

    _call(sync_tools, headers={"authorization": "Bearer configured-pat"}, mcp_tokens={URL: {"access_token": "t"}})

    assert boundary.discover_mcp_tools.call_args.kwargs["headers"] == {"authorization": "Bearer configured-pat"}


def test_without_a_token_an_unresolved_template_is_dropped_before_discovery(sync_tools, boundary):
    boundary.merge_oauth_authorization.return_value = ({"Authorization": "Bearer {github_token}"}, False)
    boundary.drop_unusable_authorization.side_effect = None
    boundary.drop_unusable_authorization.return_value = {}

    _call(sync_tools, headers={"Authorization": "Bearer {github_token}"}, mcp_tokens=None)

    boundary.drop_unusable_authorization.assert_called_once_with({"Authorization": "Bearer {github_token}"})
    kwargs = boundary.discover_mcp_tools.call_args.kwargs
    assert kwargs["headers"] == {}
    assert kwargs["configured_auth"] is False


def test_with_a_token_the_merged_headers_are_sent_untouched(sync_tools, boundary):
    _call(sync_tools, headers={}, mcp_tokens={URL: {"access_token": "oauth-token"}})

    boundary.drop_unusable_authorization.assert_not_called()
    assert boundary.discover_mcp_tools.call_args.kwargs["headers"] == MERGED_HEADERS


def test_a_blank_header_is_handed_to_the_drop_and_whatever_survives_decides_the_flag(sync_tools, boundary):
    boundary.merge_oauth_authorization.return_value = ({"Authorization": "Bearer "}, False)
    boundary.drop_unusable_authorization.side_effect = None
    boundary.drop_unusable_authorization.return_value = {}

    _call(sync_tools, headers={"Authorization": "Bearer "}, mcp_tokens=None)

    boundary.drop_unusable_authorization.assert_called_once_with({"Authorization": "Bearer "})
    kwargs = boundary.discover_mcp_tools.call_args.kwargs
    assert kwargs["headers"] == {}
    assert kwargs["configured_auth"] is False


def test_a_real_uninjected_credential_is_reported_as_configured(sync_tools, boundary):
    boundary.merge_oauth_authorization.return_value = ({"Authorization": "Bearer pat"}, False)

    _call(sync_tools, headers={"Authorization": "Bearer pat"}, mcp_tokens=None)

    kwargs = boundary.discover_mcp_tools.call_args.kwargs
    assert kwargs["headers"] == {"Authorization": "Bearer pat"}
    assert kwargs["configured_auth"] is True


def test_prebuilt_toolkits_look_the_token_up_by_type(sync_tools, boundary):
    _call(
        sync_tools,
        headers={"Authorization": "Bearer {github_token}"},
        mcp_tokens={"mcp_github": {"access_token": "oauth-token"}},
        toolkit_type="mcp_github",
    )

    boundary.merge_oauth_authorization.assert_called_once_with({"Authorization": "Bearer {github_token}"}, "oauth-token")


def test_callers_headers_dict_is_neither_aliased_nor_mutated(sync_tools, boundary):
    boundary.has_configured_authorization.return_value = True
    headers = {"Authorization": "Bearer configured-pat"}

    _call(sync_tools, headers=headers, mcp_tokens={URL: {"access_token": "oauth-token"}})

    assert headers == {"Authorization": "Bearer configured-pat"}
    assert boundary.has_configured_authorization.call_args.args[0] is not headers
    assert boundary.merge_oauth_authorization.call_args.args[0] is not headers


def test_successful_discovery_invalidates_the_servers_cached_lists_afterwards(sync_tools, boundary):
    order = Mock()
    order.attach_mock(boundary.discover_mcp_tools, "discover")
    order.attach_mock(boundary.invalidate_server_discovery, "invalidate")

    _call(sync_tools, url=URL + "/", ssl_verify=False)

    assert [c[0] for c in order.mock_calls] == ["discover", "invalidate"]
    boundary.invalidate_server_discovery.assert_called_once_with(URL + "/")


def test_a_failed_discovery_leaves_the_cached_lists_alone(sync_tools, boundary):
    boundary.discover_mcp_tools.side_effect = ConnectionError("server down")

    result = _call(sync_tools)

    assert result["success"] is False
    boundary.invalidate_server_discovery.assert_not_called()


def test_a_failed_discovery_returns_the_servers_message_as_a_sync_error(sync_tools, boundary):
    boundary.discover_mcp_tools.side_effect = ValueError("The MCP endpoint https://x/sse has been retired")

    result = _call(sync_tools)

    assert result == {
        "success": False,
        "error": "Failed to sync MCP tools: The MCP endpoint https://x/sse has been retired",
        "server_url": URL,
    }


def test_headers_that_are_not_a_mapping_produce_an_error_result_not_a_crash(sync_tools, boundary):
    result = _call(sync_tools, headers="Authorization: Bearer pasted-as-text")

    assert result["success"] is False
    assert result["error"].startswith("Failed to sync MCP tools:")
    boundary.discover_mcp_tools.assert_not_called()


def test_a_failed_cache_retirement_is_reported_as_a_warning_on_the_success_response(sync_tools, boundary):
    boundary.invalidate_server_discovery.return_value = False

    result = _call(sync_tools)

    assert result["success"] is True
    assert result["warning"] == sync_tools.CACHE_NOT_RETIRED_WARNING


def test_a_retired_cache_adds_no_warning(sync_tools, boundary):
    assert "warning" not in _call(sync_tools)


def test_an_authorization_challenge_is_recognised_by_class_name(sync_tools, boundary):
    boundary.discover_mcp_tools.side_effect = McpAuthorizationRequired("login first")

    result = _call(sync_tools)

    assert result["success"] is False
    assert result["requires_authorization"] is True
    boundary.invalidate_server_discovery.assert_not_called()


def test_an_sdk_import_failure_becomes_a_sync_error_not_a_crash(sync_tools, boundary, monkeypatch):
    monkeypatch.delitem(sys.modules, "elitea_sdk.runtime.utils.mcp_tools_discovery")

    result = _call(sync_tools)

    assert result["success"] is False
    assert result["error"].startswith("Failed to sync MCP tools: ")
    assert "mcp_tools_discovery" in result["error"]
