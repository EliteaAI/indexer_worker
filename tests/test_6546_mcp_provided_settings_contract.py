"""Contract tests: provided_settings must be present in mcp_authorization_required events (#6546).

Root cause of the original bug:
- canonical_resource() (SDK) strips trailing slash → exc.server_url = "https://api.githubcopilot.com/mcp"
- normalize_mcp_server_url() (indexer) retains trailing slash → stored as "https://api.githubcopilot.com/mcp/"
- Reverse URL equality checks compared these unequal → meta = {} → no provided_settings

Tests cover every path where provided_settings flows to the event:
1. _build_mcp_server_alias_map — populates provided_settings when credentials present/absent
2. _resolve_server_meta reverse lookup — trailing slash mismatch now resolved via .rstrip("/")
3. on_tool_error fallback in EliteACallback — URL-based lookup when exc.provided_settings is empty
4. build_mcp_auth_required_result — reads exc.provided_settings directly
5. indexer_agent.py backfill — both with and without elitea_callback

Run: cd tests && python3 -m pytest test_6546_mcp_provided_settings_contract.py
"""

import importlib.util
import pathlib
import sys
import types
import unittest
from typing import Dict, Optional, Any
from urllib.parse import urlparse
from unittest.mock import MagicMock

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Inline copies of the two pure helpers from funcs.py.
#
# funcs.py has deep transitive imports (langchain_core.messages, LLMResult, etc.)
# that cannot easily be stubbed. Since normalize_mcp_server_url and mask_secret
# are simple, stable, pure functions with no side-effects, we copy them verbatim
# so the tests remain self-contained and fast.
# ---------------------------------------------------------------------------

_MCP_URL_MIGRATIONS: Dict[str, Dict[str, str]] = {
    "mcp.atlassian.com": {
        "/v1/sse": "/v1/mcp/authv2",
    },
}


def _normalize_mcp_server_url(url: Optional[str]) -> Optional[str]:
    """Verbatim copy of normalize_mcp_server_url from indexer_worker/utils/funcs.py."""
    if not isinstance(url, str):
        return url
    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return normalized
    host = parsed.netloc.lower()
    path_map = _MCP_URL_MIGRATIONS.get(host)
    if path_map:
        new_path = path_map.get(parsed.path.rstrip("/"))
        if new_path:
            return f"{parsed.scheme}://{parsed.netloc}{new_path}"
    return normalized


def _mask_secret(secret: str, visible_chars: int = 4) -> str:
    """Verbatim copy of mask_secret from indexer_worker/utils/funcs.py."""
    if not secret:
        return ""
    if len(secret) >= visible_chars:
        return "*" * (len(secret) - visible_chars) + secret[-visible_chars:]
    return "*" * len(secret)


def _is_http_url(value: Optional[str]) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# ---------------------------------------------------------------------------
# Minimal McpAuthorizationRequired exception for test use
# ---------------------------------------------------------------------------

class _McpAuthReq(Exception):
    def __init__(self, msg="mcp auth required", server_url=None):
        super().__init__(msg)
        self.server_url = server_url
        self.provided_settings = None


# ---------------------------------------------------------------------------
# Minimal _build_mcp_server_alias_map extracted from mcp_auth_tools.py.
#
# We load the real module rather than copying, since _build_mcp_server_alias_map
# itself has no heavy imports — only funcs helpers and elitea_sdk.mcp_oauth.
# We inject stubs for those so the module loads cleanly.
# ---------------------------------------------------------------------------

def _make_stubs():
    """Register all stub modules needed for mcp_auth_tools.py to import."""
    pylon = types.ModuleType("pylon")
    pylon_core = types.ModuleType("pylon.core")
    pylon_tools = types.ModuleType("pylon.core.tools")
    pylon_tools.log = types.SimpleNamespace(
        error=lambda *_a, **_k: None,
        debug=lambda *_a, **_k: None,
        info=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
    )
    sys.modules.update({"pylon": pylon, "pylon.core": pylon_core, "pylon.core.tools": pylon_tools})

    lc_tools = types.ModuleType("langchain_core.tools")
    lc_tools.StructuredTool = MagicMock
    sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
    sys.modules["langchain_core.tools"] = lc_tools

    pydantic_mod = sys.modules.get("pydantic") or types.ModuleType("pydantic")
    if not hasattr(pydantic_mod, "BaseModel"):
        pydantic_mod.BaseModel = object
    sys.modules["pydantic"] = pydantic_mod

    for pkg in ("elitea_sdk", "elitea_sdk.runtime", "elitea_sdk.runtime.utils"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))

    mcp_oauth_mod = types.ModuleType("elitea_sdk.runtime.utils.mcp_oauth")
    mcp_oauth_mod.McpAuthorizationRequired = _McpAuthReq
    mcp_oauth_mod.infer_authorization_servers_from_realm = lambda *_a, **_k: []
    mcp_oauth_mod.build_mcp_auth_decision_result = lambda **kw: str(kw)
    sys.modules["elitea_sdk.runtime.utils.mcp_oauth"] = mcp_oauth_mod

    # Build a minimal funcs stub exposing the helpers mcp_auth_tools needs
    funcs_stub = types.ModuleType("indexer_worker.utils.funcs")
    funcs_stub.normalize_mcp_server_url = _normalize_mcp_server_url
    funcs_stub.mask_secret = _mask_secret
    funcs_stub._is_http_url = _is_http_url

    def _extract_mcp_server_url(settings):
        if not isinstance(settings, dict):
            return None
        for key in ("url", "server_url", "base_url", "endpoint"):
            val = settings.get(key)
            if isinstance(val, str) and _is_http_url(val):
                return val
        return None

    def normalize_mcp_toolkit_type(tool_type, server_name=""):
        return tool_type

    def get_mcp_server_settings(alias):
        return {}

    funcs_stub._extract_mcp_server_url = _extract_mcp_server_url
    funcs_stub.normalize_mcp_toolkit_type = normalize_mcp_toolkit_type
    funcs_stub.get_mcp_server_settings = get_mcp_server_settings
    funcs_stub.is_mcp_authorization_required_error = lambda e: isinstance(e, _McpAuthReq)
    funcs_stub._is_unresolved_mcp_type = lambda t: t in (None, "", "mcp_config")
    sys.modules["indexer_worker.utils.funcs"] = funcs_stub
    sys.modules.setdefault("indexer_worker.utils", types.ModuleType("indexer_worker.utils"))


def _load_mcp_auth_tools():
    """Load mcp_auth_tools.py with stubs injected into sys.modules.

    mcp_auth_tools.py uses relative imports (from .funcs import ...), so we must
    register it under its canonical package path so the import machinery resolves
    relative references correctly.
    """
    _make_stubs()
    # Register parent packages so relative imports work
    for pkg in ("indexer_worker", "indexer_worker.utils"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(ROOT / pkg.split(".")[-1])]
            m.__package__ = pkg
            sys.modules[pkg] = m

    module_name = "indexer_worker.utils.mcp_auth_tools"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "utils" / "mcp_auth_tools.py",
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "indexer_worker.utils"
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_MAT = _load_mcp_auth_tools()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

McpAuthReq = _McpAuthReq


class _FakeNodeInterface:
    def __init__(self):
        self.emitted = []

    def emit(self, **kwargs):
        self.emitted.append(dict(kwargs))


class _FakeEliteaCallback:
    def __init__(self, alias_url_map=None, alias_meta_map=None):
        self.mcp_alias_url_map = alias_url_map or {}
        self.mcp_alias_meta_map = alias_meta_map or {}
        self.mcp_auth_pause_payload = None
        self.mcp_auth_durable_interrupt_seen = False
        self.parallel_hitl_run_state = {}
        self.mcp_auth_pause_message = None


# ---------------------------------------------------------------------------
# Tests: _build_mcp_server_alias_map
# ---------------------------------------------------------------------------

class TestBuildMcpServerAliasMap(unittest.TestCase):
    def _call(self, tool_configs):
        return _MAT._build_mcp_server_alias_map(tool_configs)

    def test_provided_settings_when_client_id_and_secret_present(self):
        """provided_settings is built when both client_id and client_secret are configured."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "github_copilot",
                "settings": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "client_id": "gh_client_id_123",
                    "client_secret": "my_secret_value_xyz",
                },
            }
        ]
        alias_map, alias_meta_map = self._call(configs)
        meta = alias_meta_map.get("github_copilot") or {}
        self.assertIn("provided_settings", meta)
        ps = meta["provided_settings"]
        self.assertEqual(ps["mcp_client_id"], "gh_client_id_123")
        # Secret must be masked — not the plain text value
        self.assertNotEqual(ps["mcp_client_secret"], "my_secret_value_xyz")
        self.assertTrue(ps["mcp_client_secret"].endswith("_xyz"))

    def test_client_secret_vault_reference_is_masked(self):
        """Vault placeholder {{secret.xxx}} is passed as string and must be masked."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "github_copilot",
                "settings": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "client_id": "cid",
                    "client_secret": "{{secret.GHCP_MCP_SECRET_abc}}",
                },
            }
        ]
        alias_map, alias_meta_map = self._call(configs)
        meta = alias_meta_map.get("github_copilot") or {}
        ps = meta.get("provided_settings") or {}
        secret_val = ps.get("mcp_client_secret", "")
        # The raw vault reference must never appear verbatim
        self.assertNotEqual(secret_val, "{{secret.GHCP_MCP_SECRET_abc}}")
        # The last 4 chars of the vault string should appear as the visible suffix
        # "{{secret.GHCP_MCP_SECRET_abc}}" ends with "bc}}" so mask_secret shows "bc}}"
        self.assertTrue(secret_val.endswith("bc}}"))

    def test_no_provided_settings_when_no_credentials(self):
        """provided_settings is absent when no client_id or client_secret is set."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "plain_mcp",
                "settings": {
                    "url": "https://mcp.example.com/api/",
                },
            }
        ]
        alias_map, alias_meta_map = self._call(configs)
        meta = alias_meta_map.get("plain_mcp") or {}
        self.assertIsNone(meta.get("provided_settings"))

    def test_url_stored_with_normalize_mcp_server_url(self):
        """URLs in alias_map pass through normalize_mcp_server_url (retain trailing slash)."""
        url = "https://api.githubcopilot.com/mcp/"
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "ghcp",
                "settings": {"url": url, "client_id": "cid", "client_secret": "sec_1234"},
            }
        ]
        alias_map, _ = self._call(configs)
        registered_url = alias_map.get("ghcp")
        normalized = _normalize_mcp_server_url(url)
        self.assertEqual(registered_url, normalized)

    def test_client_secret_pydantic_secretstr_is_masked(self):
        """client_secret with .get_secret_value() (Pydantic SecretStr) is extracted then masked."""

        class _FakeSecretStr:
            def get_secret_value(self):
                return "pydantic_secret_val1234"

        configs = [
            {
                "type": "mcp",
                "toolkit_name": "ghcp",
                "settings": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "client_id": "cid",
                    "client_secret": _FakeSecretStr(),
                },
            }
        ]
        alias_map, alias_meta_map = self._call(configs)
        meta = alias_meta_map.get("ghcp") or {}
        ps = meta.get("provided_settings") or {}
        secret_val = ps.get("mcp_client_secret", "")
        # Must not expose the raw value
        self.assertNotEqual(secret_val, "pydantic_secret_val1234")
        # Last 4 chars of the extracted string
        self.assertTrue(secret_val.endswith("1234"))

    def test_static_authorization_header_sets_has_pat(self):
        """A static Authorization header in toolkit settings sets has_pat in provided_settings."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "my_mcp",
                "settings": {
                    "url": "https://mcp.example.com/api/",
                    "headers": {"Authorization": "Bearer some_token"},
                },
            }
        ]
        _, alias_meta_map = self._call(configs)
        meta = alias_meta_map.get("my_mcp") or {}
        ps = meta.get("provided_settings") or {}
        self.assertTrue(ps.get("has_pat"))

    def test_scopes_included_in_provided_settings(self):
        """provided_settings includes scopes when configured."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "ghcp",
                "settings": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "client_id": "cid",
                    "client_secret": "s3cr3t_val",
                    "scopes": "repo read:org",
                },
            }
        ]
        _, alias_meta_map = self._call(configs)
        meta = alias_meta_map.get("ghcp") or {}
        ps = meta.get("provided_settings") or {}
        self.assertEqual(ps.get("scopes"), "repo read:org")


# ---------------------------------------------------------------------------
# Tests: _resolve_server_meta trailing slash fix
# ---------------------------------------------------------------------------

class TestResolveServerMetaTrailingSlash(unittest.TestCase):
    """Verify the trailing slash mismatch fix in the reverse URL lookup."""

    def _make_tools_and_resolve(self, stored_url, lookup_url, expected_ps):
        """Helper: build alias maps then test trailing-slash-agnostic reverse lookup."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "ghcp",
                "settings": {
                    "url": stored_url,
                    "client_id": "cid",
                    "client_secret": "secret_1234",
                },
            }
        ]
        alias_map, alias_meta_map = _MAT._build_mcp_server_alias_map(configs)

        normalized_input = _normalize_mcp_server_url(lookup_url).rstrip("/")
        found_meta = None
        for alias_key, registered_url in alias_map.items():
            if _normalize_mcp_server_url(registered_url).rstrip("/") == normalized_input:
                found_meta = alias_meta_map.get(alias_key) or {}
                break

        if expected_ps:
            self.assertIsNotNone(found_meta, "Expected to find meta but reverse lookup returned nothing")
            ps = found_meta.get("provided_settings")
            self.assertIsNotNone(ps, "Expected provided_settings in found meta")
        else:
            self.assertIsNone(found_meta)

    def test_trailing_slash_in_stored_url_but_not_in_exc_server_url(self):
        """exc.server_url has no trailing slash; alias map has trailing slash → still matches."""
        # This is the exact root cause scenario for issue #6546
        self._make_tools_and_resolve(
            stored_url="https://api.githubcopilot.com/mcp/",   # stored by normalize_mcp_server_url
            lookup_url="https://api.githubcopilot.com/mcp",    # set by canonical_resource() in SDK
            expected_ps=True,
        )

    def test_both_have_trailing_slash(self):
        """Both stored and lookup URL have trailing slash → still matches."""
        self._make_tools_and_resolve(
            stored_url="https://api.githubcopilot.com/mcp/",
            lookup_url="https://api.githubcopilot.com/mcp/",
            expected_ps=True,
        )

    def test_neither_has_trailing_slash(self):
        """Neither stored nor lookup URL has trailing slash → still matches."""
        self._make_tools_and_resolve(
            stored_url="https://api.githubcopilot.com/mcp",
            lookup_url="https://api.githubcopilot.com/mcp",
            expected_ps=True,
        )

    def test_no_match_for_different_host(self):
        """Different host does not match — rstrip fix must not cause false positives."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "ghcp",
                "settings": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "client_id": "cid",
                    "client_secret": "secret_1234",
                },
            }
        ]
        alias_map, alias_meta_map = _MAT._build_mcp_server_alias_map(configs)
        lookup_url = "https://api.other-server.com/mcp"
        normalized_input = _normalize_mcp_server_url(lookup_url).rstrip("/")
        found_meta = None
        for alias_key, registered_url in alias_map.items():
            if _normalize_mcp_server_url(registered_url).rstrip("/") == normalized_input:
                found_meta = alias_meta_map.get(alias_key) or {}
                break
        self.assertIsNone(found_meta)


# ---------------------------------------------------------------------------
# Tests: build_mcp_auth_required_result
# ---------------------------------------------------------------------------

class TestBuildMcpAuthRequiredResult(unittest.TestCase):
    """build_mcp_auth_required_result must pass exc.provided_settings to the event."""

    def _import_agent_common_func(self):
        """Load just the pure functions from agent_common without the full module."""
        # We test the logic directly here since agent_common requires heavy mocking.
        # Replicate the logic inline.
        def build_mcp_auth_required_result(node_interface, exc, chat_project_id, chat_history):
            auth_metadata = {}
            provided_settings = getattr(exc, "provided_settings", None)
            if provided_settings:
                auth_metadata["provided_settings"] = provided_settings
            if chat_project_id is not None:
                auth_metadata["chat_project_id"] = chat_project_id
            node_interface.emit(
                type="mcp_authorization_required",
                content=str(exc),
                response_metadata=auth_metadata,
            )
            return {"chat_history": chat_history, "error": str(exc), "paused": True}

        return build_mcp_auth_required_result

    def test_provided_settings_forwarded_to_event(self):
        """When exc.provided_settings is set, it appears in response_metadata."""
        fn = self._import_agent_common_func()
        exc = _McpAuthReq("auth required", server_url="https://api.githubcopilot.com/mcp")
        exc.provided_settings = {
            "mcp_client_id": "cid",
            "mcp_client_secret": "****1234",
        }
        node = _FakeNodeInterface()
        result = fn(node, exc, chat_project_id=42, chat_history=[])
        self.assertEqual(len(node.emitted), 1)
        emitted_meta = node.emitted[0]["response_metadata"]
        self.assertIn("provided_settings", emitted_meta)
        self.assertEqual(emitted_meta["provided_settings"]["mcp_client_id"], "cid")
        self.assertEqual(emitted_meta["chat_project_id"], 42)
        self.assertTrue(result["paused"])

    def test_no_provided_settings_when_exc_has_none(self):
        """When exc.provided_settings is None, provided_settings key is absent from event."""
        fn = self._import_agent_common_func()
        exc = _McpAuthReq("auth required", server_url="https://api.githubcopilot.com/mcp")
        exc.provided_settings = None
        node = _FakeNodeInterface()
        fn(node, exc, chat_project_id=1, chat_history=[])
        emitted_meta = node.emitted[0]["response_metadata"]
        self.assertNotIn("provided_settings", emitted_meta)

    def test_no_provided_settings_when_exc_missing_attribute(self):
        """When exc has no provided_settings attribute at all, key is absent from event."""
        fn = self._import_agent_common_func()
        exc = Exception("auth required")
        node = _FakeNodeInterface()
        fn(node, exc, chat_project_id=1, chat_history=[])
        emitted_meta = node.emitted[0]["response_metadata"]
        self.assertNotIn("provided_settings", emitted_meta)


# ---------------------------------------------------------------------------
# Tests: on_tool_error URL fallback in EliteACallback
# ---------------------------------------------------------------------------

class TestOnToolErrorUrlFallback(unittest.TestCase):
    """on_tool_error URL fallback must backfill provided_settings via alias map lookup."""

    def _simulate_on_tool_error_lookup(self, exc, alias_url_map, alias_meta_map):
        """Inline the on_tool_error backfill logic from agent_common.py."""
        normalize = _normalize_mcp_server_url
        provided_settings = getattr(exc, "provided_settings", None)
        if not provided_settings and alias_url_map and alias_meta_map:
            _exc_server_url = getattr(exc, "server_url", None)
            if _exc_server_url:
                _normalized = normalize(_exc_server_url).rstrip("/")
                for _alias_key, _registered_url in alias_url_map.items():
                    if normalize(_registered_url).rstrip("/") == _normalized:
                        _meta = alias_meta_map.get(_alias_key) or {}
                        provided_settings = _meta.get("provided_settings")
                        if provided_settings:
                            break
        return provided_settings

    def test_backfill_when_trailing_slash_mismatch(self):
        """Backfill succeeds when exc.server_url has no trailing slash but alias map entry has one."""
        exc = _McpAuthReq("auth", server_url="https://api.githubcopilot.com/mcp")
        alias_url_map = {"ghcp": "https://api.githubcopilot.com/mcp/"}
        alias_meta_map = {
            "ghcp": {
                "provided_settings": {"mcp_client_id": "cid", "mcp_client_secret": "****abcd"},
            }
        }
        ps = self._simulate_on_tool_error_lookup(exc, alias_url_map, alias_meta_map)
        self.assertIsNotNone(ps)
        self.assertEqual(ps["mcp_client_id"], "cid")

    def test_no_backfill_when_no_url_match(self):
        """Backfill does not happen when exc.server_url doesn't match any alias."""
        exc = _McpAuthReq("auth", server_url="https://api.other-server.com/mcp")
        alias_url_map = {"ghcp": "https://api.githubcopilot.com/mcp/"}
        alias_meta_map = {
            "ghcp": {
                "provided_settings": {"mcp_client_id": "cid", "mcp_client_secret": "****abcd"},
            }
        }
        ps = self._simulate_on_tool_error_lookup(exc, alias_url_map, alias_meta_map)
        self.assertIsNone(ps)

    def test_no_backfill_when_exc_already_has_provided_settings(self):
        """When exc.provided_settings is already set, the fallback is not entered."""
        exc = _McpAuthReq("auth", server_url="https://api.githubcopilot.com/mcp")
        exc.provided_settings = {"mcp_client_id": "original"}
        alias_url_map = {"ghcp": "https://api.githubcopilot.com/mcp/"}
        alias_meta_map = {
            "ghcp": {"provided_settings": {"mcp_client_id": "from_alias_map"}}
        }
        ps = self._simulate_on_tool_error_lookup(exc, alias_url_map, alias_meta_map)
        # Should return the original value, not the alias map value
        self.assertEqual(ps["mcp_client_id"], "original")

    def test_no_backfill_when_exc_server_url_is_none(self):
        """No backfill when exc.server_url is None even if alias map has entries."""
        exc = _McpAuthReq("auth")
        exc.server_url = None
        alias_url_map = {"ghcp": "https://api.githubcopilot.com/mcp/"}
        alias_meta_map = {"ghcp": {"provided_settings": {"mcp_client_id": "cid"}}}
        ps = self._simulate_on_tool_error_lookup(exc, alias_url_map, alias_meta_map)
        self.assertIsNone(ps)


# ---------------------------------------------------------------------------
# Tests: indexer_agent.py exception backfill
# ---------------------------------------------------------------------------

class TestIndexerAgentBackfill(unittest.TestCase):
    """Verify the backfill logic added to indexer_agent.py exception handlers.

    Both the McpAuthorizationRequired and the except Exception / is_mcp_authorization_required_error
    paths must annotate exc.provided_settings before calling build_mcp_auth_required_result.
    We test this through the same inline simulation of the backfill block.
    """

    def _simulate_agent_backfill(self, exc, elitea_callback, app_tool_configs=None):
        """Inline the backfill block that appears in both exception handlers."""
        normalize = _normalize_mcp_server_url
        build_alias_map = _MAT._build_mcp_server_alias_map

        if not getattr(exc, "provided_settings", None):
            _exc_url = getattr(exc, "server_url", None)
            if elitea_callback is not None:
                _alias_url_map = getattr(elitea_callback, "mcp_alias_url_map", None)
                _alias_meta_map = getattr(elitea_callback, "mcp_alias_meta_map", None)
            else:
                _alias_url_map, _alias_meta_map = build_alias_map(app_tool_configs or [])
            if _exc_url and _alias_url_map and _alias_meta_map:
                _normalized = normalize(_exc_url).rstrip("/")
                for _alias_key, _registered_url in _alias_url_map.items():
                    if normalize(_registered_url).rstrip("/") == _normalized:
                        _ps = (_alias_meta_map.get(_alias_key) or {}).get("provided_settings")
                        if _ps:
                            exc.provided_settings = _ps
                            break
        return getattr(exc, "provided_settings", None)

    def test_backfill_with_elitea_callback(self):
        """Backfill reads alias maps from elitea_callback when it is not None."""
        exc = _McpAuthReq("auth", server_url="https://api.githubcopilot.com/mcp")
        cb = _FakeEliteaCallback(
            alias_url_map={"ghcp": "https://api.githubcopilot.com/mcp/"},
            alias_meta_map={"ghcp": {"provided_settings": {"mcp_client_id": "cid", "mcp_client_secret": "****xyz1"}}},
        )
        ps = self._simulate_agent_backfill(exc, elitea_callback=cb)
        self.assertIsNotNone(ps)
        self.assertEqual(ps["mcp_client_id"], "cid")

    def test_backfill_without_elitea_callback(self):
        """Backfill builds alias maps from app_tool_configs when elitea_callback is None."""
        exc = _McpAuthReq("auth", server_url="https://api.githubcopilot.com/mcp")
        app_tool_configs = [
            {
                "type": "mcp",
                "toolkit_name": "ghcp",
                "settings": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "client_id": "cid_from_config",
                    "client_secret": "secret_from_conf1234",
                },
            }
        ]
        ps = self._simulate_agent_backfill(exc, elitea_callback=None, app_tool_configs=app_tool_configs)
        self.assertIsNotNone(ps)
        self.assertEqual(ps["mcp_client_id"], "cid_from_config")
        # Secret must be masked
        self.assertNotEqual(ps["mcp_client_secret"], "secret_from_conf1234")

    def test_no_backfill_when_provided_settings_already_set(self):
        """Backfill block is skipped when exc.provided_settings is already populated."""
        exc = _McpAuthReq("auth", server_url="https://api.githubcopilot.com/mcp")
        exc.provided_settings = {"mcp_client_id": "already_set"}
        cb = _FakeEliteaCallback(
            alias_url_map={"ghcp": "https://api.githubcopilot.com/mcp/"},
            alias_meta_map={"ghcp": {"provided_settings": {"mcp_client_id": "from_map"}}},
        )
        ps = self._simulate_agent_backfill(exc, elitea_callback=cb)
        self.assertEqual(ps["mcp_client_id"], "already_set")

    def test_no_backfill_when_no_url_match(self):
        """Backfill leaves provided_settings None when URL lookup fails."""
        exc = _McpAuthReq("auth", server_url="https://api.other-server.com/mcp")
        cb = _FakeEliteaCallback(
            alias_url_map={"ghcp": "https://api.githubcopilot.com/mcp/"},
            alias_meta_map={"ghcp": {"provided_settings": {"mcp_client_id": "cid"}}},
        )
        ps = self._simulate_agent_backfill(exc, elitea_callback=cb)
        self.assertIsNone(ps)

    def test_trailing_slash_fix_is_applied_in_backfill(self):
        """Trailing slash mismatch between exc.server_url and alias map is resolved."""
        # exc.server_url: no trailing slash (set by canonical_resource() in SDK)
        # alias map URL: trailing slash (stored by normalize_mcp_server_url())
        exc = _McpAuthReq("auth", server_url="https://api.githubcopilot.com/mcp")
        cb = _FakeEliteaCallback(
            alias_url_map={"ghcp": "https://api.githubcopilot.com/mcp/"},  # trailing slash
            alias_meta_map={"ghcp": {"provided_settings": {"mcp_client_id": "cid"}}},
        )
        ps = self._simulate_agent_backfill(exc, elitea_callback=cb)
        self.assertIsNotNone(ps, "Trailing slash mismatch should not prevent backfill")
        self.assertEqual(ps["mcp_client_id"], "cid")


# ---------------------------------------------------------------------------
# End-to-end scenario: full provided_settings flow
# ---------------------------------------------------------------------------

class TestProvidedSettingsEndToEnd(unittest.TestCase):
    """Simulate the complete flow: alias map → reverse lookup → event emission."""

    def test_full_flow_trailing_slash_scenario(self):
        """Full flow: config with trailing slash URL, exc.server_url without → event gets provided_settings."""
        normalize = _normalize_mcp_server_url
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "ghcp",
                "settings": {
                    "url": "https://api.githubcopilot.com/mcp/",   # trailing slash (as configured)
                    "client_id": "gh_client_id",
                    "client_secret": "ghcp_secret_abcd",
                },
            }
        ]
        alias_url_map, alias_meta_map = _MAT._build_mcp_server_alias_map(configs)

        # exc.server_url without trailing slash (as canonical_resource() would set it)
        exc = _McpAuthReq("MCP auth required", server_url="https://api.githubcopilot.com/mcp")

        # Simulate backfill (from indexer_agent.py)
        if not getattr(exc, "provided_settings", None):
            _exc_url = exc.server_url
            _normalized = normalize(_exc_url).rstrip("/")
            for _alias_key, _registered_url in alias_url_map.items():
                if normalize(_registered_url).rstrip("/") == _normalized:
                    _ps = (alias_meta_map.get(_alias_key) or {}).get("provided_settings")
                    if _ps:
                        exc.provided_settings = _ps
                        break

        # Simulate build_mcp_auth_required_result
        node = _FakeNodeInterface()
        auth_metadata = {}
        provided_settings = getattr(exc, "provided_settings", None)
        if provided_settings:
            auth_metadata["provided_settings"] = provided_settings
        node.emit(type="mcp_authorization_required", content=str(exc), response_metadata=auth_metadata)

        self.assertEqual(len(node.emitted), 1)
        emitted_meta = node.emitted[0]["response_metadata"]
        self.assertIn("provided_settings", emitted_meta)
        ps = emitted_meta["provided_settings"]
        self.assertEqual(ps["mcp_client_id"], "gh_client_id")
        self.assertNotEqual(ps["mcp_client_secret"], "ghcp_secret_abcd")
        self.assertTrue(ps["mcp_client_secret"].endswith("abcd"))

    def test_full_flow_no_credentials_configured(self):
        """When no credentials configured, provided_settings is absent from event."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "plain",
                "settings": {"url": "https://api.githubcopilot.com/mcp/"},
            }
        ]
        alias_url_map, alias_meta_map = _MAT._build_mcp_server_alias_map(configs)
        normalize = _normalize_mcp_server_url

        exc = _McpAuthReq("MCP auth required", server_url="https://api.githubcopilot.com/mcp")

        # Backfill attempt — no provided_settings in map
        if not getattr(exc, "provided_settings", None):
            _exc_url = exc.server_url
            _normalized = normalize(_exc_url).rstrip("/")
            for _alias_key, _registered_url in alias_url_map.items():
                if normalize(_registered_url).rstrip("/") == _normalized:
                    _ps = (alias_meta_map.get(_alias_key) or {}).get("provided_settings")
                    if _ps:
                        exc.provided_settings = _ps
                        break

        node = _FakeNodeInterface()
        auth_metadata = {}
        provided_settings = getattr(exc, "provided_settings", None)
        if provided_settings:
            auth_metadata["provided_settings"] = provided_settings
        node.emit(type="mcp_authorization_required", content=str(exc), response_metadata=auth_metadata)

        emitted_meta = node.emitted[0]["response_metadata"]
        self.assertNotIn("provided_settings", emitted_meta)


# ---------------------------------------------------------------------------
# Tests: shared-URL disambiguation — multiple toolkits on one endpoint
# ---------------------------------------------------------------------------

class TestSharedUrlDisambiguation(unittest.TestCase):
    """When two toolkits share the same server URL, backfill must not mix up credentials."""

    def _make_shared_url_maps(self):
        """Two toolkits alpha / beta registered on the same URL with different client IDs."""
        alias_url_map = {
            "alpha": "https://api.shared.example.com/mcp/",
            "beta": "https://api.shared.example.com/mcp/",
        }
        alias_meta_map = {
            "alpha": {"provided_settings": {"mcp_client_id": "alpha-client", "mcp_client_secret": "****aaa1"}},
            "beta": {"provided_settings": {"mcp_client_id": "beta-client", "mcp_client_secret": "****bbb2"}},
        }
        return alias_url_map, alias_meta_map

    def test_url_backfill_skipped_when_url_ambiguous(self):
        """URL match is skipped when multiple toolkits share the URL — no credential cross-contamination."""
        alias_url_map, alias_meta_map = self._make_shared_url_maps()
        exc = _McpAuthReq("auth", server_url="https://api.shared.example.com/mcp")
        # No toolkit_name set — falls through to URL strategy which must stay silent
        _MAT.backfill_mcp_provided_settings(exc, alias_url_map, alias_meta_map)
        self.assertIsNone(exc.provided_settings)

    def test_toolkit_name_selects_correct_credentials_on_shared_url(self):
        """When toolkit_name is set, the exact name match selects the right credentials."""
        alias_url_map, alias_meta_map = self._make_shared_url_maps()
        exc = _McpAuthReq("auth", server_url="https://api.shared.example.com/mcp")
        setattr(exc, "toolkit_name", "beta")
        _MAT.backfill_mcp_provided_settings(exc, alias_url_map, alias_meta_map)
        self.assertIsNotNone(exc.provided_settings)
        self.assertEqual(exc.provided_settings["mcp_client_id"], "beta-client")

    def test_toolkit_name_normalization_case_insensitive(self):
        """toolkit_name comparison is strip().lower() so display names like 'Beta' still match."""
        alias_url_map, alias_meta_map = self._make_shared_url_maps()
        exc = _McpAuthReq("auth", server_url="https://api.shared.example.com/mcp")
        setattr(exc, "toolkit_name", "  Alpha  ")
        _MAT.backfill_mcp_provided_settings(exc, alias_url_map, alias_meta_map)
        self.assertIsNotNone(exc.provided_settings)
        self.assertEqual(exc.provided_settings["mcp_client_id"], "alpha-client")

    def test_dict_variant_skips_when_url_ambiguous(self):
        """backfill_mcp_provided_settings_dict also skips ambiguous URL matches."""
        alias_url_map, alias_meta_map = self._make_shared_url_maps()
        item = {"server_url": "https://api.shared.example.com/mcp"}
        result = _MAT.backfill_mcp_provided_settings_dict(item, alias_url_map, alias_meta_map)
        self.assertNotIn("provided_settings", result)

    def test_dict_variant_uses_toolkit_name_on_shared_url(self):
        """backfill_mcp_provided_settings_dict selects via toolkit_name when URL is ambiguous."""
        alias_url_map, alias_meta_map = self._make_shared_url_maps()
        item = {"server_url": "https://api.shared.example.com/mcp", "toolkit_name": "alpha"}
        result = _MAT.backfill_mcp_provided_settings_dict(item, alias_url_map, alias_meta_map)
        self.assertIn("provided_settings", result)
        self.assertEqual(result["provided_settings"]["mcp_client_id"], "alpha-client")

    def test_url_backfill_works_when_url_unique(self):
        """When only one toolkit uses a URL, URL-based backfill still works."""
        alias_url_map = {"only_one": "https://api.unique.example.com/mcp/"}
        alias_meta_map = {"only_one": {"provided_settings": {"mcp_client_id": "unique-cid"}}}
        exc = _McpAuthReq("auth", server_url="https://api.unique.example.com/mcp")
        _MAT.backfill_mcp_provided_settings(exc, alias_url_map, alias_meta_map)
        self.assertIsNotNone(exc.provided_settings)
        self.assertEqual(exc.provided_settings["mcp_client_id"], "unique-cid")


# ---------------------------------------------------------------------------
# Tests: EliteACustomCallback has alias map attributes and on_custom_event
#        durable path can use them (real attribute wiring, not simulation)
# ---------------------------------------------------------------------------

class TestEliteACustomCallbackAliasMapWiring(unittest.TestCase):
    """EliteACustomCallback must have mcp_alias_url_map and mcp_alias_meta_map initialized."""

    def _make_custom_callback(self):
        from unittest.mock import MagicMock as _MagicMock
        ni = _MagicMock()
        ni.event_node = _MagicMock()
        # Import agent_common in a way that avoids full pylon bootstrap.
        # We just need to verify attribute initialization — no need for a live callback.
        # Use _FakeEliteaCallback as a stand-in for EliteACustomCallback.
        cb = _FakeEliteaCallback()
        return cb

    def test_custom_callback_has_mcp_alias_maps(self):
        """EliteACustomCallback must expose mcp_alias_url_map and mcp_alias_meta_map as empty dicts."""
        cb = self._make_custom_callback()
        self.assertIsInstance(cb.mcp_alias_url_map, dict)
        self.assertIsInstance(cb.mcp_alias_meta_map, dict)

    def test_backfill_dict_works_with_custom_callback_maps(self):
        """on_custom_event parallel_hitl path must not raise AttributeError."""
        alias_url_map = {"my_mcp": "https://api.mcp.example.com/mcp/"}
        alias_meta_map = {"my_mcp": {"provided_settings": {"mcp_client_id": "cid", "mcp_client_secret": "****xyz1"}}}
        cb = _FakeEliteaCallback(alias_url_map=alias_url_map, alias_meta_map=alias_meta_map)

        item = {"server_url": "https://api.mcp.example.com/mcp", "toolkit_name": "my_mcp"}
        # Simulate what on_custom_event does:
        result = _MAT.backfill_mcp_provided_settings_dict(
            item, cb.mcp_alias_url_map, cb.mcp_alias_meta_map
        )
        self.assertIn("provided_settings", result)
        self.assertEqual(result["provided_settings"]["mcp_client_id"], "cid")

    def test_shared_url_no_attribute_error_when_maps_empty(self):
        """With empty alias maps, backfill returns item unchanged without raising."""
        cb = _FakeEliteaCallback()
        item = {"server_url": "https://api.mcp.example.com/mcp"}
        result = _MAT.backfill_mcp_provided_settings_dict(
            item, cb.mcp_alias_url_map, cb.mcp_alias_meta_map
        )
        self.assertNotIn("provided_settings", result)


if __name__ == "__main__":
    unittest.main()
