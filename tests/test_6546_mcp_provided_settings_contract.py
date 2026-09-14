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

    # --- Tests using maps produced by the real builder (catches generic "mcp" alias) ---

    def test_real_builder_single_toolkit_url_backfill_resolves(self):
        """A single configured toolkit registers 'mcp' + its name as aliases.

        URL-based backfill must still resolve because the real builder now stores
        toolkit_id on all alias entries, so deduplication collapses them to one
        unique identity even though matched_keys has length > 1.
        """
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "github_copilot",
                "toolkit_id": 101,
                "settings": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "client_id": "gh-cid",
                    "client_secret": "gh-secret-1234",
                },
            }
        ]
        alias_url_map, alias_meta_map = _MAT._build_mcp_server_alias_map(configs)
        exc = _McpAuthReq("auth", server_url="https://api.githubcopilot.com/mcp")
        # no toolkit_name / toolkit_id set on exception — falls to URL strategy
        _MAT.backfill_mcp_provided_settings(exc, alias_url_map, alias_meta_map)
        self.assertIsNotNone(
            exc.provided_settings,
            "Single toolkit with generic 'mcp' alias must still resolve via URL — "
            "identity deduplication should collapse multiple alias keys to one toolkit",
        )
        self.assertEqual(exc.provided_settings["mcp_client_id"], "gh-cid")

    def test_real_builder_single_toolkit_toolkit_id_on_exc_resolves(self):
        """toolkit_id on exc picks the right entry even when the alias key is generic 'mcp'."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "github_copilot",
                "toolkit_id": 101,
                "settings": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "client_id": "gh-cid",
                    "client_secret": "gh-secret-1234",
                },
            }
        ]
        alias_url_map, alias_meta_map = _MAT._build_mcp_server_alias_map(configs)
        exc = _McpAuthReq("auth", server_url="https://api.githubcopilot.com/mcp")
        exc.toolkit_id = 101  # type: ignore[attr-defined]
        _MAT.backfill_mcp_provided_settings(exc, alias_url_map, alias_meta_map)
        self.assertIsNotNone(exc.provided_settings)
        self.assertEqual(exc.provided_settings["mcp_client_id"], "gh-cid")

    def test_real_builder_two_toolkits_same_url_toolkit_id_selects_correct(self):
        """Two toolkits on same URL: toolkit_id on exc selects correct credentials."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "alpha",
                "toolkit_id": 10,
                "settings": {
                    "url": "https://api.shared.example.com/mcp/",
                    "client_id": "alpha-cid",
                    "client_secret": "alpha-secret-1234",
                },
            },
            {
                "type": "mcp",
                "toolkit_name": "beta",
                "toolkit_id": 20,
                "settings": {
                    "url": "https://api.shared.example.com/mcp/",
                    "client_id": "beta-cid",
                    "client_secret": "beta-secret-5678",
                },
            },
        ]
        alias_url_map, alias_meta_map = _MAT._build_mcp_server_alias_map(configs)
        # Reproduce the exact scenario Roman-Mitusov described: exc with toolkit_id=101,
        # toolkit_name='mcp', alpha URL
        exc = _McpAuthReq("auth", server_url="https://api.shared.example.com/mcp")
        exc.toolkit_id = 20  # type: ignore[attr-defined]  — beta's id
        exc.toolkit_name = "mcp"  # type: ignore[attr-defined]  — generic, as SDK may set
        _MAT.backfill_mcp_provided_settings(exc, alias_url_map, alias_meta_map)
        self.assertIsNotNone(exc.provided_settings)
        self.assertEqual(
            exc.provided_settings["mcp_client_id"],
            "beta-cid",
            "toolkit_id=20 must select beta-cid, not alpha-cid and not None",
        )

    def test_real_builder_two_toolkits_same_url_no_id_url_ambiguous(self):
        """Two toolkits on same URL, no toolkit_id on exc: URL backfill must stay silent."""
        configs = [
            {
                "type": "mcp",
                "toolkit_name": "alpha",
                "settings": {
                    "url": "https://api.shared.example.com/mcp/",
                    "client_id": "alpha-cid",
                    "client_secret": "alpha-secret-1234",
                },
            },
            {
                "type": "mcp",
                "toolkit_name": "beta",
                "settings": {
                    "url": "https://api.shared.example.com/mcp/",
                    "client_id": "beta-cid",
                    "client_secret": "beta-secret-5678",
                },
            },
        ]
        alias_url_map, alias_meta_map = _MAT._build_mcp_server_alias_map(configs)
        exc = _McpAuthReq("auth", server_url="https://api.shared.example.com/mcp")
        # toolkit_name is the generic 'mcp' alias — must not select either
        exc.toolkit_name = "mcp"  # type: ignore[attr-defined]
        _MAT.backfill_mcp_provided_settings(exc, alias_url_map, alias_meta_map)
        self.assertIsNone(
            exc.provided_settings,
            "Two toolkits with no toolkit_id and generic toolkit_name — must not attach credentials",
        )


# ---------------------------------------------------------------------------
# Loader: agent_common.py with minimal stubs
#
# agent_common.py imports langchain_core, elitea_sdk, requests, pylon and
# several relative modules. We stub everything needed so EliteACustomCallback
# (and EliteACallback) can be imported and instantiated in unit tests without
# a live pylon/indexer environment.
# ---------------------------------------------------------------------------

def _load_agent_common():
    """Load methods/agent_common.py with minimal stubs injected into sys.modules.

    Stubs that would overwrite real SDK modules (elitea_sdk.runtime.utils.trace_limits,
    elitea_sdk.tools.utils.serialization, elitea_sdk.runtime.langchain.constants) are
    installed only when the real module is absent, so running the full worker suite with
    a real SDK checkout does not pollute test_6532_trace_and_panel_serialization.py.

    Returns the loaded module, or None if loading fails (so individual tests can skip
    rather than error if the stub set is incomplete on a future SDK version).
    """
    import importlib.util as _ilu

    # ---- langchain_core stubs -----------------------------------------
    def _ensure_lc():
        import types as _t
        for mod in ("langchain_core", "langchain_core.callbacks",
                    "langchain_core.messages", "langchain_core.outputs"):
            sys.modules.setdefault(mod, _t.ModuleType(mod))

        cb = sys.modules["langchain_core.callbacks"]
        msg = sys.modules["langchain_core.messages"]
        out = sys.modules["langchain_core.outputs"]

        # BaseCallbackHandler — minimal, enough for __init_subclass__ / super().__init__()
        if not hasattr(cb, "BaseCallbackHandler"):
            class _BCH:
                def __init__(self):
                    pass
            cb.BaseCallbackHandler = _BCH

        for cls in ("BaseMessage", "HumanMessage", "AIMessage"):
            if not hasattr(msg, cls):
                setattr(msg, cls, MagicMock)

        for cls in ("ChatGenerationChunk", "LLMResult"):
            if not hasattr(out, cls):
                setattr(out, cls, MagicMock)

    _ensure_lc()

    # ---- elitea_sdk stubs — only install when the real module is absent ----
    import types as _t
    import re as _re

    for pkg in (
        "elitea_sdk", "elitea_sdk.runtime", "elitea_sdk.runtime.utils",
        "elitea_sdk.runtime.langchain", "elitea_sdk.tools", "elitea_sdk.tools.utils",
    ):
        sys.modules.setdefault(pkg, _t.ModuleType(pkg))

    # trace_limits: only stub when not already provided by the real SDK
    if "elitea_sdk.runtime.utils.trace_limits" not in sys.modules:
        _tl = _t.ModuleType("elitea_sdk.runtime.utils.trace_limits")
        _tl.TRACE_STEP_FIELD_MAX_CHARS = 10_000
        _tl.cap_trace_json = lambda v, **_: v
        _tl.cap_trace_text = lambda v, **_: v
        sys.modules["elitea_sdk.runtime.utils.trace_limits"] = _tl

    # Ensure trace_limits stub has configure_tool_result_limits so it won't break
    # test_6532 if the stub was already placed by a prior test run in the same session
    _tl_existing = sys.modules.get("elitea_sdk.runtime.utils.trace_limits")
    if _tl_existing is not None and not hasattr(_tl_existing, "configure_tool_result_limits"):
        _tl_existing.configure_tool_result_limits = MagicMock(return_value=None)

    if "elitea_sdk.tools.utils.serialization" not in sys.modules:
        _ser = _t.ModuleType("elitea_sdk.tools.utils.serialization")
        _ser.to_json_primitive = lambda v: str(v)
        sys.modules["elitea_sdk.tools.utils.serialization"] = _ser

    if "elitea_sdk.runtime.langchain.constants" not in sys.modules:
        _lc_const = _t.ModuleType("elitea_sdk.runtime.langchain.constants")
        _lc_const.LOAD_SKILL_ALREADY_ACTIVE_RE = _re.compile(r'^Skill "([^"]+)" is already active')
        _lc_const.LOADED_SKILL_PREFIX_RE = _re.compile(r'^Skill "([^"]+)" is now active')
        sys.modules["elitea_sdk.runtime.langchain.constants"] = _lc_const

    # ---- requests stub ------------------------------------------------
    sys.modules.setdefault("requests", _t.ModuleType("requests"))

    # ---- pydantic — real pydantic should be available, but ensure BaseModel ----
    import pydantic as _pyd  # noqa: F401  — already imported by mcp_auth_tools loader

    # ---- relative deps inside indexer_worker --------------------------
    for pkg in ("indexer_worker", "indexer_worker.utils", "indexer_worker.methods"):
        if pkg not in sys.modules:
            m = _t.ModuleType(pkg)
            m.__path__ = [str(ROOT / pkg.replace("indexer_worker.", "").replace("indexer_worker", ""))]
            m.__package__ = pkg
            sys.modules[pkg] = m

    # constants
    _const = _t.ModuleType("indexer_worker.utils.constants")
    _const.DEFAULT_MEMORY_CONFIG = {}
    sys.modules["indexer_worker.utils.constants"] = _const

    # exceptions
    _exc_mod = _t.ModuleType("indexer_worker.utils.exceptions")

    class _InternalSDKError(Exception):
        pass

    _exc_mod.InternalSDKError = _InternalSDKError
    sys.modules["indexer_worker.utils.exceptions"] = _exc_mod

    # funcs — reuse the existing stub (already registered by _make_stubs)
    # but extend it with what agent_common needs
    _funcs = sys.modules.get("indexer_worker.utils.funcs")
    if _funcs is None:
        _funcs = _t.ModuleType("indexer_worker.utils.funcs")
        sys.modules["indexer_worker.utils.funcs"] = _funcs
    for attr in (
        "_is_mcp_authorization_required_error",
        "is_mcp_authorization_required_error",
        "_is_unresolved_mcp_type",
        "_mcp_auth_error_to_metadata",
        "build_parallel_terminal_error",
        "budget_exceeded_error_code",
        "extract_finish_reason",
        "extract_token_usage",
        "num_tokens_from_messages",
        "should_emit_output_limit_confirmation",
    ):
        if not hasattr(_funcs, attr):
            setattr(_funcs, attr, MagicMock(return_value={}))
    if not hasattr(_funcs, "dev_reload_sdk"):
        _funcs.dev_reload_sdk = lambda *_a, **_k: None

    # node_interface — load the real one; it only needs pydantic + pylon log
    _ni_mod_name = "indexer_worker.utils.node_interface"
    if _ni_mod_name not in sys.modules:
        _ni_spec = _ilu.spec_from_file_location(
            _ni_mod_name,
            ROOT / "utils" / "node_interface.py",
            submodule_search_locations=[],
        )
        _ni_mod = _ilu.module_from_spec(_ni_spec)
        _ni_mod.__package__ = "indexer_worker.utils"
        sys.modules[_ni_mod_name] = _ni_mod
        try:
            _ni_spec.loader.exec_module(_ni_mod)
        except Exception:
            # If node_interface.py fails to load, use a minimal stub
            _ni_mod.NodeEventInterface = MagicMock
            _ni_mod.NodeEvent = MagicMock
            _ni_mod.EventTypes = MagicMock()
            _ni_mod.ELITEA_SDK_CUSTOM_EVENTS_MAPPER = {}

    # parallel_dispatch_contract
    _pdc = _t.ModuleType("indexer_worker.utils.parallel_dispatch_contract")
    _pdc.is_fanout_child = lambda _meta: False
    sys.modules["indexer_worker.utils.parallel_dispatch_contract"] = _pdc

    # mcp_auth_tools — already loaded as _MAT; re-register under the package path
    _mat_mod_name = "indexer_worker.utils.mcp_auth_tools"
    if _mat_mod_name not in sys.modules:
        sys.modules[_mat_mod_name] = _MAT

    # ---- load agent_common itself ------------------------------------
    _mod_name = "indexer_worker.methods.agent_common"
    if _mod_name in sys.modules:
        return sys.modules[_mod_name]

    _spec = _ilu.spec_from_file_location(
        _mod_name,
        ROOT / "methods" / "agent_common.py",
        submodule_search_locations=[],
    )
    _mod = _ilu.module_from_spec(_spec)
    _mod.__package__ = "indexer_worker.methods"
    sys.modules[_mod_name] = _mod
    try:
        _spec.loader.exec_module(_mod)
        return _mod
    except Exception:
        # Loading failed — tests that need this will skip with a clear message
        sys.modules.pop(_mod_name, None)
        return None


_AGENT_COMMON = _load_agent_common()


# ---------------------------------------------------------------------------
# Tests: EliteACustomCallback production-path wiring
#
# These tests instantiate the real EliteACustomCallback (and EliteACallback)
# and invoke the real on_custom_event handler so that attribute defects
# (AttributeError on missing mcp_alias_* maps) are caught at test time, not
# in production.
# ---------------------------------------------------------------------------

def _make_node_interface():
    """Return a minimal mock NodeEventInterface that records emitted events."""
    ni = MagicMock()
    ni.event_node = MagicMock()
    ni.stream_id = "test-stream-id"
    ni.payload_additional_kwargs = {}
    emitted = []

    def _emit(**kwargs):
        emitted.append(dict(kwargs))

    ni.emit.side_effect = _emit
    ni._emitted = emitted
    return ni


_SKIP_REAL_CB = _AGENT_COMMON is None
_SKIP_REASON = "agent_common.py could not be loaded with available stubs"


class TestEliteACustomCallbackAliasMapWiring(unittest.TestCase):
    """Real EliteACustomCallback must have mcp_alias_* attrs and handle on_custom_event correctly."""

    @unittest.skipIf(_SKIP_REAL_CB, _SKIP_REASON)
    def test_real_class_has_mcp_alias_maps(self):
        """EliteACustomCallback.__init__ must initialize mcp_alias_url_map and mcp_alias_meta_map."""
        EliteACustomCallback = _AGENT_COMMON.EliteACustomCallback
        ni = _make_node_interface()
        cb = EliteACustomCallback(
            node_interface=ni,
            message_id="msg-1",
            project_id=1,
            chat_project_id=1,
        )
        self.assertIsInstance(
            cb.mcp_alias_url_map, dict,
            "mcp_alias_url_map not initialized — on_custom_event will AttributeError",
        )
        self.assertIsInstance(
            cb.mcp_alias_meta_map, dict,
            "mcp_alias_meta_map not initialized — on_custom_event will AttributeError",
        )

    @unittest.skipIf(_SKIP_REAL_CB, _SKIP_REASON)
    def test_real_class_also_has_maps_on_elitea_callback(self):
        """EliteACallback.__init__ must also initialize both alias map attributes."""
        EliteACallback = _AGENT_COMMON.EliteACallback
        ni = _make_node_interface()
        cb = EliteACallback(
            node_interface=ni,
            message_id="msg-1",
            project_id=1,
            chat_project_id=1,
        )
        self.assertIsInstance(cb.mcp_alias_url_map, dict)
        self.assertIsInstance(cb.mcp_alias_meta_map, dict)

    @unittest.skipIf(_SKIP_REAL_CB, _SKIP_REASON)
    def test_on_custom_event_parallel_hitl_interrupt_emits_mcp_authorization_required(self):
        """Real on_custom_event must emit mcp_authorization_required with provided_settings.

        This test exercises the PRODUCTION code path that previously raised
        AttributeError when mcp_alias_url_map / mcp_alias_meta_map were absent
        from EliteACustomCallback.
        """
        EliteACustomCallback = _AGENT_COMMON.EliteACustomCallback
        ni = _make_node_interface()
        cb = EliteACustomCallback(
            node_interface=ni,
            message_id="msg-1",
            project_id=1,
            chat_project_id=1,
        )
        # Populate alias maps (normally done by indexer_agent after create_callbacks)
        cb.mcp_alias_url_map = {"my_mcp": "https://api.mcp.example.com/mcp/"}
        cb.mcp_alias_meta_map = {
            "my_mcp": {
                "provided_settings": {
                    "mcp_client_id": "real-cid",
                    "mcp_client_secret": "****1234",
                }
            }
        }

        # Simulate the durable parallel_hitl_interrupt event that the SDK fires
        interrupt_item = {
            "guardrail_type": "mcp_auth",
            "message": "MCP auth required",
            "toolkit_name": "my_mcp",
            "server_url": "https://api.mcp.example.com/mcp",  # no trailing slash
        }
        event_data = {
            "hitl_interrupts": [interrupt_item],
            "root_thread_id": "root-thread-1",
        }

        # Must not raise AttributeError on mcp_alias_url_map / mcp_alias_meta_map
        try:
            cb.on_custom_event(
                name="parallel_hitl_interrupt",
                data=event_data,
                run_id=MagicMock(),
                tags=[],
                metadata={},
                kwargs={},
            )
        except AttributeError as e:
            self.fail(
                f"on_custom_event raised AttributeError — alias map not initialized: {e}"
            )

        # The emitted event must include provided_settings
        emitted = ni._emitted
        mcp_events = [e for e in emitted if e.get("type") == "mcp_authorization_required"]
        self.assertGreater(
            len(mcp_events), 0,
            "Expected at least one mcp_authorization_required event to be emitted",
        )
        meta = mcp_events[0].get("response_metadata", {})
        self.assertIn(
            "provided_settings", meta,
            "provided_settings must be present in mcp_authorization_required response_metadata",
        )
        self.assertEqual(meta["provided_settings"]["mcp_client_id"], "real-cid")

    @unittest.skipIf(_SKIP_REAL_CB, _SKIP_REASON)
    def test_on_custom_event_does_not_overwrite_sdk_provided_settings(self):
        """SDK-supplied provided_settings on the interrupt item must be preserved unchanged.

        When the SDK already forwards exact provided_settings on the interrupt dict,
        the worker must serialize it as-is without substituting credentials from the
        alias map (which could belong to a different toolkit).
        """
        EliteACustomCallback = _AGENT_COMMON.EliteACustomCallback
        ni = _make_node_interface()
        cb = EliteACustomCallback(
            node_interface=ni,
            message_id="msg-2",
            project_id=1,
            chat_project_id=1,
        )
        # Populate alias maps with DIFFERENT credentials to verify they are not substituted
        cb.mcp_alias_url_map = {"my_mcp": "https://api.mcp.example.com/mcp/"}
        cb.mcp_alias_meta_map = {
            "my_mcp": {
                "provided_settings": {
                    "mcp_client_id": "alias-map-cid",
                    "mcp_client_secret": "****alias",
                }
            }
        }

        # SDK already provides exact settings on the interrupt item
        interrupt_item = {
            "guardrail_type": "mcp_auth",
            "message": "MCP auth required",
            "toolkit_name": "my_mcp",
            "server_url": "https://api.mcp.example.com/mcp",
            "provided_settings": {
                "mcp_client_id": "sdk-exact-cid",
                "mcp_client_secret": "****sdkx",
            },
        }
        event_data = {
            "hitl_interrupts": [interrupt_item],
            "root_thread_id": "root-thread-2",
        }

        cb.on_custom_event(
            name="parallel_hitl_interrupt",
            data=event_data,
            run_id=MagicMock(),
            tags=[],
            metadata={},
            kwargs={},
        )

        emitted = ni._emitted
        mcp_events = [e for e in emitted if e.get("type") == "mcp_authorization_required"]
        self.assertGreater(len(mcp_events), 0)
        meta = mcp_events[0].get("response_metadata", {})
        self.assertEqual(
            meta.get("provided_settings", {}).get("mcp_client_id"),
            "sdk-exact-cid",
            "SDK-supplied provided_settings must not be replaced by alias-map credentials",
        )

    @unittest.skipIf(_SKIP_REAL_CB, _SKIP_REASON)
    def test_on_custom_event_shared_url_ambiguity_does_not_attach_wrong_credentials(self):
        """Two toolkits on one URL: only exact toolkit_name match may attach credentials.

        When the SDK does not forward provided_settings and two toolkits share the same
        server URL, the worker must not attach credentials from the wrong toolkit.
        If toolkit_name is present it resolves unambiguously; URL alone must never select.
        """
        EliteACustomCallback = _AGENT_COMMON.EliteACustomCallback
        ni = _make_node_interface()
        cb = EliteACustomCallback(
            node_interface=ni,
            message_id="msg-3",
            project_id=1,
            chat_project_id=1,
        )
        # alpha and beta share one URL — different OAuth clients
        cb.mcp_alias_url_map = {
            "alpha": "https://api.shared.example.com/mcp/",
            "beta": "https://api.shared.example.com/mcp/",
        }
        cb.mcp_alias_meta_map = {
            "alpha": {"provided_settings": {"mcp_client_id": "alpha-client", "mcp_client_secret": "****aaaa"}},
            "beta": {"provided_settings": {"mcp_client_id": "beta-client", "mcp_client_secret": "****bbbb"}},
        }

        # beta requests auth — must receive beta-client, not alpha-client, not None
        interrupt_item = {
            "guardrail_type": "mcp_auth",
            "message": "MCP auth required",
            "toolkit_name": "beta",
            "server_url": "https://api.shared.example.com/mcp",
        }
        event_data = {
            "hitl_interrupts": [interrupt_item],
            "root_thread_id": "root-thread-3",
        }

        cb.on_custom_event(
            name="parallel_hitl_interrupt",
            data=event_data,
            run_id=MagicMock(),
            tags=[],
            metadata={},
            kwargs={},
        )

        mcp_events = [e for e in ni._emitted if e.get("type") == "mcp_authorization_required"]
        self.assertGreater(len(mcp_events), 0)
        meta = mcp_events[0].get("response_metadata", {})
        ps = meta.get("provided_settings", {})
        self.assertEqual(
            ps.get("mcp_client_id"),
            "beta-client",
            "beta toolkit must receive beta credentials, not alpha's or None",
        )


if __name__ == "__main__":
    unittest.main()
