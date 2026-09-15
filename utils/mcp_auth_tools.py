#!/usr/bin/python3
# coding=utf-8

#   Copyright 2024 EPAM Systems
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
MCP authorization tool factories.

Provides the StructuredTool objects injected into the agent when MCP toolkits
require OAuth authorization, and the helper functions they depend on.
"""

from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel as PydanticBaseModel

from pylon.core.tools import log

from elitea_sdk.runtime.utils.mcp_oauth import (
    McpAuthorizationRequired,
    build_mcp_auth_decision_result,
)

from .funcs import (
    _is_http_url,
    _is_unresolved_mcp_type,
    normalize_mcp_server_url,
    get_mcp_server_settings,
    is_mcp_authorization_required_error,
)


def _mcp_discovery_url(server_url: str) -> str:
    """Return the MCP endpoint URL to use for tool discovery/auth challenge."""
    if not _is_http_url(server_url):
        return server_url
    # Use the URL as-is. normalize_mcp_server_url already converts the deprecated
    # /v1/sse to /v1/mcp/authv2 before this function is called, so we do not
    # reverse that mapping here.
    return server_url.strip()


def build_provided_settings_from_mcp_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Build the provided_settings dict from a raw MCP toolkit settings dict.

    Handles both plain string secrets and Pydantic SecretStr values.
    Returns an empty dict when no relevant fields are present.
    """
    from .funcs import mask_secret

    result: Dict[str, Any] = {}
    client_id = settings.get("client_id")
    if client_id:
        result["mcp_client_id"] = client_id

    client_secret = settings.get("client_secret")
    if client_secret:
        secret_val = (
            client_secret.get_secret_value()
            if hasattr(client_secret, "get_secret_value")
            else str(client_secret)
        )
        result["mcp_client_secret"] = mask_secret(secret_val)

    scopes = settings.get("scopes")
    if scopes:
        result["scopes"] = scopes

    headers = settings.get("headers") or {}
    if isinstance(headers, dict):
        auth_key = next((k for k in headers if k.lower() == "authorization"), None)
        if auth_key:
            result["has_pat"] = True

    return result


def _build_mcp_server_alias_map(tool_configs: list) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Build alias maps for MCP URL resolution and toolkit metadata lookup."""
    from .funcs import _extract_mcp_server_url, normalize_mcp_toolkit_type

    alias_map: Dict[str, str] = {}
    alias_meta_map: Dict[str, Dict[str, Any]] = {}

    def _register(
        alias: Optional[str],
        url: Optional[str],
        toolkit_type: Optional[str],
        toolkit_name: Optional[str],
        provided_settings: Optional[Dict[str, Any]] = None,
        toolkit_id: Optional[Any] = None,
    ) -> None:
        if not alias or not url or not _is_http_url(url):
            return
        key = alias.strip().lower()
        alias_map[key] = normalize_mcp_server_url(url)
        entry = {
            **alias_meta_map.get(key, {}),
            "toolkit_type": toolkit_type,
            "toolkit_name": toolkit_name or alias,
            "tool_name": toolkit_name or alias,
        }
        if provided_settings:
            entry["provided_settings"] = provided_settings
        if toolkit_id is not None:
            entry["toolkit_id"] = toolkit_id
        alias_meta_map[key] = entry

    for tool in tool_configs or []:
        if not isinstance(tool, dict):
            continue

        tool_type = str(tool.get("type") or "").strip()
        if not (tool_type == "mcp" or tool_type == "mcp_config" or tool_type.startswith("mcp_")):
            continue

        settings = tool.get("settings") if isinstance(tool.get("settings"), dict) else {}
        server_name = str(settings.get("server_name") or "").strip()
        toolkit_type = normalize_mcp_toolkit_type(tool_type, server_name)
        toolkit_name = str(tool.get("toolkit_name") or tool.get("name") or server_name or toolkit_type).strip()
        if _is_http_url(toolkit_name) and server_name:
            toolkit_name = server_name
        aliases = {
            toolkit_name,
            server_name,
            tool_type,
            toolkit_type,
        }
        if tool_type.startswith("mcp_") and tool_type != "mcp_config":
            aliases.add(tool_type[4:])

        _provided = build_provided_settings_from_mcp_settings(settings)

        # Extract the raw Authorization header value separately — used by
        # _mcp_auth_control to pass the PAT to discover_mcp_tools directly.
        _headers = settings.get("headers") or {}
        _pat_headers: Optional[Dict[str, str]] = None
        if isinstance(_headers, dict):
            _auth_key = next((k for k in _headers if k.lower() == "authorization"), None)
            if _auth_key:
                _pat_headers = {_auth_key: _headers[_auth_key]}

        _provided_settings = _provided if _provided else None

        _toolkit_id = tool.get("toolkit_id") or tool.get("id")

        def _register_with_pat(
            alias: Optional[str],
            url: Optional[str],
            tk_type: Optional[str],
            tk_name: Optional[str],
            ps: Optional[Dict[str, Any]] = None,
        ) -> None:
            _register(alias, url, tk_type, tk_name, ps, toolkit_id=_toolkit_id)
            if _pat_headers and alias and url and _is_http_url(url):
                key = alias.strip().lower()
                if key in alias_meta_map:
                    alias_meta_map[key]["_auth_headers"] = _pat_headers

        direct_url = _extract_mcp_server_url(settings)
        if _is_http_url(direct_url):
            for alias in aliases:
                _register_with_pat(alias, direct_url, toolkit_type, toolkit_name, _provided_settings)
            continue

        for alias in list(aliases):
            if not alias:
                continue
            server_cfg = get_mcp_server_settings(alias) or {}
            cfg_url = _extract_mcp_server_url(server_cfg)
            _register_with_pat(alias, cfg_url, toolkit_type, toolkit_name, _provided_settings)

    return alias_map, alias_meta_map


def _enrich_mcp_auth_exc(
    exc: Any,
    normalized_url: Optional[str],
    resolved_tool_name: str,
    meta: Dict[str, Any],
) -> None:
    """Populate missing fields on a McpAuthorizationRequired exception from discovery meta.

    Called in both the direct `except McpAuthorizationRequired` catch and the
    dev-reload-safe `except Exception / is_mcp_authorization_required_error` path so
    the enrichment logic is not duplicated.
    """
    if normalized_url:
        setattr(exc, "server_url", normalized_url)
    if not getattr(exc, "tool_name", None) or _is_http_url(getattr(exc, "tool_name", None)):
        setattr(exc, "tool_name", resolved_tool_name)
    if _is_unresolved_mcp_type(getattr(exc, "toolkit_type", None)):
        setattr(exc, "toolkit_type", meta.get("toolkit_type"))
    if not getattr(exc, "toolkit_name", None):
        setattr(exc, "toolkit_name", meta.get("toolkit_name") or resolved_tool_name)
    if not getattr(exc, "provided_settings", None):
        _ps = meta.get("provided_settings")
        if _ps:
            setattr(exc, "provided_settings", _ps)


def backfill_mcp_provided_settings(
    exc: Any,
    alias_url_map: Optional[Dict[str, str]],
    alias_meta_map: Optional[Dict[str, Any]],
) -> None:
    """Attach provided_settings to exc in-place when it is not already set.

    Resolution order:
    1. Exact toolkit_id — when the SDK forwards toolkit_id on the exception and
       an alias entry carries the same id, use those credentials directly.  Generic
       alias keys (e.g. "mcp") share the same toolkit_id as the real entry, so this
       finds the right entry without counting aliases.
    2. Exact toolkit-name match — toolkit_name normalized to strip().lower() matches
       an alias key that is NOT a generic type alias (not "mcp" / "mcp_config" / a bare
       tool_type value).  Safe even when multiple toolkits share a server URL.
    3. URL match deduplicated by toolkit identity — collect all alias keys whose URL
       matches, then deduplicate by toolkit_id (when stored) or toolkit_name.  Resolves
       only when all matched keys belong to a single unique toolkit identity.
    """
    if getattr(exc, 'provided_settings', None):
        return
    if not alias_meta_map:
        return

    # Strategy 1: exact toolkit_id forwarded by the release SDK
    exc_toolkit_id = getattr(exc, 'toolkit_id', None)
    if exc_toolkit_id is not None:
        for _key, _meta in alias_meta_map.items():
            if (_meta or {}).get('toolkit_id') == exc_toolkit_id:
                ps = _meta.get('provided_settings')
                if ps:
                    exc.provided_settings = ps
                return  # found the identity; stop regardless of ps

    # Strategy 2: exact toolkit-name lookup (normalized key), skipping generic aliases
    _GENERIC_ALIASES = {"mcp", "mcp_config"}
    exc_toolkit = getattr(exc, 'toolkit_name', None)
    if exc_toolkit:
        _toolkit_key = exc_toolkit.strip().lower()
        if _toolkit_key in alias_meta_map and _toolkit_key not in _GENERIC_ALIASES:
            ps = (alias_meta_map.get(_toolkit_key) or {}).get('provided_settings')
            if ps:
                exc.provided_settings = ps
            return

    # Strategy 3: URL match, deduplicated by toolkit identity
    exc_url = getattr(exc, 'server_url', None)
    if exc_url and alias_url_map:
        _normalized = normalize_mcp_server_url(exc_url).rstrip("/")
        matched_keys = [
            alias_key for alias_key, registered_url in alias_url_map.items()
            if normalize_mcp_server_url(registered_url).rstrip("/") == _normalized
        ]
        # Deduplicate by toolkit identity: prefer toolkit_id, fall back to toolkit_name
        seen_identities: set = set()
        unique_keys: List[str] = []
        for k in matched_keys:
            _m = alias_meta_map.get(k) or {}
            identity = _m.get('toolkit_id') if _m.get('toolkit_id') is not None else _m.get('toolkit_name', k)
            if identity not in seen_identities:
                seen_identities.add(identity)
                unique_keys.append(k)
        if len(unique_keys) == 1:
            ps = (alias_meta_map.get(unique_keys[0]) or {}).get('provided_settings')
            if ps:
                exc.provided_settings = ps


def backfill_mcp_provided_settings_dict(
    item: Dict[str, Any],
    alias_url_map: Optional[Dict[str, str]],
    alias_meta_map: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return item (or a shallow copy) with provided_settings added when absent.

    Used for plain dicts (e.g. durable interrupt items) where in-place attribute
    mutation is not possible.  Returns the original dict if no backfill is needed.
    Same three-strategy resolution as backfill_mcp_provided_settings.
    """
    if item.get('provided_settings') or not alias_meta_map:
        return item

    _GENERIC_ALIASES = {"mcp", "mcp_config"}

    # Strategy 1: exact toolkit_id
    item_toolkit_id = item.get('toolkit_id')
    if item_toolkit_id is not None:
        for _key, _meta in alias_meta_map.items():
            if (_meta or {}).get('toolkit_id') == item_toolkit_id:
                ps = _meta.get('provided_settings')
                return {**item, 'provided_settings': ps} if ps else item
        return item  # found identity, no credentials → stop

    # Strategy 2: exact toolkit-name lookup, skipping generic aliases
    toolkit_name = item.get('toolkit_name')
    if toolkit_name:
        _toolkit_key = toolkit_name.strip().lower()
        if _toolkit_key in alias_meta_map and _toolkit_key not in _GENERIC_ALIASES:
            ps = (alias_meta_map.get(_toolkit_key) or {}).get('provided_settings')
            return {**item, 'provided_settings': ps} if ps else item

    # Strategy 3: URL match, deduplicated by toolkit identity
    server_url = item.get('server_url')
    if server_url and alias_url_map:
        _normalized = normalize_mcp_server_url(server_url).rstrip("/")
        matched_keys = [
            alias_key for alias_key, registered_url in alias_url_map.items()
            if normalize_mcp_server_url(registered_url).rstrip("/") == _normalized
        ]
        seen_identities: set = set()
        unique_keys: List[str] = []
        for k in matched_keys:
            _m = alias_meta_map.get(k) or {}
            identity = _m.get('toolkit_id') if _m.get('toolkit_id') is not None else _m.get('toolkit_name', k)
            if identity not in seen_identities:
                seen_identities.add(identity)
                unique_keys.append(k)
        if len(unique_keys) == 1:
            ps = (alias_meta_map.get(unique_keys[0]) or {}).get('provided_settings')
            if ps:
                return {**item, 'provided_settings': ps}
    return item


def _make_mcp_auth_tools(
    declined_servers: list,
    tool_configs: Optional[list] = None,
    mcp_tokens: Optional[Dict[str, Any]] = None,
) -> List[StructuredTool]:
    """Create MCP auth tools aligned with sensitive-tool style guidance.

    Exposes:
    - `mcp_auth_control`: primary control tool with structured decision output
    - `request_mcp_authorization`: compatibility alias for legacy prompts/checkpoints
    """
    server_metadata: Dict[str, Dict[str, Any]] = {
        normalize_mcp_server_url(s["server_url"]): {
            **s,
            "server_url": normalize_mcp_server_url(s.get("server_url")),
        }
        for s in declined_servers
        if s.get("server_url")
    }
    alias_to_server_url, alias_to_tool_meta = _build_mcp_server_alias_map(tool_configs or [])

    def _decline_reason(meta: Dict[str, Any]) -> str:
        reason = str(meta.get("skip_reason") or meta.get("denial_reason") or "").strip()
        return reason or "user skipped MCP login for this run"

    server_list = "\n".join(
        (
            f"- {s.get('server_url', '')} ({s.get('tool_name', '')}) "
            f"reason: {_decline_reason(s)}"
        )
        for s in declined_servers
    )

    class _McpAuthControlInput(PydanticBaseModel):
        action: str = "authorize"
        server_url: Optional[str] = None
        tool_name: Optional[str] = None
        reason: Optional[str] = None

    class _RequestMcpAuthorizationInput(PydanticBaseModel):
        server_url: str

    def _alias_candidates(server_url: str, tool_name: Optional[str]) -> list:
        candidates = [server_url, tool_name or ""]
        normalized_tool_name = (tool_name or "").strip()
        if normalized_tool_name.startswith("mcp_authorize_"):
            candidates.append(normalized_tool_name[len("mcp_authorize_"):])
        return [candidate for candidate in candidates if str(candidate).strip()]

    def _resolve_server_meta(server_url: str, tool_name: Optional[str] = None) -> tuple:
        """Resolve (url, metadata) with exact match then fuzzy fallback."""
        server_url = (server_url or "").strip()
        if server_url in server_metadata:
            return server_url, server_metadata[server_url]

        for alias in _alias_candidates(server_url, tool_name):
            alias_key = str(alias).strip().lower()
            resolved_url = alias_to_server_url.get(alias_key)
            if resolved_url:
                resolved_meta = dict(alias_to_tool_meta.get(alias_key) or {})
                if not resolved_meta.get("tool_name"):
                    resolved_meta["tool_name"] = tool_name or alias
                return resolved_url, resolved_meta

        if _is_http_url(server_url):
            resolved_meta: Dict[str, Any] = {}
            for alias in _alias_candidates("", tool_name):
                alias_key = str(alias).strip().lower()
                if alias_key in alias_to_tool_meta:
                    resolved_meta = dict(alias_to_tool_meta.get(alias_key) or {})
                    break
            if not resolved_meta.get("toolkit_type"):
                # Reverse-lookup: find the alias whose registered URL matches server_url.
                # This handles prebuild MCPs where the LLM passes the real OAuth server URL
                # (e.g. https://api.github.com) instead of the symbolic alias (e.g. 'github').
                # Strip trailing slash on both sides: canonical_resource() (used by SDK to set
                # exc.server_url) strips trailing slash, while alias_map stores the raw settings
                # URL which may retain it.
                normalized_input = normalize_mcp_server_url(server_url).rstrip("/")
                for alias_key, registered_url in alias_to_server_url.items():
                    if normalize_mcp_server_url(registered_url).rstrip("/") == normalized_input:
                        alias_meta = alias_to_tool_meta.get(alias_key) or {}
                        if alias_meta.get("toolkit_type"):
                            resolved_meta = {**alias_meta, **resolved_meta}
                            break
            if not resolved_meta.get("tool_name"):
                resolved_meta["tool_name"] = tool_name or server_url
            return normalize_mcp_server_url(server_url), resolved_meta

        # Last-chance resolution for symbolic server aliases (for example
        # "atlassian3") that may not be present in tool-config alias map.
        # Query global MCP server settings directly and extract an HTTP URL.
        from .funcs import _extract_mcp_server_url
        fallback_cfg = get_mcp_server_settings(server_url) or {}
        fallback_url = _extract_mcp_server_url(fallback_cfg)
        if _is_http_url(fallback_url):
            fallback_meta: Dict[str, Any] = {}
            for alias in _alias_candidates(server_url, tool_name):
                alias_key = str(alias).strip().lower()
                if alias_key in alias_to_tool_meta:
                    fallback_meta = dict(alias_to_tool_meta.get(alias_key) or {})
                    break
            if not fallback_meta.get("tool_name"):
                fallback_meta["tool_name"] = tool_name or server_url
            return normalize_mcp_server_url(fallback_url), fallback_meta

        needle = server_url.lower()
        tool_needle = (tool_name or "").lower().strip()
        for key, val in server_metadata.items():
            candidate_tool = (val.get("tool_name") or "").lower()
            key_l = key.lower()
            if needle and (needle in key_l or key_l in needle):
                return key, val
            if needle and candidate_tool and (needle in candidate_tool or candidate_tool in needle):
                return key, val
            if tool_needle and candidate_tool and (tool_needle in candidate_tool or candidate_tool in tool_needle):
                return key, val

        return server_url, {}

    def _mcp_auth_control(
        action: str = "authorize",
        server_url: Optional[str] = None,
        tool_name: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        normalized_action = (action or "authorize").strip().lower()
        resolved_url, meta = _resolve_server_meta(server_url or "", tool_name)
        resolved_tool_name = meta.get("tool_name") or (tool_name or "")
        toolkit_type = meta.get("toolkit_type") or "mcp"

        if normalized_action == "authorize":
            if not resolved_url:
                return build_mcp_auth_decision_result(
                    status="error",
                    server_url="",
                    tool_name=resolved_tool_name,
                    toolkit_type=toolkit_type,
                    message="No MCP server was provided. Choose a valid server and retry authorization.",
                    next_step="ask_user",
                    denial_reason=reason or None,
                )

            # Early return if the server was already declined in this conversation.
            # Do NOT call discover_mcp_tools — that would raise McpAuthorizationRequired
            # again and trigger another auth dialog even though the user already skipped.
            if resolved_url in server_metadata:
                decline_reason = _decline_reason(meta)
                return build_mcp_auth_decision_result(
                    status="declined",
                    server_url=resolved_url,
                    tool_name=resolved_tool_name,
                    toolkit_type=toolkit_type,
                    message=(
                        "This MCP server was already skipped in this conversation. "
                        f"Skip reason: {decline_reason}. "
                        "Do NOT request authorization again. "
                        "Continue the task without this capability, or explain to the user "
                        "why the task cannot be completed."
                    ),
                    next_step="use_other_tool",
                    denial_reason=decline_reason,
                    resource_metadata_url=meta.get("resource_metadata_url"),
                    www_authenticate=meta.get("www_authenticate"),
                )

            # Follow the same MCP auth discovery process used by other flows
            # (sync tools / toolkit discovery) so metadata shape is consistent.
            from .funcs import dev_reload_sdk
            dev_reload_sdk('elitea_sdk.runtime.utils')
            from elitea_sdk.runtime.utils.mcp_tools_discovery import discover_mcp_tools

            normalized_url = normalize_mcp_server_url(resolved_url)
            discovery_url = _mcp_discovery_url(normalized_url)

            # Extract OAuth token from mcp_tokens if available for this server.
            # mcp_tokens is already expanded with URL aliases by expand_mcp_token_aliases.
            _auth_headers: dict = {}
            _token_session_id = None
            if mcp_tokens:
                for _key in [k for k in [normalized_url, discovery_url] if k]:
                    _token_data = mcp_tokens.get(_key)
                    if _token_data:
                        if isinstance(_token_data, dict):
                            _access_token = _token_data.get('access_token')
                            _token_session_id = _token_data.get('session_id')
                        else:
                            _access_token = _token_data
                        if _access_token:
                            _auth_headers['Authorization'] = f'Bearer {_access_token}'
                        break

            # Fall back to pre-configured static headers (PAT / API key) when no OAuth
            # token is present. This ensures discover_mcp_tools uses the PAT instead of
            # running unauthenticated and triggering the OAuth modal unnecessarily.
            if not _auth_headers:
                _pat = meta.get("_auth_headers")
                if _pat:
                    _auth_headers = dict(_pat)

            try:
                discover_mcp_tools(
                    url=discovery_url,
                    headers=_auth_headers,
                    timeout=30,
                    session_id=_token_session_id,
                    ssl_verify=True,
                )
            except McpAuthorizationRequired as exc:
                _enrich_mcp_auth_exc(exc, normalized_url, resolved_tool_name, meta)
                raise
            except Exception as exc:
                if is_mcp_authorization_required_error(exc):
                    _enrich_mcp_auth_exc(exc, normalized_url, resolved_tool_name, meta)
                    raise
                log.warning("MCP auth discovery failed for %s: %s", normalized_url, exc)
                return build_mcp_auth_decision_result(
                    status="error",
                    server_url=normalized_url,
                    tool_name=resolved_tool_name,
                    toolkit_type=meta.get("toolkit_type") or toolkit_type,
                    message=(
                        "Unable to start MCP authorization flow. "
                        f"Details: {str(exc)}"
                    ),
                    next_step="ask_user",
                    denial_reason=reason or None,
                )

            # Discovery succeeded. If we injected an OAuth token, that means the
            # token was accepted — real MCP tools are now loaded. Direct the LLM
            # to proceed with the original tool call instead of skipping it.
            if _auth_headers.get("Authorization"):
                return build_mcp_auth_decision_result(
                    status="authorized",
                    server_url=normalized_url,
                    tool_name=resolved_tool_name,
                    toolkit_type=meta.get("toolkit_type") or toolkit_type,
                    message=(
                        "MCP authorization succeeded. The server is accessible with the provided credentials. "
                        f"Proceed immediately with the original tool call using the {resolved_tool_name or toolkit_type} "
                        "tools that are now available."
                    ),
                    next_step="use_mcp_tool",
                    denial_reason=reason or None,
                )
            # No token was injected — server did not require auth.
            return build_mcp_auth_decision_result(
                status="not_needed",
                server_url=normalized_url,
                tool_name=resolved_tool_name,
                toolkit_type=meta.get("toolkit_type") or toolkit_type,
                message="MCP authorization challenge was not required for this server.",
                next_step="respond_without_tool",
                denial_reason=reason or None,
            )

        if normalized_action == "status":
            status = "declined" if resolved_url in server_metadata else "not_needed"
            next_step = "use_other_tool" if status == "declined" else "respond_without_tool"
            decline_reason = _decline_reason(meta) if status == "declined" else None
            message = (
                "This MCP server was already skipped in this conversation. "
                f"Skip reason: {decline_reason}. "
                "Do NOT request authorization again. "
                "Continue the task without this capability, or explain to the user "
                "why the task cannot be completed."
                if status == "declined"
                else "No declined MCP auth state is tracked for this server in the current conversation."
            )
            return build_mcp_auth_decision_result(
                status=status,
                server_url=resolved_url or (server_url or ""),
                tool_name=resolved_tool_name,
                toolkit_type=toolkit_type,
                message=message,
                next_step=next_step,
                denial_reason=reason or decline_reason,
                resource_metadata_url=meta.get("resource_metadata_url"),
                www_authenticate=meta.get("www_authenticate"),
            )

        if normalized_action == "explain_skip":
            decline_reason = reason or _decline_reason(meta)
            return build_mcp_auth_decision_result(
                status="skipped",
                server_url=resolved_url or (server_url or ""),
                tool_name=resolved_tool_name,
                toolkit_type=toolkit_type,
                message=(
                    "Authorization was declined for THIS invocation. Do not retry the same blocked call now. "
                    "Continue with other available tools. If no alternative remains and the task fails, "
                    f"explicitly mention this skip reason: {decline_reason}."
                ),
                next_step="use_other_tool",
                denial_reason=decline_reason,
                resource_metadata_url=meta.get("resource_metadata_url"),
                www_authenticate=meta.get("www_authenticate"),
            )

        return build_mcp_auth_decision_result(
            status="error",
            server_url=resolved_url or (server_url or ""),
            tool_name=resolved_tool_name,
            toolkit_type=toolkit_type,
            message=f"Unsupported action '{normalized_action}'. Use one of: authorize, status, explain_skip.",
            next_step="ask_user",
            denial_reason=reason or None,
        )

    def _request_mcp_authorization(server_url: str) -> str:
        # Legacy compatibility alias: same behavior as mcp_auth_control(action='authorize').
        return _mcp_auth_control(action="authorize", server_url=server_url)

    mcp_auth_control_tool = StructuredTool.from_function(
        func=_mcp_auth_control,
        name="mcp_auth_control",
        description=(
            "Control MCP authorization flow with structured decisions. "
            "Use action='authorize' when an MCP capability is required and user auth is needed. "
            "Use action='status' to check declined state in this conversation. "
            "Use action='explain_skip' to generate structured skip guidance and continue with alternatives.\n"
            "This is NOT a stop signal by itself: if auth is not granted, continue with other tools when possible.\n"
            "If a required MCP capability was skipped and no alternative works, the assistant response must "
            "explicitly state the skip reason.\n"
            f"Known declined servers in this conversation:\n{server_list or '- (none)'}"
        ),
        args_schema=_McpAuthControlInput,
        handle_tool_error=False,
    )

    legacy_alias_tool = StructuredTool.from_function(
        func=_request_mcp_authorization,
        name="request_mcp_authorization",
        description=(
            "Legacy alias for MCP authorization. Prefer mcp_auth_control(action='authorize', server_url=...). "
            "Calling this tool requests MCP OAuth authorization and shows the user auth dialog."
        ),
        args_schema=_RequestMcpAuthorizationInput,
        handle_tool_error=False,
    )

    return [mcp_auth_control_tool, legacy_alias_tool]


def _has_mcp_toolkits(tool_configs: list) -> bool:
    """Return True when application tool configs include MCP toolkits."""
    for tool in tool_configs or []:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "")
        if tool_type == "mcp" or tool_type == "mcp_config" or tool_type.startswith("mcp_"):
            return True
    return False
