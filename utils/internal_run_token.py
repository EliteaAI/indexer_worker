"""The run token of a governed run (C-1009, EliteaAI/elitea_issues#7214).

Main mints it at admission and puts it on the task payload as one kwarg (``TOKEN_KWARG``): the bearer the run's
internal Elitea MCP requests authenticate with, in place of a personal or system token, and (#7229) the bearer of the
run's whole platform client when Main sent no other credential. The task takes it out of its
kwargs first (``take_run_token``), before anything reads or copies them (child and reconcile payloads are built from
the kwargs and never carry it), keeps it for the duration of the task only (``begin_run_token``), and hangs it on the
run's Elitea client (``attach_run_token``), where the SDK's MCP client sends it to the origins listed with it and to no
other. Nothing here logs it, and its ``repr`` does not show it.
"""

import contextvars
import re
from urllib.parse import urlsplit

TOKEN_KWARG = "internal_run_token"
TOKEN_PREFIX = "eio1."
MAX_TOKEN_CHARS = 4096
MAX_ORIGINS = 4

_CURRENT = contextvars.ContextVar("indexer_internal_run_auth", default=None)


class InternalRunAuth:
    """``token`` for the ``origins`` it may be sent to; ``identity`` names who it acts for (a stable cache key)."""

    __slots__ = ("token", "origins", "identity")

    def __init__(self, token, origins, identity):
        self.token, self.origins, self.identity = token, tuple(origins), identity

    def __repr__(self):
        return f"InternalRunAuth(origins={list(self.origins)})"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _origin(value):
    if not isinstance(value, str):
        return None
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.path not in ("", "/"):
        return None
    return f"{parts.scheme}://{parts.netloc}".lower()


def take_run_token(kwargs):
    """Remove the delivery from ``kwargs``; the ``InternalRunAuth`` it carried, or None (a legacy run, a malformed
    delivery: the internal tools then carry no bearer and Main refuses them)."""
    raw = kwargs.pop(TOKEN_KWARG, None) if isinstance(kwargs, dict) else None
    if not isinstance(raw, dict):
        return None
    token, identity = raw.get("token"), raw.get("identity")
    origins = raw.get("origins")
    if not isinstance(token, str) or not token.startswith(TOKEN_PREFIX) or len(token) > MAX_TOKEN_CHARS:
        return None
    if not isinstance(origins, list) or not origins or len(origins) > MAX_ORIGINS:
        return None
    parsed = [_origin(origin) for origin in origins]
    if any(origin is None for origin in parsed) or not isinstance(identity, str) or not identity:
        return None
    return InternalRunAuth(token, parsed, identity)


def begin_run_token(kwargs):
    """Take the run token out of ``kwargs`` for the duration of the task; the marker ``end_run_token`` takes back.
    Never raises: a delivery that cannot be read is no token."""
    try:
        auth = take_run_token(kwargs)
    except Exception:  # pylint: disable=W0703
        auth = None
    return _CURRENT.set(auth)


def end_run_token(marker):
    _CURRENT.reset(marker)


def _origin_of(url):
    """``scheme://host[:port]`` of a client's base URL (which may carry a path prefix), lower-case; None if unreadable."""
    if not isinstance(url, str):
        return None
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower() if parts.scheme in ("http", "https") and parts.netloc else None


def _has_own_credential(auth_token, auth_session, headers) -> bool:
    return bool(auth_token or auth_session or (isinstance(headers, dict) and any(
        isinstance(key, str) and key.lower() == "authorization" for key in headers)))


def attach_run_token(client):
    """Hang the task's run token (if any) on the run's Elitea client, where the SDK's MCP client finds it. A client
    that Main gave no credential of its own (no personal or system token, no session, no Authorization header: an
    admitted run under enforce, #7229) and that talks to one of the token's origins takes the run token as its bearer
    for every platform call; any other client keeps what it has, unchanged."""
    auth = _CURRENT.get()
    if auth is not None and client is not None:
        client.internal_run_auth = auth
        headers = getattr(client, "headers", None)
        if not _has_own_credential(getattr(client, "auth_token", None), getattr(client, "auth_session", None), headers) \
                and _origin_of(getattr(client, "base_url", None)) in auth.origins and isinstance(headers, dict) \
                and sdk_keeps_run_token_in_worker():
            client.auth_token = auth.token
            headers["Authorization"] = f"Bearer {auth.token}"
    return auth


_SDK_GUARD = {}


def sdk_keeps_run_token_in_worker() -> bool:
    """Does the installed SDK keep a run token that is its client's bearer out of the code it hands the sandbox
    (``is_run_token_bearer``, elitea-sdk #7229)? Without it the run token is never made the bearer (the run then has no
    platform identity and fails, rather than hand its bearer to generated code); said once in the log."""
    if "ok" not in _SDK_GUARD:
        try:
            from elitea_sdk.runtime.utils.internal_run_auth import is_run_token_bearer  # pylint: disable=C0415,E0401,W0611
            _SDK_GUARD["ok"] = True
        except Exception:  # pylint: disable=W0703
            _SDK_GUARD["ok"] = False
            from pylon.core.tools import log  # pylint: disable=C0415,E0401,E0611
            log.error("The installed elitea-sdk cannot keep the run token out of the code sandbox: governed runs get no "
                      "platform identity until the SDK pin with is_run_token_bearer is installed (#7229)")
    return _SDK_GUARD["ok"]


def platform_bearer(api_token, client_args):
    """The bearer of a short-lived platform client the task builds before its run client (the post-fork pgvector
    fallback): the payload's own token, else (#7229) the task's run token when the payload carries no credential and
    the client goes to one of the token's origins; None otherwise."""
    if api_token:
        return api_token
    auth = _CURRENT.get()
    args = client_args if isinstance(client_args, dict) else {}
    base_url = args.get("deployment", args.get("base_url"))
    if auth is None or _has_own_credential(None, args.get("auth_session"), args.get("api_extra_headers")):
        return None
    return auth.token if _origin_of(base_url) in auth.origins else None


# What Main answers a platform call of the run's client on its run token once the run is no longer live (#7229)
_REFUSAL = re.compile(r"\b(CONTEXT_NOT_LIVE|INTERNAL_TOKEN_EXPIRED|INTERNAL_TOKEN_INVALID)\b")
_MAX_CAUSES = 5


def run_token_refusal_code(exc):
    """The refusal code of a platform call made on this task's run token (a model call or a tool's platform call that
    Main refused because the run ended, was revoked or went idle), or None. Only while the task holds a run token; read
    from the error and its causes (an SDK or model client may wrap it)."""
    if _CURRENT.get() is None:
        return None
    for _ in range(_MAX_CAUSES):
        if exc is None:
            break
        found = _REFUSAL.search(f"{exc} {getattr(exc, 'body', '')}")
        if found:
            return found.group(1)
        exc = exc.__cause__ or exc.__context__
    return None
