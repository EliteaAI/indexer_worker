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


def attach_run_token(client):
    """Hang the task's run token (if any) on the run's Elitea client, where the SDK's MCP client finds it. A client
    that Main gave no credential of its own (no personal or system token, no session: an admitted run under enforce,
    #7229) takes the run token as its bearer for every platform call; one that has a credential keeps it unchanged."""
    auth = _CURRENT.get()
    if auth is not None and client is not None:
        client.internal_run_auth = auth
        if not getattr(client, "auth_token", None) and not getattr(client, "auth_session", None):
            client.auth_token = auth.token
            if isinstance(getattr(client, "headers", None), dict):
                client.headers["Authorization"] = f"Bearer {auth.token}"
    return auth
