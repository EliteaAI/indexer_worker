#!/usr/bin/python3
# coding=utf-8

"""Where the three voice methods send their audio calls.

WHY THIS EXISTS. `indexer_asr_whisper`, `indexer_tts` and `indexer_asr_realtime`
each held an absolute URL for a LiteLLM proxy running inside this container on
127.0.0.1:8081. The platform removed LiteLLM: the LLM data plane is now
`elitea-llm-gateway`, reached through `elitea-main` at `<deployment>/llm/v1`,
and that gateway serves `/audio/speech`, `/audio/transcriptions` and
`/realtime`.

The local proxy is not merely redundant there. It is a SECOND LLM data plane: it
applies no budget, bills nothing, and its model registry was written by
`runtime_interface_litellm` in pylon_main. A deployment without pylon_main does
not write that registry, so the local proxy answers every call from an empty
model set.

THE SWITCH DEFAULTS TO OFF, AND THAT IS DELIBERATE. A hybrid deployment still
runs pylon_main, still serves the `litellm_resolve_model` RPC and still fills the
local registry, so audio WORKS there today. Repointing unconditionally would
break the one deployment where it currently works. Set `audio_llm_base_url` to
turn it on, per deployment.

TWO THINGS MUST STILL LAND BEFORE THE SWITCH CAN BE TURNED ON. Neither is in
this repository, and neither is fixed by changing a URL:

  1. NOTHING CALLS THESE METHODS ON THE GO PLATFORM. The callers are pylon_main
     socket.io handlers (elitea_core/sio/asr.py, elitea_core/sio/tts.py). The Go
     platform has no equivalent dispatch: elitea-main answers 501 for
     /configurations/tts_voices and starts no ASR or TTS task.
  2. THE TOKEN IS THE WRONG KIND. `project_llm_key` is a per-project LiteLLM
     virtual key, read from the project vault by the `litellm_resolve_model` RPC
     that `runtime_interface_litellm` serves. The gateway sits behind
     elitea-main's authentication, which does not accept it. A caller on the Go
     platform must pass a platform credential instead.

So this module makes the indexer half correct and reversible. It does not, on
its own, make audio work on the Go platform.
"""

from urllib.parse import urlsplit, urlunsplit

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611

from tools import this  # pylint: disable=E0401


# The pre-existing local LiteLLM proxy. It stays the default so a hybrid
# deployment is unchanged by this commit.
_LEGACY_BASE_URL = "http://127.0.0.1:8081/v1"


def audio_base_url() -> str:
    """Return the base URL the voice methods post to, without a trailing slash."""
    configured = ""
    try:
        configured = this.descriptor.config.get("audio_llm_base_url", "") or ""
    except Exception:  # pylint: disable=W0703
        # A method can run before the descriptor is attached in some test
        # harnesses. Falling back to the legacy proxy keeps that path working.
        log.debug("audio_backend: no module descriptor; using the legacy proxy")
    return (configured or _LEGACY_BASE_URL).rstrip("/")


def uses_platform_gateway() -> bool:
    """Report whether audio calls go to the platform gateway rather than LiteLLM."""
    return audio_base_url() != _LEGACY_BASE_URL


def audio_model_name(project_id: int, model_name: str) -> str:
    """Return the model name to put on the wire.

    LiteLLM registered every managed model under a project-prefixed
    ``{project_id}_{name}`` group, so that prefix IS the addressable name there.

    The gateway resolves a model against the project's own configuration rows,
    by the row's `elitea_title` or by its `data.name`. It never carries the
    prefix, so sending the prefixed form asks for a model no row defines and
    earns a 404 `model_not_found`.
    """
    if uses_platform_gateway():
        return model_name
    return f"{project_id}_{model_name}"


def audio_headers(project_id: int, project_llm_key: str) -> dict:
    """Return the headers for one audio call.

    `X-Project-Id` is the part that is easy to miss. The gateway bills and
    resolves credentials per project, and elitea-main picks the project from
    this header (or from `OpenAI-Organization`). The LangChain clients the SDK
    uses send it for free through `openai_organization`; these three methods
    build their requests by hand, so nothing sends it unless it is set here.
    Without it the edge falls back to the CALLER's personal project, which bills
    the wrong project and cannot see the model the real project configured.
    """
    headers = {"Authorization": f"Bearer {project_llm_key}"}
    if uses_platform_gateway():
        headers["X-Project-Id"] = str(project_id)
    return headers


def audio_ws_url(path_and_query: str) -> str:
    """Return a WebSocket URL for `path_and_query` under the audio base URL.

    The scheme is DERIVED, never assumed: a platform deployment is https, and
    dialling ws:// at an https origin fails the upgrade. http maps to ws and
    https maps to wss.
    """
    parts = urlsplit(audio_base_url())
    scheme = "wss" if parts.scheme == "https" else "ws"
    base_path = parts.path.rstrip("/")
    if "?" in path_and_query:
        path, query = path_and_query.split("?", 1)
    else:
        path, query = path_and_query, ""
    return urlunsplit((scheme, parts.netloc, base_path + path, query, ""))
