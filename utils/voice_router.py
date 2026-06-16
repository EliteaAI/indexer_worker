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
Global voice event router.

Replaces per-sid Redis pub/sub channels:
  voice_asr_audio_input_{sid}
  voice_asr_stop_{sid}
  voice_tts_cancel_{sid}

with a single ``voice_events`` channel.  The indexer subscribes once at the
time of the first registration; routing to the correct task handler is done
entirely in-process via a (sid, event_type) keyed dict.  Redis subscription
count stays constant regardless of the number of concurrent voice sessions.

Usage (inside a task method)::

    from ..utils.voice_router import register, unregister
    from ..utils.voice_router import ASR_AUDIO_INPUT, ASR_STOP, TTS_CANCEL

    def _on_audio(payload):
        pcm = payload.get("audio", b"")
        ...

    register(event_node, sid, ASR_AUDIO_INPUT, _on_audio)
    register(event_node, sid, ASR_STOP, _on_stop)
    try:
        ...
    finally:
        unregister(sid, ASR_AUDIO_INPUT, ASR_STOP)

Handler signature: ``def my_handler(payload: dict) -> None``
"""

import threading

from pylon.core.tools import log  # pylint: disable=E0611,E0401

# Single channel on which pylon_main emits all voice control events
VOICE_EVENTS_CHANNEL = "voice_events"

# Event type identifiers — must match the values used in elitea_core/sio/asr.py
# and elitea_core/sio/tts.py
ASR_AUDIO_INPUT = "asr_audio_input"
ASR_STOP = "asr_stop"
TTS_CANCEL = "tts_cancel"
TTS_NEXT   = "tts_next"

_lock = threading.Lock()
# Keyed by (sid, event_type) to allow O(1) dispatch without nested dict overhead
_handlers: dict = {}  # {(sid, event_type): callable}
_subscribed = False


def register(event_node, sid: str, event_type: str, handler) -> None:
    """
    Register *handler* for (*sid*, *event_type*) and ensure the single global
    ``voice_events`` subscription exists.

    Thread-safe; safe to call concurrently from multiple task threads.
    """
    global _subscribed  # pylint: disable=W0603
    with _lock:
        if not _subscribed:
            event_node.subscribe(VOICE_EVENTS_CHANNEL, _route)
            _subscribed = True
        _handlers[(sid, event_type)] = handler


def unregister(sid: str, *event_types: str) -> None:
    """
    Remove handlers for the given *event_types* under *sid*.

    Called in task ``finally`` blocks to clean up after a session ends.
    """
    with _lock:
        for et in event_types:
            _handlers.pop((sid, et), None)


def _route(event, payload, *a):
    """Dispatch an incoming voice_events message to the registered handler."""
    sid = payload.get("sid")
    event_type = payload.get("type")
    if not sid or not event_type:
        return
    with _lock:
        handler = _handlers.get((sid, event_type))
    if handler is not None:
        try:
            handler(payload)
        except Exception as exc:  # pylint: disable=W0703
            log.warning(
                "voice_router: handler error sid=%s type=%s: %s",
                sid, event_type, exc,
            )
