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
Mid-turn user input injection router (Phase 0 POC).

pylon_main emits injection events on a single ``predict_events`` channel while
an agent turn is running. The indexer subscribes once and routes each event
into the in-process SDK injection registry keyed by thread_id; the running turn
drains it at the next tool-call loop boundary.

POC scope: in-process (threading runtime) only — the registry lives in SDK
memory shared with the worker thread. Fork/durable delivery is Phase 1.
"""

import threading

from pylon.core.tools import log  # pylint: disable=E0611,E0401

from elitea_sdk.runtime import _injection_registry as _registry

# Single channel on which pylon_main emits all mid-turn injection events
PREDICT_EVENTS_CHANNEL = "predict_events"

# Event type identifier — must match the value emitted by elitea_core
INJECT = "inject"

_lock = threading.Lock()
_subscribed = False


def register(event_node, thread_id: str) -> None:
    """Mark *thread_id* active for injections and ensure the global subscription."""
    global _subscribed  # pylint: disable=W0603
    with _lock:
        if not _subscribed:
            event_node.subscribe(PREDICT_EVENTS_CHANNEL, _route)
            _subscribed = True
    _registry.register(thread_id)
    log.info("predict_router: registered thread=%s (registry id=%s)",
             thread_id, id(_registry))


def unregister(thread_id: str) -> None:
    """Clear injection state for a finished turn."""
    _registry.unregister(thread_id)


def _route(event, payload, *a):
    """Push an incoming inject event into the SDK injection registry."""
    if not isinstance(payload, dict) or payload.get("type") != INJECT:
        return
    thread_id = payload.get("thread_id")
    text = payload.get("text")
    injection_id = payload.get("injection_id")
    if not thread_id or not text:
        return
    if not _registry.is_active(thread_id):
        return
    accepted = _registry.push(thread_id, text, injection_id=injection_id)
    log.info(
        "predict_router: inject thread=%s id=%s accepted=%s",
        thread_id, injection_id, accepted,
    )
