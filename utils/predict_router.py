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
Mid-turn user input injection router.

pylon_main emits injection events on a single ``predict_events`` channel while
an agent turn is running. The indexer subscribes once and routes each event
into the in-process SDK injection registry keyed by thread_id; the running turn
drains it at the next tool-call loop boundary.

The registry lives in SDK memory in the same interpreter as the running loop, so
delivery needs no cross-process drain. Under ``fork`` the arbiter EventNode clone
delivers into the forked child, whose callback thread writes that child's own
registry — which the same child's loop then drains.

Two signals go back to the UI: ``injection_ready`` once this turn is listening
(the UI only offers the affordance after it, which removes the registration
race), and the consumed injection_ids at turn end (authoritative — anything
absent was never folded in and the UI re-sends it as a normal predict).
"""

import threading

from pylon.core.tools import log  # pylint: disable=E0611,E0401

from elitea_sdk.runtime import _injection_registry as _registry
from .node_interface import EventTypes

# Single channel on which pylon_main emits all mid-turn injection events
PREDICT_EVENTS_CHANNEL = "predict_events"

# Event type identifier — must match the value emitted by elitea_core
INJECT = "inject"

_lock = threading.Lock()
_subscribed = False


def register(event_node, thread_id: str, node_interface=None, task_meta=None) -> None:
    """Mark *thread_id* active for injections and ensure the global subscription.

    Non-interactive predicts (scheduled, webhook, blocking REST) have no UI to
    inject from, so they neither register nor pay the event fan-out cost.
    """
    if task_meta and task_meta.get("non_interactive"):
        return
    global _subscribed  # pylint: disable=W0603
    with _lock:
        if not _subscribed:
            event_node.subscribe(PREDICT_EVENTS_CHANNEL, _route)
            _subscribed = True
    _registry.register(thread_id)
    log.info("predict_router: registered thread=%s", thread_id)

    if node_interface is not None:
        node_interface.emit(
            type=EventTypes.injection_ready,
            response_metadata={'thread_id': thread_id},
        )


def unregister(thread_id: str) -> None:
    """Clear injection state for a finished turn."""
    _registry.unregister(thread_id)


def report_consumed(node_interface, thread_id: str) -> None:
    """Emit the injection_ids this turn folded in, so the UI can reconcile.

    Silence is meaningful: an id the UI sent but does not see here was never
    consumed. Emitted even when empty for exactly that reason.
    """
    # Runs before unregister(), so is_active tells us whether this turn opted in
    # at all — non-interactive predicts never register and need no report.
    if not thread_id or node_interface is None or not _registry.is_active(thread_id):
        return
    consumed = _registry.consumed(thread_id)
    node_interface.emit(
        type=EventTypes.injection_consumed_report,
        response_metadata={'thread_id': thread_id, 'consumed': consumed},
    )
    log.info("predict_router: turn end thread=%s consumed=%s", thread_id, consumed)


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
