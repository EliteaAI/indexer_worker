"""Route durable parallel-HITL decisions into the active SDK supervisor."""

import threading

from pylon.core.tools import log  # pylint: disable=E0611,E0401

from elitea_sdk.runtime import _parallel_hitl_registry as _registry
from .node_interface import EventTypes


PREDICT_EVENTS_CHANNEL = "predict_events"
OFFER = "parallel_hitl_decision_offer"
COMMIT = "parallel_hitl_decision_commit"

_lock = threading.Lock()
_subscribed = False
_routes = {}


def register(event_node, thread_id: str, node_interface=None, task_meta=None) -> None:
    """Register one interactive root turn for supervised decisions."""
    if not thread_id or (task_meta and task_meta.get("non_interactive")):
        return
    global _subscribed  # pylint: disable=W0603
    with _lock:
        if not _subscribed:
            event_node.subscribe(PREDICT_EVENTS_CHANNEL, _route)
            _subscribed = True
        _routes[thread_id] = {
            "node_interface": node_interface,
            "chat_project_id": (task_meta or {}).get("chat_project_id"),
        }
    _registry.register(thread_id)
    if node_interface is not None:
        node_interface.emit(
            type=EventTypes.parallel_hitl_ready,
            response_metadata={
                "root_thread_id": thread_id,
                "chat_project_id": (task_meta or {}).get("chat_project_id"),
            },
        )


def unregister(thread_id: str) -> None:
    if not thread_id:
        return
    with _lock:
        _routes.pop(thread_id, None)
    _registry.unregister(thread_id)


def _emit_ack(thread_id, payload, phase, accepted):
    with _lock:
        route = dict(_routes.get(thread_id) or {})
    node_interface = route.get("node_interface")
    if node_interface is None:
        return
    node_interface.emit(
        type=EventTypes.parallel_hitl_decision_ack,
        response_metadata={
            "root_thread_id": thread_id,
            "thread_id": thread_id,
            "decision_id": payload.get("decision_id"),
            "interrupt_id": payload.get("interrupt_id"),
            "phase": phase,
            "accepted": bool(accepted),
            "chat_project_id": route.get("chat_project_id"),
        },
    )


def _route(event, payload, *args):
    del event, args
    if not isinstance(payload, dict):
        return
    event_type = payload.get("type")
    if event_type not in {OFFER, COMMIT}:
        return
    thread_id = payload.get("root_thread_id") or payload.get("thread_id")
    decision = payload.get("decision")
    if not thread_id or not isinstance(decision, dict):
        return
    if event_type == OFFER:
        accepted = _registry.offer(thread_id, decision)
        _emit_ack(thread_id, decision, "offered", accepted)
    else:
        accepted = _registry.commit(thread_id, decision)
        _emit_ack(thread_id, decision, "committed", accepted)
    log.info(
        "parallel_hitl_router: %s thread=%s decision=%s accepted=%s",
        event_type, thread_id, decision.get("decision_id"), accepted,
    )
