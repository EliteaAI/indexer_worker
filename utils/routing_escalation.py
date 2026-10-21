"""Auto-routing escalation signal carried from the platform to the agent run (#6722)."""
from typing import Any, Dict, Optional

ROUTING_ESCALATION_CONFIG_KEY = 'elitea_routing_escalation'


def routing_escalation(value: Any) -> Optional[Dict[str, Any]]:
    """Return ``{event, from_message_id}`` when ``value`` is a well-formed signal, else None.

    Well-formed means a dict with a non-empty string ``event``. The event vocabulary is the
    router's concern; a malformed value is omitted from the run config rather than failing the run.
    """
    if not isinstance(value, dict):
        return None
    event = value.get('event')
    if not isinstance(event, str) or not event:
        return None
    from_message_id = value.get('from_message_id')
    return {
        'event': event,
        'from_message_id': from_message_id if isinstance(from_message_id, str) else None,
    }
