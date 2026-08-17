"""Pure decisions shared by the nested parallel-dispatch runtime."""


def is_fanout_child(task_meta):
    """Whether task metadata identifies a parked parallel fan-out child."""
    return bool(
        isinstance(task_meta, dict)
        and task_meta.get('child_thread_id')
        and task_meta.get('parent_thread_id')
    )


def durable_dispatch_allowed(enabled, is_child, agent_type):
    """Only a top-level non-pipeline agent may use park+dispatch."""
    return bool(enabled and not is_child and agent_type != 'pipeline')


def normalize_hitl_pause(hitl_interrupt, hitl_interrupts):
    """Return a compatible singular plus the complete plural pause list."""
    plural = hitl_interrupts if isinstance(hitl_interrupts, list) else []
    if not plural and hitl_interrupt:
        plural = [hitl_interrupt]
    singular = hitl_interrupt or (plural[0] if plural else None)
    return singular, plural


def split_mcp_auth_interrupts(hitl_interrupt, hitl_interrupts):
    """Partition durable authorization guards from ordinary HITL prompts."""
    _, plural = normalize_hitl_pause(hitl_interrupt, hitl_interrupts)
    auth = [
        item for item in plural
        if isinstance(item, dict) and item.get('guardrail_type') == 'mcp_auth'
    ]
    review = [
        item for item in plural
        if not (isinstance(item, dict) and item.get('guardrail_type') == 'mcp_auth')
    ]
    return auth, (review[0] if review else None), review


def has_mcp_auth_interrupt(response):
    """Return whether an SDK response paused on a durable auth guard."""
    if not isinstance(response, dict):
        return False
    auth, _, _ = split_mcp_auth_interrupts(
        response.get('hitl_interrupt'), response.get('hitl_interrupts'),
    )
    return bool(auth)
