"""Pure decisions shared by the nested parallel-dispatch runtime."""


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
