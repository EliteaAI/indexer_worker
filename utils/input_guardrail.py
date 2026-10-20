"""Input guardrail (#6886) plumbing between the SDK middleware and the platform.

The SDK's InputGuardMiddleware dispatches an ``input_guardrail`` custom event with a
summary that holds rule ids and counts, never the detected value. This module merges
those summaries into the message meta and turns their audit entries into spans the
tracing plugin's AuditSpanProcessor writes as ``audit_events`` rows.
"""

from typing import Any, Dict, Iterable, List, Optional

from pylon.core.tools import log

# configurable key the SDK reads to skip segments the platform already inspected
INSPECTED_CONFIG_KEY = 'input_guardrail_inspected'
AUDIT_TRACER_NAME = 'audit-trail'
AUDIT_OBSERVATION_TYPE = 'guardrail'

_SEVERITY = {'none': 0, 'off': 0, 'record': 1, 'warn': 2, 'redact': 3, 'block': 4}
# Keys of the SDK summary that may reach the persisted message and the UI
_META_KEYS = ('action', 'categories', 'rules', 'segments', 'detection_count', 'not_inspected', 'error')


def inspected_segments(kwargs: Dict[str, Any]) -> List[str]:
    """Segment hashes the platform (checkpoints A/B) already inspected this turn."""
    value = kwargs.get(INSPECTED_CONFIG_KEY) or []
    return [str(h) for h in value if isinstance(h, str)] if isinstance(value, list) else []


def merge_verdict(current: Optional[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Fold one model call's summary into the turn's verdict. The most severe action wins."""
    incoming = {key: summary.get(key) for key in _META_KEYS}
    if not current:
        return {
            **incoming,
            'categories': sorted(set(incoming['categories'] or [])),
            'rules': sorted(set(incoming['rules'] or [])),
            'segments': sorted(set(incoming['segments'] or [])),
            'not_inspected': sorted(set(incoming['not_inspected'] or [])),
            'detection_count': int(incoming['detection_count'] or 0),
        }
    merged = dict(current)
    if _SEVERITY.get(incoming['action'] or 'none', 0) > _SEVERITY.get(current.get('action') or 'none', 0):
        merged['action'] = incoming['action']
    for key in ('categories', 'rules', 'segments', 'not_inspected'):
        merged[key] = sorted(set(current.get(key) or []) | set(incoming[key] or []))
    merged['detection_count'] = int(current.get('detection_count') or 0) + int(incoming['detection_count'] or 0)
    merged['error'] = current.get('error') or incoming['error']
    return merged


def emit_audit_spans(
    entries: Iterable[Dict[str, Any]],
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
) -> int:
    """Emit one zero-length audit span per guardrail audit entry; returns how many.

    Spans join the run's current trace, so the row lands next to the agent's own
    llm/tool rows. A failure here must never affect the run.
    """
    entries = [e for e in entries or [] if isinstance(e, dict) and e.get('action')]
    if not entries:
        return 0
    try:
        from opentelemetry import trace  # pylint: disable=C0415
        tracer = trace.get_tracer(AUDIT_TRACER_NAME)
    except Exception:  # pylint: disable=W0718
        return 0
    base = {'audit.observation.type': AUDIT_OBSERVATION_TYPE}
    for key, value in (('user.id', user_id), ('project.id', project_id)):
        try:
            if value is not None:
                base[key] = int(value)
        except (TypeError, ValueError):
            pass
    emitted = 0
    for entry in entries:
        try:
            attrs = {
                **base,
                'audit.guardrail.action': str(entry['action'])[:512],
                'audit.is_error': bool(entry.get('is_error')),
            }
            if entry.get('segment_kind'):
                attrs['audit.tool.name'] = str(entry['segment_kind'])
            tracer.start_span(name=attrs['audit.guardrail.action'], attributes=attrs).end()
            emitted += 1
        except Exception as e:  # pylint: disable=W0718
            log.debug(f"[INPUT_GUARDRAIL] Failed to emit audit span: {e}")
    return emitted
