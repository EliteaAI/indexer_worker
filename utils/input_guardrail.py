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
# not_inspected holds client-chosen block types: bound what a verdict carries
_MAX_NOT_INSPECTED = 20
_MAX_BLOCK_TYPE_CHARS = 32
_META_KEYS = ('action', 'categories', 'rules', 'segments', 'detection_count', 'not_inspected', 'error')


# --- tracing value redactor ----------------------------------------------------------
#
# The tracing plugin keeps its value redactor per process, and this worker runs the SDK
# (and so the spans carrying user_input / chat_history) in its own process, so the redactor
# the platform plugin registered in pylon_main does not reach it. The policy is the one this
# module already pushes into the SDK from its own config (``_apply_input_guardrail``, also on
# reconfig); agent tasks fork from this process and inherit it. The parsed policy is held
# here, so a span payload leaf costs one global read while the guardrail is off.

_redaction = None  # {'policy': strict Policy, 'inspect': fn, 'segment': class} while active


def set_redaction_policy(raw: Optional[Dict[str, Any]]) -> bool:
    """Parse the ``input_guardrail`` config block for span redaction; True when it is active.

    An unreadable or inactive block clears it, matching ``configure_input_guard``.
    Never logs the block's content.
    """
    global _redaction  # pylint: disable=W0603
    _redaction = None
    try:
        from dataclasses import replace  # pylint: disable=C0415
        from elitea_sdk.runtime.guardrails.input_detector import (  # pylint: disable=C0415,E0401
            Segment, inspect, policy_from_dict,
        )
        if not isinstance(raw, dict):
            return False
        policy = policy_from_dict(raw)
        if not policy.active:
            return False
        # Every detected value is replaced, whatever the configured action: this copy
        # is never meant to hold one
        strict = replace(policy, categories={
            category: ('redact' if action != 'off' else 'off')
            for category, action in policy.categories.items()
        })
        _redaction = {'policy': strict, 'inspect': inspect, 'segment': Segment}
        return True
    except Exception:  # pylint: disable=W0718
        log.debug("[INPUT_GUARDRAIL] Span redaction policy unavailable")
        return False


def redact_span_text(text: str) -> Optional[str]:
    """Value redactor registered with the tracing plugin.

    Returns the text with detected values replaced, unchanged while the guardrail is off,
    and None when the inspection failed (oversize, timeout, internal): the tracing plugin
    then masks the whole value rather than keep a raw one.
    """
    state = _redaction
    if state is None or not isinstance(text, str) or not text:
        return text
    try:
        result = state['inspect']([state['segment']('span', 0, text)], state['policy'])
    except Exception:  # pylint: disable=W0718
        return None
    if result.error:
        return None
    return result.redacted.get(('span', 0), text)


def register_tracing_redactor() -> bool:
    """Hand ``redact_span_text`` to the tracing plugin of this process; True when registered."""
    try:
        from tools import this  # pylint: disable=C0415,E0401
        this.for_module("tracing").module.register_value_redactor(redact_span_text)
        return True
    except Exception as e:  # pylint: disable=W0718
        log.debug(f"[INPUT_GUARDRAIL] Tracing redactor not registered: {e}")
        return False


def inspected_segments(kwargs: Dict[str, Any]) -> List[str]:
    """Segment hashes the platform (checkpoints A/B) already inspected this turn."""
    value = kwargs.get(INSPECTED_CONFIG_KEY) or []
    return [str(h) for h in value if isinstance(h, str)] if isinstance(value, list) else []


def _bounded_types(values) -> List[str]:
    return sorted({str(v)[:_MAX_BLOCK_TYPE_CHARS] for v in values or []})[:_MAX_NOT_INSPECTED]


def merge_verdict(current: Optional[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Fold one model call's summary into the turn's verdict. The most severe action wins."""
    incoming = {key: summary.get(key) for key in _META_KEYS}
    if not current:
        return {
            **incoming,
            'categories': sorted(set(incoming['categories'] or [])),
            'rules': sorted(set(incoming['rules'] or [])),
            'segments': sorted(set(incoming['segments'] or [])),
            'not_inspected': _bounded_types(incoming['not_inspected']),
            'detection_count': int(incoming['detection_count'] or 0),
        }
    merged = dict(current)
    if _SEVERITY.get(incoming['action'] or 'none', 0) > _SEVERITY.get(current.get('action') or 'none', 0):
        merged['action'] = incoming['action']
    for key in ('categories', 'rules', 'segments'):
        merged[key] = sorted(set(current.get(key) or []) | set(incoming[key] or []))
    merged['not_inspected'] = _bounded_types(set(current.get('not_inspected') or []) | set(incoming['not_inspected'] or []))
    merged['detection_count'] = int(current.get('detection_count') or 0) + int(incoming['detection_count'] or 0)
    merged['error'] = current.get('error') or incoming['error']
    return merged


def audit_blocked(exc: Exception, recorded: set, user_id=None, project_id=None, entity=None) -> int:
    """Audit an SDK block from the exception's own summary (#6886).

    A sub-agent's ``input_guardrail`` event can miss the parent callback (the child
    runs with its own config), so the block itself is the reliable record. Entries
    the callback already audited this run are skipped.
    """
    summary = getattr(exc, 'summary', None) or {}
    entries = [
        e for e in (summary.get('audit') or [])
        if isinstance(e, dict) and e.get('action') and _audit_key(e) not in recorded
    ]
    recorded.update(_audit_key(e) for e in entries)
    return emit_audit_spans(entries, user_id=user_id, project_id=project_id, entity=entity)


def _audit_key(entry: Dict[str, Any]) -> tuple:
    return str(entry.get('action')), str(entry.get('segment_kind'))


def _entity_attributes(entity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``entity.*`` span attributes for the audit row; empty unless type and id are known."""
    if not isinstance(entity, dict) or not entity.get('type') or entity.get('id') in (None, ''):
        return {}
    attrs = {'entity.type': str(entity['type'])[:32], 'entity.id': str(entity['id'])}
    if entity.get('name'):
        attrs['entity.name'] = str(entity['name'])[:256]
    return attrs


def emit_audit_spans(
    entries: Iterable[Dict[str, Any]],
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    entity: Optional[Dict[str, Any]] = None,
) -> int:
    """Emit one zero-length audit span per guardrail audit entry; returns how many.

    Spans join the run's current trace, so the row lands next to the agent's own
    llm/tool rows. ``entity`` is the run's agent (``type``/``id``/``name``, as
    ``entity_from_application`` builds it); it becomes the ``entity.*`` attributes the
    audit row reads, so the row names the agent. A failure here must never affect the run.
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
    base.update(_entity_attributes(entity))
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
