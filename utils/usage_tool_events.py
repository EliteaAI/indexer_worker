#!/usr/bin/python3
# coding=utf-8

#   Copyright 2026 EPAM Systems
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

"""Tool-call usage rows written straight to `centry.usage_event` (#6572).

Billing accuracy is not observability accuracy: a usage row must not ride the
lossy OTEL/Langfuse span pipeline, and must appear even with tracing fully
disabled. So this module is gated by its own `usage.mode` plugin config and
imports nothing from the `tracing` plugin.

Writing to the table directly (rather than an RPC) follows
`tracing/utils/model_pricing.py`: pylon_indexer has no cross-pylon RPC
transport and no `from tools import db`, so only the POSTGRES_* env vars are
available. Unlike that module this one writes, hence the fork guard on the
engine and the ON CONFLICT clause.
"""

import base64
import hashlib
import hmac
import json
import os
import threading
from datetime import datetime, timezone

from elitea_sdk.runtime.utils.utils import PREDICT_RUN_ID_KWARGS_KEY

from pylon.core.tools import log

MODE_OFF = "off"
MODE_OBSERVE = "observe"
MODE_ENFORCE = "enforce"
_KNOWN_MODES = (MODE_OFF, MODE_OBSERVE, MODE_ENFORCE)

EVENT_TYPE_TOOL = "tool"
ENTITY_TYPE_APPLICATION = "application"
ENTITY_TYPE_EVALUATION = "evaluation"

#: Root entity of the run, propagated to sub-agent children. Indexer-internal:
#: the indexer builds the child payload and pylon_main replays it verbatim, so
#: unlike PREDICT_RUN_ID_KWARGS_KEY this needs no shared SDK constant.
ROOT_ENTITY_KWARGS_KEY = "_elitea_root_entity"

#: Explicit leaf entity, for a caller (an eval/judge run, #6677) whose leaf identity is not
#: the `application` dict — e.g. it wants entity_type='evaluation' while root stays the
#: application/pipeline being evaluated. Same convention as ROOT_ENTITY_KWARGS_KEY.
ENTITY_KWARGS_KEY = "_elitea_entity"

#: Carries run_attribution() to the LLM interfaces in pylon_main, which write the
#: llm rows of the same run. A literal on both sides, like PREDICT_RUN_ID_HEADER
#: before the SDK held it: the interfaces are separate repos.
ATTRIBUTION_HEADER = "X-Elitea-Attribution"

#: usage_counter's convention: a NULL cannot take part in a primary key, so
#: "no individual user" is a sentinel. usage_event.user_id is NOT NULL too.
SYSTEM_USER_ID = 0

# `or` not a get() default: the var is present-but-empty in some deployments.
_SCHEMA = os.environ.get("POSTGRES_SCHEMA") or "centry"

_lock = threading.Lock()
_engine = None
_engine_pid = None
_write_errors_seen = set()

_INSERT_SQL = f"""
INSERT INTO {_SCHEMA}.usage_event (
    idempotency_key, ts, project_id, user_id, user_email,
    run_id, conversation_id,
    root_entity_type, root_entity_id, root_entity_version_id,
    entity_type, entity_id, entity_version_id, entity_name,
    event_type, tool_name, duration_ms, is_error, meta
) VALUES (
    :idempotency_key, :ts, :project_id, :user_id, :user_email,
    :run_id, :conversation_id,
    :root_entity_type, :root_entity_id, :root_entity_version_id,
    :entity_type, :entity_id, :entity_version_id, :entity_name,
    :event_type, :tool_name, :duration_ms, :is_error, CAST(:meta AS jsonb)
)
ON CONFLICT (idempotency_key, ts) DO NOTHING
"""


def normalize_mode(value):
    """Anything unrecognised means off — usage must never fail open by accident."""
    if not value:
        return MODE_OBSERVE
    mode = str(value).strip().lower()
    if mode not in _KNOWN_MODES:
        log.warning("usage: unknown mode %r, treating as observe", value)
        return MODE_OBSERVE
    return mode


def usage_mode(plugin_config):
    """Current mode from the plugin's own `usage.mode` key."""
    usage_config = (plugin_config or {}).get("usage") or {}
    if not isinstance(usage_config, dict):
        return MODE_OFF
    return normalize_mode(usage_config.get("mode"))


def enabled(plugin_config):
    """True when tool rows should be written. Rows are written in observe too."""
    return usage_mode(plugin_config) != MODE_OFF


def _get_engine():
    """Process-local write engine, rebuilt after a fork.

    A forked worker must not inherit the parent's socket, and indexer workers
    are forked — hence the pid guard rather than model_pricing's dispose-in-parent.
    """
    global _engine, _engine_pid  # pylint: disable=W0603
    pid = os.getpid()
    if _engine is not None and _engine_pid == pid:
        return _engine
    with _lock:
        if _engine is not None and _engine_pid == pid:
            return _engine
        from sqlalchemy import create_engine  # pylint: disable=C0415
        from sqlalchemy.engine import URL  # pylint: disable=C0415
        url = URL.create(
            "postgresql",
            username=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=os.environ["POSTGRES_DB"],
        )
        _engine = create_engine(url, pool_size=1, max_overflow=1, pool_pre_ping=True)
        _engine_pid = pid
        return _engine


def entity_from_application(application):
    """`{'type', 'id', 'version_id'}` for a saved Application, else None.

    A raw-LLM predict ships `application: {'instructions': ...}` with no ids —
    there is no entity to attribute, so every entity_* column stays NULL.
    """
    if not isinstance(application, dict):
        return None
    application_id = application.get("id")
    if not application_id:
        return None
    return {
        "type": ENTITY_TYPE_APPLICATION,
        "id": application_id,
        "version_id": application.get("version_id"),
        "name": application.get("name"),
    }


def run_attribution(kwargs):
    """Which conversation and which entity, from the task payload alone.

    Shared with the llm rows: these same columns ride to the interfaces in
    ATTRIBUTION_HEADER, so both row kinds of one run are attributed identically.
    """
    kwargs = kwargs or {}

    entity = kwargs.get(ENTITY_KWARGS_KEY)
    if not isinstance(entity, dict) or not entity.get("id"):
        entity = entity_from_application(kwargs.get("application")) or {}
    root = kwargs.get(ROOT_ENTITY_KWARGS_KEY)
    if not isinstance(root, dict) or not root.get("id"):
        root = entity

    return {
        "conversation_id": kwargs.get("conversation_id"),
        "entity_type": entity.get("type"),
        "entity_id": entity.get("id"),
        "entity_version_id": entity.get("version_id"),
        "entity_name": entity.get("name"),
        "root_entity_type": root.get("type"),
        "root_entity_id": root.get("id"),
        "root_entity_version_id": root.get("version_id"),
    }


#: Field of the header JSON holding its signature; the reader trusts nothing unsigned (#6762)
ATTRIBUTION_SIGNATURE_KEY = "sig"
_SIGNING_CONTEXT = b"usage-attribution-v1"


def attribution_signing_key():
    """Derived from the indexer event-node key, so the event bus key itself never signs a header."""
    try:
        from tools import worker_core  # pylint: disable=C0415,E0401
        base_key = (worker_core.descriptor.config.get("event_node") or {}).get("hmac_key")
    except Exception:  # pylint: disable=W0703
        base_key = None
    if not base_key:
        return None
    return hmac.new(str(base_key).encode("utf-8"), _SIGNING_CONTEXT, hashlib.sha256).digest()


def sign_attribution(columns, project_id, key):
    """Hex HMAC over the billed project and the canonical JSON; the reader recomputes it."""
    canonical = json.dumps(columns, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    message = f"{project_id}\n{canonical}".encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def attribution_header(kwargs, project_id=None, key=None):
    """ATTRIBUTION_HEADER value, or None when the payload names nothing to attribute.

    base64url'd compact JSON, signed for the billed project; unsigned when no key is configured.
    """
    columns = {key: value for key, value in run_attribution(kwargs).items() if value is not None}
    if not columns:
        return None
    try:
        if key and project_id:
            columns[ATTRIBUTION_SIGNATURE_KEY] = sign_attribution(columns, project_id, key)
        packed = json.dumps(columns, separators=(",", ":"), ensure_ascii=False)
        return base64.urlsafe_b64encode(packed.encode("utf-8")).decode("ascii").rstrip("=")
    except Exception as exc:  # pylint: disable=W0703
        log.warning("usage: cannot encode run attribution: %s", exc)
        return None


def build_attribution(kwargs, task_meta, task_id):
    """Everything that is constant for one run, resolved once at task entry."""
    task_meta = task_meta or {}
    user_context = task_meta.get("user_context") or {}

    return {
        "task_id": task_id,
        "project_id": task_meta.get("project_id"),
        "user_id": user_context.get("user_id") or SYSTEM_USER_ID,
        "user_email": user_context.get("user_email"),
        # stamp_predict_run_id sets both the payload key and meta; either serves
        "run_id": (kwargs or {}).get(PREDICT_RUN_ID_KWARGS_KEY)
                  or task_meta.get("platform_run_id"),
        **run_attribution(kwargs),
    }


def record_tool_event(attribution, tool_name, duration_ms, is_error, lc_run_id, meta=None):
    """Write one tool row. Never raises: a usage write must not fail a user's run."""
    if not attribution or not attribution.get("project_id"):
        return
    try:
        from sqlalchemy import text  # pylint: disable=C0415

        ts = datetime.now(tz=timezone.utc)
        correlation = attribution.get("run_id") or attribution.get("task_id")
        params = {
            "idempotency_key": f"tool:{correlation}:{lc_run_id}",
            "ts": ts,
            "project_id": attribution["project_id"],
            "user_id": attribution.get("user_id") or SYSTEM_USER_ID,
            "user_email": attribution.get("user_email"),
            "run_id": attribution.get("run_id"),
            "conversation_id": attribution.get("conversation_id"),
            "root_entity_type": attribution.get("root_entity_type"),
            "root_entity_id": attribution.get("root_entity_id"),
            "root_entity_version_id": attribution.get("root_entity_version_id"),
            "entity_type": attribution.get("entity_type"),
            "entity_id": attribution.get("entity_id"),
            "entity_version_id": attribution.get("entity_version_id"),
            "entity_name": attribution.get("entity_name"),
            "event_type": EVENT_TYPE_TOOL,
            "tool_name": (tool_name or "")[:256] or None,
            "duration_ms": duration_ms,
            "is_error": bool(is_error),
            "meta": json.dumps(meta) if meta else None,
        }
        engine = _get_engine()
        with engine.connect() as connection:
            connection.execute(text(_INSERT_SQL), params)
            connection.commit()
    except Exception as exc:  # pylint: disable=W0703
        # Dedup per unique message so a missing partition or a schema-drift
        # deploy cannot turn every tool call into a log line.
        key = str(exc)[:200]
        if key not in _write_errors_seen:
            _write_errors_seen.add(key)
            log.warning("usage: failed to write tool usage event: %s", exc)
        else:
            log.debug("usage: repeat tool usage write failure: %s", exc)
