#!/usr/bin/python3
# coding=utf-8

#   Copyright 2025 EPAM Systems
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

""" Shared LangGraph checkpoint utilities """

import hashlib
import json
from typing import Optional

from pylon.core.tools import log  # pylint: disable=E0611,E0401

from .constants import DEFAULT_MEMORY_CONFIG


def resolve_memory_config(descriptor_config: dict, pgvector_connstr: Optional[str] = None) -> dict:
    """Resolve memory config from descriptor config with optional pgvector override."""
    memory_config = descriptor_config.get("agent_memory_config", None)

    if memory_config is None:
        memory_config = DEFAULT_MEMORY_CONFIG

    if pgvector_connstr and memory_config.get("type") == "postgres":
        memory_config = {
            **memory_config,
            "connection_string": pgvector_connstr.replace("postgresql+psycopg://", "postgresql://")
        }

    return memory_config


def delete_checkpoints_by_thread_ids(memory_config: dict, thread_ids: list):
    """Delete LangGraph checkpoint rows for the given thread_ids from the configured backend."""
    memory_type = memory_config.get("type", "memory")

    if memory_type == "postgres":
        from psycopg import Connection  # pylint: disable=E0401,C0415

        with Connection.connect(
            memory_config["connection_string"],
            **memory_config["connection_kwargs"],
        ) as connection:
            delete_queries = {
                "checkpoint_writes": "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                "checkpoint_blobs": "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                "checkpoints": "DELETE FROM checkpoints WHERE thread_id = %s",
            }

            cursor = connection.cursor()
            try:
                for thread_id in thread_ids:
                    for table, query in delete_queries.items():
                        cursor.execute(query, (thread_id,))
                        log.debug(f"Deleted from {table} where thread_id={thread_id}")
                connection.commit()
            except Exception as e:
                log.error(f"Error deleting checkpoints: {e}")
            finally:
                cursor.close()

    elif memory_type == "sqlite":
        import sqlite3  # pylint: disable=C0415

        db_path = memory_config.get("path", "/data/cache/memory.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                for thread_id in thread_ids:
                    for table in ["checkpoint_writes", "checkpoint_blobs", "checkpoints"]:
                        cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
                        log.debug(f"Deleted from {table} where thread_id={thread_id}")
                conn.commit()
            except Exception as e:
                log.error(f"Error deleting checkpoints from sqlite: {e}")
            finally:
                cursor.close()
    else:
        log.debug(f"Memory type '{memory_type}' does not support checkpoint deletion")


def compute_pipeline_state_hash(version_details: Optional[dict]) -> Optional[str]:
    """Compute an MD5 fingerprint of the pipeline state default values.

    Returns None when there is no pipeline YAML or no state block (e.g. regular
    predict-agents), so callers can safely skip the checkpoint-reset logic.
    """
    import yaml  # pylint: disable=C0415

    instructions = (version_details or {}).get('instructions', '')
    if not instructions:
        return None
    try:
        schema = yaml.safe_load(instructions)
    except Exception:
        return None
    if not isinstance(schema, dict):
        return None
    state = schema.get('state', {})
    if not state:
        return None
    defaults = {
        k: v.get('value', '') if isinstance(v, dict) else ''
        for k, v in state.items()
    }
    return hashlib.md5(
        json.dumps(defaults, sort_keys=True).encode()
    ).hexdigest()


def get_stored_state_hash(memory, thread_id: str) -> Optional[str]:
    """Retrieve the pipeline_state_defaults_hash stored in the latest checkpoint metadata."""
    if memory is None:
        return None
    try:
        config = {'configurable': {'thread_id': thread_id}}
        checkpoint_tuple = memory.get_tuple(config)
        if checkpoint_tuple and checkpoint_tuple.metadata:
            return checkpoint_tuple.metadata.get('pipeline_state_defaults_hash')
    except Exception as e:
        log.warning(f"Could not read checkpoint metadata for thread {thread_id}: {e}")
    return None


def reset_checkpoint_if_state_changed(
    memory,
    thread_id: str,
    current_hash: Optional[str],
    memory_config: dict,
) -> None:
    """Delete the LangGraph checkpoint when pipeline state defaults have changed.

    Only acts when:
    - ``current_hash`` is not None (i.e. the application has a pipeline state block)
    - A previous checkpoint exists with a *different* hash

    This ensures that updated state default values take effect on the very next
    pipeline run, regardless of whether the edit came from the Pipeline page or
    the Chat Canvas.
    """
    if current_hash is None or memory is None:
        return
    stored_hash = get_stored_state_hash(memory, thread_id)
    if stored_hash is not None and stored_hash != current_hash:
        log.info(
            f"Pipeline state defaults changed "
            f"(stored={stored_hash!r} → current={current_hash!r}), "
            f"clearing checkpoint for thread {thread_id}"
        )
        delete_checkpoints_by_thread_ids(memory_config, [thread_id])
