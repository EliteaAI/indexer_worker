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

""" Method: Checkpoint management (cleanup and deletion) """

from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web  # pylint: disable=E0611,E0401

from datetime import datetime, timedelta, timezone

from ..utils.checkpoint_utils import resolve_memory_config, delete_checkpoints_by_thread_ids


class Method:
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def empty_agent_state(self, event, payload):
        """
        Removes old checkpoints and associated records from related tables based on thread_id.

        Kwargs:
            days (int): The number of days to retain checkpoints. Older checkpoints will be deleted.
        """
        self.indexer_enable_logging()

        log.debug(f'indexer_empty_agent_state start {payload=}')
        days_to_retain = payload.get("days_to_retain", 1)
        pgvector_connstr = payload.get("pgvector_connstr")
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_to_retain)).isoformat()

        try:
            memory_config = resolve_memory_config(self.descriptor.config, pgvector_connstr)
            memory_type = memory_config.get("type", "memory")

            if memory_type == "postgres":
                from psycopg import Connection  # pylint: disable=E0401,C0415
                from langgraph.checkpoint.postgres import PostgresSaver  # pylint: disable=E0401,C0415

                with Connection.connect(
                    memory_config["connection_string"],
                    **memory_config["connection_kwargs"],
                ) as connection:
                    try:
                        memory = PostgresSaver(connection)
                        memory.setup()
                    except Exception as e:
                        log.warning(f"Failed to setup PostgresSaver tables: {e}")
                        return

                    select_query = """
                        SELECT * FROM checkpoints
                        WHERE (checkpoint ->> 'ts')::timestamptz <= %s;
                    """

                    try:
                        cursor = connection.cursor()
                        cursor.execute(select_query, (cutoff_date,))
                        rows = cursor.fetchall()

                        thread_ids = []
                        for row in rows:
                            thread_ids.append(row[0])

                        if thread_ids:
                            delete_checkpoints_by_thread_ids(memory_config, thread_ids)
                            log.debug(f"Cleaned up {len(thread_ids)} old checkpoints from pgvector database")
                    except Exception as e:
                        log.error(f"Error while removing old checkpoints: {e}")
                    finally:
                        cursor.close()
        except:
            log.exception("indexer_empty_agent_state failed to start")
            raise

    @web.method()
    def delete_checkpoint(self, event, payload):
        """
        Delete checkpoint data for specific thread_ids.

        Called when chat history is cleared to ensure the LangGraph checkpoint
        is also removed, preventing stale state (including old SystemMessages)
        from leaking into subsequent conversations.

        Kwargs:
            thread_ids (list[str]): List of thread IDs whose checkpoints should be deleted.
            pgvector_connstr (str): Optional PostgreSQL connection string override.
        """
        self.indexer_enable_logging()

        thread_ids = payload.get("thread_ids", [])
        pgvector_connstr = payload.get("pgvector_connstr")

        if not thread_ids:
            log.debug("indexer_delete_checkpoint: no thread_ids provided, skipping")
            return

        log.info(f"indexer_delete_checkpoint: deleting checkpoints for {len(thread_ids)} thread(s): {thread_ids}")

        try:
            memory_config = resolve_memory_config(self.descriptor.config, pgvector_connstr)
            delete_checkpoints_by_thread_ids(memory_config, thread_ids)
            log.info(f"indexer_delete_checkpoint: successfully deleted checkpoints for {thread_ids}")
        except Exception:
            log.exception("indexer_delete_checkpoint failed")
