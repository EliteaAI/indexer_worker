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

""" Parent-side warmup of the pgvector connstr cache, so forked children inherit it (#6245) """

import queue
import threading

from pylon.core.tools import log  # pylint: disable=E0611,E0401


_warm_queue: queue.Queue = queue.Queue(maxsize=256)
_warm_thread = None
_warm_lock = threading.Lock()
_requested: set = set()


def _warm_worker():
    from ..methods.agent_common import (  # pylint: disable=C0415
        _fetch_pgvector_connstr_with_retry,
        temp_elitea_client,
    )
    #
    while True:
        project_id, client_args, api_token, api_extra_headers = _warm_queue.get()
        #
        try:
            with temp_elitea_client(client_args, api_token, api_extra_headers) as client:
                _fetch_pgvector_connstr_with_retry(client, project_id=project_id)
        except Exception as exc:  # pylint: disable=W0718
            # Leave it uncached: the child falls back to its own fetch.
            log.warning("pgvector_connstr warm failed for project_id=%s: %s", project_id, exc)
        finally:
            # Always clear: the retry helper can return None without caching, and
            # that project must stay retryable rather than be pinned forever.
            with _warm_lock:
                _requested.discard(project_id)


def request_warm(project_id, client_args, api_token, api_extra_headers):
    """ Queue a parent-side connstr fetch. Returns immediately, never raises """
    # Never blocks the caller: it runs on an eventnode callback thread (one per
    # event in spawner mode), and the SDK vault call has no HTTP timeout (#6246).
    global _warm_thread  # pylint: disable=W0603
    #
    if project_id is None:
        return
    #
    try:
        from ..methods.agent_common import _pgvector_connstr_cache  # pylint: disable=C0415
        if project_id in _pgvector_connstr_cache:
            return
        #
        with _warm_lock:
            if project_id in _requested:
                return
            _requested.add(project_id)
            #
            if _warm_thread is None:
                _warm_thread = threading.Thread(
                    target=_warm_worker, name="pgvector_connstr_warm", daemon=True,
                )
                _warm_thread.start()
        #
        _warm_queue.put_nowait((project_id, client_args, api_token, api_extra_headers))
    except queue.Full:
        with _warm_lock:
            _requested.discard(project_id)
    except Exception as exc:  # pylint: disable=W0718
        log.warning("pgvector_connstr warm request failed for project_id=%s: %s", project_id, exc)


def agent_task_approver(descriptor_config):
    """ Arbiter task approver that warms the cache before fork; always approves """
    # Called from on_start_query, which (unlike on_start_request) does not hold
    # start_task_rlock — so a slow warm here cannot gate the whole pool.
    from ..methods.agent_common import pgvector_connstr_needed  # pylint: disable=C0415
    #
    def _approver(event_name, event_payload):
        _ = event_name
        #
        try:
            if not pgvector_connstr_needed(descriptor_config):
                return True
            #
            kwargs = event_payload.get("kwargs") or {}
            client_args = (kwargs.get("llm") or {}).get("kwargs") or {}
            #
            request_warm(
                client_args.get("project_id"),
                client_args,
                kwargs.get("api_token", client_args.get("api_key")),
                kwargs.get("api_extra_headers", client_args.get("api_extra_headers", {})),
            )
        except Exception as exc:  # pylint: disable=W0718
            log.warning("pgvector_connstr warm approver failed: %s", exc)
        #
        return True
    #
    return _approver
