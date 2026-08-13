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

""" Detect a forked child that inherited a locked glibc resolver mutex (#6245) """

import os
import socket
import threading

from pylon.core.tools import log  # pylint: disable=E0611,E0401


PROBE_FAILED_ERROR = "fork_dns_probe_failed"
PROBE_USER_MESSAGE = "Temporary server error, please try again"

# Three shapes on purpose: a resolvable name, a name that must NXDOMAIN, and a bare
# IP. Each takes a different NSS path, and probing only one lets ~15% of poisoned
# children through (measured).
_PROBE_TARGETS = (
    ("localhost", 80),
    ("elitea-fork-probe.invalid", 80),
    ("127.0.0.1", 0),
)


def probe_dns_usable(timeout: float = 2.0) -> bool:
    """ True if getaddrinfo still works in this process """
    # Runs on a throwaway thread, not under SIGALRM: a Python signal handler never
    # fires on a C-level futex deadlock, but join() on a wedged thread does return,
    # leaving this thread able to report the failure.
    finished = []
    #
    def _probe():
        for host, port in _PROBE_TARGETS:
            try:
                socket.getaddrinfo(host, port)
            except Exception:  # pylint: disable=W0718
                pass  # resolution failing is fine; hanging is not
        #
        finished.append(True)
    #
    thread = threading.Thread(target=_probe, name="fork_dns_probe", daemon=True)
    thread.start()
    thread.join(timeout)
    #
    return bool(finished)


def dns_probe_enabled(descriptor_config: dict) -> bool:
    """ Feature flag, on by default """
    config = descriptor_config.get("fork_dns_probe") or {}
    return config.get("enabled", True)


def dns_probe_timeout(descriptor_config: dict) -> float:
    """ Probe timeout in seconds """
    config = descriptor_config.get("fork_dns_probe") or {}
    return float(config.get("timeout_seconds", 2.0))


def check_fork_dns(descriptor_config: dict, task_id: str = None) -> bool:
    """ True when the child is safe to continue; False when it must abort """
    if not dns_probe_enabled(descriptor_config):
        return True
    #
    if probe_dns_usable(dns_probe_timeout(descriptor_config)):
        return True
    #
    # Cannot be recovered in-process: the mutex was inherited locked and its owner
    # thread does not exist here, so it is held for this child's whole life.
    log.error(
        "DNS is unusable in forked task %s — inherited a locked resolver mutex (#6245). "
        "Aborting before the agent wedges unkillably", task_id,
    )
    #
    # Returning a result only works under the file transport. Every other transport
    # delivers it over the event node, i.e. needs the lookup this process cannot do —
    # so exit instead of wedging in the reply. No UI report, but no orphan either.
    if descriptor_config.get("agents_result_transport", "files") != "files":
        log.error("Result transport is not 'files'; exiting task %s without a result", task_id)
        os._exit(1)  # pylint: disable=W0212
    #
    return False


def build_probe_failed_result(stream_id, message_id, execution_generation=None) -> dict:
    """ Error result for a probe-failed child, read by pylon_main after the child exits """
    # Written to the result file, never emitted: reaching Redis needs a hostname
    # lookup, which is the one thing this process cannot do. stream_id and
    # execution_generation travel here because neither survives in the task meta.
    return {
        "chat_history": [],
        "error": PROBE_FAILED_ERROR,
        "human_readable": PROBE_USER_MESSAGE,
        "fork_dns_probe_failed": True,
        "stream_id": stream_id,
        "message_id": message_id,
        "execution_generation": execution_generation,
    }
