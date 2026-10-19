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

""" Per-task data memory limit """

import resource
from typing import Callable, Optional

MIB = 1024 * 1024
UNLIMITED_CGROUP_VALUE = "max"
DEFAULT_TASK_SHARE_OF_CONTAINER = 0.5
IN_PROCESS_CONTEXT = "threading"
PROC_CGROUP_PATH = "/proc/self/cgroup"
PROC_STATUS_PATH = "/proc/self/status"
CGROUP_ROOT = "/sys/fs/cgroup"


def read_container_memory_limit(proc_cgroup_path: str = PROC_CGROUP_PATH,
                                cgroup_root: str = CGROUP_ROOT) -> Optional[int]:
    try:
        with open(proc_cgroup_path, "r", encoding="utf-8") as cgroup_file:
            unified = next((line.strip().split("::", 1)[1]
                            for line in cgroup_file if line.startswith("0::")), None)
        if unified is None:
            return None
        with open(f"{cgroup_root}{unified.rstrip('/')}/memory.max", "r", encoding="utf-8") as limit_file:
            raw_limit = limit_file.read().strip()
    except OSError:
        return None
    if raw_limit == UNLIMITED_CGROUP_VALUE or not raw_limit.isdigit():
        return None
    return int(raw_limit) or None


def read_process_data_bytes(status_path: str = PROC_STATUS_PATH) -> Optional[int]:
    try:
        with open(status_path, "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmData:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def task_data_budget(configured_mb, container_limit: Optional[int]) -> Optional[int]:
    if configured_mb is not None:
        return int(configured_mb) * MIB if int(configured_mb) > 0 else None
    if container_limit:
        return int(container_limit * DEFAULT_TASK_SHARE_OF_CONTAINER)
    return None


def limit_task_memory(configured_mb, multiprocessing_context: str,
                      container_limit_reader: Callable[[], Optional[int]] = read_container_memory_limit,
                      data_bytes_reader: Callable[[], Optional[int]] = read_process_data_bytes,
                      set_limit: Callable = resource.setrlimit) -> Optional[int]:
    if multiprocessing_context == IN_PROCESS_CONTEXT:
        return None
    budget = task_data_budget(configured_mb, container_limit_reader())
    current = data_bytes_reader()
    if budget is None or current is None:
        return None
    limit = current + budget
    set_limit(resource.RLIMIT_DATA, (limit, limit))
    return limit
