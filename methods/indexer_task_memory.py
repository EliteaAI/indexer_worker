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

""" Method """

from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web  # pylint: disable=E0611,E0401

from ..utils.task_memory_limit import MIB, limit_task_memory


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def indexer_limit_task_memory(self):
        """ Cap the data memory this task process may allocate """
        try:
            import tasknode_task  # pylint: disable=E0401,C0415
            multiprocessing_context = tasknode_task.multiprocessing_context
        except (ImportError, AttributeError):
            return
        try:
            limit = limit_task_memory(
                self.descriptor.config.get("task_memory_limit_mb"),
                multiprocessing_context,
            )
        except (OSError, ValueError) as error:
            log.warning("Could not limit task memory: %s", error)
            return
        if limit is not None:
            log.info("Task data memory limited to %s MiB", limit // MIB)
