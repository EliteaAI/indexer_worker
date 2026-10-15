#!/usr/bin/python3
# coding=utf-8

#   Copyright 2024 EPAM Systems
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

""" Method for advertising the task capabilities of this worker (C-1009, #6940) """

from pylon.core.tools import web  # pylint: disable=E0611,E0401

from ..utils.envelope_guard import CAPABILITIES_EVENT, capabilities_payload


class Method:
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def task_node_capabilities_request(self, event, payload):
        """ Announce what the agent task node supports (on request and on heartbeat) """
        _ = event, payload
        self.agent_event_node.emit(
            CAPABILITIES_EVENT,
            capabilities_payload(self.agent_task_node.ident, self.agent_task_node.pool),
        )
