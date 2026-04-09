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

""" Method """

from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web  # pylint: disable=E0611,E0401



class Method:
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def toolkits_request(self, event, payload):
        from ..utils.funcs import dev_reload_sdk
        dev_reload_sdk('elitea_sdk.runtime.toolkits')
        from elitea_sdk.runtime.toolkits.tools import get_toolkits

        schemas = []
        for tk in get_toolkits():
            schema = tk.schema()
            if 'has_function_validators' not in schema.get('metadata', {}):
                total_validators = sum(
                    len(validators)
                    for validators in (
                        tk.__pydantic_decorators__.field_validators,
                        tk.__pydantic_decorators__.model_validators
                    )
                )
                schema.setdefault('metadata', {})['has_function_validators'] = bool(total_validators)
            # mark if check connection is supported
            schema.setdefault('metadata', {})['check_connection_supported'] = hasattr(tk, 'check_connection')
            schemas.append(schema)

        self.agent_event_node.emit('application_toolkits_collected', schemas)
