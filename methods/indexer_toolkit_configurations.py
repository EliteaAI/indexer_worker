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
    def toolkit_configurations_request(self, event, payload):
        from ..utils.funcs import dev_reload_sdk
        dev_reload_sdk('elitea_sdk.configurations')
        dev_reload_sdk('elitea_sdk.runtime.toolkits.mcp_config')
        from elitea_sdk.configurations import get_class_configurations
        from elitea_sdk.runtime.toolkits.mcp_config import get_mcp_config_toolkit_schemas
        schemas = {}
        for config_type, config in get_class_configurations().items():
            schema = config.model_json_schema()
            schema.setdefault('metadata', {})['check_connection_supported'] = hasattr(config, 'check_connection')
            schemas[config_type] = schema

        # Add MCP config toolkit schemas (pre-configured MCP servers from config)
        for mcp_config in get_mcp_config_toolkit_schemas():
            schema = mcp_config.model_json_schema()
            schema.setdefault('metadata', {})['check_connection_supported'] = hasattr(mcp_config, 'check_connection')
            schemas[mcp_config.__name__] = schema

        self.agent_event_node.emit('application_toolkit_configurations_collected', schemas)
