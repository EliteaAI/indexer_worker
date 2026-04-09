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

from pydantic import ValidationError

class Method:
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def indexer_configuration_check_connection(self, *args, configuration_type, settings, **kwargs):
        log.debug(f"check_connection for {configuration_type}")

        from ..utils.funcs import dev_reload_sdk
        dev_reload_sdk('elitea_sdk.configurations')
        dev_reload_sdk('elitea_sdk.runtime.toolkits.mcp_config')
        # McpAuthorizationRequired must be imported AFTER dev_reload_sdk to get the fresh class reference
        from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired
        from elitea_sdk.configurations import get_class_configurations

        configurations = get_class_configurations()
        config_model = configurations.get(configuration_type)

        # If not found in regular configurations, check MCP config schemas
        if config_model is None and configuration_type.startswith('mcp_'):
            from elitea_sdk.runtime.toolkits.mcp_config import get_mcp_config_toolkit_schemas
            for mcp_config in get_mcp_config_toolkit_schemas():
                if mcp_config.__name__ == configuration_type:
                    config_model = mcp_config
                    break

        if config_model is None:
            return f"Not supported toolkit type: {configuration_type}"
        try:
            if not hasattr(config_model, 'check_connection'):
                return f"Check connection is not implemented yet for {configuration_type}"
            res = config_model.check_connection(settings)
            log.debug(f"check_connection for {configuration_type} returned: {res}")
        except McpAuthorizationRequired as ex:
            log.info(f"MCP authorization required for configuration check_connection [{configuration_type}]: {str(ex)}")
            # Build auth metadata from the exception (no toolkit_config here — configurations page only)
            auth_metadata = ex.to_dict()
            return {
                'success': False,
                'requires_authorization': True,
                'error': str(ex),
                'auth_metadata': auth_metadata,
            }
        except ValidationError as ex:
            log.error(f"Validation error in check_connection for {configuration_type}: {ex}")
            return str(ex.errors(include_url=False, include_context=False))
        except Exception as ex:
            log.error(f"Exception in check_connection for {configuration_type}: {ex}")
            return str(ex)

        return res
