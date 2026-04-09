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

""" Method for extracting prebuilt MCP server configuration """

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
    def mcp_prebuilt_config_request(self, event, payload):
        """
        Extract prebuilt MCP server configurations from pylon config.

        Reads the `mcp_servers` section from pylon.yml configuration and returns
        all settings for prebuilt MCP toolkits.

        Example pylon.yml:
            mcp_servers:
              Epam Presales:
                client_id: "c1e199f6-ae8d-40a3-9c0c-b921e1af6474"
                client_secret: "nnSGxaeKHF1GvaL3"
                timeout: 30
                base_url: "https://api.example.com"
              GitHub Copilot:
                url: "https://api.githubcopilot.com/mcp/"

        Returns:
            Dict with MCP server configurations keyed by normalized name.
        """
        log.info("Received request for MCP prebuilt server configurations")
        mcp_configs = {}

        try:
            config = self.descriptor.config
            mcp_servers = config.get('mcp_servers', {})
            log.debug(f"Found {len(mcp_servers)} mcp_servers in pylon config")

            if mcp_servers:
                # Return all MCP server configurations with normalized keys
                for server_name, server_config in mcp_servers.items():
                    if server_config and isinstance(server_config, dict):
                        # Normalize the key (lowercase, spaces to underscores)
                        normalized_key = server_name.lower().replace(" ", "_").strip()
                        mcp_configs[normalized_key] = {
                            'original_name': server_name,
                            **server_config
                        }
                        log.debug(f"Loaded MCP config for '{server_name}' as '{normalized_key}'")

        except Exception as e:
            log.warning(f"Failed to load MCP server configurations from pylon config: {e}")

        log.info(f"MCP prebuilt configurations collected: {len(mcp_configs)} server(s)")
        self.agent_event_node.emit('application_mcp_prebuilt_config_collected', mcp_configs)

