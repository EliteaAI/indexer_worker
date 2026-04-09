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
    def file_loaders_request(self, event, payload):
        loaders_mapping = {
            "document_types": {},
            "image_types": {},
            "code_types": {},
        }
        try:
            from ..utils.funcs import dev_reload_sdk
            dev_reload_sdk('elitea_sdk.runtime.langchain.document_loaders')
            from elitea_sdk.runtime.langchain.document_loaders.constants import (
                document_loaders_map,
                image_loaders_map,
                code_loaders_map
            )
            loaders_mapping = {
                "document_types": {extension: loader_config['mime_type'] for extension, loader_config in
                                   document_loaders_map.items()},
                "image_types": {extension: loader_config['mime_type'] for extension, loader_config in
                                image_loaders_map.items()},
                "code_types": {extension: loader_config['mime_type'] for extension, loader_config in
                                code_loaders_map.items()},

            }
        except ImportError as e:
            log.warning("Failed to import elitea_sdk.runtime.langchain.document_loaders.constants: %s", e)


        self.agent_event_node.emit('application_file_loaders_collected', loaders_mapping)
