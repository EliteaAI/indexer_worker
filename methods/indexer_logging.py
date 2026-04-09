#!/usr/bin/python3
# coding=utf-8

#   Copyright 2025 EPAM Systems
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

import logging

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def indexer_enable_logging(
            self,
            level=logging.INFO,
            additional_labels=None,
    ):
        """ Enable streaming to logging_hub """
        try:
            import tasknode_task  # pylint: disable=E0401,C0415
            from tools import worker_core  # pylint: disable=E0401,C0415
            #
            labels = {
                "tasknode_task": f"id:{tasknode_task.id}",
                "stream_id": "",  # until datasources are updated
            }
            #
            if additional_labels is not None:
                labels.update(additional_labels)
            #
            log.init(
                config={
                    "level": level,
                    "filters": [
                        {
                            "type": "centry_logging.filters.truncate_base64.Base64SanitizationFilter",
                        },
                    ],
                    "handlers": [
                        {
                            "type": "logging.StreamHandler",
                        },
                        {
                            "type": "centry_logging.handlers.eventnode.EventNodeLogHandler",
                            "settings": {
                                "event_node": worker_core.event_node_config,
                                "labels": labels,
                            },
                        },
                    ],
                },
                force=True,
            )
        except:  # pylint: disable=W0702
            pass
