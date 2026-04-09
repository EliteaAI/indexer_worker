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
    def indexer_configuration_validator(self, *args, configuration_type, settings, **kwargs):
        log.debug(f"EliteA configuration validation for settings of {configuration_type}")

        from ..utils.funcs import dev_reload_sdk
        dev_reload_sdk('elitea_sdk.configurations')
        from elitea_sdk.configurations import get_class_configurations
        from pydantic import ValidationError

        configurations = get_class_configurations()
        try:
            res = configurations[configuration_type].model_validate(settings)
        except ValidationError as ex:
            return {"error": ex.errors(include_url=False, include_context=False)}
        except KeyError:
            return {"error": f"Not supported configuration type: {configuration_type}"}
        except Exception as ex:
            log.error(ex)
            return {"error": f"Error validating toolkit type: {configuration_type}"}

        return {"result": res.model_dump()}
