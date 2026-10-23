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

""" Method: governed check-connection of a saved credential (C-1009 WP-1D, #6946) """

from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web  # pylint: disable=E0611,E0401

from pydantic import ValidationError
from ..utils.context_guard import rejection_message
from ..utils.credential_resolver import TrustedCredentialResolver
from ..utils.envelope_guard import enforce_envelope, rejection_result
from ..utils.funcs import normalize_mcp_auth_metadata_urls
from ..utils.governed_ops import GOVERNED_CHECK_CONNECTION_TASK, governed_ops_capabilities
from ..utils.task_envelope import PROFILE_GOVERNED, EnvelopeRejected, read_envelope

# The slot sits directly in the stored configuration data; the SDK resolves slots one level
# down, so the data travels under this key for the duration of the call.
PROBE_FIELD = "configuration"
PROBE_TOOLKIT_INSTANCE_ID = "probe"
CREDENTIAL_REFERENCE_REQUIRED = "CREDENTIAL_REFERENCE_REQUIRED"
STALE_CONTEXT = "STALE_CONTEXT"


def _refusal(code):
    return {"success": False, "reason_code": code, "error": rejection_message(code)}


def _scrub_result(value, released, scrub):
    if isinstance(value, str):
        return scrub(value, released)
    if isinstance(value, dict):
        return {k: _scrub_result(v, released, scrub) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_result(v, released, scrub) for v in value]
    return value


def _config_model(configuration_type):
    from ..utils.funcs import dev_reload_sdk
    dev_reload_sdk('elitea_sdk.configurations')
    dev_reload_sdk('elitea_sdk.runtime.toolkits.mcp_config')
    from elitea_sdk.configurations import get_class_configurations
    config_model = get_class_configurations().get(configuration_type)
    if config_model is None and configuration_type.startswith('mcp_'):
        from elitea_sdk.runtime.toolkits.mcp_config import get_mcp_config_toolkit_schemas
        for mcp_config in get_mcp_config_toolkit_schemas():
            if mcp_config.__name__ == configuration_type:
                return mcp_config
    return config_model


class Method:
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def indexer_governed_check_connection(self, *args, configuration_type, settings, **kwargs):
        """Check a saved credential: the probe slot is resolved here, in the governed_ops fork.

        ``settings`` is the stored non-secret configuration data plus one ``credential_slot``
        carrying a ``probe_ref``. Output is scrubbed of the released values; a Main refusal
        comes back as ``{success: False, reason_code, error}`` and the MCP authorization
        result is returned as the ungoverned task returns it.
        """
        _ = args
        import tasknode_task  # pylint: disable=E0401,C0415
        try:
            enforce_envelope(kwargs, extra=governed_ops_capabilities())
            if read_envelope(kwargs).security_profile != PROFILE_GOVERNED:
                return _refusal(STALE_CONTEXT)
        except EnvelopeRejected as exc:
            return rejection_result(exc)
        from elitea_sdk.runtime.credential_slots import (  # pylint: disable=C0415,E0401
            SLOT_KEY, GovernedCredentialError, apply_credential_slots, scrub,
        )
        if not isinstance(settings, dict) or not isinstance(settings.get(SLOT_KEY), dict):
            return _refusal(CREDENTIAL_REFERENCE_REQUIRED)
        project_id = (tasknode_task.meta or {}).get("project_id")
        if project_id is None:
            project_id = kwargs.get("project_id")
        if project_id is None:
            return _refusal(STALE_CONTEXT)
        log.debug(f"governed check_connection for {configuration_type}")

        from ..utils.agent_execution_common import setup_event_node
        local_event_node = setup_event_node(tasknode_task.multiprocessing_context)
        resolver = TrustedCredentialResolver(
            local_event_node, project_id=project_id, task_name=GOVERNED_CHECK_CONNECTION_TASK,
        )
        released = frozenset()
        try:
            try:
                applied, released = apply_credential_slots(
                    {PROBE_FIELD: settings}, PROBE_TOOLKIT_INSTANCE_ID, resolver,
                )
            except GovernedCredentialError as exc:
                return _refusal(exc.reason_code)
            config_model = _config_model(configuration_type)
            if config_model is None:
                return f"Not supported toolkit type: {configuration_type}"
            if not hasattr(config_model, 'check_connection'):
                return f"Check connection is not implemented yet for {configuration_type}"
            from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired
            try:
                result = config_model.check_connection(applied[PROBE_FIELD])
            except McpAuthorizationRequired as ex:
                log.info(f"MCP authorization required for governed check_connection [{configuration_type}]")
                return _scrub_result({
                    'success': False,
                    'requires_authorization': True,
                    'error': str(ex),
                    'auth_metadata': normalize_mcp_auth_metadata_urls(ex.to_dict()),
                }, released, scrub)
            except ValidationError as ex:
                text = scrub(str(ex.errors(include_url=False, include_context=False)), released)
                log.error(f"Validation error in governed check_connection for {configuration_type}: {text}")
                return text
            except Exception as ex:  # pylint: disable=W0718
                text = scrub(str(ex), released)
                log.error(f"Exception in governed check_connection for {configuration_type}: {text}")
                return text
            return _scrub_result(result, released, scrub)
        finally:
            applied = None  # noqa: F841  # pylint: disable=W0642
            resolver.close()
            if tasknode_task.multiprocessing_context == "fork":
                local_event_node.stop()
