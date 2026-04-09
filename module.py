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

""" Module """

import os

from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import module  # pylint: disable=E0611,E0401

import arbiter  # pylint: disable=E0401

from tools import worker_core  # pylint: disable=E0401


class Module(module.ModuleModel):  # pylint: disable=R0902
    """ Pylon module """

    def __init__(self, context, descriptor):
        self.context = context
        self.descriptor = descriptor
        #
        self.agent_event_node = None
        self.agent_task_node = None
        #
        self.index_event_node = None
        self.index_maintenance_task_node = None
        self.index_task_node = None
        self.index_task_queue = None
        self.index_task_queue_proxy = None
        #
        self.toolkit_validators = None

    def preload(self):
        """ Preload handler """
        log.debug("Preloading bundles")
        #
        nltk_data_target = self.descriptor.config.get("nltk_data", None)
        #
        try:
            if nltk_data_target is None:
                raise RuntimeError("None nltk_data is not supported for bundles")
            #
            from tools import this  # pylint: disable=E0401,C0415
            #
            os.makedirs(nltk_data_target, exist_ok=True)
            #
            def _install_needed(*_args, **_kwargs):
                try:
                    return not os.path.exists(
                        os.path.join(nltk_data_target, "tokenizers/punkt/english.pickle")
                    )
                except:  # pylint: disable=W0702
                    return True
            #
            this.for_module("bootstrap").module.get_bundle(
                "nltk-data-all.tar.gz",
                install_needed=_install_needed,
                processing="tar_extract",
                extract_target=nltk_data_target,
                extract_cleanup=False,
            )
        except:  # pylint: disable=W0702
            log.exception("Failed to preload NLTK bundle")
        #
        sandbox_base = self.descriptor.config.get(
            "sandbox_base",
            os.path.join(nltk_data_target, "sandbox")
        )
        #
        os.makedirs(sandbox_base, exist_ok=True)
        #
        sandbox_deno_cache = os.path.join(sandbox_base, ".deno_dir")
        #
        os.makedirs(sandbox_deno_cache, exist_ok=True)
        #
        sandbox_deno = os.path.join(sandbox_base, "bin", "deno")
        #
        try:
            from tools import this  # pylint: disable=E0401,C0415
            #
            def _install_needed(*_args, **_kwargs):
                try:
                    return not os.path.exists(sandbox_deno)
                except:  # pylint: disable=W0702
                    return True
            #
            this.for_module("bootstrap").module.get_bundle(
                "deno-pyodide-sandbox.tar.gz",
                install_needed=_install_needed,
                processing="tar_extract",
                extract_target=sandbox_base,
                extract_cleanup=False,
            )
        except:  # pylint: disable=W0702
            pass

    def init(self):  # pylint: disable=R0915
        """ Init module """
        # Configure SDK dev reload mode from config (can also be set via ELITEA_SDK_DEV_RELOAD env var)
        if self.descriptor.config.get("sdk_dev_reload", False):
            from .utils.funcs import set_dev_reload_enabled
            set_dev_reload_enabled(True)
            log.info("SDK dev reload mode ENABLED - SDK modules will be reloaded on each call")
        # Pyodide sandbox (pre-script)
        sandbox_base = self.descriptor.config.get(
            "sandbox_base",
            os.path.join(
                self.descriptor.config.get("nltk_data", "/tmp"),
                "sandbox"
            )
        )
        #
        os.makedirs(sandbox_base, exist_ok=True)
        os.environ["SANDBOX_BASE"] = sandbox_base
        #
        sandbox_deno_cache = os.path.join(sandbox_base, ".deno_dir")
        #
        os.makedirs(sandbox_deno_cache, exist_ok=True)
        os.environ["DENO_DIR"] = sandbox_deno_cache
        #
        sandbox_bin = os.path.join(sandbox_base, "bin")
        #
        environ_path = os.environ.get("PATH", None)
        if not environ_path:
            environ_path = os.defpath
        #
        new_path = [sandbox_bin]
        new_path.extend(environ_path.split(os.pathsep))
        #
        os.environ["PATH"] = os.pathsep.join(new_path)
        #
        sandbox_deno = os.path.join(sandbox_base, "bin", "deno")
        #
        try:
            from tools import this  # pylint: disable=E0401,C0415
            #
            def _install_needed(*_args, **_kwargs):
                try:
                    return not os.path.exists(sandbox_deno)
                except:  # pylint: disable=W0702
                    return True
            #
            this.for_module("bootstrap").module.get_bundle(
                "deno-pyodide-sandbox.tar.gz",
                install_needed=_install_needed,
                processing="tar_extract",
                extract_target=sandbox_base,
                extract_cleanup=False,
            )
        except:  # pylint: disable=W0702
            pass
        # Init
        self.descriptor.init_all()
        # Pyodide sandbox - set entrypoint via environment variable
        sandbox_entrypoint = os.path.join(self.descriptor.path, "data", "sandbox", "main.js")
        os.environ["PYODIDE_SANDBOX_PKG"] = sandbox_entrypoint
        # Indexer
        nltk_data_target = self.descriptor.config.get("nltk_data", None)
        #
        try:
            if nltk_data_target is None:
                raise RuntimeError("None nltk_data is not supported for bundles")
            #
            from tools import this  # pylint: disable=E0401,C0415
            #
            os.makedirs(nltk_data_target, exist_ok=True)
            #
            def _install_needed(*_args, **_kwargs):
                try:
                    return not os.path.exists(
                        os.path.join(nltk_data_target, "tokenizers/punkt/english.pickle")
                    )
                except:  # pylint: disable=W0702
                    return True
            #
            this.for_module("bootstrap").module.get_bundle(
                "nltk-data-all.tar.gz",
                install_needed=_install_needed,
                processing="tar_extract",
                extract_target=nltk_data_target,
                extract_cleanup=False,
            )
        except:  # pylint: disable=W0702
            from elitea_sdk.runtime.langchain.tools.utils import download_nltk  # pylint: disable=C0415,E0401
            download_nltk(nltk_data_target)
        #
        for key, value in self.descriptor.config.get("env_vars", {}).items():
            os.environ[key] = value
        #
        try:
            from langchain_core.globals import set_verbose  # pylint: disable=C0415,E0401
            set_verbose(False)
        except:  # pylint: disable=W0702
            pass  # allow to fail
        #
        # Configure toolkit security blocklist from config
        toolkit_security = self.descriptor.config.get("toolkit_security", {})
        if toolkit_security:
            try:
                from elitea_sdk.runtime.toolkits.security import configure_blocklist  # pylint: disable=C0415,E0401
                blocked_toolkits = toolkit_security.get("blocked_toolkits", [])
                blocked_tools = toolkit_security.get("blocked_tools", {})
                if blocked_toolkits or blocked_tools:
                    configure_blocklist(
                        blocked_toolkits=blocked_toolkits,
                        blocked_tools=blocked_tools
                    )
                    log.info(f"[SECURITY] Configured toolkit blocklist: "
                            f"toolkits={blocked_toolkits}, tools={blocked_tools}")
            except Exception as e:  # pylint: disable=W0718
                log.warning(f"Failed to configure toolkit security blocklist: {e}")
            try:
                from elitea_sdk.runtime.toolkits.security import configure_sensitive_tools  # pylint: disable=C0415,E0401
                sensitive_tools = toolkit_security.get("sensitive_tools", {})
                company_name = toolkit_security.get("sensitive_action_company_name", None)
                message_template = toolkit_security.get("sensitive_action_message_template", None)
                if sensitive_tools:
                    configure_sensitive_tools(
                        sensitive_tools=sensitive_tools,
                        company_name=company_name,
                        message_template=message_template,
                    )
                    log.info(f"[SECURITY] Configured sensitive tools: {sensitive_tools}")
            except Exception as e:  # pylint: disable=W0718
                log.warning(f"Failed to configure sensitive tools: {e}")
        #
        if self.descriptor.config.get("worker_enabled", True):
            # Agent prereqs
            self.agent_event_node = worker_core.event_node.clone()
            self.agent_event_node.start()
            #
            self.agent_task_node = arbiter.TaskNode(
                self.agent_event_node,
                pool="agents",
                task_limit=self.descriptor.config.get(
                    "agents_task_limit", None
                ),
                ident_prefix="agents_",
                multiprocessing_context=self.descriptor.config.get(
                    "agents_multiprocessing_context", "fork"
                ),
                task_retention_period=3600,
                housekeeping_interval=60,
                thread_scan_interval=0.1,
                start_max_wait=3,
                query_wait=3,
                watcher_max_wait=3,
                stop_node_task_wait=3,
                result_max_wait=3,
                tmp_path=worker_core.tasknode_tmp,
                result_transport=self.descriptor.config.get(
                    "agents_result_transport", "files"
                ),
            )
            self.agent_task_node.start()
            # Index prereqs
            self.index_event_node = worker_core.event_node.clone()
            self.index_event_node.start()
            #
            self.index_maintenance_task_node = arbiter.TaskNode(
                self.index_event_node,
                pool="index_maintenance",
                task_limit=None,
                start_attempts=1,
                ident_prefix="index_maintenance_worker_",
                multiprocessing_context="fork",
                kill_on_stop=True,
                task_retention_period=3600,
                housekeeping_interval=60,
                start_max_wait=3,
                query_wait=3,
                watcher_max_wait=3,
                stop_node_task_wait=3,
                result_max_wait=3,
                tmp_path=worker_core.tasknode_tmp,
                result_transport="files",
            )
            self.index_maintenance_task_node.start()
            #
            index_task_limit = self.descriptor.config.get(
                "index_task_limit", int(worker_core.task_node_heavy.task_limit / 2)
            )
            log.debug("Using index task limit: %s", index_task_limit)
            #
            self.index_task_node = arbiter.TaskNode(
                self.index_event_node,
                pool="index_worker",
                task_limit=index_task_limit,
                start_attempts=1,  # Attempts and retries are made by TaskQueue
                ident_prefix="index_worker_",
                multiprocessing_context="fork",
                kill_on_stop=True,
                task_retention_period=3600,
                housekeeping_interval=60,
                start_max_wait=3,
                query_wait=3,
                watcher_max_wait=3,
                stop_node_task_wait=3,
                result_max_wait=3,
                tmp_path=worker_core.tasknode_tmp,
                result_transport="files",
            )
            self.index_task_node.start()
            #
            self.index_task_queue = arbiter.TaskQueue(
                self.index_task_node,
                debug=worker_core.descriptor.config.get("task_queue_debug", False),
            )
            self.index_task_queue.start()
            #
            def _proxy_task_id(meta):
                import tasknode_task  # pylint: disable=E0401,C0415
                meta["proxy_task_id"] = tasknode_task.id
                return meta
            #
            self.index_task_queue_proxy = self.index_task_queue.proxy(
                pool=lambda _: "index_worker",
                meta=_proxy_task_id,
            )
            # Tasks: maintenance
            self.index_maintenance_task_node.register_task(
                self.indexer_migrate, "indexer_migrate"
            )

            # toolkits
            self.toolkits_request(None, None)
            self.agent_event_node.subscribe("application_toolkits_request", self.toolkits_request)

            # file loaders
            self.file_loaders_request(None, None)
            self.agent_event_node.subscribe("application_file_loaders_request", self.file_loaders_request)

            # toolkit configurations
            self.toolkit_configurations_request(None, None)
            self.agent_event_node.subscribe("application_toolkit_configurations_request", self.toolkit_configurations_request)

            # MCP prebuilt configurations
            self.mcp_prebuilt_config_request(None, None)
            self.agent_event_node.subscribe("application_mcp_prebuilt_config_request", self.mcp_prebuilt_config_request)

            # registering other things

            worker_core.task_node_light.register_task(
                self.indexer_validator, "indexer_validator"
            )
            worker_core.task_node_light.register_task(
                self.indexer_configuration_validator, "indexer_configuration_validator"
            )
            worker_core.task_node_light.register_task(
                self.indexer_configuration_check_connection, "indexer_configuration_check_connection"
            )
            worker_core.task_node_light.register_task(
                self.indexer_toolkit_available_tools, "indexer_toolkit_available_tools"
            )
            self.agent_task_node.register_task(
                self.indexer_test_toolkit_tool, "indexer_test_toolkit_tool"
            )
            self.agent_task_node.register_task(
                self.indexer_test_mcp_connection, "indexer_test_mcp_connection"
            )
            #

            # agent state cleaning
            self.agent_event_node.subscribe("indexer_empty_agent_state", self.empty_agent_state)
            self.agent_event_node.subscribe("indexer_delete_checkpoint", self.delete_checkpoint)

            # Tasks: agent
            self.agent_task_node.register_task(
                self.indexer_agent, "indexer_agent"
            )
            self.agent_task_node.register_task(
                self.indexer_predict_agent, "indexer_predict_agent"
            )
            self.agent_task_node.register_task(
                self.indexer_mcp_sync_tools, "indexer_mcp_sync_tools"
            )

        else:
            log.debug("Worker not enabled")

    def deinit(self):
        """ De-init module """
        #
        if self.descriptor.config.get("worker_enabled", True):
            # Tasks: agent
            self.agent_task_node.unregister_task(
                self.indexer_agent, "indexer_agent"
            )
            self.agent_task_node.unregister_task(
                self.indexer_predict_agent, "indexer_predict_agent"
            )
            self.agent_task_node.unregister_task(
                self.indexer_mcp_sync_tools, "indexer_mcp_sync_tools"
            )
            worker_core.task_node_light.unregister_task(
                self.indexer_configuration_validator, "indexer_configuration_validator"
            )
            worker_core.task_node_light.unregister_task(
                self.indexer_validator, "indexer_validator"
            )
            worker_core.task_node_light.unregister_task(
                self.indexer_configuration_check_connection, "indexer_configuration_check_connection"
            )
            worker_core.task_node_light.unregister_task(
                self.indexer_toolkit_available_tools, "indexer_toolkit_available_tools"
            )
            self.agent_task_node.unregister_task(
                self.indexer_test_toolkit_tool, "indexer_test_toolkit_tool"
            )
            self.agent_task_node.unregister_task(
                self.indexer_test_mcp_connection, "indexer_test_mcp_connection"
            )
            # Tasks: maintenance
            self.index_maintenance_task_node.unregister_task(
                self.indexer_migrate, "indexer_migrate"
            )
            # Index prereqs
            self.index_task_queue.stop()
            self.index_task_node.stop()
            self.index_maintenance_task_node.stop()
            self.index_event_node.stop()
            # Agents prereqs
            self.agent_task_node.stop()

            self.agent_event_node.unsubscribe("application_toolkits_request", self.toolkits_request)
            self.agent_event_node.unsubscribe("application_toolkit_configurations_request", self.toolkit_configurations_request)
            self.agent_event_node.unsubscribe("application_file_loaders_request", self.file_loaders_request)
            self.agent_event_node.unsubscribe("application_mcp_prebuilt_config_request", self.mcp_prebuilt_config_request)

            self.agent_event_node.stop()
        # De-init
        self.descriptor.deinit_all()
