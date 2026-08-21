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

"""Code-validation execution task — EVAL-H2 (design §19.7).

Runs a single code validation's ``prelude + untrusted script`` inside the locked-down
Deno/Pyodide sandbox and returns ``{result, stdout, stderr, status, execution_time}``.
pylon_main assembles the prelude (evidence as plain literals) and dispatches here via the
arbiter task node; the untrusted script never touches the network or an ``elitea_client``
(the run helper constructs a deny-by-default ``SyncPyodideSandbox`` with no client — see
``utils/code_validation_sandbox.py``).
"""

from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web  # pylint: disable=E0611,E0401


class Method:
    """Method Resource — self points to the current Module instance."""

    @web.method()
    def indexer_code_validation(self, *args, code, **kwargs):
        """Execute one code validation script in the sandbox and return its raw result.

        The result dict is intentionally the SDK ``CodeExecutionResult`` shape
        (``{result, stdout, stderr, status, execution_time}``) so the pylon_main caller can
        map it through ``code_validation.map_execution_result`` into a verdict. Execution
        failures (timeout/OOM/exception, or Deno absent) come back as ``status`` ``'error'`` /
        ``'unavailable'`` — this task does not raise for a failed validation.
        """
        _ = args, kwargs
        log.debug("indexer_code_validation: executing code validation script")

        from ..utils.funcs import dev_reload_sdk  # pylint: disable=C0415
        dev_reload_sdk('elitea_sdk.runtime')
        from ..utils.code_validation_sandbox import run_code_in_sandbox  # pylint: disable=C0415

        return run_code_in_sandbox(code)
