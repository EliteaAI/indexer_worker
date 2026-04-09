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

import os

import requests  # pylint: disable=E0401

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
    def indexer_download_project_artifact(  # pylint: disable=R0913
            self,
            url, token, project_id, bucket, filename,
            download_dir,
            ssl_verify=False,
    ):
        """ Get artifact from Centry project """
        while url.endswith("/"):
            url = url[:-1]
        #
        # Fixme: we can use elitea_sdk.clients.client to work with artifacts
        target = f'{url}/api/v1/artifacts/artifact/default'
        target = f'{target}/{project_id}/{bucket}/{filename}'
        #
        headers = {
            "Authorization": f'Bearer {token}',
        }
        #
        data = requests.get(target, headers=headers, verify=ssl_verify, timeout=60).content
        target_path = os.path.join(download_dir, filename)
        #
        with open(target_path, "wb") as file:
            file.write(data)
        #
        return target_path

    @web.method()
    def indexer_upload_project_artifact(  # pylint: disable=R0913
            self,
            url, token, project_id, bucket, filename,
            data,
            ssl_verify=False,
    ):
        """ Upload artifact to Centry project 
        
        Note: Filenames should already be sanitized at upload time by attachment APIs
        or SDK methods. This method provides a defensive safety net.
        """
        # Defensive sanitization as safety net
        sanitized_filename, was_modified = self._sanitize_filename_defensive(filename)
        if was_modified:
            log.warning(
                f"Filename sanitization applied in indexer_upload_project_artifact: "
                f"'{filename}' -> '{sanitized_filename}'. "
                f"This indicates the filename was not sanitized at the source."
            )
        
        while url.endswith("/"):
            url = url[:-1]
        #
        # Fixme: we can use elitea_sdk.clients.client to work with artifacts
        target = f'{url}/api/v1/artifacts/artifacts/default'
        target = f'{target}/{project_id}/{bucket}'
        #
        headers = {
            "Authorization": f'Bearer {token}',
        }
        #
        files = {
            "file": (sanitized_filename, data),
        }
        #
        return requests.post(target, headers=headers, files=files, verify=ssl_verify, timeout=60)
    
    @staticmethod
    def _sanitize_filename_defensive(filename: str) -> tuple:
        """Defensive filename sanitization as a safety net."""
        import re  # pylint: disable=C0415
        from pathlib import Path  # pylint: disable=C0415
        
        if not filename or not filename.strip():
            return "unnamed_file", True
        
        original = filename
        path_obj = Path(filename)
        name = path_obj.stem
        extension = path_obj.suffix
        
        # Whitelist: alphanumeric, underscore, hyphen, space, Unicode letters/digits
        sanitized_name = re.sub(r'[^\w\s-]', '', name, flags=re.UNICODE)
        sanitized_name = re.sub(r'[-\s]+', '-', sanitized_name)
        sanitized_name = sanitized_name.strip('-').strip()
        
        if not sanitized_name:
            sanitized_name = "file"
        
        if extension:
            extension = re.sub(r'[^\w.-]', '', extension, flags=re.UNICODE)
        
        sanitized = sanitized_name + extension
        return sanitized, (sanitized != original)
