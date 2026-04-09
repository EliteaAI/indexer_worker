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

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401

from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # pylint: disable=E0401


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def indexer_check_ad_token(  # pylint: disable=R0913
            self,
            target_kwargs,
    ):
        """ Check/set AD token provider """
        #
        if "embedding_model_params" in target_kwargs and \
                "use_ad_token_scope" in target_kwargs["embedding_model_params"]:
            #
            log.info("Using managed identity / AD token provider for embeddings")
            #
            ad_token_scope = target_kwargs["embedding_model_params"].pop("use_ad_token_scope")
            ad_token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), ad_token_scope
            )
            #
            target_kwargs["embedding_model_params"]["azure_ad_token_provider"] = \
                ad_token_provider
        #
        if "ai_model_params" in target_kwargs and \
                "use_ad_token_scope" in target_kwargs["ai_model_params"]:
            #
            log.info("Using managed identity / AD token provider for AI")
            #
            ad_token_scope = target_kwargs["ai_model_params"].pop("use_ad_token_scope")
            ad_token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), ad_token_scope
            )
            #
            target_kwargs["ai_model_params"]["azure_ad_token_provider"] = \
                ad_token_provider
