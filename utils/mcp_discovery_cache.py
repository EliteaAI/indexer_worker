#!/usr/bin/python3
# coding=utf-8

#   Copyright 2026 EPAM Systems
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

"""Redis-backed store for the SDK's remote MCP discovery cache.

Registered once in the parent process so every forked agent run inherits it. The Redis
client itself is built lazily per pid: a socket must not be shared across the fork.
"""

import os

from pylon.core.tools import log

KEY_PREFIX = "mcp:discovery:"
SOCKET_TIMEOUT = 2
MAX_CONNECTIONS = 2


def build_redis_client(redis_config):
    from redis import Redis  # pylint: disable=C0415,E0401
    from redis.connection import BlockingConnectionPool  # pylint: disable=C0415,E0401
    #
    pool = BlockingConnectionPool(**{
        **redis_config,
        "max_connections": MAX_CONNECTIONS,
        "timeout": SOCKET_TIMEOUT,
        "socket_timeout": SOCKET_TIMEOUT,
        "socket_connect_timeout": SOCKET_TIMEOUT,
    })
    return Redis(connection_pool=pool)


class RedisMcpDiscoveryCacheBackend:
    """ Prefix-namespaced get/setex over a pid-bound Redis client """

    def __init__(self, redis_config, client_factory=build_redis_client, prefix=KEY_PREFIX):
        self._redis_config = dict(redis_config)
        self._client_factory = client_factory
        self._prefix = prefix
        self._client = None
        self._client_pid = None

    def _connection(self):
        pid = os.getpid()
        if self._client is None or self._client_pid != pid:
            self._client = self._client_factory(self._redis_config)
            self._client_pid = pid
        return self._client

    def get(self, key):
        value = self._connection().get(self._prefix + key)
        return value.decode() if isinstance(value, bytes) else value

    def set(self, key, value, ttl):
        self._connection().setex(self._prefix + key, ttl, value)


def build_backend_from_event_node(event_node):
    redis_config = getattr(event_node, "redis_config", None)
    if not redis_config:
        return None
    return RedisMcpDiscoveryCacheBackend(redis_config)


def configure_mcp_discovery_cache(event_node):
    from elitea_sdk.runtime.utils.mcp_discovery_cache import register_discovery_cache_backend  # pylint: disable=C0415,E0401
    #
    backend = build_backend_from_event_node(event_node)
    register_discovery_cache_backend(backend)
    log.info(
        "[MCP cache] discovery cache %s",
        "enabled (redis)" if backend else "disabled (no redis event node)",
    )
    return backend
