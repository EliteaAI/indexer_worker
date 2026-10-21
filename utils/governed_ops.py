"""The ``governed_ops`` pool: a separate fork pool for governed credential probes (C-1009 WP-1D, #6946).

Slice 1 runs it in this container beside ``agents``. It runs only
``indexer_governed_check_connection``, with its own finite ``task_limit``, so probes cannot
starve agent runs or the reverse. Off unless ``governed_ops_enabled`` is true *and* this worker
can resolve credential slots (``credential_slots_support.slots_supported``).
Both keys are worker config (``configs/indexer_worker.yml``), never an admin-editable schema.
"""

from .credential_slots_support import CAPABILITY_GOVERNED_OPS_POOL_V1, slots_supported

GOVERNED_OPS_POOL = "governed_ops"
GOVERNED_CHECK_CONNECTION_TASK = "indexer_governed_check_connection"
DEFAULT_TASK_LIMIT = 4


def governed_ops_enabled(config):
    """True only for an explicit ``true`` and a worker able to honour sealed payloads."""
    return config.get("governed_ops_enabled", False) is True and slots_supported()


def governed_ops_task_limit(config):
    """A finite positive limit; anything else (missing, null, 0, non-int) is the default."""
    value = config.get("governed_ops_task_limit", DEFAULT_TASK_LIMIT)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return DEFAULT_TASK_LIMIT
    return value


def governed_ops_capabilities():
    return (CAPABILITY_GOVERNED_OPS_POOL_V1,)


def create_governed_task_node(arbiter, event_node, config, tmp_path, task_approver):
    """Build (not start) the ``governed_ops`` TaskNode; default result transport, not ``files``."""
    return arbiter.TaskNode(
        event_node,
        pool=GOVERNED_OPS_POOL,
        task_approver=task_approver,
        task_limit=governed_ops_task_limit(config),
        start_attempts=1,
        ident_prefix="governed_ops_",
        multiprocessing_context="fork",
        kill_on_stop=True,
        task_retention_period=300,
        housekeeping_interval=60,
        start_max_wait=3,
        query_wait=3,
        watcher_max_wait=3,
        stop_node_task_wait=3,
        result_max_wait=3,
        tmp_path=tmp_path,
    )
