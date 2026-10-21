"""Can this worker resolve governed credential slots? (C-1009 WP-1D, EliteaAI/elitea_issues#6946)

True only when the installed SDK ships ``elitea_sdk.runtime.credential_slots`` and the sealed
transport round-trips on this host. The answer gates the ``credential_slots.v1`` and
``credential_slots_batch.v1`` capabilities and the ``governed_ops`` pool, so a worker that could not honour a sealed payload never claims
to. Dependency-light on purpose: the capability code imports this without the SDK or the
resolver.
"""

import functools
import os

CAPABILITY_CREDENTIAL_SLOTS_V1 = "credential_slots.v1"
# resolver.resolve_many: one release call per toolkit build
CAPABILITY_CREDENTIAL_SLOTS_BATCH_V1 = "credential_slots_batch.v1"
CAPABILITY_GOVERNED_OPS_POOL_V1 = "governed_ops_pool.v1"


def _transport_self_test():
    from .credential_transport import generate_recipient_keypair, open_sealed, seal  # pylint: disable=C0415
    private, public = generate_recipient_keypair()
    aad = b"v1|self-test"
    plaintext = os.urandom(16)
    if open_sealed(seal(plaintext, public, aad), private, aad) != plaintext:
        return False
    try:
        open_sealed(seal(plaintext, public, aad), private, b"v1|other")
    except ValueError:
        return True
    return False


@functools.lru_cache(maxsize=1)
def slots_supported():
    """SDK slot module importable and transport self-test passing; evaluated once per process."""
    try:
        import elitea_sdk.runtime.credential_slots  # pylint: disable=C0415,W0611,E0401
        return bool(_transport_self_test())
    except Exception:  # pylint: disable=W0718
        return False
