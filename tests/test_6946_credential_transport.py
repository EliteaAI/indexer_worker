"""C-1009 WP-1D (#6946): the worker side of the sealed credential transport.

The code is a verbatim copy of Main's ``shared/tools/secret_stores/transport.py``; the shared
vector below (copied verbatim from that repo) proves both sides stay interoperable.
"""

import base64
import json
import pathlib

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from utils import credential_transport as transport

VECTOR = json.loads((pathlib.Path(__file__).parent / "fixtures" / "credential_release_vector.json").read_text())


def _private(b64):
    return X25519PrivateKey.from_private_bytes(base64.b64decode(b64))


def test_shared_vector_opens_with_the_recipient_key():
    plaintext = transport.open_sealed(
        VECTOR["sealed"], _private(VECTOR["recipient_private_key"]), VECTOR["aad"].encode())
    assert plaintext.decode() == VECTOR["plaintext"]


def test_shared_vector_is_reproduced_byte_for_byte_by_seal():
    sealed = transport.seal(
        VECTOR["plaintext"].encode(), VECTOR["recipient_public_key"], VECTOR["aad"].encode(),
        _ephemeral_private_key=_private(VECTOR["ephemeral_private_key"]),
        _nonce=base64.b64decode(VECTOR["nonce"]),
    )
    assert sealed == VECTOR["sealed"]


def test_vector_public_key_matches_its_private_key():
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    raw = _private(VECTOR["recipient_private_key"]).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    assert base64.b64encode(raw).decode() == VECTOR["recipient_public_key"]


def test_round_trip_with_a_fresh_keypair():
    private, public = transport.generate_recipient_keypair()
    sealed = transport.seal(b"payload", public, b"aad")
    assert transport.open_sealed(sealed, private, b"aad") == b"payload"


def test_each_keypair_is_new():
    (_, first), (_, second) = transport.generate_recipient_keypair(), transport.generate_recipient_keypair()
    assert first != second


@pytest.mark.parametrize("mutate", ["ct", "nonce", "epk"])
def test_tampered_part_fails_with_the_fixed_error(mutate):
    private, public = transport.generate_recipient_keypair()
    sealed = transport.seal(b"payload", public, b"aad")
    raw = bytearray(base64.b64decode(sealed[mutate]))
    raw[0] ^= 1
    sealed[mutate] = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(transport.TransportError) as err:
        transport.open_sealed(sealed, private, b"aad")
    assert str(err.value) == "credential transport failure"


def test_wrong_aad_or_wrong_key_fails():
    private, public = transport.generate_recipient_keypair()
    sealed = transport.seal(b"payload", public, b"aad")
    with pytest.raises(transport.TransportError):
        transport.open_sealed(sealed, private, b"other")
    with pytest.raises(transport.TransportError):
        transport.open_sealed(sealed, transport.generate_recipient_keypair()[0], b"aad")
