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

""" Sealed transport of released credential material (C-1009 WP-1D)

Main seals the released material to a per-request X25519 public key of the recipient, so the
reply on the worker bus never carries it in clear: X25519 (fresh ephemeral key) -> HKDF-SHA256
-> AES-256-GCM, with the request context bound as AAD. The worker side runs identical code
(indexer_worker/utils/credential_transport.py); both are checked against one shared test vector.

Sealed form: {"v": 1, "epk": b64, "nonce": b64, "ct": b64}  (standard base64 with padding).
Every failure is a TransportError with a fixed message: callers and logs learn nothing about
which part was wrong, and nothing here holds or prints plaintext.
"""

import base64
import binascii
import os

from cryptography.exceptions import InvalidTag  # pylint: disable=E0401
from cryptography.hazmat.primitives import hashes  # pylint: disable=E0401
from cryptography.hazmat.primitives.asymmetric.x25519 import (  # pylint: disable=E0401
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # pylint: disable=E0401
from cryptography.hazmat.primitives.kdf.hkdf import HKDF  # pylint: disable=E0401
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat  # pylint: disable=E0401

VERSION = 1
HKDF_INFO = b"elitea-credential-release-v1"
KEY_BYTES = 32
NONCE_BYTES = 12
_FIELDS = frozenset({"v", "epk", "nonce", "ct"})


class TransportError(ValueError):
    """ Seal or open failed; the message never says why """

    def __init__(self):
        super().__init__("credential transport failure")


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(value, length=None) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except (TypeError, ValueError, binascii.Error):
        raise TransportError() from None
    if length is not None and len(raw) != length:
        raise TransportError()
    return raw


def _key(private_key: X25519PrivateKey, peer_public: bytes) -> bytes:
    try:
        shared = private_key.exchange(X25519PublicKey.from_public_bytes(peer_public))
    except ValueError:  # low-order point
        raise TransportError() from None
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None, info=HKDF_INFO).derive(shared)


def generate_recipient_keypair():
    """ (private key object, public key as base64) for one request; the private key never leaves the caller """
    private = X25519PrivateKey.generate()
    return private, _b64e(private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def seal(plaintext: bytes, recipient_key_b64: str, aad: bytes, *,
         _ephemeral_private_key=None, _nonce=None) -> dict:
    """ Seal plaintext to a recipient public key; aad is authenticated, not encrypted.

    The underscore keywords exist for the shared test vector only (fixed ephemeral key and nonce).
    """
    if not isinstance(plaintext, bytes) or not isinstance(aad, bytes):
        raise TransportError()
    recipient = _b64d(recipient_key_b64, KEY_BYTES)
    ephemeral = _ephemeral_private_key or X25519PrivateKey.generate()
    nonce = _nonce or os.urandom(NONCE_BYTES)
    if len(nonce) != NONCE_BYTES:
        raise TransportError()
    key = _key(ephemeral, recipient)
    return {
        "v": VERSION,
        "epk": _b64e(ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)),
        "nonce": _b64e(nonce),
        "ct": _b64e(AESGCM(key).encrypt(nonce, plaintext, aad)),
    }


def open_sealed(sealed: dict, recipient_private_key: X25519PrivateKey, aad: bytes) -> bytes:
    """ Open a sealed dict with the recipient private key and the same aad; TransportError on any mismatch """
    if (
            not isinstance(sealed, dict) or set(sealed) != _FIELDS
            or type(sealed["v"]) is not int or sealed["v"] != VERSION  # pylint: disable=C0123
            or not isinstance(recipient_private_key, X25519PrivateKey) or not isinstance(aad, bytes)
    ):
        raise TransportError()
    epk = _b64d(sealed["epk"], KEY_BYTES)
    nonce = _b64d(sealed["nonce"], NONCE_BYTES)
    ct = _b64d(sealed["ct"])
    key = _key(recipient_private_key, epk)
    try:
        return AESGCM(key).decrypt(nonce, ct, aad)
    except InvalidTag:
        raise TransportError() from None
