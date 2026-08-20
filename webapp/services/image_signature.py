from __future__ import annotations

import base64
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


WEBAPP_ROOT = Path(__file__).resolve().parents[1]

KEY_DIR = WEBAPP_ROOT / "signature_keys"

PRIVATE_KEY_PATH = KEY_DIR / "private_key.pem"
PUBLIC_KEY_PATH = KEY_DIR / "public_key.pem"

SIGNATURE_METADATA_FIELD = "IMAGE_SIGNATURE"


def _canonical_json(data: dict[str, Any]) -> bytes:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _load_private_key() -> Ed25519PrivateKey:
    if not PRIVATE_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Private key not found: {PRIVATE_KEY_PATH}"
        )

    return serialization.load_pem_private_key(
        PRIVATE_KEY_PATH.read_bytes(),
        password=None,
    )


def _load_public_key() -> Ed25519PublicKey:
    if not PUBLIC_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Public key not found: {PUBLIC_KEY_PATH}"
        )

    return serialization.load_pem_public_key(
        PUBLIC_KEY_PATH.read_bytes()
    )


def _pixel_hash(image_path: Path) -> str:
    """
    Hash normalized RGB pixel data rather than the complete file.

    This means adding/removing PNG metadata does not change the
    integrity hash.
    """

    image = Image.open(image_path).convert("RGB")

    pixel_bytes = image.tobytes()

    return hashlib.sha256(pixel_bytes).hexdigest()


def create_image_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    random_part = secrets.token_hex(4).upper()

    return f"IMG-{timestamp}-{random_part}"


def create_signature_record(
    image_path: Path,
    creator_id: str,
    method: str,
    prompt: str,
) -> dict[str, Any]:

    return {
        "version": 1,
        "image_id": create_image_id(),
        "creator_id": creator_id,
        "method": method,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_hash": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "pixel_sha256": _pixel_hash(image_path),
    }


def sign_record(record: dict[str, Any]) -> str:
    private_key = _load_private_key()

    payload = _canonical_json(record)

    signature = private_key.sign(payload)

    return base64.b64encode(signature).decode("ascii")


def create_signed_record(
    image_path: Path,
    creator_id: str,
    method: str,
    prompt: str,
) -> dict[str, Any]:

    record = create_signature_record(
        image_path=image_path,
        creator_id=creator_id,
        method=method,
        prompt=prompt,
    )

    signature = sign_record(record)

    return {
        "record": record,
        "signature": signature,
        "signature_algorithm": "Ed25519",
    }


def embed_signature_in_png(
    source_image: Path,
    destination_image: Path,
    signed_record: dict[str, Any],
) -> None:

    image = Image.open(source_image).convert("RGB")

    metadata = PngImagePlugin.PngInfo()

    metadata.add_text(
        SIGNATURE_METADATA_FIELD,
        json.dumps(
            signed_record,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )

    image.save(
        destination_image,
        format="PNG",
        pnginfo=metadata,
    )


def extract_signature(
    image_path: Path,
) -> dict[str, Any] | None:

    image = Image.open(image_path)

    raw = image.info.get(SIGNATURE_METADATA_FIELD)

    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def verify_image_signature(image_path: Path) -> dict[str, Any]:
    """
    Verify the embedded Image Signature.

    Returns:
        found
        signature_valid
        integrity_valid
        record
        message
    """

    signed_record = extract_signature(image_path)

    if signed_record is None:
        return {
            "found": False,
            "signature_valid": False,
            "integrity_valid": False,
            "record": None,
            "message": "No image signature found.",
        }

    record = signed_record.get("record")
    signature_b64 = signed_record.get("signature")

    if not isinstance(record, dict):
        return {
            "found": True,
            "signature_valid": False,
            "integrity_valid": False,
            "record": None,
            "message": "Image signature record is invalid.",
        }

    if not isinstance(signature_b64, str):
        return {
            "found": True,
            "signature_valid": False,
            "integrity_valid": False,
            "record": record,
            "message": "Image signature is missing.",
        }

    # --------------------------------------------------
    # Verify cryptographic signature
    # --------------------------------------------------

    try:
        signature = base64.b64decode(
            signature_b64.encode("ascii")
        )

        public_key = _load_public_key()

        public_key.verify(
            signature,
            _canonical_json(record),
        )

        signature_valid = True

    except (InvalidSignature, ValueError, TypeError):
        signature_valid = False

    # --------------------------------------------------
    # Verify image integrity
    # --------------------------------------------------

    stored_hash = record.get("pixel_sha256")
    current_hash = _pixel_hash(image_path)

    integrity_valid = (
        isinstance(stored_hash, str)
        and stored_hash == current_hash
    )

    # --------------------------------------------------
    # Final message
    # --------------------------------------------------

    if signature_valid and integrity_valid:
        message = (
            "Image signature verified and image integrity confirmed."
        )

    elif signature_valid and not integrity_valid:
        message = (
            "Image signature is valid, but the image content "
            "has changed."
        )

    else:
        message = (
            "An image signature was found, but its cryptographic "
            "signature could not be verified."
        )

    return {
        "found": True,
        "signature_valid": signature_valid,
        "integrity_valid": integrity_valid,
        "record": record,
        "message": message,
    }
