from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


WEBAPP_ROOT = Path(__file__).resolve().parent
KEY_DIR = WEBAPP_ROOT / "signature_keys"

KEY_DIR.mkdir(parents=True, exist_ok=True)

PRIVATE_KEY_PATH = KEY_DIR / "private_key.pem"
PUBLIC_KEY_PATH = KEY_DIR / "public_key.pem"


private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

private_key_bytes = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)

public_key_bytes = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)

PRIVATE_KEY_PATH.write_bytes(private_key_bytes)
PUBLIC_KEY_PATH.write_bytes(public_key_bytes)

print("Image Signature keys created successfully.")
print(f"Private key: {PRIVATE_KEY_PATH}")
print(f"Public key:  {PUBLIC_KEY_PATH}")
