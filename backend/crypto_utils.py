from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


# Generate RSA keys (for demo only – in real system, use proper key management)
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()


def sign_content(content: str) -> bytes:
    """
    Signs the given content (string) and returns signature bytes.
    """
    signature = private_key.sign(
        content.encode(),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    return signature


def verify_signature(content: str, signature: bytes) -> bool:
    """
    Verifies that signature matches the given content using the public key.
    """
    try:
        public_key.verify(
            signature,
            content.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False


def export_public_key() -> str:
    """
    Returns the PEM-formatted public key (string).
    """
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return pem.decode()
