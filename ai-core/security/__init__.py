"""
ai-core/security/__init__.py
==============================
RIVA Health Platform — Security Module
"""

from .access_control import (
    get_access_control,
    AccessControl,
    Role,
    Permission,
    User,
    Session,
    require_role,
    PAGE_ACCESS,
)
from .digital_signature import (
    get_signer,
    sign_document,
    verify_document,
    DigitalSigner,
    SignatureRecord,
)
from .encryption_handler import (
    get_encryption_handler,
    EncryptionHandler,
    ALL_DATABASES,
)
from .key_manager import (
    get_key_manager,
    get_key_status,
    setup_riva_keys,
    KeyManager,
    KeyMetadata,
)

__all__ = [
    # Access control
    "get_access_control", "AccessControl",
    "Role", "Permission", "User", "Session",
    "require_role", "PAGE_ACCESS",
    # Digital signature
    "get_signer", "sign_document", "verify_document",
    "DigitalSigner", "SignatureRecord",
    # Encryption
    "get_encryption_handler", "EncryptionHandler", "ALL_DATABASES",
    # Key manager
    "get_key_manager", "get_key_status", "setup_riva_keys",
    "KeyManager", "KeyMetadata",
]
