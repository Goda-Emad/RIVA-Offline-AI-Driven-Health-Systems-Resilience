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

# ✅ إضافة require_any_role (لأن chat.py يحتاجها)
def require_any_role(roles: list):
    """
    FastAPI decorator — requires any of the specified roles.
    
    Usage:
        @require_any_role([Role.DOCTOR, Role.NURSE, Role.ADMIN])
        async def my_endpoint(...):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            import logging
            log = logging.getLogger("riva.security")
            
            # محاولة استخراج session_id من kwargs
            session_id = kwargs.get("session_id")
            
            # إذا لم يكن موجود، حاول من request
            if not session_id:
                for arg in args:
                    if hasattr(arg, "headers"):
                        session_id = arg.headers.get("X-Session-ID")
                        break
            
            ac = get_access_control()
            session = ac.get_session(session_id) if session_id else None
            
            if not session or session.user.role not in roles:
                raise PermissionError(
                    f"هذه العملية تتطلب أحد الأدوار: {[r.value for r in roles]}"
                )
            
            # إضافة access إلى kwargs
            kwargs["access"] = session.user
            return await func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


__all__ = [
    # Access control
    "get_access_control", "AccessControl",
    "Role", "Permission", "User", "Session",
    "require_role", "require_any_role",  # ✅ أضفناها هنا
    "PAGE_ACCESS",
    # Digital signature
    "get_signer", "sign_document", "verify_document",
    "DigitalSigner", "SignatureRecord",
    # Encryption
    "get_encryption_handler", "EncryptionHandler", "ALL_DATABASES",
    # Key manager
    "get_key_manager", "get_key_status", "setup_riva_keys",
    "KeyManager", "KeyMetadata",
]
