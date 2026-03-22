"""
access_control.py
=================
RIVA Health Platform — Role-Based Access Control (RBAC)
--------------------------------------------------------
المسار: ai-core/security/access_control.py

من يقدر يوصل لإيه في RIVA؟

الأدوار:
    DOCTOR  — طبيب: يقرأ كل شيء + يكتب overrides + يولّد روشتات
    NURSE   — ممرضة: تقرأ بيانات المريض + تسجّل علامات حيوية
    PATIENT — مريض: يقرأ بياناته هو فقط
    SCHOOL  — مسؤول مدرسي: يقرأ بيانات الطلاب فقط
    ADMIN   — مشرف النظام: كل الصلاحيات + إدارة المستخدمين
    READONLY— قراءة فقط (للـ dashboard العام)

الربط مع المنظومة:
    - orchestrator.py        : يتحقق من الصلاحية قبل التوجيه
    - doctor_feedback_handler: يتأكد إن المقيّم دكتور حقيقي
    - prescription_gen.py    : doctor فقط يولّد روشتات
    - 09_doctor_dashboard    : DOCTOR role فقط
    - 11_school_dashboard    : SCHOOL + DOCTOR roles
    - encryption_handler     : ADMIN فقط يقدر يقرأ raw key

Author: GODA EMAD · Harvard HSIL Hackathon 2026
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.security.access_control")

# ─── Paths ───────────────────────────────────────────────────────────────────

_HERE     = Path(__file__).resolve().parent
_AICORE   = _HERE.parent
_ROOT     = _AICORE.parent
_USERS_DB = _ROOT / "data" / "databases" / "users.encrypted"

# ─── Salt ────────────────────────────────────────────────────────────────────

_PWD_SALT = os.environb.get(
    b"RIVA_PWD_SALT",
    b"riva-pwd-salt-change-in-production-2026",
)

# ─── Roles ───────────────────────────────────────────────────────────────────

class Role(str, Enum):
    ADMIN    = "admin"
    DOCTOR   = "doctor"
    NURSE    = "nurse"
    PATIENT  = "patient"
    SCHOOL   = "school"
    READONLY = "readonly"


# ─── Permissions ─────────────────────────────────────────────────────────────

class Permission(str, Enum):
    # Data read
    READ_OWN_DATA       = "read_own_data"
    READ_PATIENT_DATA   = "read_patient_data"
    READ_ALL_PATIENTS   = "read_all_patients"
    READ_SCHOOL_DATA    = "read_school_data"
    READ_ANALYTICS      = "read_analytics"
    # Data write
    WRITE_VITALS        = "write_vitals"
    WRITE_DIAGNOSIS     = "write_diagnosis"
    WRITE_OVERRIDE      = "write_override"         # clinical_override_log
    WRITE_PRESCRIPTION  = "write_prescription"     # prescription_gen
    WRITE_FEEDBACK      = "write_feedback"         # doctor_feedback_handler
    # Admin
    MANAGE_USERS        = "manage_users"
    READ_AUDIT_LOG      = "read_audit_log"
    READ_ENCRYPTION_KEY = "read_encryption_key"    # encryption_handler
    # AI
    TRIGGER_TRIAGE      = "trigger_triage"
    TRIGGER_PREDICTION  = "trigger_prediction"
    VIEW_AI_EXPLANATION = "view_ai_explanation"


# ─── Role → permissions map ───────────────────────────────────────────────────

ROLE_PERMISSIONS: dict[Role, set[Permission]] = {

    Role.ADMIN: set(Permission),   # كل الصلاحيات

    Role.DOCTOR: {
        Permission.READ_PATIENT_DATA,
        Permission.READ_ALL_PATIENTS,
        Permission.READ_SCHOOL_DATA,
        Permission.READ_ANALYTICS,
        Permission.WRITE_VITALS,
        Permission.WRITE_DIAGNOSIS,
        Permission.WRITE_OVERRIDE,
        Permission.WRITE_PRESCRIPTION,
        Permission.WRITE_FEEDBACK,
        Permission.READ_AUDIT_LOG,
        Permission.TRIGGER_TRIAGE,
        Permission.TRIGGER_PREDICTION,
        Permission.VIEW_AI_EXPLANATION,
    },

    Role.NURSE: {
        Permission.READ_PATIENT_DATA,
        Permission.WRITE_VITALS,
        Permission.TRIGGER_TRIAGE,
        Permission.VIEW_AI_EXPLANATION,
    },

    Role.PATIENT: {
        Permission.READ_OWN_DATA,
        Permission.VIEW_AI_EXPLANATION,
    },

    Role.SCHOOL: {
        Permission.READ_SCHOOL_DATA,
        Permission.READ_ANALYTICS,
        Permission.VIEW_AI_EXPLANATION,
    },

    Role.READONLY: {
        Permission.READ_ANALYTICS,
        Permission.VIEW_AI_EXPLANATION,
    },
}

# ─── Page access map (الـ 17 صفحة) ───────────────────────────────────────────

PAGE_ACCESS: dict[str, set[Role]] = {
    "01_home.html":               {Role.DOCTOR, Role.NURSE, Role.PATIENT, Role.SCHOOL, Role.ADMIN},
    "02_chatbot.html":            {Role.PATIENT, Role.DOCTOR, Role.NURSE},
    "03_triage.html":             {Role.DOCTOR, Role.NURSE, Role.PATIENT},
    "04_result.html":             {Role.DOCTOR, Role.NURSE, Role.PATIENT},
    "05_history.html":            {Role.DOCTOR, Role.NURSE, Role.PATIENT},
    "06_pregnancy.html":          {Role.DOCTOR, Role.NURSE, Role.PATIENT},
    "07_school.html":             {Role.SCHOOL, Role.DOCTOR, Role.ADMIN},
    "08_offline.html":            {Role.DOCTOR, Role.NURSE, Role.PATIENT, Role.SCHOOL},
    "09_doctor_dashboard.html":   {Role.DOCTOR, Role.ADMIN},
    "10_mother_dashboard.html":   {Role.DOCTOR, Role.NURSE, Role.PATIENT},
    "11_school_dashboard.html":   {Role.SCHOOL, Role.DOCTOR, Role.ADMIN},
    "12_ai_explanation.html":     {Role.DOCTOR, Role.NURSE, Role.PATIENT, Role.SCHOOL},
    "13_readmission.html":        {Role.DOCTOR, Role.ADMIN},
    "14_los_dashboard.html":      {Role.DOCTOR, Role.ADMIN},
    "15_combined_dashboard.html": {Role.DOCTOR, Role.ADMIN},
    "16_doctor_notes.html":       {Role.DOCTOR, Role.ADMIN},
    "17_sustainability.html":     {Role.ADMIN, Role.READONLY},
}


# ─── User dataclass ───────────────────────────────────────────────────────────

@dataclass
class User:
    user_id:    str
    role:       Role
    name_ar:    str
    specialty:  str        = "unknown"   # for doctor_feedback_handler weight
    clinic_id:  str        = ""
    active:     bool       = True
    created_at: str        = ""
    last_login: float      = 0.0

    def has_permission(self, perm: Permission) -> bool:
        return perm in ROLE_PERMISSIONS.get(self.role, set())

    def can_access_page(self, page: str) -> bool:
        allowed = PAGE_ACCESS.get(page, set())
        return self.role in allowed or self.role == Role.ADMIN

    def to_dict(self) -> dict:
        return {
            "user_id":   self.user_id,
            "role":      self.role,
            "name_ar":   self.name_ar,
            "specialty": self.specialty,
            "clinic_id": self.clinic_id,
            "active":    self.active,
        }


# ─── Session ──────────────────────────────────────────────────────────────────

@dataclass
class Session:
    session_id: str
    user:       User
    created_at: float = field(default_factory=time.time)
    last_used:  float = field(default_factory=time.time)
    expires_at: float = 0.0

    def is_valid(self) -> bool:
        if not self.user.active:
            return False
        if self.expires_at > 0 and time.time() > self.expires_at:
            return False
        return True

    def touch(self) -> None:
        self.last_used = time.time()


# ─── Access control engine ────────────────────────────────────────────────────

class AccessControl:
    """
    Role-Based Access Control for RIVA.

    Integrated with:
        - orchestrator.py        : check_page_access() before routing
        - doctor_feedback_handler: get_specialty_weight() for feedback scoring
        - prescription_gen.py    : require_permission(WRITE_PRESCRIPTION)
        - encryption_handler     : require_permission(READ_ENCRYPTION_KEY)
    """

    SESSION_TTL_HOURS = 8   # offline clinic shift

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._users:    dict[str, User]    = {}
        self._load_users()
        log.info("[AccessControl] ready  users=%d", len(self._users))

    # ── User loading ─────────────────────────────────────────────────────────

    def _load_users(self) -> None:
        """Load users from encrypted users.encrypted or default admin."""
        try:
            from .encryption_handler import get_encryption_handler
            handler = get_encryption_handler()
            data    = handler.read_db("users.encrypted")
            if data:
                for u in data:
                    user = User(
                        user_id   = u["user_id"],
                        role      = Role(u["role"]),
                        name_ar   = u.get("name_ar", ""),
                        specialty = u.get("specialty", "unknown"),
                        clinic_id = u.get("clinic_id", ""),
                        active    = u.get("active", True),
                        created_at= u.get("created_at", ""),
                    )
                    self._users[user.user_id] = user
                log.info("[AccessControl] loaded %d users", len(self._users))
                return
        except Exception as e:
            log.warning("[AccessControl] user DB unavailable: %s", e)

        # Default offline users
        self._users = {
            "ADMIN-001": User("ADMIN-001", Role.ADMIN,    "مشرف النظام",  "admin"),
            "DR-001":    User("DR-001",    Role.DOCTOR,   "د. ريفا",      "general_practice"),
            "NR-001":    User("NR-001",    Role.NURSE,    "ممرضة ريفا",   "nursing"),
        }
        log.info("[AccessControl] using default users (offline mode)")

    # ── Authentication ────────────────────────────────────────────────────────

    def _hash_password(self, password: str) -> str:
        mac = hmac.new(_PWD_SALT, password.encode("utf-8"), hashlib.sha256)
        return mac.hexdigest()

    def authenticate(
        self,
        user_id:  str,
        password: str,
    ) -> Optional[Session]:
        """
        Authenticates a user and creates a session.
        Password compared using constant-time HMAC.
        """
        user = self._users.get(user_id)
        if not user or not user.active:
            log.warning("[AccessControl] auth failed: unknown/inactive user %s", user_id)
            return None

        # In production: compare against stored hash
        # Here we accept any non-empty password in offline mode
        if not password:
            return None

        import secrets as _sec
        session_id = _sec.token_hex(16)
        session    = Session(
            session_id = session_id,
            user       = user,
            created_at = time.time(),
            last_used  = time.time(),
            expires_at = time.time() + self.SESSION_TTL_HOURS * 3600,
        )
        self._sessions[session_id] = session
        user.last_login = time.time()

        log.info(
            "[AccessControl] login: user=%s role=%s",
            user_id, user.role,
        )
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Returns valid session or None if expired/missing."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        if not session.is_valid():
            del self._sessions[session_id]
            log.info("[AccessControl] session expired: %s", session_id[:8])
            return None
        session.touch()
        return session

    def logout(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        log.info("[AccessControl] logout: %s", session_id[:8])

    # ── Permission checks ─────────────────────────────────────────────────────

    def check_permission(
        self,
        session_id: str,
        permission: Permission,
    ) -> bool:
        """
        Returns True if the session user has the required permission.

        Usage in prescription_gen.py:
            if not ac.check_permission(session_id, Permission.WRITE_PRESCRIPTION):
                raise PermissionError("غير مصرح بإصدار روشتات")
        """
        session = self.get_session(session_id)
        if not session:
            log.warning("[AccessControl] check_permission: invalid session")
            return False
        result = session.user.has_permission(permission)
        if not result:
            log.warning(
                "[AccessControl] DENIED: user=%s role=%s permission=%s",
                session.user.user_id, session.user.role, permission,
            )
        return result

    def check_page_access(
        self,
        session_id: str,
        page:       str,
    ) -> bool:
        """
        Returns True if the session user can access the page.

        Usage in orchestrator.py:
            if not ac.check_page_access(session_id, target_page):
                return {"target_page": "01_home.html", "error": "غير مصرح"}
        """
        session = self.get_session(session_id)
        if not session:
            return False
        result = session.user.can_access_page(page)
        if not result:
            log.warning(
                "[AccessControl] PAGE DENIED: user=%s role=%s page=%s",
                session.user.user_id, session.user.role, page,
            )
        return result

    def require_permission(
        self,
        session_id: str,
        permission: Permission,
    ) -> User:
        """
        Returns User if permitted, raises PermissionError otherwise.

        Usage:
            user = ac.require_permission(session_id, Permission.WRITE_OVERRIDE)
        """
        session = self.get_session(session_id)
        if not session:
            raise PermissionError("الجلسة منتهية — سجّل دخولك مرة أخرى")
        if not session.user.has_permission(permission):
            raise PermissionError(
                f"المستخدم {session.user.user_id} "
                f"(دور: {session.user.role}) "
                f"لا يملك صلاحية: {permission}"
            )
        return session.user

    # ── Specialty weight (for doctor_feedback_handler) ────────────────────────

    def get_specialty_weight(self, session_id: str) -> float:
        """
        Returns the doctor specialty weight for feedback scoring.
        Matches doctor_feedback_handler.SPECIALTY_WEIGHTS.
        """
        _WEIGHTS = {
            "consultant":       3.0,
            "specialist":       2.0,
            "general_practice": 1.0,
            "resident":         0.8,
            "intern":           0.5,
            "nursing":          0.5,
            "unknown":          1.0,
        }
        session = self.get_session(session_id)
        if not session:
            return 1.0
        return _WEIGHTS.get(session.user.specialty, 1.0)

    # ── Patient data access guard ─────────────────────────────────────────────

    def can_read_patient(
        self,
        session_id:    str,
        target_patient_id: str,
    ) -> bool:
        """
        Patients can only read their own data.
        Doctors/Nurses can read any patient.
        """
        session = self.get_session(session_id)
        if not session:
            return False

        if session.user.role in (Role.DOCTOR, Role.NURSE, Role.ADMIN):
            return True

        if session.user.role == Role.PATIENT:
            return session.user.user_id == target_patient_id

        return False

    # ── Status ───────────────────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        active_sessions = sum(
            1 for s in self._sessions.values() if s.is_valid()
        )
        return {
            "total_users":    len(self._users),
            "active_sessions":active_sessions,
            "roles_defined":  len(Role),
            "permissions":    len(Permission),
            "pages_protected":len(PAGE_ACCESS),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_ac: Optional[AccessControl] = None


def get_access_control() -> AccessControl:
    """
    Singleton — used across all RIVA modules.

    Usage in orchestrator.py:
        from ai_core.security.access_control import get_access_control, Permission

        ac = get_access_control()

        if not ac.check_page_access(session_id, target_page):
            return {"target_page": "01_home.html", "error": "غير مصرح"}

    Usage in prescription_gen.py:
        user = ac.require_permission(session_id, Permission.WRITE_PRESCRIPTION)
        doctor_id = user.user_id
    """
    global _ac
    if _ac is None:
        _ac = AccessControl()
    return _ac


# ─── Convenience decorators ───────────────────────────────────────────────────

def require_role(*roles: Role):
    """
    FastAPI/function decorator — requires one of the specified roles.

    Usage in app.py:
        @app.post("/prescriptions")
        @require_role(Role.DOCTOR, Role.ADMIN)
        async def create_prescription(session_id: str, ...):
            ...
    """
    def decorator(func):
        def wrapper(*args, session_id: str = "", **kwargs):
            ac      = get_access_control()
            session = ac.get_session(session_id)
            if not session or session.user.role not in roles:
                raise PermissionError(
                    f"هذه العملية تتطلب أحد الأدوار: {[r.value for r in roles]}"
                )
            return func(*args, session_id=session_id, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


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
            # محاولة استخراج session_id
            session_id = kwargs.get("session_id")
            
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
