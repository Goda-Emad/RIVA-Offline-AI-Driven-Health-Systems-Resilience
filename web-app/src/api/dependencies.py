"""
dependencies.py
================
RIVA Health Platform — Core Dependencies
-----------------------------------------
مركز التحكم المركزي لجميع التبعيات في نظام RIVA v4.2.1

Author : GODA EMAD
"""

from __future__ import annotations

import logging
from typing import Optional, AsyncGenerator, Dict, Any
from functools import lru_cache
from enum import Enum

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 0. 🔗 ربط المسارات الصحيحة
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AI_CORE_PATH = PROJECT_ROOT / "ai-core"
SECURITY_PATH = AI_CORE_PATH / "security"
STORAGE_PATH  = AI_CORE_PATH / "storage"

for path in [SECURITY_PATH, STORAGE_PATH, AI_CORE_PATH]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ─────────────────────────────────────────────────────────────────────────────
# 1. 🔐 SECURITY & AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

# تعريف المتغيرات قبل الـ try عشان لو حصل خطأ ميبقاش في مشكلة
get_access_control = None
Role = None
AccessControl = None

try:
    from access_control import get_access_control, Role, AccessControl
    print("✅ Security modules loaded successfully from ai-core/security")
except ImportError as e:
    print(f"⚠️ Warning: Security module error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 🛡️ Role Checker Dependency (التعديل السحري)
# ─────────────────────────────────────────────────────────────────────────────

def role_checker(required_role):
    """
    Dependency to check if the current user has the required role.
    """
    def role_dependency(access = Depends(get_access_control)):
        
        # لو الموديول مكنش موجود أو حصل فيه خطأ في الـ import فوق
        if access is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Security module is not loaded correctly."
            )
        
        # 💡 احتياطي: لو access لسه function لأي سبب، نشغلها إحنا
        if callable(access):
            access = access()
            
        # هنا access بقى الـ Object الفعلي، نقدر ننادي الدوال بتاعته بأمان
        current_role = access.get_user_role()
        
        if current_role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have enough permissions to access this resource."
            )
            
        return access
        
    return role_dependency
    from enum import Enum
    class Role(str, Enum):
        DOCTOR            = "doctor"
        NURSE             = "nurse"
        ADMIN             = "admin"
        SUPERVISOR        = "supervisor"
        PATIENT           = "patient"
        GENETIC_COUNSELOR = "genetic_counselor"
        SCHOOL_NURSE      = "school_nurse"
        PHARMACIST        = "pharmacist"
        PSYCHOLOGIST      = "psychologist"
        SCHOOL            = "school"
        READONLY          = "readonly"

    class AccessControl:
        def __init__(self, request=None):
            self._user_id   = "guest"
            self._user_role = Role.PATIENT
            self.current_user_role = self._user_role

        def authenticate(self):      return True
        def get_user_role(self):     return self._user_role
        def get_user_id(self):       return self._user_id
        def require_role(self, r):   return self._user_role == r
        def require_any_role(self, roles): return self._user_role in roles
        def get_session(self, session_id): return None

    def get_access_control(request):
        return AccessControl(request)

    logging.warning("Using fallback security (no ai-core/security found)")

# دالة require_role المعدلة
def require_role(required_role: Role):
    async def role_checker(
        request: Request,
        ac: AccessControl = Depends(get_access_control)
    ):
        session_id = request.headers.get("X-Session-ID")
        if not session_id:
            session_id = request.cookies.get("session_id")
        
        session = ac.get_session(session_id)
        
        if not session or session.user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"غير مصرح - مطلوب دور: {required_role.value}"
            )
        
        return session.user
    return role_checker

# ─────────────────────────────────────────────────────────────────────────────
# 2. 🗄️ DATABASE LOADER
# ─────────────────────────────────────────────────────────────────────────────

try:
    from db_loader import get_db_loader, DbLoader as DBLoader
    print("✅ Storage module (db_loader) loaded successfully from ai-core/storage")
except ImportError as e:
    print(f"⚠️ Warning: db_loader error: {e}")
    DBLoader = None

    def get_db_loader():
        return None

    logging.warning("Using fallback db_loader (no ai-core/storage found)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 🔐 AUTHENTICATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AccessControl:
    token = None
    if credentials:
        token = credentials.credentials
    if not token:
        token = request.cookies.get("access_token")
    try:
        access = get_access_control(request)
        access.authenticate()
        logging.info(f"✅ User authenticated: {access.get_user_id()} | Role: {access.get_user_role()}")
        return access
    except Exception as e:
        logging.warning(f"❌ Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: AccessControl = Depends(get_current_user)
) -> AccessControl:
    return current_user


async def get_doctor_user(
    current_user: AccessControl = Depends(get_current_active_user)
) -> AccessControl:
    if current_user.get_user_role() != Role.DOCTOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires doctor privileges"
        )
    return current_user


async def get_admin_user(
    current_user: AccessControl = Depends(get_current_active_user)
) -> AccessControl:
    if current_user.get_user_role() not in [Role.ADMIN, Role.SUPERVISOR]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires admin privileges"
        )
    return current_user

# ─────────────────────────────────────────────────────────────────────────────
# 4. 🗄️ DATABASE DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL = "sqlite+aiosqlite:///./data-storage/databases/riva.db"

engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@lru_cache()
def get_db_loader_instance():
    return get_db_loader()


async def get_patient_context(
    patient_id: str,
    db_loader_instance=Depends(get_db_loader_instance)
) -> Dict[str, Any]:
    if db_loader_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database loader not available"
        )
    try:
        patient_context = db_loader_instance.load_patient_context(patient_id)
        if not patient_context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient {patient_id} not found"
            )
        return patient_context
    except Exception as e:
        logging.error(f"Failed to load patient context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load patient data: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 5. 🧠 AI MODEL DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache()
def get_readmission_predictor():
    try:
        from ai_core.prediction.readmission_predictor import ReadmissionPredictor
        return ReadmissionPredictor()
    except ImportError:
        logging.warning("ReadmissionPredictor not available")
        return None


@lru_cache()
def get_los_predictor():
    try:
        from ai_core.prediction.los_predictor import LOSPredictor
        return LOSPredictor()
    except ImportError:
        logging.warning("LOSPredictor not available")
        return None


@lru_cache()
def get_triage_engine():
    try:
        from ai_core.local_inference.triage_engine import TriageEngine
        from pydantic_settings import BaseSettings

        class TriageSettings(BaseSettings):
            model_path:      str = "ai-core/models/triage/model_int8.onnx"
            features_path:   str = "ai-core/models/triage/features.json"
            imputer_path:    str = "ai-core/models/triage/imputer.pkl"
            scaler_path:     str = "ai-core/models/triage/scaler.pkl"
            conflicts_path:  str = "business-intelligence/medical-content/drug_conflicts.json"

        s = TriageSettings()
        return TriageEngine(
            model_path=s.model_path, imputer_path=s.imputer_path,
            scaler_path=s.scaler_path, features_path=s.features_path,
            conflicts_path=s.conflicts_path
        )
    except ImportError:
        logging.warning("TriageEngine not available")
        return None


@lru_cache()
def get_sentiment_analyzer():
    try:
        from ai_core.local_inference.sentiment_analyzer import SentimentAnalyzer
        return SentimentAnalyzer()
    except ImportError:
        logging.warning("SentimentAnalyzer not available")
        return None


@lru_cache()
def get_school_health_analyzer():
    try:
        from ai_core.local_inference.school_health import SchoolHealthAnalyzer
        from pydantic_settings import BaseSettings

        class SchoolSettings(BaseSettings):
            standards_path: str = "data/raw/who_growth/who_growth_standards.json"
            clusters_path:  str = "ai-core/models/school/cluster_centers.json"

        s = SchoolSettings()
        return SchoolHealthAnalyzer(
            standards_path=s.standards_path, clusters_path=s.clusters_path
        )
    except ImportError:
        logging.warning("SchoolHealthAnalyzer not available")
        return None


@lru_cache()
def get_drug_interaction_checker():
    try:
        from ai_core.local_inference.drug_interaction import DrugInteractionChecker
        from pydantic_settings import BaseSettings

        class DrugSettings(BaseSettings):
            csv_path:          str = "data/raw/drug_bank/drug_interactions.csv"
            interactions_path: str = "ai-core/data/interactions.json"

        s = DrugSettings()
        return DrugInteractionChecker(
            csv_path=s.csv_path, data_path=s.interactions_path
        )
    except ImportError:
        logging.warning("DrugInteractionChecker not available")
        return None


@lru_cache()
def get_prescription_generator():
    try:
        from ai_core.local_inference.prescription_gen import PrescriptionGenerator
        return PrescriptionGenerator()
    except ImportError:
        logging.warning("PrescriptionGenerator not available")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 6. 🎤 VOICE DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache()
def get_voice_encoder():
    ENCODER_PATH = AI_CORE_PATH / "models" / "chatbot" / "whisper_int8" / "encoder_model_quantized.onnx"
    if not ENCODER_PATH.exists():
        logging.warning(f"Voice encoder not found at {ENCODER_PATH}")
        return None
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        return ort.InferenceSession(
            str(ENCODER_PATH), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        logging.error(f"Failed to load voice encoder: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 7. 🚀 ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

async def get_orchestrator():
    try:
        from .routes.orchestrator import process_consultation_sync
        return process_consultation_sync
    except ImportError:
        logging.error("Orchestrator not available")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 8. 📝 LOGGING & MONITORING
# ─────────────────────────────────────────────────────────────────────────────

async def log_request(request: Request) -> None:
    logging.info(
        f"📊 Request: {request.method} {request.url.path} | "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 9. 🏥 PATIENT CONTEXT WITH AUTH
# ─────────────────────────────────────────────────────────────────────────────

async def get_patient_context_with_auth(
    patient_id: str,
    current_user: AccessControl = Depends(get_current_active_user),
    db_loader_instance=Depends(get_db_loader_instance)
) -> Dict[str, Any]:
    user_role = current_user.get_user_role()
    user_id   = current_user.get_user_id()

    if user_role == Role.PATIENT and user_id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own medical records"
        )
    if db_loader_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database loader not available"
        )
    patient_context = db_loader_instance.load_patient_context(patient_id)
    if not patient_context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    return patient_context

# ─────────────────────────────────────────────────────────────────────────────
# 10. 🏥 SCHOOL CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

async def get_school_context(
    school_id: str,
    current_user: AccessControl = Depends(get_current_active_user)
) -> Dict[str, Any]:
    try:
        db_loader_instance = get_db_loader()
        if db_loader_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database loader not available"
            )
        school_data = db_loader_instance.load_school_context(school_id)
        if not school_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"School {school_id} not found"
            )
        return school_data
    except Exception as e:
        logging.error(f"Failed to load school context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load school data: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 11. 📊 HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

async def get_system_status() -> Dict[str, Any]:
    result = {
        "database":    "unknown",
        "ai_models":   {},
        "voice_models":{},
        "security":    "ok"
    }
    try:
        db_loader_instance = get_db_loader()
        result["database"] = "ok" if db_loader_instance else "unavailable"
    except Exception:
        result["database"] = "error"

    result["ai_models"]["readmission"] = "ok" if get_readmission_predictor() else "missing"
    result["ai_models"]["los"]         = "ok" if get_los_predictor()          else "missing"
    result["ai_models"]["triage"]      = "ok" if get_triage_engine()          else "missing"
    result["ai_models"]["sentiment"]   = "ok" if get_sentiment_analyzer()     else "missing"
    result["voice_models"]["encoder"]  = "ok" if get_voice_encoder()          else "missing"

    return result

# ─────────────────────────────────────────────────────────────────────────────
# 12. 📦 EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "get_current_user",
    "get_current_active_user",
    "get_doctor_user",
    "get_admin_user",
    "get_db",
    "get_db_loader_instance",
    "get_patient_context",
    "get_patient_context_with_auth",
    "get_readmission_predictor",
    "get_los_predictor",
    "get_triage_engine",
    "get_sentiment_analyzer",
    "get_school_health_analyzer",
    "get_drug_interaction_checker",
    "get_prescription_generator",
    "get_voice_encoder",
    "get_orchestrator",
    "log_request",
    "get_system_status",
    "get_school_context",
    "Role",
    "DBLoader",
    "require_role",
]
