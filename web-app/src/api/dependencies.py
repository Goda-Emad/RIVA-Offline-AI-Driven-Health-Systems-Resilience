"""
dependencies.py
================
RIVA Health Platform — Core Dependencies
-----------------------------------------
مركز التحكم المركزي لجميع التبعيات في نظام RIVA v4.2.1

هذا الملف هو "العمود الفقري" للنظام:
    ✓ تطبيق مبدأ DRY (لا تكرار)
    ✓ مركز الأمان والصلاحيات
    ✓ إدارة موارد الـ AI وقاعدة البيانات
    ✓ حقن التبعيات (Dependency Injection) لجميع الـ 16 API

Author : GODA EMAD
"""

from __future__ import annotations

import logging
from typing import Optional, AsyncGenerator, Dict, Any
from functools import lru_cache

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

# إضافة المسار الرئيسي
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# 1. 🔐 SECURITY & AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)

# استيراد أنظمة الأمان v4.2
try:
    from access_control import get_access_control, Role, AccessControl
except ImportError:
    logging.critical("❌ access_control module not found")
    # في الإنتاج، لا نستخدم fallback - السيرفر يتوقف
    raise ImportError("access_control module is required for production")

# استيراد db_loader
try:
    from db_loader import get_db_loader, DBLoader
except ImportError:
    logging.critical("❌ db_loader module not found")
    raise ImportError("db_loader module is required for production")


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AccessControl:
    """
    🔐 استخراج المستخدم الحالي من الـ JWT Token
    
    هذه هي الدالة الأهم - كل Route محمي بيستخدمها
    تتحقق من صحة التوكن وترجع كائن AccessControl للـ Route
    
    Returns:
        AccessControl: كائن يحتوي على user_id, user_role, permissions
        
    Raises:
        HTTPException 401: إذا لم يكن المستخدم مسجلاً
        HTTPException 403: إذا كان التوكن غير صالح
    """
    # استخراج التوكن من الـ header
    token = None
    if credentials:
        token = credentials.credentials
    
    # إذا لم يكن هناك توكن، نحاول استخراجه من الـ cookie
    if not token:
        token = request.cookies.get("access_token")
    
    try:
        # تهيئة AccessControl
        access = get_access_control(request)
        
        # محاولة المصادقة
        # هذه الدالة ستقوم بفك التوكن والتحقق من صحته
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
    """
    🔐 التحقق من أن المستخدم نشط
    
    يمكن إضافة منطق إضافي هنا مثل:
        - التحقق من أن الحساب غير محظور
        - التحقق من صلاحية الـ session
        - تسجيل آخر نشاط
    """
    # يمكن إضافة منطق التحقق من النشاط هنا
    # مثلاً: if current_user.is_blocked: raise HTTPException...
    
    return current_user


async def get_doctor_user(
    current_user: AccessControl = Depends(get_current_active_user)
) -> AccessControl:
    """
    👨‍⚕️ التحقق من أن المستخدم دكتور
    
    يستخدم في المسارات التي تتطلب صلاحيات الدكتور فقط
    """
    if current_user.get_user_role() != Role.DOCTOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires doctor privileges"
        )
    return current_user


async def get_admin_user(
    current_user: AccessControl = Depends(get_current_active_user)
) -> AccessControl:
    """
    👑 التحقق من أن المستخدم Admin
    
    يستخدم في المسارات الإدارية الحساسة
    """
    if current_user.get_user_role() not in [Role.ADMIN, Role.SUPERVISOR]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires admin privileges"
        )
    return current_user


# ─────────────────────────────────────────────────────────────────────────────
# 2. 🗄️ DATABASE DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

# إعداد قاعدة البيانات (SQLAlchemy async)
# في الإنتاج، استخدم متغيرات البيئة للاتصال
DATABASE_URL = "sqlite+aiosqlite:///./data-storage/databases/riva.db"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # في الإنتاج، اجعلها False
    pool_size=10,
    max_overflow=20
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    🗄️ حقن جلسة قاعدة البيانات
    
    تفتح جلسة جديدة لكل Request وتقفلها تلقائياً بعد الانتهاء
    تمنع تسرب الذاكرة (Memory Leaks)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@lru_cache()
def get_db_loader_instance() -> DBLoader:
    """
    📦 الحصول على نسخة واحدة من DBLoader (Singleton)
    
    تستخدم لقراءة البيانات المشفرة من ملفات .encrypted
    """
    return get_db_loader()


async def get_patient_context(
    patient_id: str,
    db_loader: DBLoader = Depends(get_db_loader_instance)
) -> Dict[str, Any]:
    """
    📋 استخراج سياق المريض الكامل
    
    تجمع هذه الدالة:
        - البيانات الأساسية (patients.encrypted)
        - التاريخ الطبي (history.encrypted)
        - التحاليل (lab_results.encrypted)
        - الروشتات (prescriptions.encrypted)
        - بيانات الحمل (pregnancy.encrypted)
        - بيانات العائلة (family_links.encrypted)
    
    Returns:
        PatientContext: كائن يحتوي على جميع بيانات المريض
    """
    try:
        patient_context = db_loader.load_patient_context(patient_id, include_encrypted=True)
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
# 3. 🧠 AI MODEL DEPENDENCIES (Singletons)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache()
def get_readmission_predictor():
    """الحصول على نموذج التنبؤ بإعادة الدخول (Singleton)"""
    try:
        from ai_core.prediction.readmission_predictor import ReadmissionPredictor
        return ReadmissionPredictor()
    except ImportError:
        logging.warning("ReadmissionPredictor not available")
        return None


@lru_cache()
def get_los_predictor():
    """الحصول على نموذج التنبؤ بمدة الإقامة (Singleton)"""
    try:
        from ai_core.prediction.los_predictor import LOSPredictor
        return LOSPredictor()
    except ImportError:
        logging.warning("LOSPredictor not available")
        return None


@lru_cache()
def get_triage_engine():
    """الحصول على محرك الترياج (Singleton)"""
    try:
        from ai_core.local_inference.triage_engine import TriageEngine
        from pydantic_settings import BaseSettings
        
        class TriageSettings(BaseSettings):
            model_path: str = "ai-core/models/triage/model_int8.onnx"
            features_path: str = "ai-core/models/triage/features.json"
            imputer_path: str = "ai-core/models/triage/imputer.pkl"
            scaler_path: str = "ai-core/models/triage/scaler.pkl"
            conflicts_path: str = "business-intelligence/medical-content/drug_conflicts.json"
        
        settings = TriageSettings()
        return TriageEngine(
            model_path=settings.model_path,
            imputer_path=settings.imputer_path,
            scaler_path=settings.scaler_path,
            features_path=settings.features_path,
            conflicts_path=settings.conflicts_path
        )
    except ImportError:
        logging.warning("TriageEngine not available")
        return None


@lru_cache()
def get_sentiment_analyzer():
    """الحصول على محلل المشاعر (Singleton)"""
    try:
        from ai_core.local_inference.sentiment_analyzer import SentimentAnalyzer
        return SentimentAnalyzer()
    except ImportError:
        logging.warning("SentimentAnalyzer not available")
        return None


@lru_cache()
def get_school_health_analyzer():
    """الحصول على محلل الصحة المدرسية (Singleton)"""
    try:
        from ai_core.local_inference.school_health import SchoolHealthAnalyzer
        from pydantic_settings import BaseSettings
        
        class SchoolSettings(BaseSettings):
            standards_path: str = "data/raw/who_growth/who_growth_standards.json"
            clusters_path: str = "ai-core/models/school/cluster_centers.json"
        
        settings = SchoolSettings()
        return SchoolHealthAnalyzer(
            standards_path=settings.standards_path,
            clusters_path=settings.clusters_path
        )
    except ImportError:
        logging.warning("SchoolHealthAnalyzer not available")
        return None


@lru_cache()
def get_drug_interaction_checker():
    """الحصول على فاحص تداخلات الأدوية (Singleton)"""
    try:
        from ai_core.local_inference.drug_interaction import DrugInteractionChecker
        from pydantic_settings import BaseSettings
        
        class DrugSettings(BaseSettings):
            csv_path: str = "data/raw/drug_bank/drug_interactions.csv"
            interactions_path: str = "ai-core/data/interactions.json"
        
        settings = DrugSettings()
        return DrugInteractionChecker(
            csv_path=settings.csv_path,
            data_path=settings.interactions_path
        )
    except ImportError:
        logging.warning("DrugInteractionChecker not available")
        return None


@lru_cache()
def get_prescription_generator():
    """الحصول على مولد الروشتات (Singleton)"""
    try:
        from ai_core.local_inference.prescription_gen import PrescriptionGenerator
        return PrescriptionGenerator()
    except ImportError:
        logging.warning("PrescriptionGenerator not available")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. 🎤 VOICE DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache()
def get_voice_encoder():
    """الحصول على نموذج Whisper Encoder (Singleton)"""
    from pathlib import Path
    ENCODER_PATH = Path(__file__).parent.parent.parent / "ai-core" / "models" / "chatbot" / "whisper_int8" / "encoder_model_quantized.onnx"
    
    if not ENCODER_PATH.exists():
        logging.warning(f"Voice encoder not found at {ENCODER_PATH}")
        return None
    
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        return ort.InferenceSession(
            str(ENCODER_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        logging.error(f"Failed to load voice encoder: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. 🚀 ORCHESTRATOR DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

async def get_orchestrator():
    """
    🎯 الحصول على نسخة من الـ Orchestrator
    
    الـ Orchestrator هو العقل المدبر الذي يربط:
        - الصوت (voice)
        - المحادثة (chat)
        - التوجيه الذكي (routing)
    """
    try:
        from .routes.orchestrator import process_consultation_sync
        return process_consultation_sync
    except ImportError:
        logging.error("Orchestrator not available")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. 📝 LOGGING & MONITORING
# ─────────────────────────────────────────────────────────────────────────────

async def log_request(request: Request) -> None:
    """
    📝 تسجيل كل طلب للـ API
    
    مفيد للمراقبة والتحليل
    """
    logging.info(
        f"📊 Request: {request.method} {request.url.path} | "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. 🏥 PATIENT CONTEXT WITH AUTH
# ─────────────────────────────────────────────────────────────────────────────

async def get_patient_context_with_auth(
    patient_id: str,
    current_user: AccessControl = Depends(get_current_active_user),
    db_loader: DBLoader = Depends(get_db_loader_instance)
) -> Dict[str, Any]:
    """
    🔐 استخراج سياق المريض مع التحقق من الصلاحية
    
    تتأكد أن المستخدم الحالي لديه صلاحية لعرض بيانات هذا المريض
    """
    # التحقق من الصلاحية (الدكتور يمكنه رؤية أي مريض، المريض فقط بياناته)
    user_role = current_user.get_user_role()
    user_id = current_user.get_user_id()
    
    if user_role == Role.PATIENT and user_id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own medical records"
        )
    
    # تحميل بيانات المريض
    patient_context = db_loader.load_patient_context(patient_id, include_encrypted=True)
    if not patient_context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    return patient_context


# ─────────────────────────────────────────────────────────────────────────────
# 8. 🏥 SCHOOL CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

async def get_school_context(
    school_id: str,
    current_user: AccessControl = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    🏫 استخراج سياق المدرسة مع التحقق من الصلاحية
    
    تتأكد أن المستخدم الحالي لديه صلاحية لعرض بيانات هذه المدرسة
    """
    user_role = current_user.get_user_role()
    
    # ممرضة المدرسة يمكنها رؤية مدرستها فقط
    # في الإنتاج، هناك جدول يربط الممرضات بالمدارس
    if user_role == Role.SCHOOL_NURSE:
        # TODO: التحقق من أن الممرضة مرتبطة بهذه المدرسة
        pass
    
    # تحميل بيانات المدرسة
    try:
        db_loader = get_db_loader()
        school_data = db_loader.load_school_context(school_id)
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
# 9. 📊 HEALTH CHECK DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

async def get_system_status() -> Dict[str, Any]:
    """
    📊 الحصول على حالة النظام
    
    تستخدم في نقاط /health
    """
    status = {
        "database": "unknown",
        "ai_models": {},
        "voice_models": {},
        "security": "ok"
    }
    
    # فحص قاعدة البيانات
    try:
        db_loader = get_db_loader()
        if db_loader:
            status["database"] = "ok"
    except Exception:
        status["database"] = "error"
    
    # فحص النماذج
    status["ai_models"]["readmission"] = "ok" if get_readmission_predictor() else "missing"
    status["ai_models"]["los"] = "ok" if get_los_predictor() else "missing"
    status["ai_models"]["triage"] = "ok" if get_triage_engine() else "missing"
    status["ai_models"]["sentiment"] = "ok" if get_sentiment_analyzer() else "missing"
    
    # فحص نماذج الصوت
    status["voice_models"]["encoder"] = "ok" if get_voice_encoder() else "missing"
    
    return status


# ─────────────────────────────────────────────────────────────────────────────
# 10. 📦 EXPORT ALL DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Security
    "get_current_user",
    "get_current_active_user",
    "get_doctor_user",
    "get_admin_user",
    
    # Database
    "get_db",
    "get_db_loader_instance",
    "get_patient_context",
    "get_patient_context_with_auth",
    
    # AI Models
    "get_readmission_predictor",
    "get_los_predictor",
    "get_triage_engine",
    "get_sentiment_analyzer",
    "get_school_health_analyzer",
    "get_drug_interaction_checker",
    "get_prescription_generator",
    
    # Voice
    "get_voice_encoder",
    
    # Orchestrator
    "get_orchestrator",
    
    # Monitoring
    "log_request",
    "get_system_status",
    
    # School
    "get_school_context",
    
    # Roles
    "Role",
]
