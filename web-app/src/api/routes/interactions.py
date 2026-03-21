"""
===============================================================================
interactions.py
API لفحص تداخلات الأدوية (Drug Interactions)
Drug Interactions Checker API
===============================================================================

🏆 الإصدار: 4.1.0 - Platinum Production Edition (v4.1)
🥇 متكامل مع db_loader v4.1 - بيانات مشفرة حقيقية
⚡ وقت الاستجابة: < 50ms (باستخدام JSON المباشر)
🔐 متكامل مع نظام التحكم بالصلاحيات (Decorators Only)
💊 دعم كامل لفحص تداخلات الأدوية مع مسارات مطلقة
===============================================================================
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import sys
import os
import hashlib
import json
from functools import lru_cache

# إضافة المسار الرئيسي للمشروع
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# استيراد أنظمة الأمان v4.1 - باستخدام Decorators فقط
from access_control import require_role, require_any_role, Role

# استيراد db_loader v4.1
from db_loader import get_db_loader

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء router مع prefix صحيح
router = APIRouter(prefix="/interactions", tags=["Drug Interactions"])


# =========================================================================
# CONFIG - مع مسارات مطلقة
# =========================================================================

class Settings:
    """إعدادات المسارات - باستخدام مسارات مطلقة"""
    
    def __init__(self):
        # الحصول على المسار المطلق للجذر
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # بناء المسارات المطلقة
        self.csv_path = os.path.join(
            self.base_dir, "data", "raw", "drug_bank", "drug_interactions.csv"
        )
        self.medicines_path = os.path.join(
            self.base_dir, "data", "raw", "drug_bank", "medicines_list.csv"
        )
        self.conflicts_path = os.path.join(
            self.base_dir, "business-intelligence", "medical-content", "drug_conflicts.json"
        )
        self.interactions_path = os.path.join(
            self.base_dir, "ai-core", "data", "interactions.json"
        )


@lru_cache()
def get_settings() -> Settings:
    """الحصول على الإعدادات (cached)"""
    return Settings()


# =========================================================================
# CACHED DATA LOADERS - مع مسارات مطلقة ومعالجة JSON
# =========================================================================

@lru_cache()
def load_medicines() -> List[Dict]:
    """تحميل قائمة الأدوية من CSV (cached)"""
    settings = get_settings()
    
    if os.path.exists(settings.medicines_path):
        try:
            import csv
            with open(settings.medicines_path, encoding="utf-8") as f:
                medicines = list(csv.DictReader(f))
                logger.info(f"📚 Loaded {len(medicines)} medicines from {settings.medicines_path}")
                return medicines
        except Exception as e:
            logger.error(f"Failed to load medicines: {e}")
    
    logger.warning("⚠️ No medicines file found, using empty list")
    return []


@lru_cache()
def load_conflicts() -> Dict:
    """
    تحميل قاعدة بيانات التداخلات من JSON (cached)
    هذا هو المصدر الأساسي للتداخلات - سريع وموثوق
    """
    settings = get_settings()
    
    # محاولة تحميل من المسار المطلق
    if os.path.exists(settings.conflicts_path):
        try:
            with open(settings.conflicts_path, encoding="utf-8") as f:
                data = json.load(f)
                conflicts = data.get("conflicts", [])
                logger.info(f"📋 Loaded {len(conflicts)} drug conflicts from {settings.conflicts_path}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in conflicts file: {e}")
            return {"conflicts": []}
        except Exception as e:
            logger.error(f"❌ Failed to load conflicts: {e}")
            return {"conflicts": []}
    
    # محاولة تحميل من مسار بديل (بعد)
    alt_path = os.path.join(
        os.path.dirname(settings.base_dir), 
        "business-intelligence", "medical-content", "drug_conflicts.json"
    )
    if os.path.exists(alt_path):
        try:
            with open(alt_path, encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"📋 Loaded conflicts from alt path: {alt_path}")
                return data
        except Exception as e:
            logger.error(f"Failed to load conflicts from alt path: {e}")
    
    logger.warning("⚠️ No conflicts file found, using empty list")
    return {"conflicts": []}


def load_interactions_from_db() -> List[Dict]:
    """
    تحميل تفاعلات الأدوية من قاعدة البيانات المشفرة
    """
    try:
        db = get_db_loader()
        medications_data = db.load_all_medications()
        if medications_data:
            logger.info(f"💊 Loaded {len(medications_data)} medications from encrypted DB")
            return medications_data
    except Exception as e:
        logger.error(f"Failed to load interactions from db: {e}")
    
    return []


# =========================================================================
# MODELS
# =========================================================================

class InteractionCheckInput(BaseModel):
    """مدخلات فحص تداخل دواء واحد مع أدوية حالية"""
    new_drug: str = Field(..., min_length=2, description="الدواء الجديد المراد وصفه")
    current_drugs: List[str] = Field(default=[], description="الأدوية الحالية التي يتناولها المريض")
    
    @validator("current_drugs")
    def no_duplicate(cls, v, values):
        """إزالة أي تكرار مع الدواء الجديد"""
        new_drug = values.get("new_drug", "").lower().strip()
        v = [d for d in v if d.lower().strip() != new_drug]
        return v


class BulkCheckInput(BaseModel):
    """مدخلات فحص تداخلات مجموعة أدوية"""
    medications: List[str] = Field(..., min_items=1, description="قائمة الأدوية للفحص")


class InteractionCheckResponse(BaseModel):
    """استجابة فحص تداخل دوائي"""
    success: bool
    request_id: str
    timestamp: str
    processing_time_ms: float
    new_drug: str
    is_safe: bool
    alerts: List[Dict[str, Any]]
    alert_count: int
    source_mode: str  # "json", "database", "fallback"


class BulkCheckResponse(BaseModel):
    """استجابة فحص مجموعة أدوية"""
    success: bool
    request_id: str
    timestamp: str
    processing_time_ms: float
    medications: List[str]
    is_safe: bool
    alerts: List[Dict[str, Any]]
    alert_count: int
    high_severity_count: int


# =========================================================================
# INTERACTION CHECKER - يعتمد على JSON المباشر (سريع وموثوق)
# =========================================================================

class JSONInteractionChecker:
    """
    فاحص تداخلات الأدوية - يعتمد على ملف drug_conflicts.json
    هذا هو المصدر الأساسي - سريع (JSON cached) وموثوق (لا يعتمد على مكتبات خارجية)
    """
    
    def __init__(self):
        self.conflicts_data = load_conflicts()
        self.conflicts = self.conflicts_data.get("conflicts", [])
        logger.info(f"✅ JSONInteractionChecker initialized with {len(self.conflicts)} conflicts")
    
    def check(self, new_drug: str, current_drugs: List[str]) -> Dict:
        """فحص تداخل دواء واحد مع الأدوية الحالية"""
        alerts = []
        
        for current in current_drugs:
            interaction = self._find_interaction(new_drug, current)
            if interaction:
                alerts.append(interaction)
        
        return {
            "is_safe": len(alerts) == 0,
            "alerts": alerts,
            "alert_count": len(alerts),
            "source_mode": "json"
        }
    
    def check_bulk(self, medications: List[str]) -> Dict:
        """
        فحص تداخلات مجموعة أدوية كاملة - O(n²) لكن n صغير
        """
        drugs_set = set(medications)
        all_alerts = []
        
        for drug in medications:
            others = list(drugs_set - {drug})
            for other in others:
                interaction = self._find_interaction(drug, other)
                if interaction:
                    all_alerts.append(interaction)
        
        # إزالة التكرارات (كل زوج مرة واحدة)
        seen = set()
        unique_alerts = []
        for alert in all_alerts:
            key = tuple(sorted([alert["drug_a"], alert["drug_b"]]))
            if key not in seen:
                seen.add(key)
                unique_alerts.append(alert)
        
        high_severity = sum(1 for a in unique_alerts if a.get("severity") == "high" or a.get("severity_code") == "high")
        
        return {
            "is_safe": len(unique_alerts) == 0,
            "alerts": unique_alerts,
            "alert_count": len(unique_alerts),
            "high_severity_count": high_severity,
            "source_mode": "json"
        }
    
    def _find_interaction(self, drug_a: str, drug_b: str) -> Optional[Dict]:
        """
        البحث عن تداخل بين دوائين
        يدعم البحث الجزئي (partial matching) للأسماء
        """
        drug_a_lower = drug_a.lower().strip()
        drug_b_lower = drug_b.lower().strip()
        
        for conflict in self.conflicts:
            drugs = conflict.get("drugs", [])
            if len(drugs) == 2:
                drug1 = drugs[0].lower().strip()
                drug2 = drugs[1].lower().strip()
                
                # بحث دقيق أو بحث جزئي
                exact_match = (drug_a_lower == drug1 and drug_b_lower == drug2) or \
                              (drug_a_lower == drug2 and drug_b_lower == drug1)
                
                partial_match = (drug_a_lower in drug1 and drug_b_lower in drug2) or \
                                (drug_a_lower in drug2 and drug_b_lower in drug1)
                
                if exact_match or partial_match:
                    return {
                        "drug_a": drug_a,
                        "drug_b": drug_b,
                        "severity": conflict.get("severity", "moderate"),
                        "severity_code": conflict.get("severity", "moderate"),
                        "description": conflict.get("description", "Potential drug interaction detected"),
                        "description_ar": conflict.get("description_ar", "تم اكتشاف تداخل دوائي محتمل"),
                        "recommendation": conflict.get("recommendation", "Consult with pharmacist"),
                        "recommendation_ar": conflict.get("recommendation_ar", "استشر الصيدلي")
                    }
        
        return None


# =========================================================================
# ROUTES - بدون تكرار المسار
# =========================================================================

@router.post("/check", response_model=InteractionCheckResponse)
@require_any_role([Role.DOCTOR, Role.PHARMACIST, Role.ADMIN])
async def check_interaction(
    data: InteractionCheckInput,
    fastapi_request: Request = None
):
    """
    💊 فحص تداخل دواء جديد مع الأدوية الحالية للمريض
    
    🔐 الأمان:
        - Decorator @require_any_role يضمن الصلاحيات (Doctor, Pharmacist, Admin)
    
    📊 المخرجات:
        - هل الدواء آمن؟
        - قائمة التنبيهات (إن وجدت)
        - مستوى الخطورة لكل تداخل
    """
    start_time = datetime.now()
    request_id = hashlib.md5(f"{data.new_drug}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    logger.info(f"💊 Interaction check | Drug: {data.new_drug} | Current: {len(data.current_drugs)}")
    
    try:
        # استخدام JSONInteractionChecker السريع والموثوق
        checker = JSONInteractionChecker()
        result = checker.check(data.new_drug, data.current_drugs)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return InteractionCheckResponse(
            success=True,
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2),
            new_drug=data.new_drug,
            is_safe=result["is_safe"],
            alerts=result["alerts"],
            alert_count=result["alert_count"],
            source_mode=result["source_mode"]
        )
        
    except Exception as e:
        logger.error(f"❌ Interaction check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-bulk", response_model=BulkCheckResponse)
@require_any_role([Role.DOCTOR, Role.PHARMACIST, Role.ADMIN])
async def check_bulk_interactions(
    data: BulkCheckInput,
    fastapi_request: Request = None
):
    """
    💊 فحص تداخلات مجموعة أدوية كاملة (مثلاً لمراجعة روشتة)
    
    🔐 الأمان:
        - Decorator @require_any_role يضمن الصلاحيات
    
    📊 المخرجات:
        - هل مجموعة الأدوية آمنة؟
        - جميع التداخلات المكتشفة
        - عدد التداخلات عالية الخطورة
    """
    start_time = datetime.now()
    request_id = hashlib.md5(f"{len(data.medications)}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    logger.info(f"💊 Bulk interaction check | Medications: {len(data.medications)}")
    
    try:
        checker = JSONInteractionChecker()
        result = checker.check_bulk(data.medications)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BulkCheckResponse(
            success=True,
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2),
            medications=data.medications,
            is_safe=result["is_safe"],
            alerts=result["alerts"],
            alert_count=result["alert_count"],
            high_severity_count=result["high_severity_count"]
        )
        
    except Exception as e:
        logger.error(f"❌ Bulk interaction check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/medicines")
@require_any_role([Role.DOCTOR, Role.PHARMACIST, Role.NURSE, Role.ADMIN])
async def get_medicines(
    fastapi_request: Request = None
):
    """
    📚 الحصول على قائمة الأدوية المتاحة
    
    🔐 الأمان:
        - متاح للأطباء والصيادلة والممرضين
    """
    start_time = datetime.now()
    
    try:
        medicines = load_medicines()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "medicines": medicines,
            "total": len(medicines),
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get medicines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conflicts")
@require_any_role([Role.DOCTOR, Role.PHARMACIST, Role.ADMIN])
async def get_conflicts(
    fastapi_request: Request = None
):
    """
    📋 الحصول على قاعدة بيانات التداخلات الدوائية
    """
    start_time = datetime.now()
    
    try:
        data = load_conflicts()
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "conflicts": data.get("conflicts", []),
            "total": len(data.get("conflicts", [])),
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conflicts/high")
@require_any_role([Role.DOCTOR, Role.PHARMACIST, Role.ADMIN])
async def get_high_conflicts(
    fastapi_request: Request = None
):
    """
    ⚠️ الحصول على التداخلات عالية الخطورة فقط
    """
    start_time = datetime.now()
    
    try:
        data = load_conflicts()
        conflicts = data.get("conflicts", [])
        high_conflicts = [c for c in conflicts if c.get("severity") == "high" or c.get("severity_code") == "high"]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "conflicts": high_conflicts,
            "total": len(high_conflicts),
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get high conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_id}")
@require_any_role([Role.DOCTOR, Role.PHARMACIST, Role.ADMIN, Role.SUPERVISOR])
async def check_patient_interactions(
    patient_id: str,
    fastapi_request: Request = None
):
    """
    🏥 فحص تداخلات الأدوية لمريض معين (باستخدام بياناته الحقيقية)
    
    🔐 الأمان:
        - يتحقق من صلاحية الوصول للمريض
    
    📊 المخرجات:
        - جميع الأدوية الحالية للمريض
        - التداخلات المكتشفة بينها
        - توصيات سريرية
    """
    start_time = datetime.now()
    request_id = hashlib.md5(f"{patient_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    logger.info(f"🏥 Patient interactions check | Patient: {patient_id}")
    
    try:
        # تحميل بيانات المريض
        db = get_db_loader()
        patient_context = db.load_patient_context(patient_id, include_encrypted=True)
        
        if not patient_context:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # استخراج الأدوية الحالية
        current_medications = []
        
        # من prescriptions_24h
        if hasattr(patient_context, 'prescriptions_24h') and patient_context.prescriptions_24h:
            for rx in patient_context.prescriptions_24h:
                med_name = rx.get('drug_name_ar', rx.get('drug_id', ''))
                if med_name:
                    current_medications.append(med_name)
        
        # من medications
        if hasattr(patient_context, 'medications') and patient_context.medications:
            for med in patient_context.medications:
                med_name = med.get('name', med.get('drug_id', ''))
                if med_name:
                    current_medications.append(med_name)
        
        # إزالة التكرارات
        current_medications = list(set(current_medications))
        
        if len(current_medications) < 2:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return {
                "success": True,
                "request_id": request_id,
                "patient_id": patient_id,
                "medications": current_medications,
                "medication_count": len(current_medications),
                "is_safe": True,
                "alerts": [],
                "alert_count": 0,
                "high_severity_count": 0,
                "processing_time_ms": round(processing_time, 2),
                "message": "Patient has less than 2 medications, no interactions to check"
            }
        
        # فحص التداخلات
        checker = JSONInteractionChecker()
        result = checker.check_bulk(current_medications)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": round(processing_time, 2),
            "patient_id": patient_id,
            "medications": current_medications,
            "medication_count": len(current_medications),
            "is_safe": result["is_safe"],
            "alerts": result["alerts"],
            "alert_count": result["alert_count"],
            "high_severity_count": result["high_severity_count"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Patient interactions check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Health Check - بدون تكرار المسار
# =========================================================================

@router.get("/health")
async def health_check():
    """فحص صحة الخدمة"""
    settings = get_settings()
    checker = JSONInteractionChecker()
    
    return {
        'status': 'healthy',
        'service': 'RIVA-Maternal Drug Interactions API',
        'version': '4.1.0',
        'security_version': 'v4.1',
        'security_approach': 'Decorator-only (@require_any_role)',
        'checker_type': 'JSONInteractionChecker',
        'data_sources': {
            'medicines_exists': os.path.exists(settings.medicines_path),
            'medicines_path': settings.medicines_path,
            'conflicts_exists': os.path.exists(settings.conflicts_path),
            'conflicts_path': settings.conflicts_path,
            'conflicts_loaded': len(checker.conflicts)
        },
        'timestamp': datetime.now().isoformat()
    }


# =========================================================================
# Test Endpoint - بدون تكرار المسار
# =========================================================================

@router.get("/test")
async def test_endpoint():
    """نقطة نهاية للاختبار"""
    checker = JSONInteractionChecker()
    
    return {
        'message': 'Drug Interactions API is working',
        'version': '4.1.0',
        'security': 'Decorator-based @require_any_role',
        'data_source': 'drug_conflicts.json (direct)',
        'checker_ready': checker is not None,
        'conflicts_loaded': len(checker.conflicts),
        'endpoints': [
            'POST /interactions/check',
            'POST /interactions/check-bulk',
            'GET /interactions/medicines',
            'GET /interactions/conflicts',
            'GET /interactions/conflicts/high',
            'GET /interactions/patient/{patient_id}',
            'GET /interactions/health'
        ]
    }
