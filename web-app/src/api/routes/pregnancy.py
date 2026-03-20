# web-app/src/api/routes/pregnancy.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import logging
from typing import List, Dict, Any, Optional

# ✅ استيراد الـ predictor
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from ai_core.local_inference.pregnancy_risk import PregnancyRiskPredictor

# ================== 1️⃣ إعداد logging ==================
logger = logging.getLogger(__name__)

# ================== 2️⃣ إعداد المسارات ==================
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(os.getenv("RIVA_BASE_DIR", Path(__file__).resolve().parent.parent.parent.parent))

# المسارات للملفات الإضافية
RISK_FACTORS_PATH = BASE_DIR / "business-intelligence" / "medical-content" / "pregnancy_risk_factors.json"
SAMPLES_PATH = BASE_DIR / "data-storage" / "samples" / "pregnancy_samples.min.json"

router = APIRouter()

# ================== 3️⃣ نموذج البيانات ==================
class PregnancyRequest(BaseModel):
    """بيانات الحامل (بالعامية المصرية)"""
    patient_id: str
    age: int = Field(..., ge=15, le=50, description="السن (15-50 سنة)")
    systolic_bp: int = Field(..., ge=60, le=200, description="الضغط الانقباضي - الكبير")
    diastolic_bp: int = Field(..., ge=40, le=130, description="الضغط الانبساطي - الصغير")
    bs: float = Field(..., ge=3, le=25, description="مستوى السكر")
    body_temp: float = Field(..., ge=35, le=42, description="درجة الحرارة")
    heart_rate: int = Field(..., ge=40, le=180, description="نبض القلب")
    
    @classmethod
    @field_validator('diastolic_bp')
    def prevent_zero_division(cls, v):
        if v == 0:
            raise ValueError('الضغط الانبساطي لا يمكن أن يكون صفر')
        return v
    
    @classmethod
    @field_validator('body_temp')
    def validate_temp(cls, v):
        if v < 35 or v > 42:
            raise ValueError('درجة الحرارة خارج النطاق الطبيعي')
        return v

class PregnancyResponse(BaseModel):
    patient_id: str
    risk_level: str
    risk_level_ar: str
    confidence: float
    pulse_pressure: int
    bp_ratio: float
    temp_fever: int
    recommendations: str

# ================== 4️⃣ Singleton Predictor ==================
# ✅ يتم تحميله مرة واحدة فقط عند بدء التشغيل
_predictor: Optional[PregnancyRiskPredictor] = None
_risk_factors: Optional[Dict] = None
_samples_cache: Optional[List] = None

def get_predictor() -> PregnancyRiskPredictor:
    """الحصول على نسخة واحدة من الـ predictor (Singleton)"""
    global _predictor
    if _predictor is None:
        logger.info("🚀 إنشاء نسخة جديدة من PregnancyRiskPredictor (مرة واحدة)")
        _predictor = PregnancyRiskPredictor()
    return _predictor

async def load_extra_resources():
    """تحميل الموارد الإضافية (مرة واحدة)"""
    global _risk_factors, _samples_cache
    
    if _risk_factors is None:
        try:
            with open(RISK_FACTORS_PATH, 'r') as f:
                _risk_factors = json.load(f)
            logger.info(f"✅ تم تحميل عوامل الخطورة")
        except Exception as e:
            logger.error(f"❌ فشل تحميل عوامل الخطورة: {e}")
            _risk_factors = {}
    
    if _samples_cache is None:
        try:
            with open(SAMPLES_PATH, 'r') as f:
                _samples_cache = json.load(f)
            logger.info(f"✅ تم تحميل {len(_samples_cache)} عينة")
        except Exception as e:
            logger.error(f"❌ فشل تحميل العينات: {e}")
            _samples_cache = []

# ================== 5️⃣ نقاط النهاية ==================

@router.post("/pregnancy/predict", response_model=PregnancyResponse)
async def predict_pregnancy_risk(data: PregnancyRequest):
    """
    توقع مستوى خطورة الحمل
    """
    predictor = get_predictor()
    
    if not predictor.is_loaded():
        logger.error("محاولة توقع قبل تحميل الموارد")
        raise HTTPException(status_code=503, detail="الخدمة غير متاحة حالياً (النموذج بيتحمل)")
    
    try:
        # ✅ استخدام predictor مباشرة
        result = predictor.predict(
            age=data.age,
            systolic_bp=data.systolic_bp,
            diastolic_bp=data.diastolic_bp,
            bs=data.bs,
            body_temp=data.body_temp,
            heart_rate=data.heart_rate
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # ترجمة النتيجة
        risk_map_ar = {0: "منخفض", 1: "متوسط", 2: "مرتفع"}
        
        # توصيات
        recommendations_map = {
            0: "متابعة الحمل بشكل طبيعي مع الالتزام بالفحوصات الدورية",
            1: "زيارة طبيب النساء مرة شهرياً مع متابعة الضغط والسكر",
            2: "تدخل طبي فوري ومتابعة مكثفة في وحدة الحمل عالي الخطورة"
        }
        
        # تحويل الرقم للنص
        risk_level_encoded = result["risk_level_encoded"]
        
        logger.info(f"✅ توقع للمريض {data.patient_id}: {result['risk_level']} بثقة {result['confidence']:.2f}")
        
        return PregnancyResponse(
            patient_id=data.patient_id,
            risk_level=result["risk_level"],
            risk_level_ar=risk_map_ar[risk_level_encoded],
            confidence=result["confidence"],
            pulse_pressure=result["pulse_pressure"],
            bp_ratio=result["bp_ratio"],
            temp_fever=result["temp_fever"],
            recommendations=recommendations_map[risk_level_encoded]
        )
        
    except ValueError as ve:
        logger.warning(f"خطأ في البيانات للمريض {data.patient_id}: {ve}")
        raise HTTPException(status_code=400, detail=f"خطأ في البيانات: {str(ve)}")
    except Exception as e:
        logger.error(f"خطأ غير متوقع للمريض {data.patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطأ في التوقع: {str(e)}")

@router.get("/pregnancy/risk-factors")
async def get_risk_factors():
    """جلب عوامل خطورة الحمل"""
    await load_extra_resources()
    if not _risk_factors:
        raise HTTPException(status_code=503, detail="عوامل الخطورة غير محملة")
    return _risk_factors

@router.get("/pregnancy/samples")
async def get_samples(limit: int = 10):
    """جلب عينات الحمل"""
    await load_extra_resources()
    if not _samples_cache:
        raise HTTPException(status_code=503, detail="العينات غير محملة")
    return _samples_cache[:min(limit, len(_samples_cache))]

@router.get("/pregnancy/health")
async def health_check():
    """التحقق من صحة الخدمة"""
    predictor = get_predictor()
    return {
        "status": "✅ خدمة pregnancy شغالة",
        "predictor_loaded": predictor.is_loaded(),
        "features": len(predictor.feature_names) if predictor.feature_names else 0,
        "classes": predictor.class_names if predictor.class_names else [],
        "samples_loaded": len(_samples_cache) if _samples_cache else 0
    }

# ================== 6️⃣ دالة للاستخدام في main.py ==================
async def init_pregnancy_router():
    """تهيئة الموارد عند بدء التشغيل"""
    logger.info("🔄 تهيئة موارد pregnancy...")
    get_predictor()  # تحميل النموذج
    await load_extra_resources()  # تحميل الموارد الإضافية
    logger.info("✅ تم تهيئة جميع موارد pregnancy")
