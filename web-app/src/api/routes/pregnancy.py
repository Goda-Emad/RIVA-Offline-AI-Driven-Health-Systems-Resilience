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

# ================== 1️⃣ إعداد logging بدلاً من print ==================
logger = logging.getLogger(__name__)

# ================== 2️⃣ إعداد المسارات الديناميكية (مرنة لأي هيكل) ==================
# في Production، هنستخدم متغيرات البيئة
import os
from dotenv import load_dotenv

# تحميل ملف .env لو موجود
load_dotenv()

# المسار الأساسي: إما من .env أو من هيكل المجلدات
BASE_DIR = Path(os.getenv("RIVA_BASE_DIR", Path(__file__).resolve().parent.parent.parent.parent))

# ✅ تم التحديث للنموذج الجديد (بدل القديم)
MODEL_PATH = BASE_DIR / "ai-core" / "models" / "pregnancy" / "maternal_health_optimized_pipeline.pkl"

# ✅ باقي الملفات زي ما هي (اتحدثت مع الرفع)
FEATURES_PATH = BASE_DIR / "ai-core" / "models" / "pregnancy" / "feature_names.json"
CLASS_NAMES_PATH = BASE_DIR / "ai-core" / "models" / "pregnancy" / "class_names.json"

# ✅ الملفات التانية زي ما هي (مفيش تغيير)
RISK_FACTORS_PATH = BASE_DIR / "business-intelligence" / "medical-content" / "pregnancy_risk_factors.json"
SAMPLES_PATH = BASE_DIR / "data-storage" / "samples" / "pregnancy_samples.min.json"

router = APIRouter()

# ================== 3️⃣ نموذج البيانات مع Validations (Pydantic V2) ==================
class PregnancyRequest(BaseModel):
    """بيانات الحامل (بالعامية المصرية)"""
    patient_id: str
    age: int = Field(..., ge=15, le=50, description="السن (15-50 سنة)")
    systolic_bp: int = Field(..., ge=60, le=200, description="الضغط الانقباضي - الكبير")
    diastolic_bp: int = Field(..., ge=40, le=130, description="الضغط الانبساطي - الصغير")
    bs: float = Field(..., ge=3, le=25, description="مستوى السكر")
    body_temp: float = Field(..., ge=35, le=42, description="درجة الحرارة")
    heart_rate: int = Field(..., ge=40, le=180, description="نبض القلب")
    
    # ✅ @classmethod إلزامي مع field_validator في Pydantic V2
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
    risk_level: str                # low/mid/high
    risk_level_ar: str             # منخفض/متوسط/مرتفع
    confidence: float
    pulse_pressure: int
    bp_ratio: float
    temp_fever: int
    recommendations: str

# ================== 4️⃣ متغيرات عامة (هتتحمل في Startup مش هنا) ==================
# هتتحمل في main.py باستخدام lifespan
model = None
feature_names = None
class_names = None
risk_factors = None
samples_cache = None

# ================== 5️⃣ دوال التحميل (تتنادى من main.py) ==================
async def load_pregnancy_resources():
    """تحميل جميع موارد الحمل (تتنادى عند startup)"""
    global model, feature_names, class_names, risk_factors, samples_cache
    
    logger.info("📦 بدأ تحميل موارد الحمل...")
    
    try:
        # تحميل النموذج
        logger.info(f"تحميل النموذج من: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        
        # تحميل أسماء الميزات
        with open(FEATURES_PATH, 'r') as f:
            feature_names = json.load(f)
        
        # تحميل أسماء الفئات
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        
        # تحميل عوامل الخطورة
        with open(RISK_FACTORS_PATH, 'r') as f:
            risk_factors = json.load(f)
        
        # تحميل العينات
        with open(SAMPLES_PATH, 'r') as f:
            samples_cache = json.load(f)
        
        logger.info(f"✅ تم التحميل: {len(feature_names)} ميزة، {len(class_names)} فئة، {len(samples_cache)} عينة")
        return True
        
    except Exception as e:
        logger.error(f"❌ فشل تحميل موارد الحمل: {e}")
        return False

async def unload_pregnancy_resources():
    """تنظيف الموارد (تتنادى عند shutdown)"""
    global model, feature_names, class_names, risk_factors, samples_cache
    
    logger.info("🧹 تنظيف موارد الحمل...")
    model = None
    feature_names = None
    class_names = None
    risk_factors = None
    samples_cache = None
    logger.info("✅ تم التنظيف")

# ================== 6️⃣ دالة لتجهيز الميزات ==================
def prepare_features(data: PregnancyRequest):
    """تجهيز الميزات بالترتيب نفسه اللي اتدرب عليه الموديل"""
    
    # حساب جميع الميزات الممكنة
    features_dict = {
        "Age": data.age,
        "SystolicBP": data.systolic_bp,
        "DiastolicBP": data.diastolic_bp,
        "BS": data.bs,
        "BodyTemp": data.body_temp,
        "HeartRate": data.heart_rate,
        "PulsePressure": data.systolic_bp - data.diastolic_bp,
        "BP_ratio": round(data.systolic_bp / data.diastolic_bp, 2),
        "Temp_Fever": 1 if data.body_temp > 37.5 else 0,
        "HighBP": 1 if data.systolic_bp > 140 else 0,
        "MeanBP": (data.systolic_bp + 2 * data.diastolic_bp) / 3,
        "AgeRisk": 1 if data.age > 35 else 0,
        "HighSugar": 1 if data.bs > 7 else 0
    }
    
    # التحقق من وجود كل الميزات المطلوبة
    missing_features = [f for f in feature_names if f not in features_dict]
    if missing_features:
        raise ValueError(f"ميزات ناقصة: {missing_features}")
    
    # ترتيب الميزات حسب feature_names
    features_list = [features_dict[name] for name in feature_names]
    
    # تحويل لـ DataFrame
    df = pd.DataFrame([features_list], columns=feature_names)
    return df

# ================== 7️⃣ نقاط النهاية (Endpoints) ==================

@router.post("/pregnancy/predict", response_model=PregnancyResponse)
async def predict_pregnancy_risk(data: PregnancyRequest):
    """
    توقع مستوى خطورة الحمل
    """
    # التأكد إن الموارد محملة
    if model is None or feature_names is None:
        logger.error("محاولة توقع قبل تحميل الموارد")
        raise HTTPException(status_code=503, detail="الخدمة غير متاحة حالياً (الموارد بتتحمل)")
    
    try:
        # تجهيز الميزات
        features_df = prepare_features(data)
        logger.debug(f"تم تجهيز ميزات للمريض {data.patient_id}")
        
        # توقع
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence = float(max(probabilities))
        
        # ترجمة النتيجة
        risk_map_ar = {0: "منخفض", 1: "متوسط", 2: "مرتفع"}
        
        # توصيات
        recommendations_map = {
            0: "متابعة الحمل بشكل طبيعي مع الالتزام بالفحوصات الدورية",
            1: "زيارة طبيب النساء مرة شهرياً مع متابعة الضغط والسكر",
            2: "تدخل طبي فوري ومتابعة مكثفة في وحدة الحمل عالي الخطورة"
        }
        
        logger.info(f"✅ توقع للمريض {data.patient_id}: {class_names[prediction]} بثقة {confidence:.2f}")
        
        return PregnancyResponse(
            patient_id=data.patient_id,
            risk_level=class_names[prediction],
            risk_level_ar=risk_map_ar[prediction],
            confidence=confidence,
            pulse_pressure=data.systolic_bp - data.diastolic_bp,
            bp_ratio=round(data.systolic_bp / data.diastolic_bp, 2),
            temp_fever=1 if data.body_temp > 37.5 else 0,
            recommendations=recommendations_map[prediction]
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
    if risk_factors is None:
        raise HTTPException(status_code=503, detail="عوامل الخطورة غير محملة")
    return risk_factors

@router.get("/pregnancy/samples")
async def get_samples(limit: int = 10):
    """جلب عينات الحمل"""
    if samples_cache is None:
        raise HTTPException(status_code=503, detail="العينات غير محملة")
    return samples_cache[:min(limit, len(samples_cache))]

@router.get("/pregnancy/health")
async def health_check():
    """التحقق من صحة الخدمة"""
    return {
        "status": "✅ خدمة pregnancy شغالة",
        "model_loaded": model is not None,
        "features": len(feature_names) if feature_names else 0,
        "classes": class_names if class_names else [],
        "samples_loaded": len(samples_cache) if samples_cache else 0,
        "model_path": str(MODEL_PATH)
    }
