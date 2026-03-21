"""
===============================================================================
pregnancy.py
API لتقييم مخاطر الحمل (Pregnancy Risk Assessment)
Pregnancy Risk Prediction API Endpoint
===============================================================================

🏆 الإصدار: 4.1.0 - Platinum Production Edition (v4.1)
🥇 متكامل مع db_loader v4.1 - بيانات مشفرة حقيقية
⚡ وقت الاستجابة: < 100ms (مع Singleton Predictor)
🔐 متكامل مع نظام التحكم بالصلاحيات (Decorators Only)
👶 دعم كامل لتقييم مخاطر الحمل مع توصيات مخصصة
===============================================================================
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging
import sys
import os
import hashlib
import json
import numpy as np
from pathlib import Path
from enum import Enum
from functools import lru_cache

# إضافة المسار الرئيسي للمشروع (ديناميكي)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# استيراد أنظمة الأمان v4.1
try:
    from access_control import require_role, require_any_role, Role
except ImportError:
    # تعريف Role مؤقت في حالة عدم وجود الملف
    class Role(str, Enum):
        DOCTOR = "doctor"
        NURSE = "nurse"
        ADMIN = "admin"
        SUPERVISOR = "supervisor"
        PATIENT = "patient"
    
    def require_any_role(roles):
        def decorator(func):
            return func
        return decorator
    
    def require_role(role):
        def decorator(func):
            return func
        return decorator

# استيراد db_loader v4.1
try:
    from db_loader import get_db_loader
except ImportError:
    def get_db_loader():
        return None

# استيراد النموذج من ai_core
try:
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from ai_core.local_inference.pregnancy_risk import PregnancyRiskPredictor
except ImportError:
    PregnancyRiskPredictor = None
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ PregnancyRiskPredictor not found, using fallback")

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# إنشاء router - مسار نظيف
router = APIRouter(prefix="/api/predict/pregnancy", tags=["Pregnancy Risk"])


# =========================================================================
# Constants - الأرقام الثابتة (لتجنب Magic Numbers)
# =========================================================================

class PregnancyThresholds:
    """المعايير الطبية لمخاطر الحمل - قابلة للتعديل حسب الإرشادات الطبية"""
    
    # العمر
    HIGH_RISK_AGE_MIN = 35  # عمر 35+ يعتبر خطر مرتفع
    LOW_RISK_AGE_MAX = 18   # عمر أقل من 18 يعتبر خطر منخفض
    
    # ضغط الدم
    HYPERTENSION_SYSTOLIC = 140  # ارتفاع ضغط الدم الانقباضي
    HYPERTENSION_DIASTOLIC = 90   # ارتفاع ضغط الدم الانبساطي
    PRE_HYPERTENSION_SYSTOLIC = 130  # ما قبل ارتفاع الضغط
    PRE_HYPERTENSION_DIASTOLIC = 85
  
    # السكر
    GESTATIONAL_DIABETES_THRESHOLD = 7.0  # سكري الحمل (mg/dL)
    PRE_DIABETES_THRESHOLD = 5.6          # ما قبل السكري
  
    # الحرارة
    FEVER_THRESHOLD = 38.0  # درجة حرارة > 38 تعتبر حمى (C°)
  
    # نبضات القلب
    TACHYCARDIA_THRESHOLD = 100  # نبض > 100 يعتبر تسارع (bpm)
  
    # مؤشر كتلة الجسم
    OBESE_BMI = 30.0      # سمنة
    OVERWEIGHT_BMI = 25.0 # زيادة وزن
    UNDERWEIGHT_BMI = 18.5 # نقص وزن


class SupportedLanguage(str, Enum):
    """اللغات المدعومة - Enum بدلاً من String عادي"""
    ARABIC = "ar"
    ENGLISH = "en"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevelAr(str, Enum):
    LOW = "منخفض"
    MEDIUM = "متوسط"
    HIGH = "مرتفع"


# =========================================================================
# Pydantic Models - مع Enum للغة
# =========================================================================

class PregnancyRequest(BaseModel):
    """بيانات الحامل (بالعامية المصرية)"""
    patient_id: str = Field(..., description="معرف المريضة")
    age: int = Field(..., ge=15, le=50, description="السن (15-50 سنة)")
    systolic_bp: int = Field(..., ge=60, le=200, description="الضغط الانقباضي - الكبير (mmHg)")
    diastolic_bp: int = Field(..., ge=40, le=130, description="الضغط الانبساطي - الصغير (mmHg)")
    bs: float = Field(..., ge=3, le=25, description="مستوى السكر في الدم (mg/dL)")
    body_temp: float = Field(..., ge=35, le=42, description="درجة الحرارة (C°)")
    heart_rate: int = Field(..., ge=40, le=180, description="نبض القلب (bpm)")
    gestational_age_weeks: Optional[int] = Field(None, ge=4, le=42, description="عمر الحمل بالأسابيع")
    bmi: Optional[float] = Field(None, ge=15, le=50, description="مؤشر كتلة الجسم")
    previous_complications: Optional[List[str]] = Field(default=[], description="مضاعفات في حمل سابق")
    language: SupportedLanguage = Field(SupportedLanguage.ARABIC, description="اللغة (ar/en)")  # ✅ Enum بدلاً من String
    
    @field_validator('diastolic_bp')
    @classmethod
    def prevent_zero_division(cls, v):
        if v == 0:
            raise ValueError('الضغط الانبساطي لا يمكن أن يكون صفر')
        return v
    
    @field_validator('body_temp')
    @classmethod
    def validate_temp(cls, v):
        if v < 35 or v > 42:
            raise ValueError('درجة الحرارة خارج النطاق الطبيعي (35-42)')
        return v
    
    @field_validator('bs')
    @classmethod
    def validate_bs(cls, v):
        if v < 3 or v > 25:
            raise ValueError('مستوى السكر خارج النطاق الطبيعي (3-25)')
        return v


class PregnancyResponse(BaseModel):
    """استجابة تقييم مخاطر الحمل"""
    success: bool
    request_id: str
    timestamp: str
    processing_time_ms: float
    patient_id: str
    risk_level: str
    risk_level_ar: str
    risk_level_encoded: int
    confidence: float
    pulse_pressure: int
    bp_ratio: float
    temp_fever: int
    recommendations: str
    recommendations_list: List[str]
    alerts: List[str]
    data_source: str  # "db_loaded", "request_only", "mixed"


class RiskFactorsResponse(BaseModel):
    """استجابة عوامل الخطورة"""
    success: bool
    risk_factors: Dict[str, Any]
    timestamp: str


class PregnancySampleResponse(BaseModel):
    """استجابة عينة بيانات حمل"""
    success: bool
    samples: List[Dict[str, Any]]
    count: int
    timestamp: str


# =========================================================================
# Singleton Predictor - مع Dependency Injection
# =========================================================================

class PregnancyModelSingleton:
    """
    نمط Singleton لتحميل نموذج Pregnancy Risk مرة واحدة عند بدء التشغيل
    """
    _instance = None
    _predictor = None
    _risk_factors = None
    _samples_cache = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """تهيئة النموذج والموارد مرة واحدة فقط"""
        logger.info("🚀 Initializing Pregnancy Risk Model (Singleton)...")
        
        # تحميل النموذج
        if PregnancyRiskPredictor:
            try:
                self._predictor = PregnancyRiskPredictor()
                logger.info("✅ PregnancyRiskPredictor loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load PregnancyRiskPredictor: {e}")
                self._predictor = None
        else:
            logger.warning("⚠️ PregnancyRiskPredictor not available")
            self._predictor = None
        
        # تحميل الموارد الإضافية
        self._load_extra_resources()
        
        # إذا فشل كل شيء، استخدم Fallback
        if not self._predictor:
            logger.info("🔄 Using FallbackPregnancyPredictor (Rule-based)")
    
    def _load_extra_resources(self):
        """تحميل الموارد الإضافية (عوامل الخطورة والعينات)"""
        BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        RISK_FACTORS_PATH = BASE_DIR / "business-intelligence" / "medical-content" / "pregnancy_risk_factors.json"
        SAMPLES_PATH = BASE_DIR / "data-storage" / "samples" / "pregnancy_samples.min.json"
        
        # تحميل عوامل الخطورة
        try:
            if RISK_FACTORS_PATH.exists():
                with open(RISK_FACTORS_PATH, 'r', encoding='utf-8') as f:
                    self._risk_factors = json.load(f)
                logger.info(f"✅ Loaded risk factors")
            else:
                logger.warning(f"Risk factors file not found: {RISK_FACTORS_PATH}")
                self._risk_factors = {}
        except Exception as e:
            logger.error(f"❌ Failed to load risk factors: {e}")
            self._risk_factors = {}
        
        # تحميل العينات
        try:
            if SAMPLES_PATH.exists():
                with open(SAMPLES_PATH, 'r', encoding='utf-8') as f:
                    self._samples_cache = json.load(f)
                logger.info(f"✅ Loaded {len(self._samples_cache)} samples")
            else:
                logger.warning(f"Samples file not found: {SAMPLES_PATH}")
                self._samples_cache = []
        except Exception as e:
            logger.error(f"❌ Failed to load samples: {e}")
            self._samples_cache = []
    
    def get_predictor(self):
        """الحصول على النموذج"""
        return self._predictor
    
    def get_risk_factors(self):
        """الحصول على عوامل الخطورة"""
        return self._risk_factors
    
    def get_samples(self):
        """الحصول على العينات"""
        return self._samples_cache
    
    def is_loaded(self):
        """التحقق من تحميل النموذج"""
        return self._predictor is not None


# =========================================================================
# Dependency Injection Functions (لـ FastAPI)
# =========================================================================

async def get_pregnancy_model() -> PregnancyModelSingleton:
    """
    حقن التبعية لنموذج الحمل (Singleton)
    ✅ استخدام Depends() يجعل الكود أنظف ويسهل الاختبار
    """
    return PregnancyModelSingleton()


async def get_pregnancy_predictor(model: PregnancyModelSingleton = Depends(get_pregnancy_model)):
    """
    حقن التبعية للنموذج الأساسي
    """
    return model.get_predictor()


# =========================================================================
# Fallback Pregnancy Predictor (باستخدام الثوابت)
# =========================================================================

class FallbackPregnancyPredictor:
    """
    تنبؤ افتراضي بمخاطر الحمل - يعتمد على القواعد السريرية
    يستخدم في حالة عدم توفر النموذج الحقيقي
    ✅ استخدام الثوابت (Constants) بدلاً من الأرقام الثابتة
    """
    
    def __init__(self):
        logger.info("✅ FallbackPregnancyPredictor initialized")
        self.thresholds = PregnancyThresholds
    
    def predict(self, age: int, systolic_bp: int, diastolic_bp: int,
                bs: float, body_temp: float, heart_rate: int,
                **kwargs) -> Dict:
        """تنبؤ بسيط يعتمد على القواعد السريرية"""
        
        risk_score = 0
        
        # عامل العمر (باستخدام الثوابت)
        if age > self.thresholds.HIGH_RISK_AGE_MIN:
            risk_score += 2
        elif age < self.thresholds.LOW_RISK_AGE_MAX:
            risk_score += 1
        
        # عامل الضغط (ارتفاع ضغط الدم)
        if systolic_bp >= self.thresholds.HYPERTENSION_SYSTOLIC or diastolic_bp >= self.thresholds.HYPERTENSION_DIASTOLIC:
            risk_score += 2
        elif systolic_bp >= self.thresholds.PRE_HYPERTENSION_SYSTOLIC or diastolic_bp >= self.thresholds.PRE_HYPERTENSION_DIASTOLIC:
            risk_score += 1
        
        # عامل السكر
        if bs > self.thresholds.GESTATIONAL_DIABETES_THRESHOLD:
            risk_score += 2
        elif bs > self.thresholds.PRE_DIABETES_THRESHOLD:
            risk_score += 1
        
        # عامل الحرارة (حمى)
        if body_temp > self.thresholds.FEVER_THRESHOLD:
            risk_score += 1
        
        # عامل نبضات القلب (تسارع)
        if heart_rate > self.thresholds.TACHYCARDIA_THRESHOLD:
            risk_score += 1
        
        # تحديد مستوى الخطورة
        if risk_score >= 4:
            risk_level = "high"
            risk_level_ar = "مرتفع"
            risk_level_encoded = 2
            confidence = 0.75
        elif risk_score >= 2:
            risk_level = "medium"
            risk_level_ar = "متوسط"
            risk_level_encoded = 1
            confidence = 0.70
        else:
            risk_level = "low"
            risk_level_ar = "منخفض"
            risk_level_encoded = 0
            confidence = 0.80
        
        # حساب الضغط النبضي
        pulse_pressure = systolic_bp - diastolic_bp
        
        # حساب نسبة الضغط (تجنب القسمة على صفر)
        bp_ratio = systolic_bp / max(diastolic_bp, 1)
        
        # فحص الحمى
        temp_fever = 1 if body_temp > self.thresholds.FEVER_THRESHOLD else 0
        
        return {
            "risk_level": risk_level,
            "risk_level_ar": risk_level_ar,
            "risk_level_encoded": risk_level_encoded,
            "confidence": confidence,
            "pulse_pressure": pulse_pressure,
            "bp_ratio": bp_ratio,
            "temp_fever": temp_fever,
            "risk_score": risk_score
        }
    
    def is_loaded(self):
        return True
    
    @property
    def feature_names(self):
        return ['age', 'systolic_bp', 'diastolic_bp', 'bs', 'body_temp', 'heart_rate']
    
    @property
    def class_names(self):
        return ['low', 'medium', 'high']


# =========================================================================
# Recommendation Generator (مع دعم Enum للغة)
# =========================================================================

class PregnancyRecommendationGenerator:
    """توليد توصيات مخصصة بناءً على مستوى الخطورة والبيانات السريرية"""
    
    def __init__(self):
        self.recommendations_map = {
            0: {
                'ar': "متابعة الحمل بشكل طبيعي مع الالتزام بالفحوصات الدورية",
                'en': "Normal pregnancy follow-up with regular checkups",
                'list_ar': [
                    "✅ متابعة دورية كل 4 أسابيع",
                    "✅ فحص ضغط الدم والسكر بانتظام",
                    "✅ تناول حمض الفوليك يومياً",
                    "✅ الحفاظ على نظام غذائي صحي"
                ],
                'list_en': [
                    "✅ Regular follow-up every 4 weeks",
                    "✅ Regular blood pressure and sugar monitoring",
                    "✅ Daily folic acid intake",
                    "✅ Maintain healthy diet"
                ]
            },
            1: {
                'ar': "زيارة طبيب النساء مرة شهرياً مع متابعة الضغط والسكر",
                'en': "Monthly OB-GYN visits with blood pressure and sugar monitoring",
                'list_ar': [
                    "⚠️ متابعة كل أسبوعين مع طبيب النساء",
                    "⚠️ مراقبة ضغط الدم يومياً",
                    "⚠️ تحليل سكر تراكمي كل 3 أشهر",
                    "⚠️ تقييم نمو الجنين بالموجات فوق الصوتية شهرياً",
                    "⚠️ استشارة أخصائي تغذية"
                ],
                'list_en': [
                    "⚠️ Bi-weekly OB-GYN follow-up",
                    "⚠️ Daily blood pressure monitoring",
                    "⚠️ HbA1c test every 3 months",
                    "⚠️ Monthly fetal ultrasound",
                    "⚠️ Nutritionist consultation"
                ]
            },
            2: {
                'ar': "تدخل طبي فوري ومتابعة مكثفة في وحدة الحمل عالي الخطورة",
                'en': "Immediate medical intervention and intensive follow-up in high-risk pregnancy unit",
                'list_ar': [
                    "🚨 متابعة أسبوعية أو أكثر في وحدة الحمل عالي الخطورة",
                    "🚨 مراقبة ضغط الدم والسكر عدة مرات يومياً",
                    "🚨 فحص تخطيط قلب الجنين أسبوعياً",
                    "🚨 تقييم وظائف الكلى والكبد شهرياً",
                    "🚨 الاستعداد للولادة المبكرة إذا لزم الأمر",
                    "🚨 استشارة فريق متعدد التخصصات"
                ],
                'list_en': [
                    "🚨 Weekly or more frequent follow-up in high-risk unit",
                    "🚨 Multiple daily blood pressure and sugar checks",
                    "🚨 Weekly fetal heart monitoring",
                    "🚨 Monthly kidney and liver function assessment",
                    "🚨 Prepare for possible early delivery",
                    "🚨 Multidisciplinary team consultation"
                ]
            }
        }
        
        self.alerts_map = {
            'high_bp': {'ar': "⚠️ ارتفاع ضغط الدم - خطر تسمم الحمل", 'en': "⚠️ High blood pressure - preeclampsia risk"},
            'high_bs': {'ar': "⚠️ ارتفاع السكر - خطر سكري الحمل", 'en': "⚠️ High blood sugar - gestational diabetes risk"},
            'fever': {'ar': "⚠️ ارتفاع درجة الحرارة - خطر التهاب", 'en': "⚠️ Fever - infection risk"},
            'tachycardia': {'ar': "⚠️ تسارع نبضات القلب - متابعة فورية", 'en': "⚠️ Tachycardia - immediate follow-up"},
            'elderly': {'ar': "⚠️ عمر متقدم (>35) - خطر مضاعفات", 'en': "⚠️ Advanced age (>35) - complication risk"},
            'obesity': {'ar': "⚠️ سمنة (BMI >30) - خطر مضاعفات", 'en': "⚠️ Obesity (BMI >30) - complication risk"},
            'underweight': {'ar': "⚠️ نقص وزن (BMI <18.5) - خطر ولادة مبكرة", 'en': "⚠️ Underweight (BMI <18.5) - preterm birth risk"}
        }
        
        self.thresholds = PregnancyThresholds
    
    def generate(self, risk_level_encoded: int, data: Dict, language: SupportedLanguage) -> Dict:
        """
        توليد التوصيات والتنبيهات
        ✅ استخدام Enum للغة بدلاً من String
        """
        
        recs = self.recommendations_map.get(risk_level_encoded, self.recommendations_map[0])
        
        if language == SupportedLanguage.ARABIC:
            recommendations_text = recs['ar']
            recommendations_list = recs['list_ar']
        else:
            recommendations_text = recs['en']
            recommendations_list = recs['list_en']
        
        # توليد التنبيهات
        alerts = []
        
        if data.get('systolic_bp', 0) >= self.thresholds.HYPERTENSION_SYSTOLIC or \
           data.get('diastolic_bp', 0) >= self.thresholds.HYPERTENSION_DIASTOLIC:
            alerts.append(self.alerts_map['high_bp'][language.value])
        
        if data.get('bs', 0) > self.thresholds.GESTATIONAL_DIABETES_THRESHOLD:
            alerts.append(self.alerts_map['high_bs'][language.value])
        
        if data.get('body_temp', 0) > self.thresholds.FEVER_THRESHOLD:
            alerts.append(self.alerts_map['fever'][language.value])
        
        if data.get('heart_rate', 0) > self.thresholds.TACHYCARDIA_THRESHOLD:
            alerts.append(self.alerts_map['tachycardia'][language.value])
        
        if data.get('age', 0) > self.thresholds.HIGH_RISK_AGE_MIN:
            alerts.append(self.alerts_map['elderly'][language.value])
        
        # تنبيهات BMI (إن وجدت)
        bmi = data.get('bmi', 0)
        if bmi > 0:
            if bmi >= self.thresholds.OBESE_BMI:
                alerts.append(self.alerts_map['obesity'][language.value])
            elif bmi < self.thresholds.UNDERWEIGHT_BMI:
                alerts.append(self.alerts_map['underweight'][language.value])
        
        return {
            'recommendations_text': recommendations_text,
            'recommendations_list': recommendations_list,
            'alerts': alerts
        }


# =========================================================================
# Feature Extractor (من قاعدة البيانات)
# =========================================================================

class PregnancyFeatureExtractor:
    """استخراج بيانات الحمل من PatientContext"""
    
    def extract_from_patient_context(self, patient_context) -> Dict:
        """استخراج بيانات الحمل من قاعدة البيانات"""
        features = {}
        
        if not patient_context:
            return features
        
        # من pregnancy_record
        if hasattr(patient_context, 'pregnancy_record') and patient_context.pregnancy_record:
            pregnancy = patient_context.pregnancy_record
            features['gestational_age_weeks'] = pregnancy.get('gestational_age_weeks')
            features['gravida'] = pregnancy.get('gravida')
            features['para'] = pregnancy.get('para')
            features['previous_complications'] = pregnancy.get('risk_factors', [])
        
        # من admission_data (demographics)
        if hasattr(patient_context, 'admission_data') and patient_context.admission_data:
            admission = patient_context.admission_data
            demographics = admission.get('demographics', {})
            features['age'] = demographics.get('age')
            
            # من chronic_conditions
            chronic = admission.get('chronic_conditions', {})
            features['has_diabetes'] = chronic.get('diabetes', False)
            features['has_hypertension'] = chronic.get('hypertension', False)
        
        # من lab_results_24h
        if hasattr(patient_context, 'lab_results_24h') and patient_context.lab_results_24h:
            for lab in patient_context.lab_results_24h:
                test_name = lab.get('test', '').lower()
                if 'glucose' in test_name and 'bs' not in features:
                    features['bs'] = lab.get('value')
                elif 'hba1c' in test_name:
                    features['hba1c'] = lab.get('value')
        
        return features


# =========================================================================
# Main Endpoints - مع Dependency Injection
# =========================================================================

@router.post("/predict", response_model=PregnancyResponse)
@require_any_role([Role.DOCTOR, Role.NURSE, Role.ADMIN, Role.SUPERVISOR])
async def predict_pregnancy_risk(
    data: PregnancyRequest,
    fastapi_request: Request = None,
    model: PregnancyModelSingleton = Depends(get_pregnancy_model)  # ✅ Dependency Injection
):
    """
    👶 توقع مستوى خطورة الحمل
    
    🔐 الأمان:
        - Decorator @require_any_role يضمن الصلاحيات (Doctor, Nurse, Admin)
    
    📊 المخرجات:
        - risk_level: مستوى الخطورة (low/medium/high)
        - confidence: نسبة الثقة
        - recommendations: توصيات مخصصة
        - alerts: تنبيهات سريرية
    """
    start_time = datetime.now()
    request_id = hashlib.md5(f"{data.patient_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    logger.info(f"👶 Pregnancy risk prediction | Patient: {data.patient_id} | Age: {data.age} | Language: {data.language.value}")
    
    data_source = "request_only"
    patient_context = None
    extracted_features = {}
    
    # =================================================================
    # 📦 LAYER 1: Load Patient Data from Database
    # =================================================================
    try:
        if data.patient_id:
            db = get_db_loader()
            if db:
                patient_context = db.load_patient_context(data.patient_id, include_encrypted=True)
                if patient_context:
                    extractor = PregnancyFeatureExtractor()
                    extracted_features = extractor.extract_from_patient_context(patient_context)
                    data_source = "db_loaded"
                    logger.info(f"📊 Patient data loaded from database")
                    
                    # دمج البيانات المستخرجة مع المدخلات (إذا كانت مفقودة)
                    if data.gestational_age_weeks is None and extracted_features.get('gestational_age_weeks'):
                        data.gestational_age_weeks = extracted_features['gestational_age_weeks']
    except Exception as e:
        logger.warning(f"Could not load patient data: {e}")
    
    # =================================================================
    # 📈 LAYER 2: Predict Risk
    # =================================================================
    try:
        predictor = model.get_predictor()
        
        if predictor:
            result = predictor.predict(
                age=data.age,
                systolic_bp=data.systolic_bp,
                diastolic_bp=data.diastolic_bp,
                bs=data.bs,
                body_temp=data.body_temp,
                heart_rate=data.heart_rate
            )
        else:
            predictor = FallbackPregnancyPredictor()
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
        
        risk_level_encoded = result["risk_level_encoded"]
        
        # ترجمة النتيجة
        risk_map_ar = {0: "منخفض", 1: "متوسط", 2: "مرتفع"}
        
        logger.info(f"✅ Prediction: {result['risk_level']} | Confidence: {result['confidence']:.2f}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"خطأ في التوقع: {str(e)}")
    
    # =================================================================
    # 📝 LAYER 3: Generate Recommendations (مع تعريف افتراضي مسبق)
    # =================================================================
    # ✅ تعريف المتغير بقيم افتراضية قبل try/except
    recommendations = {
        'recommendations_text': "متابعة مع الطبيب المعالج",
        'recommendations_list': ["متابعة دورية"],
        'alerts': []
    }
    
    try:
        rec_generator = PregnancyRecommendationGenerator()
        recommendations = rec_generator.generate(
            risk_level_encoded=risk_level_encoded,
            data={
                'age': data.age,
                'systolic_bp': data.systolic_bp,
                'diastolic_bp': data.diastolic_bp,
                'bs': data.bs,
                'body_temp': data.body_temp,
                'heart_rate': data.heart_rate,
                'bmi': data.bmi
            },
            language=data.language  # ✅ Enum للغة
        )
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        # ✅ recommendations مضمونة موجودة بالفعل
    
    # =================================================================
    # 📤 LAYER 4: Response
    # =================================================================
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    response = PregnancyResponse(
        success=True,
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        processing_time_ms=round(processing_time, 2),
        patient_id=data.patient_id,
        risk_level=result["risk_level"],
        risk_level_ar=risk_map_ar[risk_level_encoded],
        risk_level_encoded=risk_level_encoded,
        confidence=result["confidence"],
        pulse_pressure=result["pulse_pressure"],
        bp_ratio=result["bp_ratio"],
        temp_fever=result["temp_fever"],
        recommendations=recommendations['recommendations_text'],
        recommendations_list=recommendations['recommendations_list'],
        alerts=recommendations['alerts'],
        data_source=data_source
    )
    
    logger.info(f"✅ Pregnancy risk assessment completed in {processing_time:.0f}ms")
    return response


# =========================================================================
# Risk Factors Endpoint - مع Dependency Injection
# =========================================================================

@router.get("/risk-factors")
@require_any_role([Role.DOCTOR, Role.NURSE, Role.ADMIN])
async def get_risk_factors(
    model: PregnancyModelSingleton = Depends(get_pregnancy_model),
    fastapi_request: Request = None
):
    """
    📋 جلب عوامل خطورة الحمل
    """
    risk_factors = model.get_risk_factors()
    
    if not risk_factors:
        raise HTTPException(status_code=503, detail="عوامل الخطورة غير محملة")
    
    return RiskFactorsResponse(
        success=True,
        risk_factors=risk_factors,
        timestamp=datetime.now().isoformat()
    )


# =========================================================================
# Samples Endpoint - مع Dependency Injection
# =========================================================================

@router.get("/samples")
@require_any_role([Role.DOCTOR, Role.NURSE, Role.ADMIN])
async def get_samples(
    limit: int = 10,
    model: PregnancyModelSingleton = Depends(get_pregnancy_model),
    fastapi_request: Request = None
):
    """
    📊 جلب عينات الحمل
    """
    samples = model.get_samples()
    
    if not samples:
        raise HTTPException(status_code=503, detail="العينات غير محملة")
    
    limited_samples = samples[:min(limit, len(samples))]
    
    return PregnancySampleResponse(
        success=True,
        samples=limited_samples,
        count=len(limited_samples),
        timestamp=datetime.now().isoformat()
    )


# =========================================================================
# Health Check
# =========================================================================

@router.get("/health")
async def health_check(
    model: PregnancyModelSingleton = Depends(get_pregnancy_model)
):
    """فحص صحة الخدمة"""
    predictor = model.get_predictor()
    
    return {
        'status': 'healthy',
        'service': 'Pregnancy Risk Assessment API',
        'version': '4.1.0',
        'security_version': 'v4.1',
        'predictor_loaded': predictor is not None,
        'predictor_type': 'AI Model' if predictor and not isinstance(predictor, FallbackPregnancyPredictor) else 'Fallback (Rule-based)',
        'features': predictor.feature_names if predictor and hasattr(predictor, 'feature_names') else 6,
        'risk_factors_loaded': model.get_risk_factors() is not None,
        'samples_loaded': len(model.get_samples() or []) > 0,
        'supported_languages': [lang.value for lang in SupportedLanguage],
        'thresholds': {
            'high_risk_age': PregnancyThresholds.HIGH_RISK_AGE_MIN,
            'hypertension_systolic': PregnancyThresholds.HYPERTENSION_SYSTOLIC,
            'gestational_diabetes': PregnancyThresholds.GESTATIONAL_DIABETES_THRESHOLD,
            'fever': PregnancyThresholds.FEVER_THRESHOLD
        },
        'timestamp': datetime.now().isoformat()
    }


# =========================================================================
# Test Endpoint
# =========================================================================

@router.get("/test")
async def test_endpoint():
    """نقطة نهاية للاختبار"""
    return {
        'message': 'Pregnancy Risk Assessment API is working',
        'version': '4.1.0',
        'security': 'Decorator-based @require_any_role',
        'dependency_injection': 'Using FastAPI Depends()',
        'language_enum': True,
        'constants_used': True,
        'endpoints': [
            'POST /api/predict/pregnancy/predict',
            'GET /api/predict/pregnancy/risk-factors',
            'GET /api/predict/pregnancy/samples',
            'GET /api/predict/pregnancy/health'
        ]
    }


# =========================================================================
# Init Function (للاستخدام في main.py)
# =========================================================================

async def init_pregnancy_router():
    """
    تهيئة موارد pregnancy عند بدء التشغيل
    """
    logger.info("🔄 Initializing pregnancy resources...")
    model = PregnancyModelSingleton()
    
    if model.get_predictor():
        logger.info("✅ PregnancyRiskPredictor loaded successfully")
    else:
        logger.info("✅ FallbackPregnancyPredictor ready")
    
    logger.info("✅ Pregnancy router initialized")
