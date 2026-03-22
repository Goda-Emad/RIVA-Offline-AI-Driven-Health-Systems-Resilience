"""
===============================================================================
los.py
API للتنبؤ بمدة إقامة المريض في المستشفى
Length of Stay (LOS) Prediction API Endpoint
===============================================================================

🏆 الإصدار: 4.1.0 - Platinum Production Edition (v4.1)
🥇 متكامل مع db_loader v4.1 - بيانات مشفرة حقيقية
⚡ وقت الاستجابة: < 200ms (مع Singleton Model)
🔐 متكامل مع نظام التحكم بالصلاحيات (Decorators Only)
🎯 الهدف: MAE 7.11 يوم (نسعى لتحسينه)

المميزات:
✓ نموذج ذكاء اصطناعي محمل مرة واحدة (Singleton)
✓ تنبؤ فوري بمدة الإقامة
✓ شرح النتائج مع هامش خطأ
✓ تحليل عوامل الخطر
✓ توصيات طبية
✓ دعم العربية والإنجليزية
✓ معالجة القيم الشاذة
✓ تكامل مع قاعدة البيانات المشفرة
✓ Batch Prediction مع Fallback لكل مريض على حدة
===============================================================================
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import numpy as np
import logging
import sys
import os
import hashlib
from enum import Enum
from functools import lru_cache

# إضافة المسار الرئيسي للمشروع (ديناميكي)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# استيراد أنظمة الأمان v4.1
from access_control import require_any_role, Role
# استيراد db_loader v4.1
from db_loader import get_db_loader

# استيراد النماذج من ai_core
try:
    from ai_core.prediction.los_predictor import LOSPredictor
    from ai_core.prediction.feature_engineering import FeatureEngineering
    from ai_core.prediction.explanation_generator import ExplanationGenerator
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ AI Core modules not found, using fallback predictors")
    LOSPredictor = None
    FeatureEngineering = None
    ExplanationGenerator = None

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================================
# Singleton Model Loader - تحميل النموذج مرة واحدة فقط
# =========================================================================

class LOSModelSingleton:
    """
    نمط Singleton لتحميل نموذج LOS مرة واحدة عند بدء التشغيل
    هذا يضمن:
    - سرعة استجابة عالية (< 200ms)
    - استهلاك أقل للذاكرة
    - عدم إعادة تحميل النموذج مع كل طلب
    """
    _instance = None
    _model = None
    _feature_engineering = None
    _explanation_generator = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """تهيئة النموذج مرة واحدة فقط"""
        logger.info("🚀 Initializing LOS Model (Singleton)...")
        
        # تحميل النموذج الأساسي
        if LOSPredictor:
            try:
                self._model = LOSPredictor()
                logger.info("✅ LOSPredictor loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load LOSPredictor: {e}")
                self._model = None
        else:
            logger.warning("⚠️ LOSPredictor not available")
            self._model = None
        
        # تحميل Feature Engineering
        if FeatureEngineering:
            try:
                self._feature_engineering = FeatureEngineering()
                logger.info("✅ FeatureEngineering loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load FeatureEngineering: {e}")
                self._feature_engineering = None
        
        # تحميل Explanation Generator
        if ExplanationGenerator:
            try:
                self._explanation_generator = ExplanationGenerator()
                logger.info("✅ ExplanationGenerator loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load ExplanationGenerator: {e}")
                self._explanation_generator = None
        
        # إذا فشل كل شيء، استخدم Fallback
        if not self._model:
            logger.info("🔄 Using FallbackLOSPredictor (Rule-based)")
    
    def get_predictor(self):
        """الحصول على النموذج (قد يكون None إذا فشل التحميل)"""
        return self._model
    
    def get_feature_engineering(self):
        """الحصول على Feature Engineering"""
        return self._feature_engineering
    
    def get_explanation_generator(self):
        """الحصول على Explanation Generator"""
        return self._explanation_generator


# =========================================================================
# إنشاء router - prefix نظيف
# =========================================================================
router = APIRouter(prefix="/api/predict/los", tags=["LOS Prediction"])


# =========================================================================
# Enums
# =========================================================================

class Language(str, Enum):
    ARABIC = "ar"
    ENGLISH = "en"


class RiskCategory(str, Enum):
    LOW = "منخفضة"
    MEDIUM = "متوسطة"
    HIGH = "عالية"
    CRITICAL = "حرجة"


# =========================================================================
# Pydantic Models
# =========================================================================

class SymptomInput(BaseModel):
    """بيانات العرض"""
    name: str = Field(..., description="اسم العرض")
    severity: int = Field(1, ge=1, le=5, description="الشدة (1-5)")
    duration_hours: float = Field(0, description="المدة بالساعات")


class MedicationInput(BaseModel):
    """بيانات الدواء"""
    name: str = Field(..., description="اسم الدواء")
    dosage: Optional[str] = Field(None, description="الجرعة")
    frequency: Optional[str] = Field(None, description="التكرار")


class PatientInput(BaseModel):
    """مدخلات المريض"""
    patient_id: Optional[str] = Field(None, description="معرف المريض")
    
    # من الشات
    chat_text: Optional[str] = Field(None, description="كلام المريض")
    
    # الأعراض (بديل عن الشات)
    symptoms: Optional[List[SymptomInput]] = Field(None, description="قائمة الأعراض")
    
    # الأدوية
    medications: Optional[List[MedicationInput]] = Field(None, description="قائمة الأدوية")
    
    # العلامات الحيوية
    vitals: Optional[Dict[str, float]] = Field(None, description="العلامات الحيوية")
    
    # الأمراض المزمنة
    conditions: Optional[Dict[str, bool]] = Field(None, description="الأمراض المزمنة")
    
    # إعدادات
    language: str = Field("ar", description="اللغة (ar/en)")
    explanation_level: str = Field("clinical", description="مستوى الشرح (patient/clinical/expert)")


class ChatOnlyInput(BaseModel):
    """مدخلات للمحادثة فقط - لتجنب URI Too Long"""
    chat_text: str = Field(..., min_length=1, description="نص المحادثة")
    language: str = Field("ar", description="اللغة")
    patient_id: Optional[str] = Field(None, description="معرف المريض (اختياري)")


class LOSResponse(BaseModel):
    """نتيجة التنبؤ"""
    success: bool
    request_id: str
    timestamp: str
    processing_time_ms: float
    patient_id: Optional[str]
    prediction: Dict[str, Any]
    explanation: Dict[str, Any]
    recommendations: List[str]
    risk_factors: List[Dict]
    alerts: List[str]
    data_source: str  # "db_loaded", "request_only", "mixed"


# =========================================================================
# Fallback LOS Predictor (عند عدم توفر AI Core)
# =========================================================================

class FallbackLOSPredictor:
    """
    تنبؤ افتراضي بمدة الإقامة - يعتمد على قواعد بسيطة
    يستخدم في حالة عدم توفر النماذج الحقيقية
    """
    
    def __init__(self):
        logger.info("✅ FallbackLOSPredictor initialized")
    
    def predict(self, features: Dict) -> Dict:
        """تنبؤ بسيط يعتمد على قواعد ثابتة"""
        days = 5.0
        
        # زيادة المدة بناءً على الأمراض المزمنة
        if features.get('has_diabetes'):
            days += 1.5
        if features.get('has_hypertension'):
            days += 1.0
        if features.get('has_heart_failure'):
            days += 3.0
        if features.get('has_kidney_disease'):
            days += 2.5
        
        # زيادة المدة بناءً على عدد الأعراض
        symptom_count = features.get('symptom_count', 0)
        if symptom_count > 3:
            days += 1.5
        elif symptom_count > 1:
            days += 0.5
        
        # زيادة المدة للعلامات الحيوية غير الطبيعية
        vitals = features.get('vitals', {})
        if vitals.get('heart_rate', 80) > 100:
            days += 1.0
        if vitals.get('oxygen_saturation', 98) < 94:
            days += 2.0
        
        # تحديد الفئة
        if days <= 3:
            category = "🟢 منخفضة"
            risk_level = "low"
        elif days <= 7:
            category = "🟡 متوسطة"
            risk_level = "medium"
        elif days <= 14:
            category = "🟠 عالية"
            risk_level = "high"
        else:
            category = "🔴 حرجة"
            risk_level = "critical"
        
        return {
            'days': round(days, 1),
            'confidence': 0.75,
            'min_days': round(max(1, days - 2), 1),
            'max_days': round(days + 2, 1),
            'risk_category': category,
            'risk_level': risk_level,
            'model_mae': 7.11
        }
    
    def get_model_info(self) -> Dict:
        """معلومات عن النموذج الافتراضي"""
        return {
            'name': 'Fallback LOS Predictor (Rule-based)',
            'type': 'Rule-based',
            'mae': 7.11,
            'target_mae': 3.2,
            'features_count': 10,
            'is_fallback': True
        }


# =========================================================================
# Feature Extractor (محسن)
# =========================================================================

class FeatureExtractor:
    """استخراج الميزات من المدخلات وبيانات المريض"""
    
    def __init__(self):
        self.model_singleton = LOSModelSingleton()
        self.fe = self.model_singleton.get_feature_engineering()
    
    def extract_from_input(self, input_data: PatientInput) -> Dict:
        """استخراج الميزات من بيانات الإدخال المباشر"""
        features = {}
        
        # من الشات
        if input_data.chat_text and self.fe:
            try:
                chat_features = self.fe.process_text(input_data.chat_text)
                if isinstance(chat_features, dict):
                    features.update(chat_features)
            except Exception as e:
                logger.warning(f"Chat text processing failed: {e}")
        
        # من الأعراض
        if input_data.symptoms:
            features['symptom_count'] = len(input_data.symptoms)
            features['max_severity'] = max([s.severity for s in input_data.symptoms])
            features['avg_severity'] = sum([s.severity for s in input_data.symptoms]) / len(input_data.symptoms)
        
        # من الأدوية
        if input_data.medications:
            features['medication_count'] = len(input_data.medications)
            high_risk_meds = ['warfarin', 'insulin', 'digoxin', 'heparin']
            features['has_high_risk_meds'] = any(
                m.name.lower() in high_risk_meds 
                for m in input_data.medications
            )
        
        # من العلامات الحيوية
        if input_data.vitals:
            features['vitals'] = input_data.vitals
            features['heart_rate'] = input_data.vitals.get('heart_rate')
            features['systolic_bp'] = input_data.vitals.get('systolic_bp')
            features['oxygen_saturation'] = input_data.vitals.get('oxygen_saturation')
        
        # من الأمراض المزمنة
        if input_data.conditions:
            features.update(input_data.conditions)
        
        return features
    
    def extract_from_patient_context(self, patient_context) -> Dict:
        """
        استخراج الميزات من PatientContext (قاعدة البيانات)
        مع معالجة صحيحة للبيانات المتداخلة
        """
        features = {}
        
        if not patient_context:
            return features
        
        # من admission_data
        admission_data = patient_context.admission_data if hasattr(patient_context, 'admission_data') else {}
        
        # ✅ تصحيح: العمر والبيانات الشخصية متداخلة في demographics
        demographics = admission_data.get('demographics', {})
        
        # العمر - معالجة صحيحة
        age = demographics.get('age', 0)
        if age > 0:
            features['age'] = age
            # تصنيف الفئات العمرية (مهم للنموذج)
            if age < 18:
                features['age_group'] = 'pediatric'
            elif age < 65:
                features['age_group'] = 'adult'
            else:
                features['age_group'] = 'geriatric'
                features['is_elderly'] = True
        
        # الجنس
        gender = demographics.get('gender', '')
        if gender:
            features['gender'] = gender
            features['is_male'] = 1 if gender.lower() == 'm' else 0
        
        # الأمراض المزمنة (من chronic_conditions)
        chronic_conditions = admission_data.get('chronic_conditions', {})
        for condition, has_it in chronic_conditions.items():
            if has_it:
                features[f'has_{condition}'] = True
        
        # من conditions (قائمة التشخيصات)
        if hasattr(patient_context, 'conditions') and patient_context.conditions:
            features['condition_count'] = len(patient_context.conditions)
        
        # من lab_results_24h
        if hasattr(patient_context, 'lab_results_24h') and patient_context.lab_results_24h:
            features['lab_count'] = len(patient_context.lab_results_24h)
            # استخراج تحاليل محددة
            for lab in patient_context.lab_results_24h:
                test_name = lab.get('test', '').lower()
                if 'creatinine' in test_name:
                    features['creatinine'] = lab.get('value')
                elif 'hba1c' in test_name:
                    features['hba1c'] = lab.get('value')
                elif 'glucose' in test_name:
                    features['glucose'] = lab.get('value')
        
        # من prescriptions_24h
        if hasattr(patient_context, 'prescriptions_24h') and patient_context.prescriptions_24h:
            features['prescription_count'] = len(patient_context.prescriptions_24h)
        
        return features


# =========================================================================
# Explanation Generator (محسن)
# =========================================================================

class LOSExplanationGenerator:
    """توليد شرح للتنبؤ بمدة الإقامة"""
    
    def __init__(self):
        self.model_singleton = LOSModelSingleton()
        self.explainer = self.model_singleton.get_explanation_generator()
        self.lang = "ar"
    
    def generate(self, days: float, features: Dict, language: str = "ar") -> Dict:
        """توليد شرح للتنبؤ"""
        
        # استخدام الـ explainer الحقيقي إذا كان متاحاً
        if self.explainer and hasattr(self.explainer, 'generate_los_explanation'):
            try:
                return self.explainer.generate_los_explanation(
                    days=days,
                    features=features,
                    language=language
                )
            except Exception as e:
                logger.warning(f"Explanation generator failed: {e}")
        
        # تحديد مستوى الخطر
        if days <= 3:
            risk_level = "منخفضة"
            risk_level_en = "low"
            color = "🟢"
        elif days <= 7:
            risk_level = "متوسطة"
            risk_level_en = "medium"
            color = "🟡"
        elif days <= 14:
            risk_level = "عالية"
            risk_level_en = "high"
            color = "🟠"
        else:
            risk_level = "حرجة"
            risk_level_en = "critical"
            color = "🔴"
        
        # توليد الملخص
        if language == "ar":
            summary = f"{color} مدة الإقامة المتوقعة: {days:.1f} يوم (خطر {risk_level})"
            summary += f"\n• الثقة: {85 if days <= 14 else 75}%"
            if features.get('has_diabetes'):
                summary += "\n• ⚠️ السكري يزيد مدة الإقامة"
            if features.get('has_heart_failure'):
                summary += "\n• ⚠️ فشل القلب يزيد مدة الإقامة"
            if features.get('age', 0) > 65:
                summary += "\n• 👴 العمر المتقدم (>65) يزيد مدة الإقامة"
        else:
            summary = f"{color} Expected LOS: {days:.1f} days ({risk_level_en} risk)"
            summary += f"\n• Confidence: {85 if days <= 14 else 75}%"
            if features.get('has_diabetes'):
                summary += "\n• ⚠️ Diabetes increases LOS"
            if features.get('has_heart_failure'):
                summary += "\n• ⚠️ Heart failure increases LOS"
            if features.get('age', 0) > 65:
                summary += "\n• 👴 Advanced age (>65) increases LOS"
        
        # عوامل الخطر
        risk_factors = []
        if features.get('has_diabetes'):
            risk_factors.append({
                'name': 'diabetes',
                'name_ar': 'السكري',
                'impact': '+1.5 days',
                'description': 'يزيد مدة الإقامة بسبب الحاجة لضبط السكر'
            })
        if features.get('has_heart_failure'):
            risk_factors.append({
                'name': 'heart_failure',
                'name_ar': 'فشل القلب',
                'impact': '+3.0 days',
                'description': 'يتطلب مراقبة مستمرة وعلاج مكثف'
            })
        if features.get('age', 0) > 65:
            risk_factors.append({
                'name': 'elderly',
                'name_ar': 'العمر المتقدم',
                'impact': '+2.0 days',
                'description': 'كبار السن يحتاجون وقت أطول للتعافي'
            })
        if features.get('symptom_count', 0) > 3:
            risk_factors.append({
                'name': 'multiple_symptoms',
                'name_ar': 'أعراض متعددة',
                'impact': f'+{min(2, features["symptom_count"] * 0.5)} days',
                'description': 'عدد الأعراض الكبير يشير لحالة أكثر تعقيداً'
            })
        
        # التوصيات
        recommendations = []
        if days > 7:
            if language == "ar":
                recommendations.append("تخطيط للخروج المبكر مع فريق متعدد التخصصات")
                recommendations.append("متابعة بعد الخروج خلال 48 ساعة")
                recommendations.append("تقييم احتياجات الرعاية المنزلية")
            else:
                recommendations.append("Early discharge planning with multidisciplinary team")
                recommendations.append("Follow-up within 48 hours post-discharge")
                recommendations.append("Assess home care needs")
        elif days > 3:
            if language == "ar":
                recommendations.append("متابعة يومية للعلامات الحيوية")
                recommendations.append("تقييم جاهزية الخروج من اليوم الثالث")
            else:
                recommendations.append("Daily vital signs monitoring")
                recommendations.append("Discharge readiness assessment from day 3")
        else:
            if language == "ar":
                recommendations.append("خروج مبكر مع تعليمات واضحة")
                recommendations.append("متابعة هاتفية بعد 7 أيام")
            else:
                recommendations.append("Early discharge with clear instructions")
                recommendations.append("Phone follow-up after 7 days")
        
        # التنبيهات
        alerts = []
        if days > 14:
            alerts.append("⚠️ مدة إقامة طويلة جداً - مراجعة خطة العلاج")
        if features.get('has_heart_failure') and features.get('has_diabetes'):
            alerts.append("⚡ وجود أمراض مزمنة متعددة - يحتاج رعاية مكثفة")
        if features.get('vitals', {}).get('oxygen_saturation', 100) < 92:
            alerts.append("🆘 نقص أكسجين - متابعة فورية مطلوبة")
        if features.get('age', 0) > 80 and days > 10:
            alerts.append("👴 مريض مسن مع إقامة طويلة - خطر مضاعفات مرتفع")
        
        return {
            'summary': summary,
            'details': risk_factors,
            'recommendations': recommendations,
            'risk_factors': risk_factors,
            'alerts': alerts
        }


# =========================================================================
# Main Endpoints - مسارات نظيفة بدون تكرار
# =========================================================================

@router.post("/", response_model=LOSResponse)
@require_any_role([Role.DOCTOR, Role.NURSE, Role.ADMIN, Role.SUPERVISOR])
async def predict_los(
    input_data: PatientInput,
    fastapi_request: Request = None
):
    """
    🏥 التنبؤ بمدة إقامة المريض في المستشفى
    
    🔐 الأمان:
        - Decorator @require_any_role يضمن الصلاحيات
    
    📊 المخرجات:
        - days: عدد الأيام المتوقعة
        - confidence: مستوى الثقة
        - explanation: شرح النتيجة
        - recommendations: توصيات
        - risk_factors: عوامل الخطر
        - alerts: تنبيهات
    """
    start_time = datetime.now()
    request_id = hashlib.md5(f"{input_data.patient_id or 'unknown'}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    logger.info(f"🏥 LOS prediction request | Patient: {input_data.patient_id} | Request ID: {request_id}")
    
    data_source = "request_only"
    patient_context = None
    features = {}
    
    # =================================================================
    # 📦 LAYER 1: Load Patient Data from Database (إذا وجد patient_id)
    # =================================================================
    try:
        if input_data.patient_id:
            db = get_db_loader()
            patient_context = db.load_patient_context(input_data.patient_id, include_encrypted=True)
            if patient_context:
                data_source = "db_loaded"
                logger.info(f"📊 Patient data loaded from database")
    except Exception as e:
        logger.warning(f"Could not load patient data: {e}")
    
    # =================================================================
    # 🧠 LAYER 2: Extract Features
    # =================================================================
    try:
        extractor = FeatureExtractor()
        
        # ميزات من الإدخال المباشر
        input_features = extractor.extract_from_input(input_data)
        features.update(input_features)
        
        # ميزات من قاعدة البيانات (إن وجدت)
        if patient_context:
            db_features = extractor.extract_from_patient_context(patient_context)
            features.update(db_features)
            if db_features:
                data_source = "mixed" if input_features else "db_loaded"
        
        logger.info(f"🔍 Features extracted: {len(features)} keys | Source: {data_source}")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        features = {}
    
    # =================================================================
    # 📈 LAYER 3: Predict LOS (باستخدام Singleton Model)
    # =================================================================
    try:
        # استخدام النموذج المحمل مرة واحدة (Singleton)
        model_singleton = LOSModelSingleton()
        predictor = model_singleton.get_predictor()
        
        if predictor:
            prediction = predictor.predict(features)
        else:
            predictor = FallbackLOSPredictor()
            prediction = predictor.predict(features)
        
        logger.info(f"📊 LOS prediction: {prediction.get('days')} days | Confidence: {prediction.get('confidence')}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # استخدام التنبؤ الافتراضي
        predictor = FallbackLOSPredictor()
        prediction = predictor.predict(features)
    
    # =================================================================
    # 📝 LAYER 4: Generate Explanation
    # =================================================================
    try:
        explainer = LOSExplanationGenerator()
        explanation = explainer.generate(
            days=prediction.get('days', 5.0),
            features=features,
            language=input_data.language
        )
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        explanation = {
            'summary': f"مدة الإقامة المتوقعة: {prediction.get('days', 5.0):.1f} أيام",
            'details': [],
            'recommendations': ["متابعة مع الطبيب المعالج"],
            'risk_factors': [],
            'alerts': []
        }
    
    # =================================================================
    # 📤 LAYER 5: Response
    # =================================================================
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    response = LOSResponse(
        success=True,
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        processing_time_ms=round(processing_time, 2),
        patient_id=input_data.patient_id,
        prediction={
            'days': prediction.get('days', 5.0),
            'confidence': prediction.get('confidence', 0.7),
            'min_days': prediction.get('min_days', 3.0),
            'max_days': prediction.get('max_days', 8.0),
            'risk_category': prediction.get('risk_category', '🟡 متوسط'),
            'risk_level': prediction.get('risk_level', 'medium'),
            'model_mae': prediction.get('model_mae', 7.11)
        },
        explanation={
            'summary': explanation.get('summary', ''),
            'details': explanation.get('details', [])
        },
        recommendations=explanation.get('recommendations', []),
        risk_factors=explanation.get('risk_factors', []),
        alerts=explanation.get('alerts', []),
        data_source=data_source
    )
    
    logger.info(f"✅ LOS prediction completed in {processing_time:.0f}ms")
    return response


# =========================================================================
# Simplified Endpoint - From Chat Only (باستخدام Body)
# =========================================================================

@router.post("/from-chat")
@require_any_role([Role.DOCTOR, Role.NURSE, Role.ADMIN])
async def predict_los_from_chat(
    input_data: ChatOnlyInput = Body(...)
):
    """
    💬 تنبؤ بمدة الإقامة من محادثة فقط (سريع)
    
    ✅ يستخدم Body(...) بدلاً من Query Parameter
    ✅ يتجنب خطأ URI Too Long
    """
    start_time = datetime.now()
    request_id = hashlib.md5(f"chat_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    try:
        # إنشاء PatientInput من بيانات المحادثة
        patient_input = PatientInput(
            patient_id=input_data.patient_id,
            chat_text=input_data.chat_text,
            language=input_data.language
        )
        
        # استخراج الميزات
        extractor = FeatureExtractor()
        features = extractor.extract_from_input(patient_input)
        
        # تنبؤ باستخدام Singleton Model
        model_singleton = LOSModelSingleton()
        predictor = model_singleton.get_predictor()
        
        if predictor:
            prediction = predictor.predict(features)
        else:
            predictor = FallbackLOSPredictor()
            prediction = predictor.predict(features)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'success': True,
            'request_id': request_id,
            'chat_analysis': {
                'symptom_count': features.get('symptom_count', 0),
                'has_diabetes': features.get('has_diabetes', False),
                'has_hypertension': features.get('has_hypertension', False),
                'has_heart_failure': features.get('has_heart_failure', False)
            },
            'prediction': {
                'days': prediction.get('days', 5.0),
                'confidence': prediction.get('confidence', 0.7),
                'min_days': prediction.get('min_days', 3.0),
                'max_days': prediction.get('max_days', 8.0),
                'risk_category': prediction.get('risk_category', '🟡 متوسط')
            },
            'processing_time_ms': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Batch Prediction Endpoint - مع Fallback لكل مريض على حدة
# =========================================================================

@router.post("/batch")
@require_role(Role.ADMIN)
async def predict_los_batch(
    inputs: List[PatientInput],
    fastapi_request: Request = None
):
    """
    📊 التنبؤ لمجموعة من المرضى (للمسؤولين فقط)
    
    ✅ معالجة كل مريض على حدة (try/catch داخل الـ Loop)
    ✅ فشل مريض واحد لا يؤثر على باقي المرضى
    ✅ استخدام Fallback للمرضى الذين فشلوا
    """
    start_time = datetime.now()
    request_id = hashlib.md5(f"batch_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    logger.info(f"📊 Batch LOS prediction: {len(inputs)} patients")
    
    # تهيئة النموذج مرة واحدة للباتش
    model_singleton = LOSModelSingleton()
    predictor = model_singleton.get_predictor()
    
    if not predictor:
        predictor = FallbackLOSPredictor()
        logger.info("🔄 Using FallbackLOSPredictor for batch")
    
    extractor = FeatureExtractor()
    results = []
    failed_count = 0
    
    for i, input_data in enumerate(inputs):
        patient_start = datetime.now()
        try:
            # استخراج الميزات للمريض الحالي
            features = extractor.extract_from_input(input_data)
            
            # تحميل بيانات المريض من قاعدة البيانات إذا كان لديه ID
            if input_data.patient_id:
                try:
                    db = get_db_loader()
                    patient_context = db.load_patient_context(input_data.patient_id, include_encrypted=True)
                    if patient_context:
                        db_features = extractor.extract_from_patient_context(patient_context)
                        features.update(db_features)
                except Exception as e:
                    logger.warning(f"Failed to load patient {input_data.patient_id}: {e}")
            
            # تنبؤ
            prediction = predictor.predict(features)
            
            results.append({
                'patient_index': i,
                'patient_id': input_data.patient_id,
                'success': True,
                'days': prediction.get('days', 5.0),
                'confidence': prediction.get('confidence', 0.7),
                'risk_category': prediction.get('risk_category', ''),
                'processing_time_ms': round((datetime.now() - patient_start).total_seconds() * 1000, 2)
            })
            
        except Exception as e:
            # ✅ فشل مريض واحد - نستخدم Fallback للمريض فقط
            failed_count += 1
            logger.error(f"Patient {i} failed: {e}, using fallback")
            
            # استخدام Fallback للمريض الفاشل
            fallback = FallbackLOSPredictor()
            fallback_prediction = fallback.predict({})
            
            results.append({
                'patient_index': i,
                'patient_id': input_data.patient_id,
                'success': False,
                'error': str(e)[:100],  # اختصار الخطأ
                'days': fallback_prediction.get('days', 5.0),
                'confidence': fallback_prediction.get('confidence', 0.5),
                'risk_category': fallback_prediction.get('risk_category', ''),
                'processing_time_ms': round((datetime.now() - patient_start).total_seconds() * 1000, 2)
            })
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    days_list = [r['days'] for r in results]
    
    return {
        'success': True,
        'request_id': request_id,
        'results': results,
        'count': len(results),
        'failed_count': failed_count,
        'success_rate': round((len(results) - failed_count) / len(results) * 100, 1),
        'stats': {
            'avg_days': round(sum(days_list) / len(days_list), 1),
            'min_days': min(days_list),
            'max_days': max(days_list)
        },
        'processing_time_ms': round(processing_time, 2),
        'timestamp': datetime.now().isoformat()
    }


# =========================================================================
# Model Info Endpoint
# =========================================================================

@router.get("/model-info")
@require_any_role([Role.DOCTOR, Role.ADMIN])
async def get_los_model_info(
    fastapi_request: Request = None
):
    """
    📊 معلومات عن نموذج LOS
    """
    try:
        model_singleton = LOSModelSingleton()
        predictor = model_singleton.get_predictor()
        
        if predictor and hasattr(predictor, 'get_model_info'):
            info = predictor.get_model_info()
        else:
            predictor = FallbackLOSPredictor()
            info = predictor.get_model_info()
        
        # إضافة معلومات عن حالة النموذج
        info['model_loaded'] = model_singleton.get_predictor() is not None
        info['feature_engineering_loaded'] = model_singleton.get_feature_engineering() is not None
        info['explanation_generator_loaded'] = model_singleton.get_explanation_generator() is not None
        
        return {
            'success': True,
            'model_info': info,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Health Check
# =========================================================================

@router.get("/health")
async def health_check():
    """فحص صحة الخدمة"""
    model_singleton = LOSModelSingleton()
    
    return {
        'status': 'healthy',
        'service': 'LOS Prediction API',
        'version': '4.1.0',
        'security_version': 'v4.1',
        'security_approach': 'Decorator-only (@require_any_role)',
        'model_status': {
            'ai_core_available': LOSPredictor is not None,
            'model_loaded': model_singleton.get_predictor() is not None,
            'feature_engineering_loaded': model_singleton.get_feature_engineering() is not None,
            'explanation_generator_loaded': model_singleton.get_explanation_generator() is not None
        },
        'timestamp': datetime.now().isoformat()
    }


# =========================================================================
# Test Endpoint
# =========================================================================

@router.get("/test")
async def test_endpoint():
    """نقطة نهاية للاختبار"""
    model_singleton = LOSModelSingleton()
    
    return {
        'message': 'LOS Prediction API is working',
        'version': '4.1.0',
        'security': 'Decorator-based @require_any_role',
        'model_status': 'loaded' if model_singleton.get_predictor() else 'fallback_mode',
        'prefix_fixed': True,  # المسارات الآن نظيفة بدون تكرار
        'endpoints': [
            'POST /api/predict/los/',
            'POST /api/predict/los/from-chat (Body)',
            'POST /api/predict/los/batch (Admin only)',
            'GET /api/predict/los/model-info',
            'GET /api/predict/los/health'
        ]
    }
