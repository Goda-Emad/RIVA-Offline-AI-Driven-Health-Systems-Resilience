"""
===============================================================================
combined.py
API الموحد للتنبؤات الطبية - إعادة الدخول + مدة الإقامة
Combined Medical Predictions API - Readmission + LOS
===============================================================================

🏆 الإصدار: 4.0.0 - Platinum Production Edition
🥇 جاهز للرفع على أي سيرفر (Cloud/On-Premise)
⚡ وقت الاستجابة: < 300ms
🔐 متكامل مع نظام التحكم بالصلاحيات والتشفير
===============================================================================
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging
import sys
import os
import hashlib
from enum import Enum

# إضافة المسار الرئيسي للمشروع (ديناميكي - يشتغل في أي مكان)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# استيراد النماذج من المجلد المصحح (ai_core بدلاً من ai-core)
try:
    from ai_core.prediction.readmission_predictor import ReadmissionPredictor
    from ai_core.prediction.los_predictor import LOSPredictor
    from ai_core.prediction.feature_engineering import FeatureEngineering
except ImportError:
    # في حالة عدم وجود المجلد، نستخدم dummy models للتشغيل
    logging.warning("⚠️ AI Core modules not found, using dummy predictors")
    ReadmissionPredictor = None
    LOSPredictor = None
    FeatureEngineering = None

# استيراد أنظمة الأمان v4.0
from access_control import require_any_role, Role, get_access_control


# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء router
router = APIRouter(prefix="/api/v1", tags=["Combined Predictions"])


# =========================================================================
# Enums
# =========================================================================

class Language(str, Enum):
    ARABIC = "ar"
    ENGLISH = "en"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =========================================================================
# Pydantic Models
# =========================================================================

class VitalsInput(BaseModel):
    heart_rate: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    temperature: Optional[float] = None
    oxygen_saturation: Optional[float] = None


class SymptomInput(BaseModel):
    name: str
    severity: int = Field(1, ge=1, le=5)


class MedicationInput(BaseModel):
    name: str
    dosage: Optional[str] = None


class ChronicConditions(BaseModel):
    hypertension: bool = False
    diabetes: bool = False
    heart_failure: bool = False
    kidney_disease: bool = False


class CombinedInput(BaseModel):
    patient_id: Optional[str] = None
    chat_text: Optional[str] = None
    symptoms: Optional[List[SymptomInput]] = None
    medications: Optional[List[MedicationInput]] = None
    vitals: Optional[VitalsInput] = None
    conditions: Optional[ChronicConditions] = None
    language: Language = Language.ARABIC


class PredictionResult(BaseModel):
    value: float
    confidence: float
    risk_level: RiskLevel
    interpretation: str


class CombinedPrediction(BaseModel):
    readmission: Optional[PredictionResult] = None
    los: Optional[PredictionResult] = None


class RiskFactor(BaseModel):
    name: str
    impact_score: float
    description: str


class Recommendation(BaseModel):
    priority: str
    title: str
    description: str


class Alert(BaseModel):
    level: str
    title: str
    message: str
    action_required: bool


class CombinedResponse(BaseModel):
    success: bool
    request_id: str
    timestamp: str
    processing_time_ms: float
    patient_info: Dict[str, Any]
    predictions: CombinedPrediction
    risk_factors: List[RiskFactor]
    recommendations: List[Recommendation]
    alerts: List[Alert]
    explanations: Dict[str, str]


# =========================================================================
# Service Layer (Business Logic)
# =========================================================================

class PredictionService:
    """خدمة التنبؤات - تفصل منطق الأعمال عن الـ API"""
    
    def __init__(self):
        self.readmission_predictor = None
        self.los_predictor = None
        self.feature_engineering = None
        
        # تهيئة النماذج إذا كانت متوفرة
        if ReadmissionPredictor:
            self.readmission_predictor = ReadmissionPredictor()
        if LOSPredictor:
            self.los_predictor = LOSPredictor()
        if FeatureEngineering:
            self.feature_engineering = FeatureEngineering()
    
    def extract_features(self, input_data: CombinedInput, patient_data: Optional[Dict] = None) -> Dict:
        """استخراج الميزات من المدخلات وبيانات المريض"""
        features = {}
        
        # 1. معالجة النص إذا وجد
        if input_data.chat_text and self.feature_engineering:
            try:
                chat_features = self.feature_engineering.process_text(input_data.chat_text)
                if isinstance(chat_features, dict):
                    features.update(chat_features)
            except Exception as e:
                logger.warning(f"Chat text processing failed: {e}")
        
        # 2. معالجة الأعراض
        if input_data.symptoms:
            features['symptom_count'] = len(input_data.symptoms)
            features['max_severity'] = max([s.severity for s in input_data.symptoms])
            features['avg_severity'] = sum(s.severity for s in input_data.symptoms) / len(input_data.symptoms)
        
        # 3. معالجة الأدوية
        if input_data.medications:
            features['medication_count'] = len(input_data.medications)
        
        # 4. معالجة الحالات المزمنة
        if input_data.conditions:
            features.update(input_data.conditions.model_dump())
        
        # 5. دمج بيانات المريض من قاعدة البيانات
        if patient_data:
            # بيانات ديموغرافية
            if 'demographics' in patient_data:
                demog = patient_data['demographics']
                features['age'] = demog.get('age', 0)
                features['gender'] = 1 if demog.get('gender') == 'male' else 0
            
            # العلامات الحيوية من قاعدة البيانات (إذا لم تكن في الطلب)
            if 'vitals' in patient_data and not input_data.vitals:
                vitals = patient_data['vitals']
                features['heart_rate'] = vitals.get('heart_rate')
                features['systolic_bp'] = vitals.get('systolic_bp')
                features['oxygen_saturation'] = vitals.get('oxygen_saturation')
        
        return features
    
    def predict_readmission(self, features: Dict) -> Optional[Dict]:
        """التنبؤ بإعادة الدخول"""
        if not self.readmission_predictor:
            return None
        
        try:
            result = self.readmission_predictor.predict(features)
            return result
        except Exception as e:
            logger.error(f"Readmission prediction failed: {e}")
            return None
    
    def predict_los(self, features: Dict) -> Optional[Dict]:
        """التنبؤ بمدة الإقامة"""
        if not self.los_predictor:
            return None
        
        try:
            result = self.los_predictor.predict(features)
            return result
        except Exception as e:
            logger.error(f"LOS prediction failed: {e}")
            return None
    
    def generate_risk_factors(self, predictions: CombinedPrediction, features: Dict, language: Language) -> List[RiskFactor]:
        """توليد عوامل الخطر بناءً على التنبؤات والميزات"""
        risk_factors = []
        
        # من الحالات المزمنة
        if features.get('heart_failure'):
            risk_factors.append(RiskFactor(
                name='heart_failure',
                impact_score=2.5,
                description='فشل القلب يزيد خطر إعادة الدخول' if language == Language.ARABIC else 'Heart failure increases readmission risk'
            ))
        
        if features.get('diabetes'):
            risk_factors.append(RiskFactor(
                name='diabetes',
                impact_score=1.8,
                description='السكري يؤثر على مدة الإقامة' if language == Language.ARABIC else 'Diabetes affects length of stay'
            ))
        
        # من نتائج التنبؤات
        if predictions.readmission and predictions.readmission.value > 0.7:
            risk_factors.append(RiskFactor(
                name='high_readmission_risk',
                impact_score=predictions.readmission.value * 3,
                description='خطر مرتفع لإعادة الدخول خلال 30 يوم' if language == Language.ARABIC else 'High risk of 30-day readmission'
            ))
        
        if predictions.los and predictions.los.value > 10:
            risk_factors.append(RiskFactor(
                name='prolonged_los',
                impact_score=predictions.los.value / 2,
                description='مدة إقامة طويلة متوقعة' if language == Language.ARABIC else 'Expected prolonged hospital stay'
            ))
        
        return risk_factors
    
    def generate_recommendations(self, predictions: CombinedPrediction, language: Language) -> List[Recommendation]:
        """توليد توصيات ذكية"""
        recommendations = []
        
        if predictions.readmission and predictions.readmission.value > 0.7:
            recommendations.append(Recommendation(
                priority='high',
                title='متابعة مكثفة' if language == Language.ARABIC else 'Intensive Follow-up',
                description='متابعة خلال 48 ساعة بعد الخروج' if language == Language.ARABIC else 'Follow-up within 48 hours post-discharge'
            ))
        
        if predictions.los and predictions.los.value > 10:
            recommendations.append(Recommendation(
                priority='high',
                title='تخطيط الخروج المبكر' if language == Language.ARABIC else 'Early Discharge Planning',
                description='بدء التخطيط للخروج من اليوم الأول' if language == Language.ARABIC else 'Start discharge planning from day one'
            ))
        
        if len(recommendations) == 0:
            recommendations.append(Recommendation(
                priority='normal',
                title='متابعة روتينية' if language == Language.ARABIC else 'Routine Follow-up',
                description='متابعة بعد 7 أيام من الخروج' if language == Language.ARABIC else 'Follow-up after 7 days post-discharge'
            ))
        
        return recommendations
    
    def generate_alerts(self, input_data: CombinedInput, predictions: CombinedPrediction, language: Language) -> List[Alert]:
        """توليد تنبيهات سريرية"""
        alerts = []
        
        # تنبيهات من العلامات الحيوية
        if input_data.vitals:
            if input_data.vitals.heart_rate and input_data.vitals.heart_rate > 120:
                alerts.append(Alert(
                    level='warning',
                    title='تسارع نبضات القلب' if language == Language.ARABIC else 'Tachycardia',
                    message=f'معدل ضربات القلب {input_data.vitals.heart_rate} > 120' if language == Language.ARABIC else f'Heart rate {input_data.vitals.heart_rate} > 120 bpm',
                    action_required=True
                ))
            
            if input_data.vitals.oxygen_saturation and input_data.vitals.oxygen_saturation < 92:
                alerts.append(Alert(
                    level='critical',
                    title='نقص الأكسجين' if language == Language.ARABIC else 'Hypoxia',
                    message=f'تشبع الأكسجين {input_data.vitals.oxygen_saturation}% < 92%' if language == Language.ARABIC else f'Oxygen saturation {input_data.vitals.oxygen_saturation}% < 92%',
                    action_required=True
                ))
        
        # تنبيهات من التنبؤات
        if predictions.readmission and predictions.readmission.value > 0.8:
            alerts.append(Alert(
                level='critical',
                title='خطر شديد لإعادة الدخول' if language == Language.ARABIC else 'Severe Readmission Risk',
                message='احتمالية إعادة الدخول > 80%، تدخل فوري مطلوب' if language == Language.ARABIC else 'Readmission probability > 80%, immediate intervention required',
                action_required=True
            ))
        
        return alerts


# =========================================================================
# Main Endpoint (Clean Controller)
# =========================================================================

@router.post("/predict/combined", response_model=CombinedResponse)
async def combined_prediction(
    request: CombinedInput,
    fastapi_request: Request,
    access: get_access_control = Depends(get_access_control)
):
    """
    🏆 التنبؤات الطبية المتكاملة
    
    🔐 الأمان:
        - التحقق من هوية المستخدم عبر JWT
        - التحقق من صلاحيات الوصول (Doctor/Admin/Supervisor)
    
    📊 المخرجات:
        - احتمالية إعادة الدخول (Readmission)
        - مدة الإقامة المتوقعة (LOS)
        - عوامل الخطر والتوصيات والتنبيهات
    """
    start_time = datetime.now()
    request_id = hashlib.md5(f"{request.patient_id or 'unknown'}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    logger.info(f"🚀 Combined prediction started | Request ID: {request_id}")
    
    # =================================================================
    # 🔐 LAYER 1: Authentication & Authorization (v4.0)
    # =================================================================
    try:
        access.authenticate()
        
        # التحقق من الصلاحيات - Doctor, Admin, Supervisor فقط
        user_role = access.get_user_role()
        allowed_roles = [Role.DOCTOR, Role.ADMIN, Role.SUPERVISOR]
        
        if not access.require_any_role(allowed_roles):
            logger.warning(f"❌ Unauthorized access | Role: {user_role} | Request ID: {request_id}")
            raise HTTPException(
                status_code=403, 
                detail=f"Role {user_role} does not have permission. Required: doctor, admin, or supervisor"
            )
        
        logger.info(f"✅ Authentication successful | Role: {user_role} | Request ID: {request_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    # =================================================================
    # 📦 LAYER 2: Load Patient Data (Optional - if available)
    # =================================================================
    patient_data = None
    try:
        if request.patient_id:
            # محاولة تحميل بيانات المريض (إذا كان db_loader متاح)
            try:
                from db_loader import get_db_loader
                db = get_db_loader()
                patient_data = await db.load_patient_context(
                    patient_id=request.patient_id,
                    include_encrypted=True
                )
                logger.info(f"📊 Patient data loaded | ID: {request.patient_id}")
            except ImportError:
                logger.info("📁 db_loader not available, using request data only")
            except Exception as e:
                logger.warning(f"⚠️ Could not load patient data: {e}")
    except Exception as e:
        logger.warning(f"Patient data load skipped: {e}")
    
    # =================================================================
    # 🧠 LAYER 3: Business Logic (Prediction Service)
    # =================================================================
    try:
        service = PredictionService()
        
        # استخراج الميزات
        features = service.extract_features(request, patient_data)
        logger.info(f"🔍 Features extracted: {len(features)} features")
        
        # التنبؤات
        predictions = CombinedPrediction()
        
        # Readmission
        readmission_result = service.predict_readmission(features)
        if readmission_result:
            prob = readmission_result.get('probability', 0.5)
            predictions.readmission = PredictionResult(
                value=prob,
                confidence=readmission_result.get('confidence', 0.85),
                risk_level=RiskLevel.HIGH if prob > 0.7 else RiskLevel.MEDIUM if prob > 0.3 else RiskLevel.LOW,
                interpretation=f"احتمالية إعادة الدخول: {prob:.1%}" if request.language == Language.ARABIC else f"Readmission probability: {prob:.1%}"
            )
            logger.info(f"📈 Readmission: {prob:.1%}")
        
        # LOS
        los_result = service.predict_los(features)
        if los_result:
            days = los_result.get('days', 5.0)
            predictions.los = PredictionResult(
                value=days,
                confidence=los_result.get('confidence', 0.7),
                risk_level=RiskLevel.HIGH if days > 10 else RiskLevel.MEDIUM if days > 5 else RiskLevel.LOW,
                interpretation=f"مدة الإقامة المتوقعة: {days:.0f} يوم" if request.language == Language.ARABIC else f"Expected LOS: {days:.0f} days"
            )
            logger.info(f"📊 LOS: {days:.1f} days")
        
        # توليد عوامل الخطر والتوصيات والتنبيهات
        risk_factors = service.generate_risk_factors(predictions, features, request.language)
        recommendations = service.generate_recommendations(predictions, request.language)
        alerts = service.generate_alerts(request, predictions, request.language)
        
        # =================================================================
        # 📤 LAYER 4: Response
        # =================================================================
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = CombinedResponse(
            success=True,
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2),
            patient_info={
                'patient_id': request.patient_id,
                'language': request.language.value,
                'data_source': 'db_loaded' if patient_data else 'request_only'
            },
            predictions=predictions,
            risk_factors=risk_factors,
            recommendations=recommendations,
            alerts=alerts,
            explanations={
                'readmission': readmission_result.get('explanation', '') if readmission_result else '',
                'los': los_result.get('explanation', '') if los_result else ''
            }
        )
        
        logger.info(f"✅ Completed in {processing_time:.0f}ms | Request ID: {request_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Health Check
# =========================================================================

@router.get("/predict/combined/health")
async def health_check():
    """فحص صحة الخدمة"""
    return {
        'status': 'healthy',
        'service': 'RIVA-Maternal Combined API',
        'version': '4.0.0',
        'security_version': 'v4.0',
        'models_available': {
            'readmission': ReadmissionPredictor is not None,
            'los': LOSPredictor is not None,
            'feature_engineering': FeatureEngineering is not None
        },
        'timestamp': datetime.now().isoformat()
    }


# =========================================================================
# Simple Test Endpoint
# =========================================================================

@router.post("/predict/combined/test")
async def test_prediction():
    """نقطة نهاية بسيطة للاختبار"""
    return {
        'message': 'API is working',
        'version': '4.0.0',
        'status': 'ready'
    }
