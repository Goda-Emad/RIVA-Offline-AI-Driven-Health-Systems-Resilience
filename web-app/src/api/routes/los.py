"""
===============================================================================
los.py
API للتنبؤ بمدة إقامة المريض في المستشفى
Length of Stay (LOS) Prediction API Endpoint
===============================================================================

🏆 الإصدار: 2.0.0 - للمسابقات العالمية
🎯 الهدف: MAE 7.11 يوم (نسعى لتحسينه)
⚡ وقت الاستجابة: < 200ms

المميزات:
✓ تنبؤ فوري بمدة الإقامة
✓ شرح النتائج مع هامش خطأ
✓ تحليل عوامل الخطر
✓ توصيات طبية
✓ دعم العربية والإنجليزية
✓ معالجة القيم الشاذة
===============================================================================
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import numpy as np
import joblib
import json
import os
import logging
import sys

# إضافة المسار الرئيسي للمشروع
sys.path.append('/content/drive/MyDrive/RIVA-Maternal')

from ai-core.prediction.los_predictor import LOSPredictor
from ai-core.prediction.feature_engineering import FeatureEngineering
from ai-core.prediction.explanation_generator import ExplanationGenerator

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# إنشاء router
router = APIRouter(prefix="/api/predict", tags=["LOS Prediction"])


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
    explanation_level: str = Field("patient", description="مستوى الشرح")


class LOSResponse(BaseModel):
    """نتيجة التنبؤ"""
    success: bool = Field(..., description="نجاح العملية")
    prediction: Dict[str, Any] = Field(..., description="نتيجة التنبؤ")
    explanation: Dict[str, Any] = Field(..., description="شرح النتيجة")
    recommendations: List[str] = Field(..., description="التوصيات")
    risk_factors: List[Dict] = Field(..., description="عوامل الخطر")
    alerts: List[str] = Field(..., description="تنبيهات")
    processing_time_ms: float = Field(..., description="وقت المعالجة")
    timestamp: str = Field(..., description="الوقت")


# =========================================================================
# دوال مساعدة للتحقق
# =========================================================================

async def verify_api_key(api_key: Optional[str] = None):
    """التحقق من مفتاح API (للتبسيط)"""
    return True


async def get_db():
    """الحصول على اتصال قاعدة البيانات"""
    return None


# =========================================================================
# المسارات (Endpoints)
# =========================================================================

@router.post("/los", response_model=LOSResponse)
async def predict_los(
    input_data: PatientInput,
    api_key: str = Depends(verify_api_key)
):
    """
    التنبؤ بمدة إقامة المريض في المستشفى
    
    المدخلات:
    - chat_text: كلام المريض
    - symptoms: قائمة الأعراض
    - medications: قائمة الأدوية
    - vitals: العلامات الحيوية
    - conditions: الأمراض المزمنة
    
    المخرجات:
    - days: عدد الأيام المتوقعة
    - confidence: مستوى الثقة
    - explanation: شرح النتيجة
    - recommendations: توصيات
    - min_days: الحد الأدنى المتوقع
    - max_days: الحد الأقصى المتوقع
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"📥 LOS prediction request received")
        
        # =================================================================
        # 1. استخراج الميزات
        # =================================================================
        features = {}
        
        # من الشات
        if input_data.chat_text:
            fe = FeatureEngineering()
            chat_features = fe.process_text(input_data.chat_text)
            if isinstance(chat_features, dict):
                features.update(chat_features)
                
                # استخراج العلامات الحيوية من النص
                if 'vitals' in chat_features:
                    features['vitals'] = chat_features['vitals']
        
        # من الأعراض المدخلة
        if input_data.symptoms:
            features['symptom_count'] = len(input_data.symptoms)
            features['max_severity'] = max([s.severity for s in input_data.symptoms])
            features['avg_severity'] = sum([s.severity for s in input_data.symptoms]) / len(input_data.symptoms)
        
        # من الأدوية
        if input_data.medications:
            features['medication_count'] = len(input_data.medications)
            features['has_high_risk_meds'] = any(
                m.name.lower() in ['warfarin', 'insulin', 'digoxin'] 
                for m in input_data.medications
            )
        
        # من العلامات الحيوية
        if input_data.vitals:
            features['vitals'] = input_data.vitals
        
        # من الأمراض المزمنة (تأثير كبير على LOS)
        if input_data.conditions:
            features.update(input_data.conditions)
        
        logger.info(f"📊 Features extracted: {len(features)} keys")
        
        # =================================================================
        # 2. التنبؤ
        # =================================================================
        predictor = LOSPredictor()
        prediction = predictor.predict(features)
        
        # =================================================================
        # 3. توليد الشرح
        # =================================================================
        explainer = ExplanationGenerator(language=input_data.language)
        explanation_result = explainer.generate_los_explanation(
            days=prediction.get('days', 5.0),
            features=features
        )
        
        # =================================================================
        # 4. حساب وقت المعالجة
        # =================================================================
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # =================================================================
        # 5. إعداد الاستجابة
        # =================================================================
        response = LOSResponse(
            success=True,
            prediction={
                'days': prediction.get('days', 5.0),
                'confidence': prediction.get('confidence', 0.7),
                'min_days': prediction.get('min_days', 3.0),
                'max_days': prediction.get('max_days', 8.0),
                'risk_category': prediction.get('risk_category', '🟡 متوسط'),
                'model_mae': 7.11
            },
            explanation={
                'summary': explanation_result.get('summary', ''),
                'details': explanation_result.get('details', [])
            },
            recommendations=explanation_result.get('recommendations', []),
            risk_factors=explanation_result.get('risk_factors', []),
            alerts=explanation_result.get('alerts', []),
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ LOS prediction completed in {processing_time:.0f}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': 'فشل التنبؤ بمدة الإقامة',
                'details': str(e),
                'timestamp': datetime.now().isoformat()
            }
        )


@router.post("/los/from-chat")
async def predict_los_from_chat(
    chat_text: str,
    language: str = "ar",
    api_key: str = Depends(verify_api_key)
):
    """
    تنبؤ بمدة الإقامة من محادثة فقط
    
    المدخلات:
    - chat_text: كلام المريض
    
    المخرجات:
    - days: عدد الأيام المتوقعة
    - explanation: شرح
    - recommendations: توصيات
    """
    try:
        # استخراج الميزات من المحادثة
        fe = FeatureEngineering()
        features = fe.process_text(chat_text)
        
        # تنبؤ
        predictor = LOSPredictor()
        prediction = predictor.predict(features)
        
        # شرح
        explainer = ExplanationGenerator(language=language)
        explanation = explainer.generate_los_explanation(
            days=prediction.get('days', 5.0),
            features=features
        )
        
        return {
            'success': True,
            'chat_analysis': {
                'symptoms': features.get('symptoms', []),
                'medications': features.get('medications', []),
                'symptom_count': features.get('symptom_count', 0),
                'vitals': features.get('vitals', {})
            },
            'prediction': {
                'days': prediction.get('days', 5.0),
                'confidence': prediction.get('confidence', 0.7),
                'min_days': prediction.get('min_days', 3.0),
                'max_days': prediction.get('max_days', 8.0)
            },
            'recommendations': explanation.get('recommendations', []),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/los/batch")
async def predict_los_batch(
    inputs: List[PatientInput],
    api_key: str = Depends(verify_api_key)
):
    """
    التنبؤ لمجموعة من المرضى
    
    المدخلات:
    - list of PatientInput
    
    المخرجات:
    - list of predictions with stats
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"📥 Batch LOS prediction: {len(inputs)} patients")
        
        results = []
        predictor = LOSPredictor()
        explainer = ExplanationGenerator()
        
        for i, input_data in enumerate(inputs):
            # استخراج الميزات (مبسط للسرعة)
            features = {}
            if input_data.symptoms:
                features['symptom_count'] = len(input_data.symptoms)
            if input_data.conditions:
                features.update(input_data.conditions)
            
            # تنبؤ
            prediction = predictor.predict(features)
            
            # شرح
            explanation = explainer.generate_los_explanation(
                days=prediction.get('days', 5.0),
                features=features
            )
            
            results.append({
                'patient_index': i,
                'patient_id': input_data.patient_id,
                'days': prediction.get('days', 5.0),
                'confidence': prediction.get('confidence', 0.7),
                'risk_category': prediction.get('risk_category', ''),
                'recommendations': explanation.get('recommendations', [])[:2]
            })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # إحصائيات
        days_list = [r['days'] for r in results]
        
        return {
            'success': True,
            'results': results,
            'count': len(results),
            'stats': {
                'avg_days': round(sum(days_list) / len(days_list), 1),
                'min_days': min(days_list),
                'max_days': max(days_list)
            },
            'processing_time_ms': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/los/model-info")
async def get_los_model_info(api_key: str = Depends(verify_api_key)):
    """
    معلومات عن نموذج LOS
    
    المخرجات:
    - model details
    - performance metrics
    - features importance
    """
    try:
        predictor = LOSPredictor()
        info = predictor.get_model_info() if hasattr(predictor, 'get_model_info') else {}
        
        return {
            'success': True,
            'model_info': {
                'name': 'LOS Prediction Model',
                'type': 'XGBoost Regressor',
                'mae': 7.11,
                'target_mae': 3.2,
                'features_count': 22,
                'training_samples': 129,
                'max_los': 30,
                'use_log_transform': True,
                'top_features': info.get('top_features', [])
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/los/health")
async def health_check():
    """
    فحص صحة الخدمة
    """
    return {
        'status': 'healthy',
        'service': 'LOS Prediction API',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat()
    }
