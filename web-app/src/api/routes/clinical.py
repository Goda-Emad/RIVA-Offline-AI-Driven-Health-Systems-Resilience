#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 RIVA Clinical API - Unified Clinical Prediction Endpoint
═══════════════════════════════════════════════════════════════
نظام API موحد للتنبؤات السريرية يجمع:
    • Readmission Risk (خطر إعادة الدخول)
    • Length of Stay (مدة الإقامة)
    • توصيات سريرية ذكية
    • شرح القرارات (Explainability)
    • متوافق مع FastAPI

الإصدار: 2.0.0
التاريخ: 2026-03-17
المطور: فريق RIVA
═══════════════════════════════════════════════════════════════
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import sys
import os
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np

# ================== إعداد المسارات ==================
# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# استيراد الـ predictor من الملف السابق
try:
    from ai_core.local_inference.unified_predictor_pro import UnifiedPredictor
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    PREDICTOR_AVAILABLE = False
    print(f"⚠️ تحذير: {e}")

# ================== إعداد logging ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('/content/drive2/MyDrive/RIVA-Maternal/logs/clinical_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ClinicalAPI')

# ================== تهيئة الـ router ==================
router = APIRouter(prefix="/api/v1/clinical", tags=["التنبؤات السريرية"])

# ================== تهيئة الـ predictor ==================
predictor = None
if PREDICTOR_AVAILABLE:
    try:
        # ✅ نفس المسار اللي استخدمناه في الملف السابق
        base_path = '/content/drive2/MyDrive/RIVA-Maternal'
        predictor = UnifiedPredictor(
            base_path=base_path,
            enable_shap=True,
            enable_pdf=True
        )
        logger.info("✅ RIVA Clinical API جاهز للعمل")
    except Exception as e:
        logger.error(f"❌ فشل تهيئة الـ predictor: {e}")


# ================== نماذج البيانات ==================
class PatientClinicalData(BaseModel):
    """
    نموذج بيانات المريض السريرية
    """
    patient_id: str = Field(..., description="رقم المريض الفريد")
    age: int = Field(..., ge=0, le=120, description="عمر المريض (0-120 سنة)")
    gender_M: int = Field(1, ge=0, le=1, description="الجنس: 1=ذكر, 0=أنثى")
    admit_EMERGENCY: int = Field(1, ge=0, le=1, description="دخول طارئ: 1=نعم, 0=لا")
    num_diagnoses: int = Field(..., ge=0, description="عدد التشخيصات")
    num_procedures: int = Field(..., ge=0, description="عدد الإجراءات")
    num_medications: int = Field(..., ge=0, description="عدد الأدوية")
    charlson_index: int = Field(..., ge=0, description="مؤشر تشارلسون للأمراض المصاحبة")
    total_visits: int = Field(..., ge=0, description="عدد الزيارات السابقة")
    
    @validator('gender_M')
    def validate_gender(cls, v):
        if v not in [0, 1]:
            raise ValueError('الجنس يجب أن يكون 0 (أنثى) أو 1 (ذكر)')
        return v
    
    @validator('admit_EMERGENCY')
    def validate_admit(cls, v):
        if v not in [0, 1]:
            raise ValueError('نوع الدخول يجب أن يكون 0 (اختياري) أو 1 (طارئ)')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "age": 65,
                "gender_M": 1,
                "admit_EMERGENCY": 1,
                "num_diagnoses": 5,
                "num_procedures": 2,
                "num_medications": 8,
                "charlson_index": 3,
                "total_visits": 4
            }
        }


class ClinicalPredictionResponse(BaseModel):
    """
    نموذج استجابة التنبؤ السريري
    """
    patient_id: str
    timestamp: str
    processing_time_ms: float
    
    # Readmission
    readmission_risk: str
    readmission_probability: float
    readmission_confidence: float
    
    # LOS
    predicted_los_days: float
    los_confidence_interval: Dict[str, float]
    
    # Recommendations
    recommendations: List[str]
    
    # Clinical notes
    clinical_notes: str


class BatchPredictionRequest(BaseModel):
    """
    نموذج طلب تنبؤ جماعي
    """
    patients: List[PatientClinicalData]


class BatchPredictionResponse(BaseModel):
    """
    نموذج استجابة التنبؤ الجماعي
    """
    predictions: List[ClinicalPredictionResponse]
    total_patients: int
    processing_time_ms: float


# ================== نقاط النهاية (Endpoints) ==================

@router.get("/health", summary="فحص صحة الخدمة")
async def health_check():
    """
    فحص حالة الخدمة والتأكد من جاهزية النماذج
    """
    if not PREDICTOR_AVAILABLE or predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "النماذج غير محملة",
                "predictor_available": False
            }
        )
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "predictor_available": True,
        "model_info": predictor.get_model_info() if predictor else None
    }


@router.post("/predict", response_model=ClinicalPredictionResponse, summary="تنبؤ سريري شامل")
async def predict_clinical(patient: PatientClinicalData):
    """
    تنبؤ سريري شامل لمريض واحد:
    
    - 🔄 **Readmission Risk**: خطر إعادة الدخول خلال 30 يوم
    - ⏱️ **Length of Stay**: مدة الإقامة المتوقعة
    - 💡 **Recommendations**: توصيات سريرية مخصصة
    - 📋 **Clinical Notes**: ملاحظات طبية
  
    """
    if not PREDICTOR_AVAILABLE or predictor is None:
        raise HTTPException(
            status_code=503,
            detail="الخدمة غير متاحة حالياً - النماذج قيد التحميل"
        )
    
    try:
        # تحويل البيانات إلى dict
        patient_data = patient.dict()
        
        # تنبؤ
        result = predictor.predict(patient_data)
        
        # تجهيز التوصيات
        recommendations = []
        if 'recommendations' in result:
            for rec in result['recommendations']['readmission']:
                recommendations.append(rec['action'])
        
        # تجهيز الرد
        response = ClinicalPredictionResponse(
            patient_id=patient.patient_id,
            timestamp=result['metadata']['timestamp'],
            processing_time_ms=result['metadata']['processing_time_ms'],
            readmission_risk=result['readmission']['risk_level_ar'],
            readmission_probability=result['readmission']['probability'],
            readmission_confidence=result['readmission']['confidence'],
            predicted_los_days=result['los']['predicted_days'],
            los_confidence_interval=result['los']['confidence_interval'],
            recommendations=recommendations[:3],  # أهم 3 توصيات
            clinical_notes=f"مريض {patient.age} سنة | {result['readmission']['risk_level_ar']} خطر | {result['los']['predicted_days']} يوم متوقعة"
        )
        
        # تسجيل العملية
        logger.info(f"✅ تنبؤ ناجح للمريض {patient.patient_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ خطأ في التنبؤ للمريض {patient.patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ في التنبؤ: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse, summary="تنبؤ جماعي")
async def predict_batch(request: BatchPredictionRequest):
    """
    تنبؤ سريري لمجموعة من المرضى دفعة واحدة
    
    - مفيد لتحليل قاعدة بيانات كاملة
    - يقلل وقت المعالجة
    - يعطي نتائج متسقة
    """
    if not PREDICTOR_AVAILABLE or predictor is None:
        raise HTTPException(
            status_code=503,
            detail="الخدمة غير متاحة حالياً - النماذج قيد التحميل"
        )
    
    start_time = datetime.now()
    
    try:
        # تحويل القائمة إلى DataFrame
        patients_data = [p.dict() for p in request.patients]
        df = pd.DataFrame(patients_data)
        
        # تنبؤ جماعي
        results = predictor.predict_batch(df)
        
        # تجهيز الردود
        predictions = []
        for i, result in enumerate(results):
            recommendations = []
            if 'recommendations' in result:
                for rec in result['recommendations']['readmission']:
                    recommendations.append(rec['action'])
            
            pred = ClinicalPredictionResponse(
                patient_id=patients_data[i]['patient_id'],
                timestamp=result['metadata']['timestamp'],
                processing_time_ms=result['metadata']['processing_time_ms'],
                readmission_risk=result['readmission']['risk_level_ar'],
                readmission_probability=result['readmission']['probability'],
                readmission_confidence=result['readmission']['confidence'],
                predicted_los_days=result['los']['predicted_days'],
                los_confidence_interval=result['los']['confidence_interval'],
                recommendations=recommendations[:3],
                clinical_notes=f"مريض {patients_data[i]['age']} سنة | {result['readmission']['risk_level_ar']} خطر"
            )
            predictions.append(pred)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_patients=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"✅ تنبؤ جماعي ناجح لـ {len(predictions)} مريض")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ خطأ في التنبؤ الجماعي: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ في التنبؤ الجماعي: {str(e)}")


@router.get("/stats", summary="إحصائيات الأداء")
async def get_performance_stats():
    """
    إحصائيات أداء النظام:
    - عدد التوقعات
    - متوسط وقت المعالجة
    - نسبة الحالات عالية الخطورة
    - متوسط مدة الإقامة المتوقعة
    """
    if not PREDICTOR_AVAILABLE or predictor is None:
        raise HTTPException(
            status_code=503,
            detail="الخدمة غير متاحة حالياً"
        )
    
    try:
        stats = predictor.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"❌ خطأ في جلب الإحصائيات: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info", summary="معلومات النماذج")
async def get_models_info():
    """
    معلومات تفصيلية عن النماذج المستخدمة:
    - عدد الميزات
    - دقة النماذج
    - إصدارات النماذج
    """
    if not PREDICTOR_AVAILABLE or predictor is None:
        raise HTTPException(
            status_code=503,
            detail="الخدمة غير متاحة حالياً"
        )
    
    try:
        info = predictor.get_model_info()
        return info
    except Exception as e:
        logger.error(f"❌ خطأ في جلب معلومات النماذج: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/history", summary="تصدير سجل التوقعات")
async def export_history(format: str = "csv"):
    """
    تصدير سجل التوقعات كملف CSV أو JSON
    
    - format: csv أو json
    """
    if not PREDICTOR_AVAILABLE or predictor is None:
        raise HTTPException(
            status_code=503,
            detail="الخدمة غير متاحة حالياً"
        )
    
    try:
        filename = predictor.export_history(format)
        
        if os.path.exists(filename):
            return FileResponse(
                filename,
                media_type='text/csv' if format == 'csv' else 'application/json',
                filename=f'predictions_history.{format}'
            )
        else:
            raise HTTPException(status_code=404, detail="الملف غير موجود")
            
    except Exception as e:
        logger.error(f"❌ خطأ في تصدير السجل: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/guide", summary="دليل استخدام API")
async def api_guide():
    """
    دليل كامل لاستخدام API مع أمثلة
    """
    guide = {
        "api_name": "RIVA Clinical Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "/health": {
                "method": "GET",
                "description": "فحص صحة الخدمة",
                "example_response": {
                    "status": "healthy",
                    "timestamp": "2026-03-17T15:30:00",
                    "predictor_available": True
                }
            },
            "/predict": {
                "method": "POST",
                "description": "تنبؤ سريري شامل لمريض واحد",
                "request_example": {
                    "patient_id": "P001",
                    "age": 65,
                    "gender_M": 1,
                    "admit_EMERGENCY": 1,
                    "num_diagnoses": 5,
                    "num_procedures": 2,
                    "num_medications": 8,
                    "charlson_index": 3,
                    "total_visits": 4
                }
            },
            "/predict/batch": {
                "method": "POST",
                "description": "تنبؤ جماعي لمجموعة من المرضى"
            },
            "/stats": {
                "method": "GET",
                "description": "إحصائيات الأداء"
            },
            "/info": {
                "method": "GET",
                "description": "معلومات النماذج"
            }
        }
    }
    return guide


# ================== Error Handlers ==================
@router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


# =================- Middleware للتسجيل ==================
@router.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
    
    return response
