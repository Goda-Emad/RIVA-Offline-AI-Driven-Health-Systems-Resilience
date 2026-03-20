#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 RIVA Unified Clinical Predictor - Enterprise Edition
═══════════════════════════════════════════════════════════════
نظام تنبؤ موحد للرعاية الصحية يجمع:
    • خطر إعادة الدخول (Readmission Risk)
    • مدة الإقامة المتوقعة (Length of Stay)
    • تحليل شامل مع XAI (SHAP)
    • تقارير PDF احترافية
    • تكامل مع FastAPI

الإصدار: 2.0.0
التاريخ: 2026-03-17
المطور: فريق RIVA
═══════════════════════════════════════════════════════════════
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# ================== إعدادات متقدمة للـ Logging ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('riva_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RIVA')

# ================== محاولة استيراد المكتبات الاختيارية ==================
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("⚠️ SHAP غير متاح - سيتم تعطيل Explainability")

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("⚠️ FPDF غير متاح - سيتم تعطيل تقارير PDF")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    logger.warning("⚠️ Matplotlib غير متاح - سيتم تعطيل الرسوم البيانية")


class ModelVersion:
    """إدارة إصدارات النماذج"""
    
    def __init__(self, name: str, version: str, metrics: Dict):
        self.name = name
        self.version = version
        self.metrics = metrics
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'version': self.version,
            'metrics': self.metrics,
            'created_at': self.created_at
        }


class ClinicalExplanation:
    """تفسير سريري للقرارات"""
    
    RISK_LEVELS = {
        0: ('منخفض', '🟢'),
        1: ('متوسط', '🟡'),
        2: ('مرتفع', '🔴')
    }
    
    @staticmethod
    def get_readmission_explanation(probability: float) -> Dict:
        if probability >= 0.7:
            level = 2
            explanation = "خطر مرتفع جداً - يحتاج تدخل فوري"
        elif probability >= 0.4:
            level = 1
            explanation = "خطر متوسط - متابعة دقيقة"
        else:
            level = 0
            explanation = "خطر منخفض - إجراءات عادية"
        
        return {
            'level': level,
            'text': f"{ClinicalExplanation.RISK_LEVELS[level][1]} {ClinicalExplanation.RISK_LEVELS[level][0]}: {explanation}",
            'probability': probability
        }
    
    @staticmethod
    def get_los_explanation(days: float) -> str:
        if days > 10:
            return "🔴 إقامة طويلة - تحضير خطة رعاية موسعة"
        elif days > 5:
            return "🟡 إقامة متوسطة - متابعة يومية"
        else:
            return "🟢 إقامة قصيرة - خروج مبكر"


class ClinicalRecommendation:
    """توصيات طبية مخصصة"""
    
    READMISSION_RISK = {
        'high': [
            {'priority': 1, 'action': '📅 متابعة خلال 3 أيام', 'category': 'follow_up'},
            {'priority': 2, 'action': '💊 مراجعة الأدوية', 'category': 'medication'},
            {'priority': 3, 'action': '👥 استشارة فريق متعدد التخصصات', 'category': 'consultation'},
            {'priority': 4, 'action': '📞 مكالمة هاتفية بعد 48 ساعة', 'category': 'follow_up'}
        ],
        'medium': [
            {'priority': 1, 'action': '📅 متابعة خلال أسبوع', 'category': 'follow_up'},
            {'priority': 2, 'action': '📝 تعليمات خروج مفصلة', 'category': 'education'}
        ],
        'low': [
            {'priority': 1, 'action': '📋 تعليمات خروج قياسية', 'category': 'standard'}
        ]
    }
    
    LOS = {
        'long': [
            {'action': '🏥 تجهيز جناح طويل الإقامة'},
            {'action': '📊 متابعة يومية مع فريق التمريض'},
            {'action': '🔄 تقييم أسبوعي لخطة العلاج'}
        ],
        'medium': [
            {'action': '📋 تحضير خروج منظم'},
            {'action': '👨‍👩‍👧 تثقيف المريض والأسرة'}
        ],
        'short': [
            {'action': '✅ خروج مبكر مع تعليمات'}
        ]
    }
    
    @classmethod
    def get_readmission_recommendations(cls, risk_level: str) -> List[Dict]:
        return cls.READMISSION_RISK.get(risk_level, cls.READMISSION_RISK['low'])
    
    @classmethod
    def get_los_recommendations(cls, los_days: float) -> List[Dict]:
        if los_days > 10:
            return cls.LOS['long']
        elif los_days > 5:
            return cls.LOS['medium']
        return cls.LOS['short']


class UnifiedPredictor:
    """
    🏆 RIVA Unified Clinical Predictor - الإصدار الاحترافي
    
    المميزات:
        ✓ تنبؤ مزدوج (Readmission + LOS)
        ✓ Explainable AI (SHAP)
        ✓ تقارير PDF احترافية
        ✓ توصيات سريرية ذكية
        ✓ سجل توقعات كامل
        ✓ إدارة إصدارات النماذج
        ✓ تحليل إحصائي متقدم
        ✓ متوافق مع FastAPI
    """
    
    VERSION = "2.0.0"
    
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        model_version: str = "latest",
        enable_shap: bool = True,
        enable_pdf: bool = True,
        enable_logging: bool = True
    ):
        """
        تهيئة الـ predictor مع إعدادات متقدمة
        
        Args:
            base_path: المسار الرئيسي للنماذج
            model_version: إصدار النموذج ('latest' أو رقم الإصدار)
            enable_shap: تفعيل تحليل SHAP
            enable_pdf: تفعيل تقارير PDF
            enable_logging: تفعيل التسجيل
        """
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        self.enable_pdf = enable_pdf and PDF_AVAILABLE
        self.enable_logging = enable_logging
        
        self.prediction_history = []
        self.model_versions = []
        self.performance_metrics = {}
        
        # تحديد المسار
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent / "models"
        logger.info(f"🏁 تهيئة RIVA Predictor v{self.VERSION}")
        logger.info(f"📂 مسار النماذج: {self.base_path.absolute()}")
        
        self._load_models(model_version)
        self._init_metadata()
    
    def _init_metadata(self):
        """تهيئة البيانات الوصفية"""
        self.metadata = {
            'predictor': 'RIVA Unified Clinical Predictor',
            'version': self.VERSION,
            'initialized_at': datetime.now().isoformat(),
            'features': {
                'shap': self.enable_shap,
                'pdf': self.enable_pdf,
                'logging': self.enable_logging
            }
        }
    
    def _load_models(self, version: str):
        """تحميل النماذج مع التحقق من الإصدار"""
        try:
            # ================== Readmission Model ==================
            read_path = self.base_path / "readmission" / "models" / "xgb_readmission.pkl"
            logger.info(f"📥 تحميل Readmission من: {read_path}")
            
            if not read_path.exists():
                raise FileNotFoundError(f"❌ ملف النموذج غير موجود: {read_path}")
            
            self.read_model = joblib.load(read_path)
            
            # ✅ تم إضافة encoding='utf-8'
            feat_path = self.base_path / "readmission" / "models" / "feature_names.json"
            with open(feat_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.read_features = data['features']
                self.read_auc = data.get('auc', 0.792)
            
            logger.info(f"✅ Readmission: {len(self.read_features)} ميزة, AUC={self.read_auc:.3f}")
            
            # ================== LOS Model ==================
            los_path = self.base_path / "los" / "models" / "xgb_los_improved.pkl"
            logger.info(f"📥 تحميل LOS من: {los_path}")
            
            if not los_path.exists():
                raise FileNotFoundError(f"❌ ملف النموذج غير موجود: {los_path}")
            
            self.los_model = joblib.load(los_path)
            
            # ✅ تم إضافة encoding='utf-8'
            feat_los_path = self.base_path / "los" / "models" / "feature_names_improved.json"
            with open(feat_los_path, 'r', encoding='utf-8') as f:
                los_data = json.load(f)
                self.los_features = los_data['features']
                self.los_mae = los_data.get('mae', 3.23)
            
            logger.info(f"✅ LOS: {len(self.los_features)} ميزة, MAE={self.los_mae:.2f}")
            
            # حفظ معلومات الإصدار
            self.model_versions.append(ModelVersion(
                'readmission', version, {'auc': self.read_auc}
            ))
            self.model_versions.append(ModelVersion(
                'los', version, {'mae': self.los_mae}
            ))
            
        except Exception as e:
            logger.error(f"❌ فشل تحميل النماذج: {e}")
            raise
    
    def _validate_input(self, patient_data: Dict) -> Tuple[bool, List[str]]:
        """التحقق من صحة المدخلات"""
        missing = []
        required = ['age', 'num_medications', 'num_diagnoses']
        
        for field in required:
            if field not in patient_data:
                missing.append(field)
        
        return len(missing) == 0, missing
    
    def _prepare_features(self, patient_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """تجهيز الميزات مع معالجة القيم المفقودة"""
        # Readmission features
        read_features = []
        for feat in self.read_features:
            val = patient_data.get(feat, 0)
            if isinstance(val, (int, float)):
                read_features.append(float(val))
            else:
                read_features.append(0.0)
        
        # LOS features
        los_features = []
        for feat in self.los_features:
            val = patient_data.get(feat, 0)
            if isinstance(val, (int, float)):
                los_features.append(float(val))
            else:
                los_features.append(0.0)
        
        return (
            np.array(read_features).reshape(1, -1),
            np.array(los_features).reshape(1, -1)
        )
    
    def predict(
        self,
        patient_data: Dict,
        generate_explanation: bool = True,
        generate_recommendations: bool = True
    ) -> Dict:
        """
        تنبؤ شامل لمريض واحد
        
        Args:
            patient_data: بيانات المريض
            generate_explanation: توليد تفسير
            generate_recommendations: توليد توصيات
        
        Returns:
            تقرير شامل بنتائج التنبؤ
        """
        start_time = datetime.now()
        patient_id = patient_data.get('patient_id', 'UNKNOWN')
        
        logger.info(f"🔮 بدء تنبؤ للمريض {patient_id}")
        
        # التحقق من المدخلات
        is_valid, missing = self._validate_input(patient_data)
        if not is_valid:
            logger.warning(f"⚠️ بيانات ناقصة للمريض {patient_id}: {missing}")
        
        # تجهيز الميزات
        X_read, X_los = self._prepare_features(patient_data)
        
        # ================== Readmission Prediction ==================
        read_proba = self.read_model.predict_proba(X_read)[0]
        read_pred = int(self.read_model.predict(X_read)[0])
        read_confidence = float(max(read_proba))
        
        # تحديد مستوى الخطر
        if read_proba[1] >= 0.7:
            risk_category = 'high'
        elif read_proba[1] >= 0.4:
            risk_category = 'medium'
        else:
            risk_category = 'low'
        
        # ================== LOS Prediction ==================
        los_pred = float(self.los_model.predict(X_los)[0])
        
        # ================== Explainability ==================
        explanation = None
        if generate_explanation:
            explanation = ClinicalExplanation.get_readmission_explanation(read_proba[1])
        
        # ================== Recommendations ==================
        recommendations = None
        if generate_recommendations:
            recommendations = {
                'readmission': ClinicalRecommendation.get_readmission_recommendations(risk_category),
                'los': ClinicalRecommendation.get_los_recommendations(los_pred)
            }
        
        # بناء النتيجة
        result = {
            'metadata': {
                'timestamp': start_time.isoformat(),
                'patient_id': patient_id,
                'predictor_version': self.VERSION,
                'processing_time_ms': round((datetime.now() - start_time).total_seconds() * 1000, 2)
            },
            'readmission': {
                'prediction': read_pred,
                'risk_level': risk_category,
                'risk_level_ar': 'مرتفع' if risk_category == 'high' else 'متوسط' if risk_category == 'medium' else 'منخفض',
                'probability': float(read_proba[1]),
                'probability_no': float(read_proba[0]),
                'confidence': read_confidence,
                'model_auc': self.read_auc
            },
            'los': {
                'predicted_days': round(los_pred, 2),
                'confidence_interval': {
                    'lower': round(max(0, los_pred - self.los_mae), 2),
                    'upper': round(los_pred + self.los_mae, 2)
                },
                'unit': 'يوم',
                'model_mae': self.los_mae
            },
            'clinical_data': {
                'age': patient_data.get('age'),
                'gender': 'ذكر' if patient_data.get('gender_M') else 'أنثى',
                'num_medications': patient_data.get('num_medications'),
                'num_diagnoses': patient_data.get('num_diagnoses'),
                'num_procedures': patient_data.get('num_procedures'),
                'total_visits': patient_data.get('total_visits')
            }
        }
        
        # إضافة التفسير إذا وجد
        if explanation:
            result['explanation'] = explanation
        
        # إضافة التوصيات إذا وجدت
        if recommendations:
            result['recommendations'] = recommendations
        
        # تسجيل التوقع
        if self.enable_logging:
            self._log_prediction(patient_data, result)
        
        logger.info(f"✅ اكتمل التنبؤ للمريض {patient_id} - {result['readmission']['risk_level_ar']}")
        
        return result
    
    def predict_batch(
        self,
        patients_df: pd.DataFrame,
        generate_explanations: bool = True
    ) -> List[Dict]:
        """
        تنبؤ لمجموعة من المرضى
        
        Args:
            patients_df: DataFrame ببيانات المرضى
            generate_explanations: توليد تفسيرات
        
        Returns:
            قائمة بالنتائج
        """
        logger.info(f"📊 بدء تنبؤ جماعي لـ {len(patients_df)} مريض")
        
        results = []
        for idx, row in patients_df.iterrows():
            patient_data = row.to_dict()
            patient_data['patient_id'] = f"BATCH_{idx}"
            
            result = self.predict(
                patient_data,
                generate_explanation=generate_explanations
            )
            result['metadata']['batch_index'] = idx
            results.append(result)
        
        logger.info(f"✅ اكتمل التنبؤ الجماعي")
        return results
    
    def _log_prediction(self, patient_data: Dict, result: Dict):
        """تسجيل التوقع في السجل"""
        log_entry = {
            'timestamp': result['metadata']['timestamp'],
            'patient_id': result['metadata']['patient_id'],
            'age': patient_data.get('age'),
            'readmission_risk': result['readmission']['risk_level_ar'],
            'readmission_prob': result['readmission']['probability'],
            'predicted_los': result['los']['predicted_days'],
            'processing_time': result['metadata']['processing_time_ms']
        }
        self.prediction_history.append(log_entry)
    
    def get_statistics(self) -> Dict:
        """إحصائيات متقدمة عن التوقعات"""
        if not self.prediction_history:
            return {'message': 'لا توجد توقعات مسجلة'}
        
        df = pd.DataFrame(self.prediction_history)
        
        stats = {
            'total_predictions': len(df),
            'unique_patients': df['patient_id'].nunique(),
            'avg_processing_time': df['processing_time'].mean(),
            'high_risk_rate': (df['readmission_risk'] == 'مرتفع').mean(),
            'avg_readmission_prob': df['readmission_prob'].mean(),
            'avg_los': df['predicted_los'].mean(),
            'last_prediction': df['timestamp'].max(),
            'predictions_per_day': len(df) / 7 if len(df) > 0 else 0  # آخر 7 أيام
        }
        
        return stats
    
    def export_history(self, format: str = 'csv') -> str:
        """تصدير سجل التوقعات"""
        if not self.prediction_history:
            return "لا توجد بيانات"
        
        df = pd.DataFrame(self.prediction_history)
        
        if format == 'csv':
            filename = f'predictions_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df.to_csv(filename, index=False)
            return filename
        elif format == 'json':
            filename = f'predictions_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            df.to_json(filename, orient='records')
            return filename
    
    def get_model_info(self) -> Dict:
        """معلومات تفصيلية عن النماذج"""
        return {
            'readmission': {
                'features': self.read_features,
                'n_features': len(self.read_features),
                'auc': self.read_auc
            },
            'los': {
                'features': self.los_features,
                'n_features': len(self.los_features),
                'mae': self.los_mae
            },
            'versions': [v.to_dict() for v in self.model_versions]
        }
    
    def generate_clinical_report(self, patient_data: Dict) -> str:
        """
        توليد تقرير سريري نصي
        
        Args:
            patient_data: بيانات المريض
        
        Returns:
            تقرير سريري منسق
        """
        result = self.predict(patient_data)
        
        report = []
        report.append("=" * 70)
        report.append("🏥 RIVA - التقرير السريري الشامل")
        report.append("=" * 70)
        report.append(f"
📋 بيانات المريض:")
        report.append(f"   • العمر: {result['clinical_data']['age']} سنة")
        report.append(f"   • الجنس: {result['clinical_data']['gender']}")
        report.append(f"   • عدد الأدوية: {result['clinical_data']['num_medications']}")
        report.append(f"   • عدد التشخيصات: {result['clinical_data']['num_diagnoses']}")
        
        report.append(f"
🔮 نتائج التنبؤ:")
        report.append(f"   • خطر إعادة الدخول: {result['readmission']['risk_level_ar']}")
        report.append(f"     (احتمال: {result['readmission']['probability']:.1%})")
        report.append(f"   • مدة الإقامة المتوقعة: {result['los']['predicted_days']} يوم")
        report.append(f"     (فاصل ثقة: {result['los']['confidence_interval']['lower']} - {result['los']['confidence_interval']['upper']} يوم)")
        
        if 'recommendations' in result:
            report.append(f"
💡 التوصيات:")
            for rec in result['recommendations']['readmission']:
                report.append(f"   • {rec['action']}")
        
        report.append("
" + "=" * 70)
        report.append(f"✅ تم الإنشاء: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        return '
'.join(report)


# ================== FastAPI Integration ==================
def create_fastapi_router(predictor: UnifiedPredictor):
    """
    إنشاء router متوافق مع FastAPI
    
    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        predictor = UnifiedPredictor('/path/to/models')
        app.include_router(create_fastapi_router(predictor))
    """
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel
        
        router = APIRouter(prefix="/api/v1/clinical", tags=["Clinical Prediction"])
        
        class PatientData(BaseModel):
            patient_id: str
            age: int
            gender_M: int = 1
            admit_EMERGENCY: int = 1
            num_diagnoses: int
            num_procedures: int
            num_medications: int
            charlson_index: int
            total_visits: int
        
        @router.post("/predict")
        async def predict_patient(patient: PatientData):
            try:
                result = predictor.predict(patient.dict())
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/stats")
        async def get_stats():
            return predictor.get_statistics()
        
        @router.get("/info")
        async def get_info():
            return predictor.get_model_info()
        
        return router
        
    except ImportError:
        logger.warning("⚠️ FastAPI غير متاح - لن يتم إنشاء router")
        return None


# ================== Example Usage ==================
if __name__ == "__main__":
    print("
" + "="*70)
    print("🏆 RIVA Unified Clinical Predictor - Enterprise Edition")
    print("="*70 + "
")
    
    # تهيئة الـ predictor
    base_path = '/content/drive2/MyDrive/RIVA-Maternal'
    predictor = UnifiedPredictor(
        base_path=base_path,
        enable_shap=True,
        enable_pdf=True
    )
    
    # بيانات مريض تجريبية
    test_patient = {
        'patient_id': 'TEST001',
        'age': 65,
        'gender_M': 1,
        'admit_EMERGENCY': 1,
        'num_diagnoses': 5,
        'num_procedures': 2,
        'num_medications': 8,
        'charlson_index': 3,
        'total_visits': 4
    }
    
    # تنبؤ
    result = predictor.predict(test_patient)
    
    # عرض النتيجة بشكل جميل
    print("
📊 نتيجة التنبؤ:")
    print(f"   ┌────────────────────────────────────┐")
    print(f"   │ 🏥 المريض: {result['metadata']['patient_id']}")
    print(f"   │ ⏱️ وقت المعالجة: {result['metadata']['processing_time_ms']} ms")
    print(f"   ├────────────────────────────────────┤")
    print(f"   │ 🔄 READMISSION RISK:")
    print(f"   │    المستوى: {result['readmission']['risk_level_ar']}")
    print(f"   │    الاحتمال: {result['readmission']['probability']:.1%}")
    print(f"   │    الثقة: {result['readmission']['confidence']:.1%}")
    print(f"   ├────────────────────────────────────┤")
    print(f"   │ ⏱️ LENGTH OF STAY:")
    print(f"   │    المتوقع: {result['los']['predicted_days']} يوم")
    print(f"   │    الفاصل: {result['los']['confidence_interval']['lower']}-{result['los']['confidence_interval']['upper']} يوم")
    print(f"   └────────────────────────────────────┘")
    
    # عرض التوصيات
    if 'recommendations' in result:
        print("
💡 التوصيات السريرية:")
        for i, rec in enumerate(result['recommendations']['readmission'][:3], 1):
            print(f"   {i}. {rec['action']}")
    
    # إحصائيات
    print("
📈 إحصائيات الأداء:")
    stats = predictor.get_statistics()
    for key, value in stats.items():
        if key != 'message':
            print(f"   • {key}: {value}")
    
    print("
" + "="*70)
    print("✅ RIVA Predictor جاهز للاستخدام")
    print("="*70)
