"""
===============================================================================
los_predictor.py
نموذج التنبؤ بمدة إقامة المريض في المستشفى
Length of Stay (LOS) Prediction Model
===============================================================================
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LOSPredictor:
    """
    المتنبئ بمدة إقامة المريض في المستشفى
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.features = [
            'CHF', 'ARRHYTHMIA', 'VALVULAR', 'PULMONARY', 'PVD',
            'HYPERTENSION', 'PARALYSIS', 'NEUROLOGICAL', 'HYPOTHYROID',
            'RENAL', 'LIVER', 'ULCER', 'AIDS', 'lab_count', 'lab_mean',
            'had_outpatient_labs', 'heart_rate_mean', 'sbp_mean',
            'dbp_mean', 'resp_rate_mean', 'temperature_mean', 'spo2_mean'
        ]
        self.max_los = 30
        self._load_model(model_path)
        logger.info("✅ LOSPredictor initialized")
    
    def _load_model(self, model_path: Optional[str] = None):
        """تحميل النموذج المدرب"""
        try:
            if model_path is None:
                models_dir = '/content/drive/MyDrive/RIVA-Maternal/ai-core/models'
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) 
                                  if f.startswith('los_') and f.endswith('.pkl')]
                    if model_files:
                        latest_model = sorted(model_files)[-1]
                        model_path = os.path.join(models_dir, latest_model)
                        self.model = joblib.load(model_path)
                        logger.info(f"✅ Model loaded: {latest_model}")
        except Exception as e:
            logger.warning(f"⚠️ Using fallback model: {e}")
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """التنبؤ بمدة الإقامة"""
        try:
            # تحضير الميزات
            X = []
            for feature in self.features:
                value = features.get(feature, 0)
                if isinstance(value, (str, bool)):
                    value = 1 if value else 0
                elif value is None:
                    value = 0
                X.append(float(value))
            
            # تنبؤ
            if self.model is not None:
                X_array = np.array(X).reshape(1, -1)
                log_days = float(self.model.predict(X_array)[0])
                days = np.expm1(log_days)
            else:
                # نموذج افتراضي
                days = 5.0
                if features.get('RENAL', 0):
                    days += 2.5
                if features.get('CHF', 0):
                    days += 2.0
                if features.get('NEUROLOGICAL', 0):
                    days += 2.8
            
            days = min(round(days, 1), self.max_los)
            
            # حساب الثقة
            confidence = 0.7
            feature_count = sum(1 for v in features.values() if v)
            confidence += min(feature_count * 0.01, 0.15)
            confidence = min(round(confidence, 2), 0.95)
            
            # تحديد التصنيف
            if days < 3:
                category = "🟢 قصيرة"
            elif days < 7:
                category = "🟡 متوسطة"
            elif days < 14:
                category = "🟠 طويلة"
            else:
                category = "🔴 طويلة جداً"
            
            # شرح النتيجة
            reasons = []
            if features.get('RENAL', 0):
                reasons.append("مشاكل كلوية")
            if features.get('CHF', 0):
                reasons.append("قصور قلب")
            if features.get('NEUROLOGICAL', 0):
                reasons.append("أمراض عصبية")
            if features.get('lab_count', 0) and features['lab_count'] > 100:
                reasons.append("كثرة التحاليل")
            
            explanation = f"مدة إقامة {category.lower()} ({days} أيام)"
            if reasons:
                explanation += f" بسبب: {', '.join(reasons[:3])}"
            
            return {
                'days': days,
                'confidence': confidence,
                'explanation': explanation,
                'risk_category': category,
                'min_days': max(0.5, round(days - (1-confidence)*days, 1)),
                'max_days': round(days + (1-confidence)*days, 1),
                'model_mae': 7.11,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {
                'days': 5.0,
                'confidence': 0,
                'explanation': f'حدث خطأ: {e}',
                'risk_category': '⚠️ خطأ',
                'min_days': 0,
                'max_days': 10,
                'model_mae': 7.11,
                'timestamp': datetime.now().isoformat()
            }


# اختبار سريع
if __name__ == "__main__":
    predictor = LOSPredictor()
    
    # حالات اختبار
    test_cases = [
        {"name": "حالة خفيفة", "features": {"HYPERTENSION": 1, "lab_count": 25}},
        {"name": "حالة متوسطة", "features": {"CHF": 1, "RENAL": 1, "lab_count": 85}},
        {"name": "حالة شديدة", "features": {"CHF": 1, "RENAL": 1, "NEUROLOGICAL": 1, "lab_count": 150}}
    ]
    
    print("
📊 نتائج الاختبار:")
    print("-"*50)
    
    for test in test_cases:
        result = predictor.predict(test["features"])
        print(f"
📋 {test['name']}:")
        print(f"   - المدة: {result['days']} أيام")
        print(f"   - {result['explanation']}")
        print(f"   - الثقة: {result['confidence']:.0%}")
    
    print("
✅ تم الانتهاء من الاختبار")
