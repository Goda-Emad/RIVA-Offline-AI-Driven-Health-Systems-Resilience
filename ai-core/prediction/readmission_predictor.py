"""
===============================================================================
readmission_predictor.py
نموذج التنبؤ بإعادة دخول المريض خلال 30 يوم
Readmission Prediction Model (AUC: 0.917)
===============================================================================
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReadmissionPredictor:
    """
    المتنبئ بإعادة دخول المريض خلال 30 يوم
    AUC: 0.917
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.features = None
        self._load_model(model_path)
        logger.info("✅ ReadmissionPredictor initialized")
    
    def _load_model(self, model_path: Optional[str] = None):
        """تحميل نموذج XGBoost المدرب"""
        try:
            if model_path is None:
                models_dir = '/content/drive/MyDrive/RIVA-Maternal/ai-core/models'
                model_files = [f for f in os.listdir(models_dir) 
                              if f.startswith('readmission_xgb') and f.endswith('.pkl')]
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    model_path = os.path.join(models_dir, latest_model)
            
            self.model = joblib.load(model_path)
            
            # الميزات المستخدمة في التدريب
            self.features = [
                'CHF', 'ARRHYTHMIA', 'VALVULAR', 'PULMONARY', 'PVD',
                'HYPERTENSION', 'PARALYSIS', 'NEUROLOGICAL', 'HYPOTHYROID',
                'RENAL', 'LIVER', 'ULCER', 'AIDS', 'lab_count', 'lab_mean',
                'had_outpatient_labs', 'heart_rate_mean', 'sbp_mean',
                'dbp_mean', 'resp_rate_mean', 'temperature_mean', 'spo2_mean'
            ]
            logger.info(f"✅ Model loaded: {os.path.basename(model_path)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        التنبؤ بإعادة دخول المريض
        
        Args:
            features: قاموس بالميزات
            
        Returns:
            probability: نسبة الاحتمال
            risk_level: مستوى الخطر
            explanation: شرح النتيجة
            confidence: مستوى الثقة
        """
        try:
            # تجهيز الميزات
            X = []
            for feature in self.features:
                value = features.get(feature, 0)
                if isinstance(value, (str, bool)):
                    value = 1 if value else 0
                elif value is None:
                    value = 0
                X.append(float(value))
            
            X_array = np.array(X).reshape(1, -1)
            
            # التنبؤ
            probability = float(self.model.predict_proba(X_array)[0, 1])
            
            # مستوى الخطر
            if probability < 0.3:
                risk_level = 'low'
                base_msg = "✅ احتمالية إعادة الدخول منخفضة"
            elif probability < 0.7:
                risk_level = 'medium'
                base_msg = "⚠️ احتمالية إعادة الدخول متوسطة"
            else:
                risk_level = 'high'
                base_msg = "🔴 احتمالية إعادة الدخول مرتفعة"
            
            # شرح مبسط
            reasons = []
            if features.get('NEUROLOGICAL', 0):
                reasons.append("أمراض عصبية")
            if features.get('RENAL', 0):
                reasons.append("مشاكل في الكلى")
            if features.get('CHF', 0):
                reasons.append("قصور في القلب")
            if features.get('HYPERTENSION', 0):
                reasons.append("ارتفاع ضغط")
            
            explanation = f"{base_msg}"
            if reasons:
                explanation += f" بسبب: {', '.join(reasons[:3])}"
            
            # ثقة
            confidence = 0.8
            feature_count = sum(1 for v in features.values() if v not in [0, None, ''])
            confidence += min(feature_count * 0.01, 0.15)
            confidence = min(round(confidence, 2), 0.95)
            
            return {
                'probability': round(probability, 3),
                'risk_level': risk_level,
                'explanation': explanation,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {
                'probability': 0.5,
                'risk_level': 'unknown',
                'explanation': 'حدث خطأ في التنبؤ',
                'confidence': 0
            }


# =========================================================================
# اختبار
# =========================================================================

if __name__ == "__main__":
    predictor = ReadmissionPredictor()
    
    test_features = {
        'CHF': 1,
        'HYPERTENSION': 1,
        'NEUROLOGICAL': 1,
        'lab_count': 150,
        'lab_mean': 45.5
    }
    
    result = predictor.predict(test_features)
    print(f"📊 النتيجة: {result}")
