# ai-core/local-inference/pregnancy_risk.py
import joblib
import pandas as pd
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Union

# ✅ إعداد الـ logging للاستخدام المستقل (Standalone)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # يظهر في التيرمنال
    ]
)
logger = logging.getLogger(__name__)

class PregnancyRiskPredictor:
    def __init__(self, model_path: Optional[Path] = None):
        """
        تحميل النموذج والمعلومات مع معالجة الأخطاء
        """
        # ✅ تعريف المتغيرات بـ None أولاً
        self.model = None
        self.feature_names = None
        self.class_names = None
        
        base = Path(__file__).parent.parent
        self.model_path = model_path or base / "models" / "pregnancy" / "maternal_health_xgb_pipeline.pkl"
        self.features_path = base / "models" / "pregnancy" / "feature_names.json"
        self.classes_path = base / "models" / "pregnancy" / "class_names.json"
        
        # ✅ محاولة التحميل مع try/except (الكلاس مش هيوقع)
        self._load_model()
    
    def _load_model(self) -> bool:
        """تحميل الموارد مع معالجة الأخطاء"""
        try:
            logger.info(f"📦 تحميل النموذج من: {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"📦 تحميل الميزات من: {self.features_path}")
            with open(self.features_path, 'r') as f:
                self.feature_names = json.load(f)
            
            logger.info(f"📦 تحميل الفئات من: {self.classes_path}")
            with open(self.classes_path, 'r') as f:
                self.class_names = json.load(f)
            
            logger.info(f"✅ تم التحميل: {len(self.feature_names)} ميزات، {len(self.class_names)} فئات")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"❌ ملف غير موجود: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ فشل تحميل النموذج: {e}")
            return False
    
    def _compute_engineered_features(self, row):
        """حساب الميزات المشتقة لصف واحد"""
        systolic = row['systolic_bp'] if 'systolic_bp' in row else row.get('SystolicBP', 0)
        diastolic = row['diastolic_bp'] if 'diastolic_bp' in row else row.get('DiastolicBP', 0)
        age = row['age'] if 'age' in row else row.get('Age', 0)
        bs = row['bs'] if 'bs' in row else row.get('BS', 0)
        temp = row['body_temp'] if 'body_temp' in row else row.get('BodyTemp', 0)
        
        return {
            'PulsePressure': systolic - diastolic,
            'BP_ratio': round(systolic / diastolic, 2) if diastolic != 0 else 0,
            'Temp_Fever': 1 if temp > 37.5 else 0,
            'HighBP': 1 if systolic > 140 else 0,
            'MeanBP': (systolic + 2 * diastolic) / 3,
            'AgeRisk': 1 if age > 35 else 0,
            'HighSugar': 1 if bs > 7 else 0
        }
    
    def _prepare_single(self, age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
        """تجهيز مريض واحد (للاستخدام العادي)"""
        features = {
            "Age": age,
            "SystolicBP": systolic_bp,
            "DiastolicBP": diastolic_bp,
            "BS": bs,
            "BodyTemp": body_temp,
            "HeartRate": heart_rate,
            **self._compute_engineered_features({
                'age': age, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
                'bs': bs, 'body_temp': body_temp
            })
        }
        
        features_list = [features[name] for name in self.feature_names]
        return pd.DataFrame([features_list], columns=self.feature_names)
    
    def _prepare_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ تجهيز مجموعة من المرضى مع حساب جميع الميزات المشتقة
        (Vectorized operation - سريع وآمن)
        """
        # نسخة من الداتا فريم عشان نضيف أعمدة جديدة
        df_copy = df.copy()
        
        # توحيد أسماء الأعمدة (تسامح مع الحالات المختلفة)
        column_map = {
            'Age': 'age', 'SystolicBP': 'systolic_bp', 'DiastolicBP': 'diastolic_bp',
            'BS': 'bs', 'BodyTemp': 'body_temp', 'HeartRate': 'heart_rate',
            'age': 'age', 'systolic_bp': 'systolic_bp', 'diastolic_bp': 'diastolic_bp',
            'bs': 'bs', 'body_temp': 'body_temp', 'heart_rate': 'heart_rate'
        }
        
        # إعادة تسمية الأعمدة لصيغة موحدة
        df_copy = df_copy.rename(columns=column_map)
        
        # التأكد من وجود الأعمدة الأساسية
        required = ['age', 'systolic_bp', 'diastolic_bp', 'bs', 'body_temp', 'heart_rate']
        for col in required:
            if col not in df_copy.columns:
                raise ValueError(f"العمود المطلوب {col} غير موجود في البيانات")
        
        # حساب الميزات المشتقة (Vectorized)
        df_copy['PulsePressure'] = df_copy['systolic_bp'] - df_copy['diastolic_bp']
        df_copy['BP_ratio'] = (df_copy['systolic_bp'] / df_copy['diastolic_bp']).round(2)
        df_copy['Temp_Fever'] = (df_copy['body_temp'] > 37.5).astype(int)
        df_copy['HighBP'] = (df_copy['systolic_bp'] > 140).astype(int)
        df_copy['MeanBP'] = (df_copy['systolic_bp'] + 2 * df_copy['diastolic_bp']) / 3
        df_copy['AgeRisk'] = (df_copy['age'] > 35).astype(int)
        df_copy['HighSugar'] = (df_copy['bs'] > 7).astype(int)
        
        # ترتيب الميزات حسب feature_names
        # نتأكد إن كل الميزات المطلوبة موجودة
        missing = [f for f in self.feature_names if f not in df_copy.columns]
        if missing:
            raise ValueError(f"الميزات التالية غير موجودة بعد التجهيز: {missing}")
        
        return df_copy[self.feature_names]
    
    def predict(self, age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate) -> Dict:
        """
        توقع مستوى خطورة لمريض واحد
        """
        if self.model is None:
            logger.error("النموذج غير محمل")
            return {"error": "النموذج غير محمل", "risk_level": "unknown", "confidence": 0.0}
        
        try:
            df = self._prepare_single(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate)
            pred = self.model.predict(df)[0]
            prob = self.model.predict_proba(df)[0]
            
            return {
                "risk_level": self.class_names[pred],
                "risk_level_encoded": int(pred),
                "confidence": float(max(prob)),
                "probabilities": prob.tolist(),
                "pulse_pressure": systolic_bp - diastolic_bp,
                "bp_ratio": round(systolic_bp / diastolic_bp, 2),
                "temp_fever": 1 if body_temp > 37.5 else 0
            }
            
        except Exception as e:
            logger.error(f"خطأ في التوقع: {e}")
            return {"error": str(e), "risk_level": "error", "confidence": 0.0}
    
    def predict_batch(self, patients_df: pd.DataFrame) -> List[Dict]:
        """
        ✅ توقع لمجموعة من المرضى (Batch prediction)
        مع حساب جميع الميزات المشتقة أوتوماتيكياً
        """
        if self.model is None:
            logger.error("النموذج غير محمل")
            return [{"error": "النموذج غير محمل"} for _ in range(len(patients_df))]
        
        try:
            # تجهيز الداتا فريم (حساب كل الميزات المشتقة)
            logger.info(f"📊 تجهيز {len(patients_df)} مريض للـ batch prediction")
            X = self._prepare_batch(patients_df)
            
            # توقع
            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)
            
            results = []
            for i, pred in enumerate(preds):
                results.append({
                    "risk_level": self.class_names[pred],
                    "risk_level_encoded": int(pred),
                    "confidence": float(max(probs[i])),
                    "probabilities": probs[i].tolist()
                })
            
            logger.info(f"✅ تم توقع {len(results)} مريض بنجاح")
            return results
            
        except Exception as e:
            logger.error(f"خطأ في batch prediction: {e}")
            return [{"error": str(e)} for _ in range(len(patients_df))]
    
    def is_loaded(self) -> bool:
        """التحقق من حالة النموذج"""
        return self.model is not None

# ================== مثال للاستخدام ==================
if __name__ == "__main__":
    # اختبار سريع
    predictor = PregnancyRiskPredictor()
    
    if predictor.is_loaded():
        # توقع فردي
        result = predictor.predict(
            age=32,
            systolic_bp=120,
            diastolic_bp=80,
            bs=5.5,
            body_temp=37.0,
            heart_rate=75
        )
        print("\n📊 توقع فردي:", result)
        
        # توقع مجموعة (اختبار)
        test_data = pd.DataFrame([
            {"age": 28, "systolic_bp": 110, "diastolic_bp": 70, "bs": 5.0, "body_temp": 36.8, "heart_rate": 72},
            {"age": 38, "systolic_bp": 140, "diastolic_bp": 90, "bs": 8.2, "body_temp": 37.3, "heart_rate": 80}
        ])
        
        batch_results = predictor.predict_batch(test_data)
        print("\n📊 توقع مجموعة:", batch_results)
