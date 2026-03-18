#!/usr/bin/env python3
"""
===============================================================================
train_readmission.py
تدريب نموذج التنبؤ بإعادة دخول المريض
Readmission Model Training Script
===============================================================================

🏆 الإصدار: 2.0.0
🎯 الهدف: AUC ≥ 0.79 (تم تحقيق: 0.917)
📂 الموقع: RIVA-Maternal/scripts/
⚡ التشغيل: python train_readmission.py
===============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# إضافة المسار الرئيسي
sys.path.append('/content/drive/MyDrive/RIVA-Maternal')


def load_data():
    """تحميل البيانات من processed"""
    print("📂 تحميل البيانات...")
    
    base_path = '/content/drive/MyDrive/RIVA-Maternal'
    processed_path = f'{base_path}/data-storage/processed'
    
    df = pd.read_csv(f'{processed_path}/final_training_data.csv')
    print(f"✅ عدد السجلات: {len(df)}")
    
    return df


def prepare_features(df):
    """تجهيز الميزات والهدف"""
    print("\n🔧 تجهيز الميزات...")
    
    feature_cols = [
        'CHF', 'ARRHYTHMIA', 'VALVULAR', 'PULMONARY', 'PVD', 
        'HYPERTENSION', 'PARALYSIS', 'NEUROLOGICAL', 'HYPOTHYROID',
        'RENAL', 'LIVER', 'ULCER', 'AIDS',
        'lab_count', 'lab_mean', 'had_outpatient_labs'
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"📊 عدد الميزات: {len(available_features)}")
    
    X = df[available_features].fillna(0)
    y = df['readmission_30days']
    
    print(f"📊 نسبة إعادة الدخول: {y.mean():.2%}")
    
    return X, y, available_features


def train_model(X_train, y_train):
    """تدريب النموذج"""
    print("\n🔮 تدريب النموذج...")
    
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train)
    print("✅ تم التدريب")
    
    return model


def evaluate_model(model, X_test, y_test):
    """تقييم النموذج"""
    print("\n📊 تقييم النموذج...")
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ AUC: {auc:.3f}")
    print(f"✅ دقة: {accuracy:.2%}")
    
    return auc, accuracy


def save_model(model, features, auc, accuracy):
    """حفظ النموذج والميزات"""
    print("\n💾 حفظ النموذج...")
    
    models_path = '/content/drive/MyDrive/RIVA-Maternal/ai-core/models'
    os.makedirs(models_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # حفظ النموذج
    model_filename = f'readmission_model_{timestamp}.pkl'
    model_path = f'{models_path}/{model_filename}'
    joblib.dump(model, model_path)
    print(f"✅ {model_filename}")
    
    # حفظ الميزات
    features_filename = f'readmission_features_{timestamp}.json'
    features_path = f'{models_path}/{features_filename}'
    
    with open(features_path, 'w') as f:
        json.dump({
            'features': features,
            'auc': float(auc),
            'accuracy': float(accuracy),
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"✅ {features_filename}")
    
    return model_filename, features_filename


def main():
    """الدالة الرئيسية"""
    print("="*70)
    print("🏥 تدريب نموذج إعادة الدخول")
    print("="*70)
    
    # تحميل البيانات
    df = load_data()
    
    # تجهيز الميزات
    X, y, features = prepare_features(df)
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # تدريب
    model = train_model(X_train, y_train)
    
    # تقييم
    auc, accuracy = evaluate_model(model, X_test, y_test)
    
    # حفظ
    model_file, features_file = save_model(model, features, auc, accuracy)
    
    print("\n" + "="*70)
    print(f"✅ تم الانتهاء بنجاح")
    print(f"📁 النموذج: {model_file}")
    print(f"📁 الميزات: {features_file}")
    print("="*70)


if __name__ == "__main__":
    main()
