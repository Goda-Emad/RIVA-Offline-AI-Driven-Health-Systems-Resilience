# ai-core/training/train_pregnancy.py
"""
تدريب نموذج XGBoost للتنبؤ بمخاطر الحمل
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ================== المسارات ==================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "data-storage" / "raw" / "uci_maternal" / "maternal_health_clean.csv"
MODEL_DIR = BASE_DIR / "ai-core" / "models" / "pregnancy"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
REPORTS_DIR = BASE_DIR / "business-intelligence" / "performance"

# إنشاء المجلدات
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# إعداد logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """تحميل وتجهيز البيانات"""
    logger.info("📦 تحميل البيانات...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"✅ تم تحميل {len(df)} عينة")
    
    # الميزات الأساسية
    feature_names = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
    X = df[feature_names].copy()
    y = df['RiskLevel_encoded'].copy()
    
    # الميزات المشتقة
    X['PulsePressure'] = X['SystolicBP'] - X['DiastolicBP']
    X['BP_ratio'] = (X['SystolicBP'] / (X['DiastolicBP'] + 1e-10)).round(2)
    X['Temp_Fever'] = (X['BodyTemp'] > 37.5).astype(int)
    X['HighBP'] = (X['SystolicBP'] > 140).astype(int)
    X['MeanBP'] = (X['SystolicBP'] + 2 * X['DiastolicBP']) / 3
    X['AgeRisk'] = (X['Age'] > 35).astype(int)
    X['HighSugar'] = (X['BS'] > 7).astype(int)
    
    final_features = [
        'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate',
        'PulsePressure', 'BP_ratio', 'Temp_Fever', 'HighBP', 'MeanBP', 'AgeRisk', 'HighSugar'
    ]
    
    X = X[final_features]
    return X, y, final_features

def train_model(X, y):
    """تدريب النموذج"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42))
    ])
    
    param_grid = {
        'xgb__n_estimators': [200, 300],
        'xgb__max_depth': [4, 6],
        'xgb__learning_rate': [0.05, 0.1]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    logger.info(f"✅ أفضل باراميترات: {grid.best_params_}")
    return grid.best_estimator_, X_train, X_test, y_train, y_test

def main():
    logger.info("🚀 بدأ تدريب النموذج")
    X, y, features = load_and_prepare_data()
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # تقييم
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['low', 'mid', 'high']))
    
    # حفظ
    joblib.dump(model, MODEL_DIR / "maternal_health_xgb_pipeline.pkl")
    with open(MODEL_DIR / "feature_names.json", 'w') as f:
        json.dump(features, f)
    
    logger.info("✅ تم حفظ النموذج")

if __name__ == "__main__":
    main()
