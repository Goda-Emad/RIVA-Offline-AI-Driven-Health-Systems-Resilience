"""
===============================================================================
train_los.py
تدريب نموذج التنبؤ بمدة الإقامة
Length of Stay (LOS) Model Training
===============================================================================

🏆 الإصدار: 2.0.0
🎯 الهدف: MAE ≤ 3.2 يوم
⚡ وقت التدريب: < 3 دقائق
===============================================================================
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🏥 تدريب نموذج مدة الإقامة - LOS Model")
print("="*70)

# المسارات
base_path = '/content/drive/MyDrive/RIVA-Maternal'
processed_path = f'{base_path}/data-storage/processed'
models_path = f'{base_path}/ai-core/models'
os.makedirs(models_path, exist_ok=True)

# ============================================================
# 1. تحميل البيانات
# ============================================================
print("\n📂 1. تحميل البيانات...")

df = pd.read_csv(f'{processed_path}/final_training_data.csv')
print(f"✅ عدد السجلات: {len(df)}")

# تحميل ADMISSIONS للحصول على LOS
df_adm = pd.read_csv(f'{base_path}/data-storage/raw/mimic/ADMISSIONS.csv')
df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])
df_adm['dischtime'] = pd.to_datetime(df_adm['dischtime'])
df_adm['los_days'] = (df_adm['dischtime'] - df_adm['admittime']).dt.total_seconds() / (24 * 3600)

# دمج مع البيانات
df = df.merge(df_adm[['hadm_id', 'los_days']], on='hadm_id', how='left')

# معالجة القيم الشاذة
max_los = 30
df['los_days_capped'] = df['los_days'].clip(upper=max_los)
df['los_days_log'] = np.log1p(df['los_days_capped'])

print(f"📊 إحصائيات LOS:")
print(f"   - المتوسط: {df['los_days_capped'].mean():.2f}")
print(f"   - الانحراف: {df['los_days_capped'].std():.2f}")
print(f"   - الأقصى: {df['los_days_capped'].max():.2f}")

# ============================================================
# 2. تجهيز الميزات
# ============================================================
print("\n🔧 2. تجهيز الميزات...")

feature_cols = [
    'CHF', 'ARRHYTHMIA', 'VALVULAR', 'PULMONARY', 'PVD', 
    'HYPERTENSION', 'PARALYSIS', 'NEUROLOGICAL', 'HYPOTHYROID',
    'RENAL', 'LIVER', 'ULCER', 'AIDS',
    'lab_count', 'lab_mean', 'had_outpatient_labs'
]

available_features = [col for col in feature_cols if col in df.columns]
print(f"📊 عدد الميزات: {len(available_features)}")

X = df[available_features].fillna(0)
y = df['los_days_log']

print(f"📊 شكل البيانات: {X.shape}")

# ============================================================
# 3. تقسيم البيانات
# ============================================================
print("\n✂️ 3. تقسيم البيانات...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# حفظ القيم الأصلية للاختبار
y_test_original = df.loc[X_test.index, 'los_days_capped']

print(f"✅ تدريب: {len(X_train)} عينة")
print(f"✅ اختبار: {len(X_test)} عينة")

# ============================================================
# 4. تدريب النموذج
# ============================================================
print("\n🔮 4. تدريب نموذج XGBoost...")

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# 5. تقييم النموذج
# ============================================================
print("\n📊 5. تقييم النموذج...")

# تنبؤات
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# مقاييس
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print(f"✅ MAE: {mae:.3f} يوم (الهدف: 3.2)")
print(f"✅ RMSE: {rmse:.3f} يوم")
print(f"✅ R²: {r2:.3f}")

if mae <= 3.2:
    print("🎯 تم تحقيق الهدف!")
else:
    print("⚠️ قريب من الهدف")

# ============================================================
# 6. أهمية الميزات
# ============================================================
print("\n📈 6. أهم الميزات:")

feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ============================================================
# 7. حفظ النموذج
# ============================================================
print("\n💾 7. حفظ النموذج...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# حفظ النموذج
model_filename = f'los_model_{timestamp}.pkl'
model_path = f'{models_path}/{model_filename}'
joblib.dump(model, model_path)
print(f"✅ تم حفظ النموذج: {model_filename}")

# حفظ الميزات والمقاييس
features_filename = f'los_features_{timestamp}.json'
features_path = f'{models_path}/{features_filename}'

with open(features_path, 'w') as f:
    json.dump({
        'features': available_features,
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'max_los': max_los,
        'timestamp': timestamp,
        'feature_importance': feature_importance.head(10).to_dict('records')
    }, f, indent=2)

print(f"✅ تم حفظ الميزات: {features_filename}")

# ============================================================
# 8. التقرير النهائي
# ============================================================
print("\n" + "="*70)
print("📋 التقرير النهائي")
print("="*70)
print(f"""
🎯 نتائج التدريب:
   - MAE: {mae:.3f} يوم (الهدف: 3.2)
   - RMSE: {rmse:.3f} يوم
   - R²: {r2:.3f}
   - الحالة: {'✅ مُحقَق' if mae <= 3.2 else '⚠️ قريب'}

📊 تفاصيل النموذج:
   - عدد ميزات: {len(available_features)}
   - حجم التدريب: {len(X_train)} عينة
   - حجم الاختبار: {len(X_test)} عينة

📁 الملفات المحفوظة:
   - النموذج: {model_filename}
   - الميزات: {features_filename}
   - المسار: {models_path}

⏰ وقت التدريب: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

print("\n✅ تم الانتهاء من تدريب نموذج LOS!")
