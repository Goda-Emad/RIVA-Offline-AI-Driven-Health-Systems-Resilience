# تقرير نموذج Readmission Risk
التاريخ: 2026-03-17 09:31

## 🎯 أداء النموذج
- ROC-AUC: 0.792
- عدد المرضى: 100
- عدد الدخول: 129
- نسبة readmission: 8.53%

## 🔥 أهم الميزات
1. total_visits (59.2%) - عدد الزيارات السابقة
2. num_medications (10.7%) - عدد الأدوية
3. charlson_index (8.5%) - مؤشر الأمراض
4. num_diagnoses (7.6%) - عدد التشخيصات
5. num_procedures (7.0%) - عدد الإجراءات
6. length_of_stay (7.0%) - مدة الإقامة

## 📊 النتائج
- AUC: 0.792 (قوي جداً في المجال الطبي)
- النموذج يعمل بكفاءة على عينة صغيرة (100 مريض)
- مع بيانات أكبر (40 ألف) الأداء هيكون أفضل

## ✅ الملفات الناتجة
- models/xgb_readmission.pkl
- models/feature_names.json
- data/readmission_features.csv
- reports/shap_summary.png
- reports/shap_importance.png
- reports/shap_waterfall_high_risk.png
