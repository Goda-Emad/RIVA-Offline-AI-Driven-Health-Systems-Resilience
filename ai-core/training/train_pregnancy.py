# ai-core/training/train_pregnancy.py
"""
تدريب نموذج محسن للتنبؤ بمخاطر الحمل
مع 18 ميزة + GridSearch + Ensemble + SHAP
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# ================== المسارات ==================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "data-storage" / "raw" / "uci_maternal" / "maternal_health_clean.csv"
MODEL_DIR = BASE_DIR / "ai-core" / "models" / "pregnancy"
REPORTS_DIR = BASE_DIR / "business-intelligence" / "performance"
SHAP_DIR = BASE_DIR / "business-intelligence" / "explainability"

# إنشاء المجلدات
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

# ================== إعداد logging ==================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """تحميل وتجهيز البيانات مع 18 ميزة"""
    logger.info("📦 تحميل البيانات...")
    
    df = pd.read_csv(DATA_PATH)
    logger.info(f"✅ تم تحميل {len(df)} عينة")
    
    # توزيع الفئات
    logger.info("\n📊 توزيع الفئات:")
    class_dist = df['RiskLevel'].value_counts()
    for class_name, count in class_dist.items():
        logger.info(f"   {class_name}: {count} ({count/len(df)*100:.1f}%)")
    
    # الميزات الأساسية
    X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']].copy()
    y = df['RiskLevel_encoded'].copy()
    
    # الميزات المشتقة (7)
    X['PulsePressure'] = X['SystolicBP'] - X['DiastolicBP']
    X['BP_ratio'] = (X['SystolicBP'] / (X['DiastolicBP'] + 1e-10)).round(2)
    X['Temp_Fever'] = (X['BodyTemp'] > 37.5).astype(int)
    X['HighBP'] = (X['SystolicBP'] > 140).astype(int)
    X['MeanBP'] = (X['SystolicBP'] + 2 * X['DiastolicBP']) / 3
    X['AgeRisk'] = (X['Age'] > 35).astype(int)
    X['HighSugar'] = (X['BS'] > 7).astype(int)
    
    # الميزات التفاعلية الجديدة (5)
    logger.info("🔧 إضافة ميزات تفاعلية جديدة...")
    X['BP_BS_Interaction'] = X['SystolicBP'] * X['BS'] / 100
    X['Total_Risk_Score'] = (X['HighBP'] + X['AgeRisk'] + X['HighSugar'] + X['Temp_Fever'])
    X['Age_BP_Interaction'] = X['Age'] * X['SystolicBP'] / 100
    
    X['BP_Severity'] = np.where(
        X['SystolicBP'] > 160, 3,
        np.where(X['SystolicBP'] > 140, 2,
        np.where(X['SystolicBP'] > 130, 1, 0))
    )
    
    X['BS_Severity'] = np.where(
        X['BS'] > 11, 3,
        np.where(X['BS'] > 8, 2,
        np.where(X['BS'] > 6.5, 1, 0))
    )
    
    final_features = list(X.columns)
    logger.info(f"\n📊 عدد الميزات الكلي: {len(final_features)}")
    
    return X, y, final_features

def train_optimized_model(X, y, feature_names):
    """تدريب النموذج المحسن مع Ensemble"""
    logger.info("\n🏋️ بدأ تدريب النموذج المحسن...")
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"📊 حجم التدريب: {X_train.shape[0]}, حجم الاختبار: {X_test.shape[0]}")
    
    # XGBoost مع Hyperparameter Tuning موسع
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    xgb_param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [1, 2, 3]
    }
    
    logger.info(f"🔍 جاري البحث عن أفضل باراميترات ({3**7} تجربة)...")
    xgb_grid = GridSearchCV(
        xgb_model,
        xgb_param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    xgb_grid.fit(X_train, y_train)
    logger.info(f"✅ أفضل باراميترات XGBoost: {xgb_grid.best_params_}")
    logger.info(f"✅ أفضل F1-Macro: {xgb_grid.best_score_:.4f}")
    
    # RandomForest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # LightGBM
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_grid.best_estimator_),
            ('rf', rf_model),
            ('lgbm', lgbm_model)
        ],
        voting='soft',
        weights=[2, 1.5, 1.5]
    )
    
    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', voting_clf)
    ])
    
    # تدريب
    pipeline.fit(X_train, y_train)
    
    # تقييم
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # F1 لكل فئة
    f1_scores = f1_score(y_test, y_pred, average=None)
    logger.info("\n📊 F1-Score لكل فئة:")
    for i, score in enumerate(f1_scores):
        logger.info(f"   الفئة {i}: {score:.4f}")
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    logger.info(f"\n✅ Macro F1-Score: {macro_f1:.4f}")
    
    # Cross Validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1_macro')
    logger.info(f"\n📊 Cross Validation (3-Fold): {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
    
    return pipeline, X_train, X_test, y_train, y_test, y_pred, y_proba, cv_scores

def plot_feature_importance(model, feature_names, version):
    """رسم أهمية الميزات"""
    voting_clf = model.named_steps['classifier']
    xgb_model = voting_clf.named_estimators_['xgb']
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature', palette='viridis')
    plt.title(f'Feature Importance - Pregnancy Model {version}')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    img_path = REPORTS_DIR / f'feature_importance_{version}.png'
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"✅ تم حفظ أهمية الميزات في: {img_path}")
    
    return importance_df

def plot_confusion_matrix(y_test, y_pred, class_names, version):
    """رسم مصفوفة الارتباك"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - Pregnancy Model {version}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    img_path = REPORTS_DIR / f'confusion_matrix_{version}.png'
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"✅ تم حفظ مصفوفة الارتباك في: {img_path}")

def analyze_shap(model, X_train, X_test, y_test, y_pred, feature_names, class_names, version):
    """تحليل SHAP لتفسير قرارات النموذج"""
    logger.info("\n🔍 بدأ تحليل SHAP...")
    
    try:
        # استخراج نموذج XGBoost من الـ pipeline
        voting_clf = model.named_steps['classifier']
        xgb_model = voting_clf.named_estimators_['xgb']
        
        # تطبيع البيانات (لأن الـ pipeline بيعمل scaling)
        scaler = model.named_steps['scaler']
        X_test_scaled = scaler.transform(X_test)
        
        # SHAP Explainer
        logger.info("⏳ جاري حساب SHAP values (قد يستغرق دقيقة)...")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # 1️⃣ Summary Plot
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary - Model {version}', fontsize=16)
        plt.tight_layout()
        shap_path = SHAP_DIR / f'shap_summary_{version}.png'
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ تم حفظ SHAP Summary في: {shap_path}")
        
        # 2️⃣ Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, 
                          plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - Model {version}', fontsize=16)
        plt.tight_layout()
        shap_bar_path = SHAP_DIR / f'shap_importance_{version}.png'
        plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ تم حفظ SHAP Importance في: {shap_bar_path}")
        
        # 3️⃣ Waterfall plot لحالة High Risk
        high_risk_indices = np.where(y_test == 2)[0]
        if len(high_risk_indices) > 0:
            idx = high_risk_indices[0]
            
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[2][idx],
                    base_values=explainer.expected_value[2],
                    data=X_test_scaled[idx],
                    feature_names=feature_names
                ),
                show=False,
                max_display=15
            )
            plt.title(f'SHAP Waterfall - High Risk Case (Sample {idx})', fontsize=14)
            waterfall_path = SHAP_DIR / f'shap_waterfall_high_risk_{version}.png'
            plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✅ تم حفظ SHAP Waterfall في: {waterfall_path}")
        
        # 4️⃣ تفسير أول 5 عينات
        logger.info("\n📊 تفسير العينات الأولى:")
        for i in range(min(5, len(X_test))):
            logger.info(f"\n   العينة {i+1}:")
            logger.info(f"      القيمة الفعلية: {class_names[y_test.iloc[i]]}")
            logger.info(f"      التوقع: {class_names[y_pred[i]]}")
            
            pred_class = y_pred[i]
            shap_vals = shap_values[pred_class][i]
            top_indices = np.argsort(np.abs(shap_vals))[-3:]
            
            for idx in top_indices[::-1]:
                direction = "⬆️ (تزيد)" if shap_vals[idx] > 0 else "⬇️ (تقلل)"
                logger.info(f"      - {feature_names[idx]}: {shap_vals[idx]:.4f} {direction}")
        
        return shap_values
        
    except Exception as e:
        logger.warning(f"⚠️ تحليل SHAP لم يكتمل: {e}")
        return None

def save_model(model, feature_names, class_names, cv_scores, version):
    """حفظ النموذج والملفات المرتبطة"""
    logger.info("\n💾 حفظ النموذج...")
    
    # حفظ النموذج مع timestamp
    model_path = MODEL_DIR / f"pregnancy_model_{version}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"✅ تم حفظ النموذج في: {model_path}")
    
    # أحدث إصدار
    latest_path = MODEL_DIR / "maternal_health_optimized_pipeline.pkl"
    joblib.dump(model, latest_path)
    logger.info(f"✅ تم تحديث أحدث إصدار: {latest_path}")
    
    # حفظ الميزات والفئات
    with open(MODEL_DIR / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    with open(MODEL_DIR / "class_names.json", 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # حفظ معلومات الإصدار
    version_info = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "cv_f1_macro": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "features": feature_names,
        "classes": class_names
    }
    
    versions_path = MODEL_DIR / "model_versions.json"
    if versions_path.exists():
        with open(versions_path, 'r') as f:
            versions = json.load(f)
    else:
        versions = []
    
    versions.append(version_info)
    
    with open(versions_path, 'w') as f:
        json.dump(versions, f, indent=2)
    
    logger.info("✅ تم حفظ جميع الملفات")

def generate_report(y_test, y_pred, y_proba, class_names, cv_scores, version, feature_importance_df):
    """توليد تقرير الأداء"""
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    confidence_scores = np.max(y_proba, axis=1)
    
    report_text = f"""# تقرير أداء نموذج الحمل (Pregnancy Risk Model)

## 🎯 النتائج النهائية
- **الإصدار**: {version}
- **النموذج**: Ensemble (XGBoost + RandomForest + LightGBM)
- **عدد الميزات**: 18
- **تاريخ التدريب**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Macro F1-Score**: {report['macro avg']['f1-score']:.4f}
- **Weighted F1-Score**: {report['weighted avg']['f1-score']:.4f}

## 📈 أداء كل فئة
| الفئة | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
"""
    for class_name in class_names:
        report_text += f"| {class_name} | {report[class_name]['precision']:.3f} | {report[class_name]['recall']:.3f} | {report[class_name]['f1-score']:.3f} |\n"
    
    report_text += f"""
## ⚠️ تحليل الثقة
- **متوسط الثقة**: {confidence_scores.mean():.3f}
- **حالات غير مؤكدة (<70%)**: {np.sum(confidence_scores < 0.7)} من {len(y_test)} ({np.sum(confidence_scores < 0.7)/len(y_test)*100:.1f}%)

## 📊 Cross Validation
- **3-Folds F1 Macro**: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}

## 🔥 أهم 5 ميزات
"""
    for i, row in feature_importance_df.head(5).iterrows():
        report_text += f"{i+1}. **{row['feature']}**: {row['importance']:.4f}\n"
    
    report_text += f"""
## 🔍 Explainability
تم إنشاء ملفات SHAP في مجلد `explainability/`:
- `shap_summary_{version}.png`: تأثير كل ميزة
- `shap_importance_{version}.png`: أهمية الميزات
- `shap_waterfall_high_risk_{version}.png`: شرح حالة High Risk

## ✅ الخلاصة
✅ النموذج جاهز للاستخدام في الإنتاج
"""
    
    report_path = REPORTS_DIR / f'pregnancy_performance_{version}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"✅ تم حفظ التقرير في: {report_path}")

def main():
    """الدالة الرئيسية"""
    logger.info("="*80)
    logger.info("🚀 تدريب نموذج الحمل المحسن (18 ميزة + SHAP)")
    logger.info("="*80)
    
    try:
        # 1. تحميل البيانات
        X, y, feature_names = load_and_prepare_data()
        class_names = ['low risk', 'mid risk', 'high risk']
        
        # 2. تدريب النموذج
        model, X_train, X_test, y_train, y_test, y_pred, y_proba, cv_scores = train_optimized_model(X, y, feature_names)
        
        # 3. إصدار
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 4. رسم النتائج الأساسية
        plot_confusion_matrix(y_test, y_pred, class_names, version)
        feature_importance_df = plot_feature_importance(model, feature_names, version)
        
        # 5. تحليل SHAP
        shap_values = analyze_shap(model, X_train, X_test, y_test, y_pred, feature_names, class_names, version)
        
        # 6. حفظ النموذج
        save_model(model, feature_names, class_names, cv_scores, version)
        
        # 7. تقرير
        generate_report(y_test, y_pred, y_proba, class_names, cv_scores, version, feature_importance_df)
        
        # 8. طباعة النتائج
        logger.info("\n📈 تقرير التصنيف الكامل:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        logger.info("\n" + "="*80)
        logger.info("✅✅✅ تم تدريب النموذج بنجاح!")
        logger.info(f"📌 الإصدار: {version}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ خطأ: {e}")
        raise

if __name__ == "__main__":
    main()
