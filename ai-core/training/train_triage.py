import json, os, pickle
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt

# ================================================================
# CONFIG
# ================================================================
DATA_PATH    = "data/raw/pima/diabetes.csv"
MODEL_PATH   = "ai-core/models/triage/model_int8.onnx"
FEATURES_PATH= "ai-core/models/triage/features.json"
IMPUTER_PATH = "ai-core/models/triage/imputer.pkl"
SCALER_PATH  = "ai-core/models/triage/scaler.pkl"

FEATURES  = ["Pregnancies","Glucose","BloodPressure",
             "SkinThickness","Insulin","BMI",
             "DiabetesPedigreeFunction","Age"]
ZERO_COLS = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
RANDOM_STATE = 42

# ================================================================
# STEP 1: تحميل البيانات
# ================================================================
print("STEP 1: Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"OK: {df.shape[0]} rows x {df.shape[1]} cols")

# ================================================================
# STEP 2: تنظيف + KNNImputer
# ================================================================
print("
STEP 2: Cleaning + KNNImputer...")
df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)
print(f"Missing values: {df[ZERO_COLS].isnull().sum().sum()}")

imputer = KNNImputer(n_neighbors=5)
df[FEATURES] = imputer.fit_transform(df[FEATURES])
print(f"After imputation: {df.isnull().sum().sum()} missing")

# ================================================================
# STEP 3: Split قبل SMOTE
# ================================================================
print("
STEP 3: Splitting data...")
X = df[FEATURES].values.astype(np.float32)
y = df["Outcome"].values

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train_raw)} | Test: {len(X_test)}")
print(f"Test — No Diabetes: {np.bincount(y_test)[0]} | Diabetes: {np.bincount(y_test)[1]}")

# ================================================================
# STEP 4: SMOTE على Train فقط
# ================================================================
print("
STEP 4: SMOTE on Train only...")
smote = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = smote.fit_resample(X_train_raw, y_train_raw)
n_synthetic  = len(y_res) - len(y_train_raw)
print(f"Before: {np.bincount(y_train_raw)}")
print(f"After:  {np.bincount(y_res)}")
print(f"Synthetic samples added: {n_synthetic}")

# ================================================================
# STEP 5: Scaling
# ================================================================
print("
STEP 5: Scaling...")
scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_res).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

# ================================================================
# STEP 6: تدريب XGBoost
# ================================================================
print("
STEP 6: Training XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=1,
    random_state=RANDOM_STATE,
    eval_metric="aucpr",
    n_jobs=-1
)
model.fit(X_train, y_res, verbose=False)

y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:,1]

acc = round(accuracy_score(y_test, y_pred)*100, 1)
f1  = round(f1_score(y_test, y_pred)*100, 1)
auc = round(roc_auc_score(y_test, y_proba)*100, 1)

print(f"Accuracy: {acc}% | F1: {f1}% | ROC-AUC: {auc}%")
print(classification_report(y_test, y_pred,
      target_names=["No Diabetes","Diabetes"]))

# ================================================================
# STEP 7: Cross Validation
# ================================================================
print("STEP 7: Cross Validation (5-Fold)...")
model_cv = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=1, random_state=RANDOM_STATE,
    eval_metric="aucpr", n_jobs=-1
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
f1_cv  = cross_val_score(model_cv, X_train, y_res, cv=cv, scoring="f1")
auc_cv = cross_val_score(model_cv, X_train, y_res, cv=cv, scoring="roc_auc")
print(f"CV F1:  {round(f1_cv.mean()*100,1)}% +/- {round(f1_cv.std()*100,1)}%")
print(f"CV AUC: {round(auc_cv.mean()*100,1)}% +/- {round(auc_cv.std()*100,1)}%")

# ================================================================
# STEP 8: تصدير ONNX
# ================================================================
print("
STEP 8: Exporting ONNX...")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
initial_type = [("float_input", FloatTensorType([None, len(FEATURES)]))]
onnx_model   = convert_xgboost(model, initial_types=initial_type)

with open(MODEL_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"OK: {MODEL_PATH} — {round(os.path.getsize(MODEL_PATH)/1024,1)} KB")

# ================================================================
# STEP 9: حفظ features.json + imputer + scaler
# ================================================================
feat_imp = sorted(zip(FEATURES, model.feature_importances_),
                  key=lambda x: x[1], reverse=True)

features_data = {
    "features":    FEATURES,
    "target":      "Outcome",
    "n_features":  len(FEATURES),
    "classes":     ["No Diabetes", "Diabetes"],
    "metrics": {
        "accuracy":  acc,
        "f1_score":  f1,
        "roc_auc":   auc,
        "cv_f1":     round(f1_cv.mean()*100,1),
        "cv_auc":    round(auc_cv.mean()*100,1)
    },
    "top_features":  [f[0] for f in feat_imp[:5]],
    "imputation":    "KNNImputer (n_neighbors=5)",
    "smote":         f"SMOTE on Train only — {n_synthetic} synthetic samples",
    "scaler":        {
        "mean": scaler.mean_.tolist(),
        "std":  scaler.scale_.tolist()
    }
}

with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(features_data, f, ensure_ascii=False, indent=2)

with open(IMPUTER_PATH, "wb") as f:
    pickle.dump(imputer, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"OK: features.json + imputer.pkl + scaler.pkl")

# ================================================================
# STEP 10: اختبار ONNX
# ================================================================
print("
STEP 10: Testing ONNX...")
sess = rt.InferenceSession(MODEL_PATH)
inp  = sess.get_inputs()[0].name
print(f"Input name: {inp}")

test_cases = [
    ([6, 148, 72, 35, 0,  33.6, 0.627, 50], "Diabetes"),
    ([1,  85, 66, 29, 0,  26.6, 0.351, 31], "No Diabetes"),
    ([8, 183, 64,  0, 0,  23.3, 0.672, 32], "Diabetes"),
    ([1,  89, 66, 23, 94, 28.1, 0.167, 21], "No Diabetes"),
]

correct = 0
for feat, expected in test_cases:
    feat_df  = pd.DataFrame([feat], columns=FEATURES)
    feat_imp = imputer.transform(feat_df)
    feat_s   = scaler.transform(feat_imp).astype(np.float32)
    pred     = sess.run(None, {inp: feat_s})[0][0]
    res      = "Diabetes" if pred==1 else "No Diabetes"
    ok       = "OK" if res==expected else "FAIL"
    if ok == "OK": correct += 1
    print(f"  {ok}: {res} (expected: {expected})")

print(f"
Test cases: {correct}/{len(test_cases)}")
print(f"\nDone — Accuracy: {acc}% | F1: {f1}% | ROC-AUC: {auc}%")
