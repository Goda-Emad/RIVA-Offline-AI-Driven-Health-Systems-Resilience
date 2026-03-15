import json
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt
import os

WHO_PATH     = "data/raw/who_growth/who_growth_standards.json"
MODEL_PATH   = "ai-core/models/school/model_int8.onnx"
CLUSTER_PATH = "ai-core/models/school/cluster_centers.json"
FEATURES     = ["age_months", "gender", "bmi", "z_score"]
N_SAMPLES    = 2000
N_CLUSTERS   = 3
RANDOM_STATE = 42

print("STEP 1: Loading WHO data...")
with open(WHO_PATH, encoding="utf-8") as f:
    who_data = json.load(f)
print("OK: WHO data loaded")

print("STEP 2: Generating training data...")
def generate_training_data(n=N_SAMPLES):
    import random
    rows = []
    for _ in range(n):
        gender = random.choice(["boys","girls"])
        table  = who_data[f"bmi_{gender}"]
        row    = random.choice(table)
        bmi    = np.random.normal(row["M"], row["M"]*0.15)
        bmi    = max(10, min(45, bmi))
        L,M,S  = row["L"],row["M"],row["S"]
        z      = (((bmi/M)**L)-1)/(L*S) if L!=0 else np.log(bmi/M)/S
        label  = 2 if z>2 else 1 if z<-2 else 0
        rows.append({"age_months":row["months"],"gender":1 if gender=="boys" else 0,
                     "bmi":round(bmi,2),"z_score":round(z,2),"label":label})
    return pd.DataFrame(rows)

df = generate_training_data(N_SAMPLES)
print(f"OK: {len(df)} samples")

print("STEP 3: Balancing...")
df0 = resample(df[df.label==0],n_samples=500,random_state=RANDOM_STATE)
df1 = resample(df[df.label==1],n_samples=500,random_state=RANDOM_STATE)
df2 = resample(df[df.label==2],n_samples=500,random_state=RANDOM_STATE)
df_bal = pd.concat([df0,df1,df2]).reset_index(drop=True)

X = df_bal[FEATURES].values.astype(np.float32)
y = df_bal["label"].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE,stratify=y)

print("STEP 4: Training XGBoost...")
model = XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,
                      random_state=RANDOM_STATE,eval_metric="mlogloss")
model.fit(X_train,y_train,eval_set=[(X_test,y_test)],verbose=False)
y_pred = model.predict(X_test)
acc = round(accuracy_score(y_test,y_pred)*100,1)
print(f"Accuracy: {acc}%")
print(classification_report(y_test,y_pred,target_names=["normal","thin","overweight"]))

print("STEP 5: K-Means clustering...")
kmeans = KMeans(n_clusters=N_CLUSTERS,random_state=RANDOM_STATE,n_init=10)
kmeans.fit(X)
centers = kmeans.cluster_centers_
labels_map = {}
for i,c in enumerate(centers):
    z = c[3]
    if z>1: labels_map[i]="overweight"
    elif z<-1: labels_map[i]="thin"
    else: labels_map[i]="normal"
cluster_data = {"n_clusters":N_CLUSTERS,"features":FEATURES,"accuracy":acc,
    "centers":[{"cluster_id":i,"label":labels_map[i],
                "age_months":round(float(centers[i][0]),2),
                "gender":round(float(centers[i][1]),2),
                "bmi":round(float(centers[i][2]),2),
                "z_score":round(float(centers[i][3]),2)} for i in range(N_CLUSTERS)]}
os.makedirs(os.path.dirname(CLUSTER_PATH),exist_ok=True)
with open(CLUSTER_PATH,"w",encoding="utf-8") as f:
    json.dump(cluster_data,f,ensure_ascii=False,indent=2)

print("STEP 6: Exporting ONNX...")
initial_type = [("float_input",FloatTensorType([None,len(FEATURES)]))]
onnx_model   = convert_xgboost(model,initial_types=initial_type)
os.makedirs(os.path.dirname(MODEL_PATH),exist_ok=True)
with open(MODEL_PATH,"wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"OK: {MODEL_PATH} — {round(os.path.getsize(MODEL_PATH)/1024,1)} KB")

print("STEP 7: Testing ONNX...")
sess = rt.InferenceSession(MODEL_PATH)
inp  = sess.get_inputs()[0].name
for feat,exp in [([61,1,15.2,-0.05],"normal"),([84,0,24.3,3.23],"overweight"),([120,1,11.8,-4.2],"thin")]:
    pred = sess.run(None,{inp:np.array([feat],dtype=np.float32)})[0][0]
    labels = ["normal","thin","overweight"]
    print(f"  {'OK' if labels[pred]==exp else 'FAIL'}: {feat} -> {labels[pred]}")

print(f"Done — Accuracy: {acc}%")
