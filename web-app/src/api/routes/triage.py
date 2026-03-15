from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Any
from functools import lru_cache
import numpy as np
import pandas as pd
import pickle
import logging

logger = logging.getLogger("RIVA.Triage")
router = APIRouter(prefix="/triage", tags=["Triage"])

# ================================================================
# CONFIG
# ================================================================
class Settings(BaseSettings):
    model_path:   str = "ai-core/models/triage/model_int8.onnx"
    features_path:str = "ai-core/models/triage/features.json"
    imputer_path: str = "ai-core/models/triage/imputer.pkl"
    scaler_path:  str = "ai-core/models/triage/scaler.pkl"
    samples_path: str = "data/samples/triage_samples.json"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# ================================================================
# DEPENDENCIES
# ================================================================
@lru_cache()
def get_classifier():
    from ai_core.local_inference.triage_classifier import TriageClassifier
    s = get_settings()
    return TriageClassifier(
        model_path    = s.model_path,
        features_path = s.features_path,
        imputer_path  = s.imputer_path,
        scaler_path   = s.scaler_path
    )

# ================================================================
# MODELS
# ================================================================
class TriageInput(BaseModel):
    age:                     int   = Field(..., ge=1,  le=120)
    gender:                  str   = Field(..., pattern="^(male|female)$")
    pregnancies:             int   = Field(default=0, ge=0, le=20)
    glucose:                 float = Field(default=0, ge=0, le=500)
    blood_pressure:          float = Field(default=0, ge=0, le=200)
    skin_thickness:          float = Field(default=0, ge=0, le=100)
    insulin:                 float = Field(default=0, ge=0, le=900)
    bmi:                     float = Field(default=0, ge=0, le=70)
    diabetes_pedigree:       float = Field(default=0, ge=0, le=3)
    symptoms:                List[str] = Field(default=[])
    chronic_diseases:        List[str] = Field(default=[])
    current_medications:     List[str] = Field(default=[])

    @model_validator(mode="after")
    def set_pregnancies_for_male(self) -> "TriageInput":
        if self.gender == "male":
            self.pregnancies = 0
        return self

class BatchTriageInput(BaseModel):
    patients: List[TriageInput] = Field(..., min_length=1)

# ================================================================
# CACHED DATA
# ================================================================
@lru_cache()
def load_samples() -> list:
    import json, os
    path = get_settings().samples_path
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", [])

# ================================================================
# ROUTES
# ================================================================
@router.post("/predict", response_model=dict)
async def predict_triage(
    data:       TriageInput,
    classifier = Depends(get_classifier)
):
    try:
        features = {
            "Pregnancies":             data.pregnancies,
            "Glucose":                 data.glucose,
            "BloodPressure":           data.blood_pressure,
            "SkinThickness":           data.skin_thickness,
            "Insulin":                 data.insulin,
            "BMI":                     data.bmi,
            "DiabetesPedigreeFunction":data.diabetes_pedigree,
            "Age":                     data.age
        }

        result = classifier.predict(features)

        # تحديد مستوى الفرز
        confidence = result.get("confidence", 0)
        diabetes   = result.get("prediction") == 1

        if diabetes and confidence > 0.8:
            triage_level = "عاجل"
            triage_label = 1
        elif diabetes and confidence > 0.6:
            triage_level = "غير عاجل"
            triage_label = 0
        else:
            triage_level = "غير عاجل"
            triage_label = 0

        # لو في أعراض خطيرة
        emergency_symptoms = ["ألم صدر","ضيق تنفس","فقدان وعي","تشنجات","نزيف"]
        if any(s in emergency_symptoms for s in data.symptoms):
            triage_level = "طارئ"
            triage_label = 2

        logger.info(f"Triage: age={data.age} | diabetes={diabetes} | level={triage_level}")

        return {
            "success":         True,
            "triage_level":    triage_level,
            "triage_label":    triage_label,
            "diabetes":        diabetes,
            "confidence":      round(confidence, 3),
            "diagnosis":       "سكري" if diabetes else "سليم",
            "recommended_action": {
                "طارئ":     "اتصل بالإسعاف فوراً",
                "عاجل":     "كشف طبيب خلال ساعات",
                "غير عاجل": "راحة + متابعة عيادة"
            }[triage_level],
            "features_used": features
        }
    except Exception as e:
        logger.error(f"Triage error: {e}")
        raise HTTPException(status_code=500, detail=f"Triage error: {str(e)}")

@router.post("/predict-batch", response_model=dict)
async def predict_batch(
    data:       BatchTriageInput,
    classifier = Depends(get_classifier)
):
    try:
        results = []
        for patient in data.patients:
            result = await predict_triage(patient, classifier)
            results.append(result)

        emergency_count = sum(1 for r in results if r["triage_level"] == "طارئ")
        diabetes_count  = sum(1 for r in results if r["diabetes"])

        return {
            "success":         True,
            "total":           len(results),
            "emergency_count": emergency_count,
            "diabetes_count":  diabetes_count,
            "results":         results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")

@router.get("/samples", response_model=dict)
async def get_samples():
    try:
        samples = load_samples()
        if not samples:
            raise HTTPException(status_code=404, detail="Samples not found")
        return {"samples": samples[:5], "total": len(samples)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features", response_model=dict)
async def get_features():
    try:
        import json, os
        path = get_settings().features_path
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Features file not found")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
