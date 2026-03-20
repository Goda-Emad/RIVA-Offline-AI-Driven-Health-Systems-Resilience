from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import json, os, logging

logger = logging.getLogger("RIVA.Triage")
router = APIRouter(prefix="/triage", tags=["Triage"])

# ================================================================
# CONFIG
# ================================================================
class Settings(BaseSettings):
    model_path:    str = "ai-core/models/triage/model_int8.onnx"
    features_path: str = "ai-core/models/triage/features.json"
    imputer_path:  str = "ai-core/models/triage/imputer.pkl"
    scaler_path:   str = "ai-core/models/triage/scaler.pkl"
    conflicts_path:str = "business-intelligence/medical-content/drug_conflicts.json"
    samples_path:  str = "data/samples/triage_samples.json"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# ================================================================
# DEPENDENCY — TriageEngine بدل TriageClassifier
# ================================================================
@lru_cache()
def get_engine():
    from ai_core.local_inference.triage_engine import TriageEngine
    s = get_settings()
    return TriageEngine(
        model_path    = s.model_path,
        imputer_path  = s.imputer_path,
        scaler_path   = s.scaler_path,
        features_path = s.features_path,
        conflicts_path= s.conflicts_path
    )

# ================================================================
# CACHED DATA
# ================================================================
@lru_cache()
def load_samples() -> list:
    path = get_settings().samples_path
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("samples", [])

# ================================================================
# MODELS
# ================================================================
class TriageInput(BaseModel):
    age:               int   = Field(..., ge=1, le=120)
    gender:            str   = Field(..., pattern="^(male|female)$")
    pregnancies:       int   = Field(default=0, ge=0, le=20)
    glucose:           float = Field(default=0, ge=0, le=500)
    blood_pressure:    float = Field(default=0, ge=0, le=200)
    skin_thickness:    float = Field(default=0, ge=0, le=100)
    insulin:           float = Field(default=0, ge=0, le=900)
    bmi:               float = Field(default=0, ge=0, le=70)
    diabetes_pedigree: float = Field(default=0, ge=0, le=3)
    symptoms:          List[str] = Field(default=[])
    chronic_diseases:  List[str] = Field(default=[])
    current_medications: List[str] = Field(default=[])

    @model_validator(mode="after")
    def set_pregnancies_for_male(self) -> "TriageInput":
        if self.gender == "male":
            self.pregnancies = 0
        return self

class BatchTriageInput(BaseModel):
    patients: List[TriageInput] = Field(..., min_length=1)

# ================================================================
# ROUTES
# ================================================================
@router.post("/predict", response_model=dict)
async def predict_triage(
    data:   TriageInput,
    engine = Depends(get_engine)
):
    try:
        features = {
            "Pregnancies":              data.pregnancies,
            "Glucose":                  data.glucose,
            "BloodPressure":            data.blood_pressure,
            "SkinThickness":            data.skin_thickness,
            "Insulin":                  data.insulin,
            "BMI":                      data.bmi,
            "DiabetesPedigreeFunction": data.diabetes_pedigree,
            "Age":                      data.age
        }

        result = engine.decide(
            features         = features,
            symptoms         = data.symptoms,
            pain_level       = 0,
            chronic_diseases = data.chronic_diseases,
            medications      = data.current_medications
        )

        logger.info(
            f"Triage: age={data.age} | "
            f"level={result.triage_level} | "
            f"score={result.final_score} | "
            f"ms={result.inference_ms}"
        )

        return {
            "success":             True,
            "triage_level":        result.triage_level,
            "triage_label":        result.triage_label,
            "final_score":         result.final_score,
            "diabetes":            result.diabetes,
            "diabetes_confidence": result.diabetes_confidence,
            "diagnosis":           result.diagnosis,
            "recommended_action":  result.recommended_action,
            "specialty":           result.specialty,
            "drug_alerts":         result.drug_alerts,
            "explanation":         result.explanation,
            "scores":              result.scores,
            "inference_ms":        result.inference_ms
        }
    except Exception as e:
        logger.error(f"Triage error: {e}")
        raise HTTPException(status_code=500, detail=f"Triage error: {str(e)}")

@router.post("/predict-batch", response_model=dict)
async def predict_batch(
    data:   BatchTriageInput,
    engine = Depends(get_engine)
):
    try:
        results = []
        for patient in data.patients:
            r = await predict_triage(patient, engine)
            results.append(r)

        return {
            "success":         True,
            "total":           len(results),
            "emergency_count": sum(1 for r in results if r["triage_level"] == "طارئ"),
            "urgent_count":    sum(1 for r in results if r["triage_level"] == "عاجل"),
            "diabetes_count":  sum(1 for r in results if r["diabetes"]),
            "avg_score":       round(sum(r["final_score"] for r in results)/len(results), 3),
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
        path = get_settings().features_path
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Features not found")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
