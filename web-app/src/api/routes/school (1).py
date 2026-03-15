from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import json, os

router = APIRouter()

# ================================================================
# CONFIG — بدل hardcoded paths
# ================================================================
class Settings(BaseSettings):
    who_standards_path: str = "data/raw/who_growth/who_growth_standards.json"
    cluster_centers_path: str = "ai-core/models/school/cluster_centers.json"
    school_samples_path: str = "data/samples/school_samples.json"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# ================================================================
# DEPENDENCY — بدل global instance
# ================================================================
@lru_cache()
def get_analyzer():
    from ai_core.local_inference.school_health import SchoolHealthAnalyzer
    settings = get_settings()
    if not os.path.exists(settings.who_standards_path):
        raise RuntimeError(f"WHO standards not found: {settings.who_standards_path}")
    if not os.path.exists(settings.cluster_centers_path):
        raise RuntimeError(f"Cluster centers not found: {settings.cluster_centers_path}")
    return SchoolHealthAnalyzer(
        standards_path = settings.who_standards_path,
        clusters_path  = settings.cluster_centers_path
    )

# ================================================================
# MODELS
# ================================================================
class StudentInput(BaseModel):
    name:       Optional[str] = Field(default="unknown")
    age_months: int           = Field(..., ge=60, le=228)
    gender:     str           = Field(..., pattern="^(boys|girls)$")
    height_cm:  float         = Field(..., gt=50, lt=250)
    weight_kg:  float         = Field(..., gt=5,  lt=200)

class ClassInput(BaseModel):
    school_name: Optional[str] = "unknown"
    class_name:  Optional[str] = "unknown"
    students:    List[StudentInput]

# ================================================================
# ROUTES
# ================================================================
@router.post("/analyze-student")
def analyze_student(
    data:     StudentInput,
    analyzer = Depends(get_analyzer)
):
    try:
        result = analyzer.analyze(
            age_months = data.age_months,
            gender     = data.gender,
            height_cm  = data.height_cm,
            weight_kg  = data.weight_kg
        )
        result["name"] = data.name
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-class")
def analyze_class(
    data:     ClassInput,
    analyzer = Depends(get_analyzer)
):
    try:
        students = [s.dict() for s in data.students]
        report   = analyzer.analyze_class(students)
        return {
            "success": True,
            "school":  data.school_name,
            "class":   data.class_name,
            "report":  report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/who-standards")
def get_who_standards(settings: Settings = Depends(get_settings)):
    try:
        with open(settings.who_standards_path, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "source":     data.get("source"),
            "age_range":  data.get("age_range"),
            "categories": data.get("z_score_categories")
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="WHO standards file not found")

@router.get("/samples")
def get_samples(settings: Settings = Depends(get_settings)):
    try:
        with open(settings.school_samples_path, encoding="utf-8") as f:
            samples = json.load(f)
        return {"samples": samples[:10], "total": len(samples)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Samples file not found")
