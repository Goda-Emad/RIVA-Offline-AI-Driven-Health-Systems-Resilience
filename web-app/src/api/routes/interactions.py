from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import json, os, csv

router = APIRouter(prefix="/interactions", tags=["Drug Interactions"])

# ================================================================
# CONFIG
# ================================================================
class Settings(BaseSettings):
    csv_path:          str = "data/raw/drug_bank/drug_interactions.csv"
    medicines_path:    str = "data/raw/drug_bank/medicines_list.csv"
    conflicts_path:    str = "business-intelligence/medical-content/drug_conflicts.json"
    interactions_path: str = "ai-core/data/interactions.json"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# ================================================================
# DEPENDENCY — بدون lru_cache على الـ function
# ================================================================
def get_checker():
    from ai_core.local_inference.drug_interaction import DrugInteractionChecker
    settings = get_settings()
    return DrugInteractionChecker(
        csv_path  = settings.csv_path,
        data_path = settings.interactions_path
    )

# ================================================================
# CACHED DATA LOADERS
# ================================================================
@lru_cache()
def load_medicines() -> list:
    settings = get_settings()
    if not os.path.exists(settings.medicines_path):
        return []
    with open(settings.medicines_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

@lru_cache()
def load_conflicts() -> dict:
    settings = get_settings()
    if not os.path.exists(settings.conflicts_path):
        return {"conflicts": []}
    with open(settings.conflicts_path, encoding="utf-8") as f:
        return json.load(f)

# ================================================================
# MODELS
# ================================================================
class InteractionCheckInput(BaseModel):
    new_drug:      str       = Field(..., min_length=2)
    current_drugs: List[str] = Field(default=[])

    @validator("current_drugs")
    def no_duplicate(cls, v, values):
        new_drug = values.get("new_drug", "").lower().strip()
        v = [d for d in v if d.lower().strip() != new_drug]
        return v

class BulkCheckInput(BaseModel):
    medications: List[str] = Field(..., min_items=1)

# ================================================================
# ROUTES
# ================================================================
@router.post("/check")
async def check_interaction(
    data:    InteractionCheckInput,
    checker = Depends(get_checker)
):
    try:
        result = checker.check(data.new_drug, data.current_drugs)
        return {
            "success":     True,
            "new_drug":    data.new_drug,
            "is_safe":     result["is_safe"],
            "alerts":      result["alerts"],
            "alert_count": result["alert_count"],
            "source_mode": result["source_mode"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-bulk")
async def check_bulk(
    data:    BulkCheckInput,
    checker = Depends(get_checker)
):
    try:
        # تحسين O(n) بدل O(n²)
        drugs_set  = set(data.medications)
        all_alerts = []

        for drug in data.medications:
            others = list(drugs_set - {drug})
            result = checker.check_offline(drug, others)
            all_alerts.extend(result)

        # شيل التكرار
        seen, unique = set(), []
        for a in all_alerts:
            key = tuple(sorted([a["drug_a"], a["drug_b"]]))
            if key not in seen:
                seen.add(key)
                unique.append(a)

        return {
            "success":     True,
            "medications": data.medications,
            "is_safe":     not any(a["severity_code"]=="high" for a in unique),
            "alerts":      unique,
            "alert_count": len(unique)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/medicines")
async def get_medicines():
    try:
        medicines = load_medicines()
        if not medicines:
            raise HTTPException(status_code=404, detail="Medicines file not found")
        return {"medicines": medicines, "total": len(medicines)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conflicts")
async def get_conflicts():
    try:
        data = load_conflicts()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conflicts/high")
async def get_high_conflicts():
    try:
        data    = load_conflicts()
        high    = [c for c in data.get("conflicts", []) if c.get("severity") == "high"]
        return {"conflicts": high, "total": len(high)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
