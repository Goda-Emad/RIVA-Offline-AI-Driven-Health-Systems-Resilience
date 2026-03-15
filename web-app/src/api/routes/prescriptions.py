from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Any
from functools import lru_cache
import json, os, csv, logging

logger = logging.getLogger("RIVA.Prescriptions")
router = APIRouter(prefix="/prescriptions", tags=["Prescriptions"])

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
# DEPENDENCIES
# ================================================================
@lru_cache()
def get_checker():
    from ai_core.local_inference.drug_interaction import DrugInteractionChecker
    settings = get_settings()
    return DrugInteractionChecker(
        csv_path  = settings.csv_path,
        data_path = settings.interactions_path
    )

@lru_cache()
def get_generator():
    from ai_core.local_inference.prescription_gen import PrescriptionGenerator
    return PrescriptionGenerator()

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
class Medication(BaseModel):
    name:  str = Field(..., min_length=2)
    dose:  str
    times: str
    days:  int = Field(..., gt=0, le=365)

class PrescriptionInput(BaseModel):
    patient_id:    str
    doctor_id:     str
    diagnosis:     str
    medications:   List[Medication] = Field(..., min_length=1)
    current_drugs: List[str]        = Field(default=[])
    notes:         str              = Field(default="")

    @model_validator(mode="after")
    def no_duplicate_drugs(self) -> "PrescriptionInput":
        med_names = {m.name.lower().strip() for m in self.medications}
        self.current_drugs = [
            d for d in self.current_drugs
            if d.lower().strip() not in med_names
        ]
        return self

class InteractionCheckInput(BaseModel):
    new_drug:      str       = Field(..., min_length=2)
    current_drugs: List[str] = Field(default=[])

    @field_validator("current_drugs", mode="after")
    @classmethod
    def no_duplicate(cls, v: List[str], info: Any) -> List[str]:
        new_drug = (info.data.get("new_drug") or "").lower().strip()
        return [d for d in v if d.lower().strip() != new_drug]

# ================================================================
# ROUTES
# ================================================================
@router.post("/create", response_model=dict)
async def create_prescription(
    data:      PrescriptionInput,
    checker  = Depends(get_checker),
    generator = Depends(get_generator)
):
    try:
        # الإصلاح: فحص الأدوية الجديدة مع بعضها + مع القديمة
        drugs_set = (
            {d.lower().strip() for d in data.current_drugs} |
            {m.name.lower().strip() for m in data.medications}
        )

        all_alerts = []
        for med in data.medications:
            # شيل الدواء نفسه من الفحص
            others = list(drugs_set - {med.name.lower().strip()})
            result = checker.check(med.name, others)
            if "alerts" in result:
                all_alerts.extend(result["alerts"])

        # شيل التكرار
        seen, unique = set(), []
        for a in all_alerts:
            key = tuple(sorted([a["drug_a"], a["drug_b"]]))
            if key not in seen:
                seen.add(key)
                unique.append(a)

        # توليد الروشتة
        try:
            rx = generator.generate(
                patient_id  = data.patient_id,
                doctor_id   = data.doctor_id,
                diagnosis   = data.diagnosis,
                medications = [m.model_dump() for m in data.medications],
                notes       = data.notes
            )
            qr = generator.to_qr_payload(rx)
        except Exception as gen_err:
            logger.error(f"Generator error: {gen_err}")
            raise HTTPException(
                status_code=500,
                detail=f"Prescription generation error: {str(gen_err)}"
            )

        has_high = any(a.get("severity_code") == "high" for a in unique)
        logger.info(f"Prescription created | patient={data.patient_id} | safe={not has_high}")

        return {
            "success":      True,
            "prescription": rx,
            "drug_alerts":  unique,
            "alert_count":  len(unique),
            "is_safe":      not has_high,
            "qr_payload":   qr
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prescription error: {e}")
        raise HTTPException(status_code=500, detail=f"Prescription error: {str(e)}")

@router.post("/check-interactions", response_model=dict)
async def check_interactions(
    data:    InteractionCheckInput,
    checker = Depends(get_checker)
):
    try:
        result = checker.check(data.new_drug, data.current_drugs)
        return {
            "success":     True,
            "safe":        result["is_safe"],
            "alerts":      result.get("alerts", []),
            "alert_count": result.get("alert_count", 0),
            "source":      result.get("source_mode", "offline")
        }
    except Exception as e:
        logger.error(f"Interaction check error: {e}")
        raise HTTPException(status_code=500, detail=f"Interaction check error: {str(e)}")

@router.get("/medicines", response_model=dict)
async def get_medicines():
    try:
        medicines = load_medicines()
        if not medicines:
            raise HTTPException(status_code=404, detail="Medicines file not found")
        return {"medicines": medicines, "total": len(medicines)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Medicines error: {str(e)}")

@router.get("/conflicts", response_model=dict)
async def get_conflicts():
    try:
        return load_conflicts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conflicts error: {str(e)}")

@router.get("/conflicts/high", response_model=dict)
async def get_high_conflicts():
    try:
        data = load_conflicts()
        high = [c for c in data.get("conflicts", [])
                if c.get("severity") == "high"]
        return {"conflicts": high, "total": len(high)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conflicts error: {str(e)}")
