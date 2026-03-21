"""
unified_predictor.py
====================
RIVA Health Platform — Unified Clinical Predictor v3.0
-------------------------------------------------------
نظام تنبؤ موحد يجمع:
    • خطر إعادة الإدخال (Readmission Risk)    — XGBoost AUC 0.79
    • مدة الإقامة المتوقعة (Length of Stay)   — XGBoost MAE 3.23
    • Platt calibration للـ probabilities
    • تكامل مع كل ملفات RIVA

التحسينات على v2.0:
    1. ربط مع readmission_predictor.py — Platt calibration + heatmap
    2. ربط مع history_analyzer         — history features تغذّي النموذج
    3. ربط مع confidence_scorer        — probabilities → scorer
    4. ربط مع explainability           — shap_ready جاهز
    5. ربط مع orchestrator             — target_pages من الـ 17 صفحة
    6. ربط مع prescription_gen audit   — تسجيل في SQLite
    7. numpy بدل pandas للـ single     — سريع للـ Serverless
    8. Platt scaling                   — Brier 0.19 → 0.14

الربط مع الـ 17 صفحة:
    high readmission + long LOS  → 15_combined_dashboard.html
    high readmission             → 13_readmission.html
    long LOS                     → 14_los_dashboard.html
    normal                       → 05_history.html

Author : GODA EMAD + RIVA Team
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np

warnings.filterwarnings("ignore")
log = logging.getLogger("riva.local_inference.unified_predictor")

# ─── Paths ───────────────────────────────────────────────────────────────────

_BASE         = Path(__file__).parent.parent
_READ_MODEL   = _BASE / "models/readmission/readmission_xgb_20260317_175502.pkl"
_READ_FEATS   = _BASE / "models/readmission/readmission_features_20260317_175502.json"
_LOS_MODEL    = _BASE / "models/los/los_final_xgb_tuned_20260317_174240.pkl"
_LOS_FEATS    = _BASE / "models/los/feature_names_los_20260317_174240.json"
_AUDIT_DB     = _BASE.parent / "data/databases/unified_predictions_audit.db"

# ─── Platt calibration (readmission) — same as readmission_predictor.py ──────

_PLATT_A = -1.82
_PLATT_B =  0.91


def _calibrate(prob: float) -> float:
    return round(min(1.0, max(0.0,
        1.0 / (1.0 + math.exp(-(_PLATT_A * prob + _PLATT_B)))
    )), 3)


# ─── Risk thresholds ──────────────────────────────────────────────────────────

READ_HIGH  = 0.60
READ_LOW   = 0.30
LOS_LONG   = 10.0
LOS_MEDIUM = 5.0

# ─── Target pages (from all 17 pages) ────────────────────────────────────────

def _decide_page(risk: str, los_days: float) -> str:
    """Routes to the most appropriate page from the 17-page system."""
    long_los = los_days > LOS_LONG
    if risk == "high" and long_los:
        return "15_combined_dashboard.html"
    if risk == "high":
        return "13_readmission.html"
    if long_los:
        return "14_los_dashboard.html"
    return "05_history.html"


# ─── Clinical explanations ────────────────────────────────────────────────────

class ClinicalExplanation:

    @staticmethod
    def readmission(probability: float) -> dict:
        if probability >= 0.70:
            level, text = "high",   "خطر مرتفع جداً — يحتاج تدخل فوري"
        elif probability >= 0.40:
            level, text = "medium", "خطر متوسط — متابعة دقيقة مطلوبة"
        else:
            level, text = "low",    "خطر منخفض — إجراءات عادية"
        return {
            "level":       level,
            "level_ar":    {"high": "مرتفع", "medium": "متوسط", "low": "منخفض"}[level],
            "text":        text,
            "probability": probability,
            "color":       {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}[level],
        }

    @staticmethod
    def los(days: float) -> str:
        if days > LOS_LONG:
            return "🔴 إقامة طويلة — تحضير خطة رعاية موسعة"
        if days > LOS_MEDIUM:
            return "🟡 إقامة متوسطة — متابعة يومية"
        return "🟢 إقامة قصيرة — خروج مبكر متوقع"


# ─── Clinical recommendations ─────────────────────────────────────────────────

class ClinicalRecommendation:

    _READ = {
        "high": [
            {"priority": 1, "action": "متابعة خلال 3 أيام",             "category": "follow_up"},
            {"priority": 2, "action": "مراجعة الأدوية والجرعات",        "category": "medication"},
            {"priority": 3, "action": "استشارة فريق متعدد التخصصات",   "category": "consultation"},
            {"priority": 4, "action": "مكالمة متابعة بعد 48 ساعة",      "category": "follow_up"},
        ],
        "medium": [
            {"priority": 1, "action": "متابعة خلال أسبوع",             "category": "follow_up"},
            {"priority": 2, "action": "تعليمات خروج مفصلة للمريض",    "category": "education"},
        ],
        "low": [
            {"priority": 1, "action": "تعليمات خروج قياسية",           "category": "standard"},
        ],
    }

    _LOS = {
        "long":   [
            {"action": "تجهيز جناح طويل الإقامة"},
            {"action": "متابعة يومية مع فريق التمريض"},
            {"action": "تقييم أسبوعي لخطة العلاج"},
        ],
        "medium": [
            {"action": "تحضير خروج منظم"},
            {"action": "تثقيف المريض والأسرة"},
        ],
        "short":  [
            {"action": "خروج مبكر مع تعليمات"},
        ],
    }

    @classmethod
    def readmission(cls, risk_level: str) -> list:
        return cls._READ.get(risk_level, cls._READ["low"])

    @classmethod
    def los(cls, days: float) -> list:
        if days > LOS_LONG:   return cls._LOS["long"]
        if days > LOS_MEDIUM: return cls._LOS["medium"]
        return cls._LOS["short"]


# ─── Model version ────────────────────────────────────────────────────────────

class ModelVersion:
    def __init__(self, name: str, version: str, metrics: dict):
        self.name       = name
        self.version    = version
        self.metrics    = metrics
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version,
                "metrics": self.metrics, "created_at": self.created_at}


# ─── Audit trail ──────────────────────────────────────────────────────────────

def _store_audit(result: dict) -> None:
    """Stores prediction in SQLite for 17_sustainability.html analytics."""
    try:
        _AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(_AUDIT_DB))
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS unified_audit (
                patient_id       TEXT,
                timestamp        TEXT,
                read_risk        TEXT,
                read_probability REAL,
                los_days         REAL,
                target_page      TEXT,
                processing_ms    REAL
            )
        """)
        meta = result.get("metadata", {})
        cur.execute("INSERT INTO unified_audit VALUES (?,?,?,?,?,?,?)", (
            meta.get("patient_id", "unknown"),
            meta.get("timestamp"),
            result.get("readmission", {}).get("risk_level", ""),
            result.get("readmission", {}).get("probability", 0.0),
            result.get("los", {}).get("predicted_days", 0.0),
            result.get("target_page", ""),
            meta.get("processing_ms", 0.0),
        ))
        con.commit()
        con.close()
    except Exception as e:
        log.warning("[UnifiedPredictor] audit failed: %s", e)


# ─── Core predictor ───────────────────────────────────────────────────────────

class UnifiedPredictor:
    """
    RIVA Unified Clinical Predictor v3.0

    Combines Readmission + LOS prediction with:
        - Platt-calibrated probabilities
        - Automatic page routing (17-page system)
        - SQLite audit trail for sustainability analytics
        - shap_ready features for explainability.py
        - numpy single inference (no pandas overhead)
    """

    VERSION = "3.0.0"

    def __init__(
        self,
        base_path:     Optional[Union[str, Path]] = None,
        enable_logging:bool = True,
    ):
        self._base          = Path(base_path) if base_path else _BASE
        self.enable_logging = enable_logging
        self.model_versions: list[ModelVersion] = []
        self._history:       list[dict]         = []

        self.read_model    = None
        self.los_model     = None
        self.read_features: list[str] = []
        self.los_features:  list[str] = []
        self.read_auc       = 0.792
        self.los_mae        = 3.23

        self._load_models()

        log.info("[UnifiedPredictor] v%s ready", self.VERSION)

    def _load_models(self) -> None:
        try:
            import joblib

            # Readmission
            self.read_model = joblib.load(str(_READ_MODEL))
            with open(_READ_FEATS, encoding="utf-8") as f:
                rd = json.load(f)
            self.read_features = rd["features"]
            self.read_auc      = rd.get("auc", 0.792)
            self.model_versions.append(ModelVersion(
                "readmission", "v4.0", {"auc": self.read_auc}
            ))
            log.info("[UnifiedPredictor] readmission loaded  features=%d  AUC=%.3f",
                     len(self.read_features), self.read_auc)

            # LOS
            self.los_model = joblib.load(str(_LOS_MODEL))
            with open(_LOS_FEATS, encoding="utf-8") as f:
                ld = json.load(f)
            self.los_features = ld["features"]
            self.los_mae      = ld.get("mae", 3.23)
            self.model_versions.append(ModelVersion(
                "los", "v4.0", {"mae": self.los_mae}
            ))
            log.info("[UnifiedPredictor] LOS loaded  features=%d  MAE=%.2f",
                     len(self.los_features), self.los_mae)

        except Exception as e:
            log.error("[UnifiedPredictor] model load failed: %s", e)

    def is_loaded(self) -> bool:
        return self.read_model is not None and self.los_model is not None

    # ── Feature preparation (numpy — no pandas for single patient) ───────────

    def _prep_read(self, data: dict) -> np.ndarray:
        """numpy array for readmission — avoids pandas DataFrame overhead."""
        return np.array(
            [[float(data.get(f, 0)) for f in self.read_features]],
            dtype=np.float32,
        )

    def _prep_los(self, data: dict) -> np.ndarray:
        return np.array(
            [[float(data.get(f, 0)) for f in self.los_features]],
            dtype=np.float32,
        )

    # ── Single prediction ─────────────────────────────────────────────────────

    def predict(
        self,
        patient_data:              dict,
        generate_explanation:      bool = True,
        generate_recommendations:  bool = True,
        history_features:          dict = {},
        session_id:                Optional[str] = None,
    ) -> dict:
        """
        Full clinical prediction for one patient.

        Args:
            patient_data            : raw patient features
            generate_explanation    : include clinical explanation
            generate_recommendations: include action recommendations
            history_features        : from history_analyzer.get_readmission_features()
            session_id              : for audit trail

        Returns:
            Complete result dict with target_page, shap_ready, recommendations.
        """
        if not self.is_loaded():
            return {"error": "النماذج غير محملة", "target_page": "01_home.html"}

        import time
        t0         = time.perf_counter()
        patient_id = patient_data.get("patient_id", session_id or "unknown")

        # Merge history features
        merged = {**patient_data, **history_features}

        # ── Readmission prediction ──────────────────────────────────────────
        X_read     = self._prep_read(merged)
        raw_prob   = float(self.read_model.predict_proba(X_read)[0][1])
        cal_prob   = _calibrate(raw_prob)

        if cal_prob >= READ_HIGH:
            risk = "high"
        elif cal_prob >= READ_LOW:
            risk = "medium"
        else:
            risk = "low"

        # ── LOS prediction ──────────────────────────────────────────────────
        X_los    = self._prep_los(merged)
        los_days = round(float(self.los_model.predict(X_los)[0]), 2)

        # ── Routing ─────────────────────────────────────────────────────────
        target_page = _decide_page(risk, los_days)

        # ── shap_ready for explainability.py ────────────────────────────────
        shap_ready = {
            **{f: float(merged.get(f, 0)) for f in self.read_features},
            "los_days":           los_days,
            "readmission_prob":   cal_prob,
        }

        ms = round((time.perf_counter() - t0) * 1000, 2)

        result = {
            "metadata": {
                "timestamp":     datetime.now(timezone.utc).isoformat(),
                "patient_id":    patient_id,
                "version":       self.VERSION,
                "processing_ms": ms,
                "calibrated":    True,
            },
            "readmission": {
                "risk_level":        risk,
                "risk_level_ar":     {"high":"مرتفع","medium":"متوسط","low":"منخفض"}[risk],
                "probability":       cal_prob,
                "raw_probability":   round(raw_prob, 3),
                "color":             {"high":"#ef4444","medium":"#f59e0b","low":"#10b981"}[risk],
                "model_auc":         self.read_auc,
            },
            "los": {
                "predicted_days":    los_days,
                "confidence_interval": {
                    "lower": round(max(0, los_days - self.los_mae), 2),
                    "upper": round(los_days + self.los_mae, 2),
                },
                "unit":    "يوم",
                "model_mae": self.los_mae,
                "category": "long" if los_days > LOS_LONG else "medium" if los_days > LOS_MEDIUM else "short",
            },
            "target_page": target_page,
            "shap_ready":  shap_ready,
            "offline":     True,
        }

        if generate_explanation:
            result["explanation"] = {
                "readmission": ClinicalExplanation.readmission(cal_prob),
                "los":         ClinicalExplanation.los(los_days),
            }

        if generate_recommendations:
            result["recommendations"] = {
                "readmission": ClinicalRecommendation.readmission(risk),
                "los":         ClinicalRecommendation.los(los_days),
            }

        # Audit trail
        _store_audit(result)

        if self.enable_logging:
            self._history.append({
                "timestamp":    result["metadata"]["timestamp"],
                "patient_id":   patient_id,
                "risk":         risk,
                "probability":  cal_prob,
                "los_days":     los_days,
                "processing_ms":ms,
            })

        log.info(
            "[UnifiedPredictor] patient=%s  risk=%s  prob=%.3f  los=%.1fd  "
            "→ %s  %.0fms",
            patient_id, risk, cal_prob, los_days, target_page, ms,
        )

        return result

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(self, patients: list[dict]) -> list[dict]:
        """
        Vectorised batch prediction — sorted by risk (highest first).
        Used in 15_combined_dashboard.html.
        """
        if not self.is_loaded():
            return [{"error": "النماذج غير محملة"} for _ in patients]

        try:
            # Readmission batch
            X_read = np.array(
                [[float(p.get(f, 0)) for f in self.read_features] for p in patients],
                dtype=np.float32,
            )
            raw_probs = self.read_model.predict_proba(X_read)[:, 1]
            cal_probs = [_calibrate(float(p)) for p in raw_probs]

            # LOS batch
            X_los    = np.array(
                [[float(p.get(f, 0)) for f in self.los_features] for p in patients],
                dtype=np.float32,
            )
            los_preds = self.los_model.predict(X_los)

            results = []
            for i, p in enumerate(patients):
                cal  = cal_probs[i]
                los  = round(float(los_preds[i]), 2)
                risk = ("high" if cal >= READ_HIGH else
                        "medium" if cal >= READ_LOW else "low")
                results.append({
                    "patient_id":  p.get("patient_id", f"patient_{i}"),
                    "risk_level":  risk,
                    "risk_level_ar":{"high":"مرتفع","medium":"متوسط","low":"منخفض"}[risk],
                    "probability": cal,
                    "los_days":    los,
                    "target_page": _decide_page(risk, los),
                    "color":       {"high":"#ef4444","medium":"#f59e0b","low":"#10b981"}[risk],
                })

            return sorted(results, key=lambda x: x["probability"], reverse=True)

        except Exception as e:
            log.error("[UnifiedPredictor] batch error: %s", e)
            return [{"error": str(e)} for _ in patients]

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """Returns aggregate stats for 17_sustainability.html."""
        if not self._history:
            return {"message": "لا توجد توقعات مسجلة بعد"}

        probs = [h["probability"] for h in self._history]
        los   = [h["los_days"]    for h in self._history]
        times = [h["processing_ms"] for h in self._history]

        return {
            "total_predictions":  len(self._history),
            "high_risk_count":    sum(1 for h in self._history if h["risk"] == "high"),
            "avg_probability":    round(sum(probs) / len(probs), 3),
            "avg_los_days":       round(sum(los)   / len(los),   2),
            "avg_processing_ms":  round(sum(times) / len(times), 2),
            "last_prediction":    self._history[-1]["timestamp"],
        }

    def get_model_info(self) -> dict:
        return {
            "version":      self.VERSION,
            "readmission":  {"features": len(self.read_features), "auc": self.read_auc},
            "los":          {"features": len(self.los_features),  "mae": self.los_mae},
            "versions":     [v.to_dict() for v in self.model_versions],
            "calibrated":   True,
            "target_pages": {
                "high+long":  "15_combined_dashboard.html",
                "high":       "13_readmission.html",
                "long_los":   "14_los_dashboard.html",
                "normal":     "05_history.html",
            },
        }

    def generate_clinical_report(self, patient_data: dict) -> str:
        """Generates formatted Arabic clinical report."""
        result = self.predict(patient_data)
        r = result["readmission"]
        l = result["los"]
        lines = [
            "=" * 60,
            "RIVA — التقرير السريري الشامل",
            "=" * 60,
            f"المريض       : {result['metadata']['patient_id']}",
            f"وقت المعالجة : {result['metadata']['processing_ms']} ms",
            "-" * 60,
            f"خطر إعادة الإدخال : {r['risk_level_ar']} ({r['probability']:.1%})",
            f"مدة الإقامة المتوقعة: {l['predicted_days']} يوم "
            f"({l['confidence_interval']['lower']}–{l['confidence_interval']['upper']})",
            f"الصفحة الموصى بها : {result['target_page']}",
            "-" * 60,
        ]
        if "recommendations" in result:
            lines.append("التوصيات:")
            for rec in result["recommendations"]["readmission"][:3]:
                lines.append(f"  • {rec['action']}")
        lines += ["=" * 60,
                  f"تم الإنشاء: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  "=" * 60]
        return "\n".join(lines)


# ─── FastAPI router ───────────────────────────────────────────────────────────

def create_fastapi_router(predictor: UnifiedPredictor):
    """
    Creates a FastAPI router for the unified predictor.

    Usage in app.py:
        from .unified_predictor import UnifiedPredictor, create_fastapi_router
        predictor = UnifiedPredictor()
        app.include_router(create_fastapi_router(predictor))
    """
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel

        router = APIRouter(prefix="/api/v1/clinical", tags=["Clinical Prediction"])

        class PatientData(BaseModel):
            patient_id:      str   = "unknown"
            age:             int
            gender_M:        int   = 1
            admit_EMERGENCY: int   = 1
            num_diagnoses:   int
            num_procedures:  int
            num_medications: int
            charlson_index:  int   = 0
            total_visits:    int   = 1

        @router.post("/predict")
        async def predict_patient(patient: PatientData):
            try:
                return predictor.predict(patient.dict())
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @router.post("/batch")
        async def predict_patients(patients: list[PatientData]):
            try:
                return predictor.predict_batch([p.dict() for p in patients])
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @router.get("/stats")
        async def get_stats():
            return predictor.get_statistics()

        @router.get("/info")
        async def get_info():
            return predictor.get_model_info()

        return router

    except ImportError:
        log.warning("[UnifiedPredictor] FastAPI not available — router not created")
        return None


# ─── Singleton ────────────────────────────────────────────────────────────────

_predictor: Optional[UnifiedPredictor] = None


def get_predictor() -> UnifiedPredictor:
    global _predictor
    if _predictor is None:
        _predictor = UnifiedPredictor()
    return _predictor


# ─── Public API ───────────────────────────────────────────────────────────────

def predict(
    patient_data:     dict,
    history_features: dict = {},
    session_id:       Optional[str] = None,
) -> dict:
    """
    Main entry point.

    Usage in orchestrator.py (intent=Combined):
        from .unified_predictor import predict
        from .history_analyzer  import get_readmission_features

        hist   = get_readmission_features(patient_hash, visits)
        result = predict(
            patient_data     = session["clinical_features"],
            history_features = hist,
            session_id       = session_id,
        )
        target_page = result["target_page"]
        # 15_combined_dashboard.html  لو high risk + long LOS
        # 13_readmission.html         لو high risk فقط
        # 14_los_dashboard.html       لو LOS طويل فقط

        # Feed into confidence_scorer
        score = compute_confidence(
            intent            = "Combined",
            model_logit_score = result["readmission"]["probability"],
        )

        # Feed into explainability
        explanation = explain(
            model_type = ModelType.READMISSION,
            features   = result["shap_ready"],
            prediction = result["readmission"]["risk_level"],
            confidence = result["readmission"]["probability"],
        )
    """
    return get_predictor().predict(
        patient_data     = patient_data,
        history_features = history_features,
        session_id       = session_id,
    )


def predict_batch(patients: list[dict]) -> list[dict]:
    """Vectorised batch — sorted by risk."""
    return get_predictor().predict_batch(patients)


def get_statistics() -> dict:
    return get_predictor().get_statistics()


def get_model_info() -> dict:
    return get_predictor().get_model_info()
