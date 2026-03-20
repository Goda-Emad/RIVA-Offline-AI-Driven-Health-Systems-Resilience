"""
triage_engine.py
================
RIVA Health Platform — Triage Engine v3.2 (MCDM)
-------------------------------------------------
Multi-Criteria Decision Making:
1. Physiological Assessment → ONNX Optimized (numpy - Fix A)
2. Symptom Assessment       → Max Weight للأعراض الطارئة (v3.2)
3. Interaction Check        → Drug Conflicts

Fixes:
    Fix A: numpy preprocessing (no pandas overhead)
    Fix B: mutable default arguments → None guards
    Fix C: JSON loaders (no pickle security risk)
    v3.2: Max Weight for emergency symptoms

Author: GODA EMAD
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("riva.local_inference.triage_engine")

# ================================================================
# PATHS
# ================================================================
_HERE          = Path(__file__).resolve().parent
_AICORE        = _HERE.parent
_MODEL_PATH    = _AICORE / "models/triage/model_int8.onnx"
_IMPUTER_JSON  = _AICORE / "models/triage/imputer_weights.json"
_SCALER_JSON   = _AICORE / "models/triage/scaler_weights.json"
_IMPUTER_PKL   = _AICORE / "models/triage/imputer.pkl"
_SCALER_PKL    = _AICORE / "models/triage/scaler.pkl"
_FEATURES_PATH = _AICORE / "models/triage/features.json"
_CONFLICTS_PATH= _AICORE.parent / "business-intelligence/medical-content/drug_conflicts.json"

# ================================================================
# WEIGHTS
# ================================================================
WEIGHTS = {
    "physiological": 0.50,
    "clinical":      0.35,
    "interaction":   0.15,
}

# ================================================================
# SYMPTOM WEIGHTS
# ================================================================
SYMPTOM_WEIGHTS: dict[str, float] = {
    # طارئ — 1.0
    "ألم في الصدر":          1.0,
    "ضيق تنفس":              1.0,
    "فقدان وعي":             1.0,
    "تشنجات":                1.0,
    "نزيف شديد":             1.0,
    "شلل مفاجئ":             1.0,
    "chest_pain":            1.0,
    "breathing_difficulty":  1.0,
    "loss_of_consciousness": 1.0,
    "seizures":              1.0,
    "severe_bleeding":       1.0,
    # عاجل — 0.6-0.8
    "حمى شديدة":             0.8,
    "ضعف مفاجئ":             0.7,
    "ارتباك":                0.7,
    "قيء متكرر":             0.6,
    "دوخة شديدة":            0.6,
    "تورم حاد":              0.6,
    "high_fever":            0.8,
    "sudden_weakness":       0.7,
    # متوسط — 0.2-0.4
    "صداع":                  0.3,
    "غثيان":                 0.3,
    "كحة":                   0.2,
    "رشح":                   0.1,
    "التهاب حلق":            0.25,
    "طفح جلدي":              0.2,
    "ألم بطن":               0.35,
    "إسهال":                 0.3,
    "تعب عام":               0.2,
}

# ================================================================
# CHRONIC DISEASE WEIGHTS
# ================================================================
CHRONIC_WEIGHTS: dict[str, float] = {
    "سكري":      0.30,
    "ضغط مرتفع": 0.25,
    "قلب":       0.35,
    "ربو":       0.20,
    "صرع":       0.30,
    "فشل كلوي":  0.35,
    "سرطان":     0.40,
}

# ================================================================
# TARGET PAGES — ربط مع 17 صفحة
# ================================================================
TARGET_PAGES: dict[str, str] = {
    "طارئ":           "04_result.html",
    "عاجل":           "03_triage.html",
    "غير عاجل":       "05_history.html",
    "doctor":         "09_doctor_dashboard.html",
    "mother":         "10_mother_dashboard.html",
    "school":         "11_school_dashboard.html",
    "ai_explanation": "12_ai_explanation.html",
    "readmission":    "13_readmission.html",
    "los":            "14_los_dashboard.html",
    "combined":       "15_combined_dashboard.html",
    "doctor_notes":   "16_doctor_notes.html",
    "sustainability": "17_sustainability.html",
    "home":           "01_home.html",
    "chatbot":        "02_chatbot.html",
    "history":        "05_history.html",
    "pregnancy":      "06_pregnancy.html",
    "offline":        "08_offline.html",
}

# ================================================================
# JSON LOADERS (Fix C — no pickle)
# ================================================================
class _JsonImputer:
    def __init__(self, fit_X: list, n_neighbors: int = 5):
        self._fit_X       = np.array(fit_X, dtype=np.float64)
        self._n_neighbors = n_neighbors

    def transform(self, X: np.ndarray) -> np.ndarray:
        out      = X.copy().astype(np.float64)
        nan_mask = np.isnan(out)
        if not nan_mask.any():
            return out
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if nan_mask[i, j]:
                    col      = self._fit_X[:, j]
                    valid    = col[~np.isnan(col)]
                    out[i,j] = float(np.median(valid)) if len(valid) else 0.0
        return out

class _JsonScaler:
    def __init__(self, mean: list, scale: list):
        self._mean  = np.array(mean,  dtype=np.float64)
        self._scale = np.array(scale, dtype=np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._scale

def _load_imputer():
    if _IMPUTER_JSON.exists():
        data = json.loads(_IMPUTER_JSON.read_text(encoding="utf-8"))
        log.info("imputer loaded from JSON (secure)")
        return _JsonImputer(data["fit_X"], data.get("n_neighbors", 5))
    if _IMPUTER_PKL.exists():
        import pickle
        log.warning("imputer.pkl — migrate to JSON for production security")
        with open(_IMPUTER_PKL, "rb") as f:
            return pickle.load(f)
    return None

def _load_scaler():
    if _SCALER_JSON.exists():
        data = json.loads(_SCALER_JSON.read_text(encoding="utf-8"))
        log.info("scaler loaded from JSON (secure)")
        return _JsonScaler(data["mean_"], data["scale_"])
    if _SCALER_PKL.exists():
        import pickle
        log.warning("scaler.pkl — migrate to JSON for production security")
        with open(_SCALER_PKL, "rb") as f:
            return pickle.load(f)
    return None

# ================================================================
# RESULT
# ================================================================
@dataclass
class TriageResult:
    triage_level:        str
    triage_label:        int
    final_score:         float
    diabetes:            bool
    diabetes_confidence: float
    diagnosis:           str
    recommended_action:  str
    specialty:           str
    drug_alerts:         list
    explanation:         str
    scores:              dict
    symptom_scores:      dict  = field(default_factory=dict)
    inference_ms:        float = 0.0
    target_page:         str   = "03_triage.html"
    shap_ready:          dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "triage_level":        self.triage_level,
            "triage_label":        self.triage_label,
            "final_score":         self.final_score,
            "diabetes":            self.diabetes,
            "diabetes_confidence": self.diabetes_confidence,
            "diagnosis":           self.diagnosis,
            "recommended_action":  self.recommended_action,
            "specialty":           self.specialty,
            "drug_alerts":         self.drug_alerts,
            "explanation":         self.explanation,
            "scores":              self.scores,
            "symptom_scores":      self.symptom_scores,
            "inference_ms":        self.inference_ms,
            "target_page":         self.target_page,
            "shap_ready":          self.shap_ready,
            "offline":             True,
        }

# ================================================================
# TRIAGE ENGINE
# ================================================================
class TriageEngine:
    def __init__(self) -> None:
        self.sess:      object    = None
        self.inp_name:  str       = None
        self.imputer:   object    = None
        self.scaler:    object    = None
        self.features:  list[str] = []
        self.conflicts: list      = []
        self._try_load()

    def _try_load(self) -> None:
        try:
            import onnxruntime as rt
            opts = rt.SessionOptions()
            opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads     = 1
            opts.execution_mode           = rt.ExecutionMode.ORT_SEQUENTIAL

            self.sess     = rt.InferenceSession(
                str(_MODEL_PATH),
                sess_options = opts,
                providers    = ["CPUExecutionProvider"]
            )
            self.inp_name = self.sess.get_inputs()[0].name
            self.imputer  = _load_imputer()
            self.scaler   = _load_scaler()

            if _FEATURES_PATH.exists():
                self.features = json.loads(
                    _FEATURES_PATH.read_text(encoding="utf-8")
                )["features"]

            if _CONFLICTS_PATH.exists():
                self.conflicts = json.loads(
                    _CONFLICTS_PATH.read_text(encoding="utf-8")
                ).get("conflicts", [])

            log.info("TriageEngine v3.2 loaded | features=%d", len(self.features))
        except Exception as e:
            log.error("TriageEngine load failed: %s", e)

    def is_loaded(self) -> bool:
        return self.sess is not None

    # ── 1. PHYSIOLOGICAL (Fix A: numpy) ─────────
    def _assess_physiological(
        self, features: dict
    ) -> tuple[float, bool, float, float]:
        t0  = time.perf_counter()
        row = np.array(
            [[features.get(f, np.nan) for f in self.features]],
            dtype=np.float64
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if self.imputer: row = self.imputer.transform(row)
            if self.scaler:  row = self.scaler.transform(row)

        proba    = self.sess.run(None, {self.inp_name: row.astype(np.float32)})[1][0]
        diabetes = bool(np.argmax(proba) == 1)
        conf     = float(proba[1])
        score    = conf if diabetes else (1 - conf) * 0.3
        ms       = round((time.perf_counter() - t0) * 1000, 2)

        log.info("Inference: %.1fms | diabetes=%s | conf=%.3f", ms, diabetes, conf)
        return round(score, 3), diabetes, round(conf, 3), ms

    # ── 2. CLINICAL (v3.2: Max Weight) ──────────
    def _assess_clinical(
        self,
        symptoms:         list,
        pain_level:       int,
        chronic_diseases: list,
    ) -> tuple[float, str, dict, float]:
        symptom_scores = {s: SYMPTOM_WEIGHTS.get(s, 0.15) for s in symptoms}
        max_weight     = max(symptom_scores.values(), default=0.0)

        # v3.2: Max Weight — لو في عرض طارئ مش بنخففه بالـ sum
        if max_weight >= 1.0:
            symptom_norm = 1.0
        else:
            symptom_norm = min(sum(symptom_scores.values()) / 2.0, 1.0)

        pain_score    = pain_level / 10
        chronic_score = min(
            sum(CHRONIC_WEIGHTS.get(d, 0.1) for d in chronic_diseases), 0.4
        )
        score = symptom_norm*0.50 + pain_score*0.30 + chronic_score*0.20

        reasons = []
        if max_weight >= 1.0:
            emergency = [s for s,w in symptom_scores.items() if w >= 1.0]
            reasons.append(f"أعراض طارئة: {emergency}")
        if pain_level >= 8:
            reasons.append(f"ألم شديد {pain_level}/10")
        if chronic_diseases:
            reasons.append(f"أمراض مزمنة: {chronic_diseases}")

        explanation = " | ".join(reasons) if reasons else "أعراض خفيفة"
        return round(min(score, 1.0), 3), explanation, symptom_scores, max_weight

    # ── 3. INTERACTION CHECK ─────────────────────
    def _assess_interactions(self, medications: list) -> tuple[float, list]:
        if len(medications) < 2:
            return 0.0, []

        meds_lower = {m.lower().strip() for m in medications if m}
        alerts     = []

        for c in self.conflicts:
            if (c["drug_a"].lower() in meds_lower and
                c["drug_b"].lower() in meds_lower):
                alerts.append({
                    "drug_a":         c["drug_a"],
                    "drug_b":         c["drug_b"],
                    "severity":       c["severity"],
                    "severity_score": c.get("severity_score", 1),
                    "effect_ar":      c["effect_ar"],
                    "recommendation": c.get("recommendation_ar", "")
                })

        if not alerts:
            return 0.0, []

        score_map = {"high": 1.0, "medium": 0.5, "low": 0.2}
        max_score = max(score_map.get(a["severity"], 0) for a in alerts)
        return round(max_score, 3), sorted(
            alerts, key=lambda x: x["severity_score"], reverse=True
        )

    # ── FINAL DECISION ───────────────────────────
    def decide(
        self,
        features:         Optional[dict] = None,   # Fix B
        symptoms:         Optional[list] = None,   # Fix B
        pain_level:       int            = 0,
        chronic_diseases: Optional[list] = None,   # Fix B
        medications:      Optional[list] = None,   # Fix B
        session_id:       Optional[str]  = None,
    ) -> TriageResult:
        # Fix B: mutable default arguments guard
        features         = features         or {}
        symptoms         = symptoms         or []
        chronic_diseases = chronic_diseases or []
        medications      = medications      or []

        if not self.is_loaded():
            return TriageResult(
                triage_level="غير محدد", triage_label=0, final_score=0.0,
                diabetes=False, diabetes_confidence=0.0, diagnosis="خطأ",
                recommended_action="النموذج غير محمل", specialty="",
                drug_alerts=[], explanation="النموذج غير محمل",
                scores={}, target_page="01_home.html"
            )

        phys_score, diabetes, conf, ms        = self._assess_physiological(features)
        clin_score, clin_exp, sym_sc, max_w   = self._assess_clinical(
            symptoms, pain_level, chronic_diseases
        )
        inter_score, alerts = self._assess_interactions(medications)

        final_score = round(
            phys_score  * WEIGHTS["physiological"] +
            clin_score  * WEIGHTS["clinical"] +
            inter_score * WEIGHTS["interaction"],
            3
        )

        has_emergency  = max_w >= 1.0
        has_high_inter = any(a["severity"] == "high" for a in alerts)

        if has_emergency or final_score >= 0.75:
            level, label = "طارئ", 2
            action       = "اتصل بالإسعاف فوراً"
            specialty    = "طوارئ"
            page         = TARGET_PAGES["طارئ"]
        elif final_score >= 0.45 or has_high_inter:
            level, label = "عاجل", 1
            action       = "كشف طبيب خلال ساعات"
            specialty    = "باطنة غدد صماء" if diabetes else "باطنة عامة"
            page         = TARGET_PAGES["عاجل"]
        else:
            level, label = "غير عاجل", 0
            action       = "راحة + متابعة عيادة"
            specialty    = "عيادة عامة"
            page         = TARGET_PAGES["غير عاجل"]

        # v3.2: SHAP Ready محسّن لـ 12_ai_explanation.html
        shap_ready = {
            **{f: float(features.get(f, 0)) for f in self.features},
            "symptom_max_weight":   max_w,
            "symptom_count":        float(len(symptoms)),
            "chronic_count":        float(len(chronic_diseases)),
            "pain_level":           float(pain_level),
            "has_drug_interaction": float(len(alerts) > 0),
            "score_physiological":  phys_score,
            "score_clinical":       clin_score,
            "score_interaction":    inter_score,
        }

        explanation = (
            f"Final={final_score} | Phys={phys_score} | "
            f"Clin={clin_score} | Inter={inter_score} | "
            f"Inference={ms}ms | {clin_exp}"
        )

        log.info(
            "Decision: %s | score=%.3f | diabetes=%s | alerts=%d | session=%s",
            level, final_score, diabetes, len(alerts), session_id
        )

        return TriageResult(
            triage_level        = level,
            triage_label        = label,
            final_score         = final_score,
            diabetes            = diabetes,
            diabetes_confidence = conf,
            diagnosis           = "سكري" if diabetes else "سليم",
            recommended_action  = action,
            specialty           = specialty,
            drug_alerts         = alerts,
            explanation         = explanation,
            scores = {
                "physiological": phys_score,
                "clinical":      clin_score,
                "interaction":   inter_score,
                "final":         final_score
            },
            symptom_scores = sym_sc,
            inference_ms   = ms,
            target_page    = page,
            shap_ready     = shap_ready,
        )


# ================================================================
# Singleton
# ================================================================
_engine: Optional[TriageEngine] = None

def get_engine() -> TriageEngine:
    global _engine
    if _engine is None:
        _engine = TriageEngine()
    return _engine


# ================================================================
# Public API
# ================================================================
def decide(
    features:         Optional[dict] = None,   # Fix B
    symptoms:         Optional[list] = None,   # Fix B
    pain_level:       int            = 0,
    chronic_diseases: Optional[list] = None,   # Fix B
    medications:      Optional[list] = None,   # Fix B
    session_id:       Optional[str]  = None,
) -> dict:
    """
    Main entry point للـ orchestrator.

    Usage:
        from ai_core.local_inference.triage_engine import decide

        result = decide(
            features         = triage_features,
            symptoms         = ["ألم في الصدر"],
            pain_level       = 8,
            chronic_diseases = ["سكري"],
            medications      = ["warfarin", "aspirin"],
            session_id       = session_id,
        )
        target_page = result["target_page"]  # 04_result.html
        shap_data   = result["shap_ready"]   # → 12_ai_explanation.html
    """
    return get_engine().decide(
        features         = features,
        symptoms         = symptoms,
        pain_level       = pain_level,
        chronic_diseases = chronic_diseases,
        medications      = medications,
        session_id       = session_id,
    ).to_dict()


def get_target_page(context: str) -> str:
    """يرجع الصفحة المناسبة حسب السياق."""
    return TARGET_PAGES.get(context, "01_home.html")
