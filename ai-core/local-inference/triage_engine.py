"""
RIVA - Triage Engine v2.1 (Optimized) 🧠⚡
===========================================
Multi-Criteria Decision Making
1. Physiological Assessment  → ONNX Optimized
2. Symptom Assessment        → Score-based Weights
3. Interaction Check         → Drug Conflicts
Author: GODA EMAD
"""
import json
import pickle
import logging
import numpy as np
import pandas as pd
import onnxruntime as rt
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("RIVA.TriageEngine")

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
    symptom_scores:      dict = field(default_factory=dict)
    inference_ms:        float = 0.0

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
        }

# ================================================================
# TRIAGE ENGINE
# ================================================================
class TriageEngine:
    def __init__(
        self,
        model_path:    str = "ai-core/models/triage/model_int8.onnx",
        imputer_path:  str = "ai-core/models/triage/imputer.pkl",
        scaler_path:   str = "ai-core/models/triage/scaler.pkl",
        features_path: str = "ai-core/models/triage/features.json",
        conflicts_path:str = "business-intelligence/medical-content/drug_conflicts.json"
    ):
        import time
        start = time.time() * 1000

        # ONNX Session — Optimized
        opts = rt.SessionOptions()
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads     = 1
        opts.execution_mode           = rt.ExecutionMode.ORT_SEQUENTIAL

        self.sess     = rt.InferenceSession(
            model_path,
            sess_options = opts,
            providers    = ["CPUExecutionProvider"]
        )
        self.inp_name = self.sess.get_inputs()[0].name

        # Imputer + Scaler
        with open(imputer_path, "rb") as f: self.imputer = pickle.load(f)
        with open(scaler_path,  "rb") as f: self.scaler  = pickle.load(f)

        # Features
        with open(features_path, encoding="utf-8") as f:
            self.features = json.load(f)["features"]

        # Drug Conflicts
        self.conflicts: list = []
        try:
            with open(conflicts_path, encoding="utf-8") as f:
                self.conflicts = json.load(f).get("conflicts", [])
        except FileNotFoundError:
            logger.warning(f"conflicts not found: {conflicts_path}")

        elapsed = round(time.time() * 1000 - start, 1)
        logger.info(f"TriageEngine v2.1 initialized in {elapsed}ms")

    # ── 1. PHYSIOLOGICAL ────────────────────────
    def _assess_physiological(
        self, features: dict
    ) -> tuple[float, bool, float, float]:
        import time
        start = time.time() * 1000

        feat_df  = pd.DataFrame([features], columns=self.features)
        feat_imp = self.imputer.transform(feat_df)
        feat_s   = self.scaler.transform(feat_imp).astype(np.float32)
        proba    = self.sess.run(None, {self.inp_name: feat_s})[1][0]

        diabetes = bool(np.argmax(proba) == 1)
        conf     = float(proba[1])
        score    = conf if diabetes else (1 - conf) * 0.3

        ms = round(time.time() * 1000 - start, 2)
        logger.info(f"Inference: {ms}ms | diabetes={diabetes} | conf={conf:.3f}")

        return round(score, 3), diabetes, round(conf, 3), ms

    # ── 2. CLINICAL (Score-based) ────────────────
    def _assess_clinical(
        self,
        symptoms:         list,
        pain_level:       int,
        chronic_diseases: list
    ) -> tuple[float, str, dict]:
        symptom_scores = {s: SYMPTOM_WEIGHTS.get(s, 0.15) for s in symptoms}
        symptom_norm   = min(sum(symptom_scores.values()) / 2.0, 1.0)
        pain_score     = pain_level / 10
        chronic_score  = min(
            sum(CHRONIC_WEIGHTS.get(d, 0.1) for d in chronic_diseases),
            0.4
        )

        score = (
            symptom_norm  * 0.50 +
            pain_score    * 0.30 +
            chronic_score * 0.20
        )

        reasons = []
        if any(w >= 1.0 for w in symptom_scores.values()):
            reasons.append(f"أعراض طارئة: {[s for s,w in symptom_scores.items() if w>=1.0]}")
        if pain_level >= 8:
            reasons.append(f"ألم شديد {pain_level}/10")
        if chronic_diseases:
            reasons.append(f"أمراض مزمنة: {chronic_diseases}")

        explanation = " | ".join(reasons) if reasons else "أعراض خفيفة"
        return round(min(score, 1.0), 3), explanation, symptom_scores

    # ── 3. INTERACTION CHECK ─────────────────────
    def _assess_interactions(
        self, medications: list
    ) -> tuple[float, list]:
        if len(medications) < 2:
            return 0.0, []

        meds_lower = {m.lower().strip() for m in medications}
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
        alerts    = sorted(alerts,
                           key=lambda x: x["severity_score"],
                           reverse=True)

        return round(max_score, 3), alerts

    # ── FINAL DECISION ───────────────────────────
    def decide(
        self,
        features:         dict,
        symptoms:         list  = [],
        pain_level:       int   = 0,
        chronic_diseases: list  = [],
        medications:      list  = []
    ) -> TriageResult:

        phys_score, diabetes, conf, ms = self._assess_physiological(features)
        clin_score, clin_exp, sym_sc   = self._assess_clinical(
            symptoms, pain_level, chronic_diseases
        )
        inter_score, alerts = self._assess_interactions(medications)

        final_score = round(
            phys_score  * WEIGHTS["physiological"] +
            clin_score  * WEIGHTS["clinical"] +
            inter_score * WEIGHTS["interaction"],
            3
        )

        has_emergency  = any(SYMPTOM_WEIGHTS.get(s,0) >= 1.0 for s in symptoms)
        has_high_inter = any(a["severity"] == "high" for a in alerts)

        if has_emergency or final_score >= 0.75:
            level, label = "طارئ", 2
            action       = "اتصل بالإسعاف فوراً"
            specialty    = "طوارئ"
        elif final_score >= 0.45 or has_high_inter:
            level, label = "عاجل", 1
            action       = "كشف طبيب خلال ساعات"
            specialty    = "باطنة غدد صماء" if diabetes else "باطنة عامة"
        else:
            level, label = "غير عاجل", 0
            action       = "راحة + متابعة عيادة"
            specialty    = "عيادة عامة"

        explanation = (
            f"Final={final_score} | "
            f"Phys={phys_score} | "
            f"Clin={clin_score} | "
            f"Inter={inter_score} | "
            f"Inference={ms}ms | "
            f"{clin_exp}"
        )

        logger.info(
            f"Decision: {level} | score={final_score} | "
            f"diabetes={diabetes} | alerts={len(alerts)} | {ms}ms"
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
            inference_ms   = ms
        )


# Singleton
_engine: Optional[TriageEngine] = None

def get_engine() -> TriageEngine:
    global _engine
    if _engine is None:
        _engine = TriageEngine()
    return _engine
