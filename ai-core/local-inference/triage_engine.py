"""
RIVA - Triage Engine (MCDM) 🧠
================================
Multi-Criteria Decision Making
1. Physiological Assessment  → ONNX Model
2. Symptom Assessment        → Pain + Symptoms
3. Interaction Check         → Drug Conflicts
Author: GODA EMAD
"""
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("RIVA.TriageEngine")

# ================================================================
# WEIGHTS — أوزان كل معيار في القرار النهائي
# ================================================================
WEIGHTS = {
    "physiological": 0.50,   # نتيجة الموديل
    "clinical":      0.35,   # الأعراض + الألم
    "interaction":   0.15,   # تعارض الأدوية
}

# أعراض خطيرة تلقائياً → طارئ
EMERGENCY_SYMPTOMS = {
    "ألم في الصدر", "ضيق تنفس", "فقدان وعي",
    "تشنجات", "نزيف شديد", "شلل", "سكتة دماغية",
    "chest_pain", "breathing_difficulty", "loss_of_consciousness",
    "seizures", "severe_bleeding"
}

# أعراض تزيد الخطورة
HIGH_RISK_SYMPTOMS = {
    "حمى شديدة", "قيء متكرر", "دوخة شديدة",
    "ضعف مفاجئ", "ارتباك", "تورم حاد"
}

# ================================================================
# RESULT
# ================================================================
@dataclass
class TriageResult:
    triage_level:       str      # طارئ / عاجل / غير عاجل
    triage_label:       int      # 2 / 1 / 0
    final_score:        float    # 0.0 → 1.0
    diabetes:           bool
    diabetes_confidence:float
    diagnosis:          str
    recommended_action: str
    specialty:          str
    drug_alerts:        list
    explanation:        str
    scores: dict         # breakdown of scores

# ================================================================
# TRIAGE ENGINE
# ================================================================
class TriageEngine:
    """
    Multi-Criteria Decision Making Engine
    ======================================
    يأخذ بيانات المريض ويطلع قرار نهائي بناءً على 3 معايير
    """

    def __init__(
        self,
        model_path:    str = "ai-core/models/triage/model_int8.onnx",
        imputer_path:  str = "ai-core/models/triage/imputer.pkl",
        scaler_path:   str = "ai-core/models/triage/scaler.pkl",
        features_path: str = "ai-core/models/triage/features.json",
        conflicts_path:str = "business-intelligence/medical-content/drug_conflicts.json"
    ):
        import onnxruntime as rt

        self.sess     = rt.InferenceSession(model_path,
                        providers=["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name

        with open(imputer_path,  "rb") as f: self.imputer  = pickle.load(f)
        with open(scaler_path,   "rb") as f: self.scaler   = pickle.load(f)
        with open(features_path, "rb") as f: self.features = json.load(f)["features"]

        self.conflicts = []
        try:
            with open(conflicts_path, encoding="utf-8") as f:
                self.conflicts = json.load(f).get("conflicts", [])
        except FileNotFoundError:
            logger.warning("drug_conflicts.json not found")

        logger.info("TriageEngine initialized")

    # ── 1. PHYSIOLOGICAL ASSESSMENT ─────────────
    def _assess_physiological(self, features: dict) -> tuple[float, bool, float]:
        """
        يشغّل الـ ONNX Model ويرجع:
        - score: خطورة فسيولوجية (0-1)
        - diabetes: هل السكري محتمل؟
        - confidence: نسبة الثقة
        """
        feat_df  = pd.DataFrame([features], columns=self.features)
        feat_imp = self.imputer.transform(feat_df)
        feat_s   = self.scaler.transform(feat_imp).astype(np.float32)

        proba    = self.sess.run(None, {self.inp_name: feat_s})[1][0]
        diabetes = bool(np.argmax(proba) == 1)
        conf     = float(proba[1])

        # Score: لو سكري محتمل بنسبة عالية → score أعلى
        score = conf if diabetes else (1 - conf) * 0.3
        return score, diabetes, conf

    # ── 2. CLINICAL ASSESSMENT ───────────────────
    def _assess_clinical(
        self,
        symptoms:  list,
        pain_level: int,
        chronic_diseases: list
    ) -> tuple[float, str]:
        """
        يحسب خطورة الأعراض السريرية
        - score: 0-1
        - explanation: شرح السبب
        """
        score      = 0.0
        reasons    = []

        # أعراض طارئة فورية
        emergency = [s for s in symptoms if s in EMERGENCY_SYMPTOMS]
        if emergency:
            score = 1.0
            reasons.append(f"أعراض طارئة: {', '.join(emergency)}")
            return score, " | ".join(reasons)

        # مستوى الألم
        pain_score = pain_level / 10
        score     += pain_score * 0.4
        if pain_level >= 8:
            reasons.append(f"ألم شديد ({pain_level}/10)")

        # أعراض عالية الخطورة
        high_risk = [s for s in symptoms if s in HIGH_RISK_SYMPTOMS]
        if high_risk:
            score += 0.3
            reasons.append(f"أعراض خطيرة: {', '.join(high_risk)}")

        # عدد الأعراض
        if len(symptoms) >= 4:
            score += 0.2
            reasons.append(f"أعراض متعددة ({len(symptoms)})")

        # أمراض مزمنة
        if len(chronic_diseases) >= 2:
            score += 0.1
            reasons.append(f"أمراض مزمنة: {', '.join(chronic_diseases)}")

        score = min(score, 1.0)
        explanation = " | ".join(reasons) if reasons else "أعراض خفيفة"
        return score, explanation

    # ── 3. INTERACTION CHECK ─────────────────────
    def _assess_interactions(self, medications: list) -> tuple[float, list]:
        """
        يفحص تعارضات الأدوية
        - score: خطورة التعارض (0-1)
        - alerts: قائمة التحذيرات
        """
        if not medications or len(medications) < 2:
            return 0.0, []

        alerts = []
        meds_lower = [m.lower().strip() for m in medications]

        for conflict in self.conflicts:
            drug_a = conflict["drug_a"].lower()
            drug_b = conflict["drug_b"].lower()
            if drug_a in meds_lower and drug_b in meds_lower:
                alerts.append({
                    "drug_a":    conflict["drug_a"],
                    "drug_b":    conflict["drug_b"],
                    "severity":  conflict["severity"],
                    "effect_ar": conflict["effect_ar"],
                    "recommendation": conflict.get("recommendation_ar","")
                })

        if not alerts:
            return 0.0, []

        # Score بناءً على الخطورة
        score_map = {"high": 1.0, "medium": 0.5, "low": 0.2}
        max_score = max(score_map.get(a["severity"], 0) for a in alerts)
        return max_score, alerts

    # ── FINAL DECISION ───────────────────────────
    def decide(
        self,
        features:         dict,
        symptoms:         list = [],
        pain_level:       int  = 0,
        chronic_diseases: list = [],
        medications:      list = []
    ) -> TriageResult:
        """
        القرار النهائي بناءً على MCDM
        """
        # 1. Physiological
        phys_score, diabetes, conf = self._assess_physiological(features)

        # 2. Clinical
        clin_score, clin_explanation = self._assess_clinical(
            symptoms, pain_level, chronic_diseases
        )

        # 3. Interaction
        inter_score, alerts = self._assess_interactions(medications)

        # Final Score
        final_score = (
            phys_score  * WEIGHTS["physiological"] +
            clin_score  * WEIGHTS["clinical"] +
            inter_score * WEIGHTS["interaction"]
        )
        final_score = round(final_score, 3)

        # تحديد مستوى الفرز
        emergency_symp = any(s in EMERGENCY_SYMPTOMS for s in symptoms)
        high_inter     = any(a["severity"] == "high" for a in alerts)

        if emergency_symp or final_score >= 0.75:
            level, label = "طارئ", 2
            action = "اتصل بالإسعاف فوراً"
            specialty = "طوارئ"
        elif final_score >= 0.45 or high_inter:
            level, label = "عاجل", 1
            action = "كشف طبيب خلال ساعات"
            specialty = "باطنة" if diabetes else "عيادة عامة"
        else:
            level, label = "غير عاجل", 0
            action = "راحة + متابعة عيادة"
            specialty = "عيادة عامة"

        explanation = (
            f"Score={final_score} | "
            f"Physiological={round(phys_score,2)} | "
            f"Clinical={round(clin_score,2)} | "
            f"Interaction={round(inter_score,2)} | "
            f"{clin_explanation}"
        )

        logger.info(
            f"Decision: {level} | score={final_score} | "
            f"diabetes={diabetes} | alerts={len(alerts)}"
        )

        return TriageResult(
            triage_level        = level,
            triage_label        = label,
            final_score         = final_score,
            diabetes            = diabetes,
            diabetes_confidence = round(conf, 3),
            diagnosis           = "سكري" if diabetes else "سليم",
            recommended_action  = action,
            specialty           = specialty,
            drug_alerts         = alerts,
            explanation         = explanation,
            scores = {
                "physiological": round(phys_score,  3),
                "clinical":      round(clin_score,  3),
                "interaction":   round(inter_score, 3),
                "final":         final_score
            }
        )
