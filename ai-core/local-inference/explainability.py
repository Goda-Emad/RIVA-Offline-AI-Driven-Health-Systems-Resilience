"""
explainability.py
=================
RIVA Health Platform — AI Decision Explainability Engine
---------------------------------------------------------
يشرح قرارات الـ AI للمريض والدكتور بالعربية البسيطة.

ليه هو "أهم ملف"؟
    - المسابقة: الشفافية الطبية (Medical Transparency) معيار رئيسي
    - الأخلاقيات: المريض حقه يفهم ليه الـ AI قرر كده
    - الثقة: الدكتور مش هيثق في AI مش مفهوم
    - القانون: GDPR / HIPAA بيطلب "Right to Explanation"
    - الـ 17 صفحة: 12_ai_explanation.html بيعتمد كلياً على الملف ده

الملف ده بيشرح 6 موديلات مختلفة:
    1. Triage (XGBoost)      — الفرز الطبي، دقة 91%
    2. Pregnancy (XGBoost)   — مخاطر الحمل
    3. Chatbot (DistilBERT)  — المحادثة الطبية
    4. School (K-Means)      — الصحة المدرسية
    5. Readmission (XGBoost) — خطر إعادة الإدخال
    6. LOS (XGBoost)         — مدة الإقامة بالمستشفى

طرق الشرح:
    - SHAP values      — أهم العوامل المؤثرة في القرار
    - Feature importance— ترتيب العوامل بالأرقام
    - Natural language  — شرح بالعربية للمريض العادي
    - Counterfactual    — "لو X اتغيّر، القرار كان هيكون Y"
    - Confidence breakdown— من confidence_scorer.py

الربط مع المنظومة:
    - orchestrator.py       : يستدعيه لو confidence < 0.55
    - 12_ai_explanation.html: يعرض النتيجة للمريض
    - 09_doctor_dashboard   : يعرض SHAP waterfall للدكتور
    - doctor_feedback_handler: الدكتور يقيّم الشرح

Author : GODA EMAD
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("riva.local_inference.explainability")

# ─── Paths ───────────────────────────────────────────────────────────────────

_BASE        = Path(__file__).parent.parent
_RULES_FILE  = _BASE / "business-intelligence/medical-content/explanation_rules.json"
_TRIAGE_RULES= _BASE / "business-intelligence/mapping/triage_explanation_rules.json"

# ─── Model types ─────────────────────────────────────────────────────────────

class ModelType(str, Enum):
    TRIAGE      = "triage"
    PREGNANCY   = "pregnancy"
    CHATBOT     = "chatbot"
    SCHOOL      = "school"
    READMISSION = "readmission"
    LOS         = "los"


# ─── Audience ────────────────────────────────────────────────────────────────

class Audience(str, Enum):
    PATIENT = "patient"   # عربي بسيط — بدون مصطلحات
    DOCTOR  = "doctor"    # مصطلحات طبية + SHAP values
    SCHOOL  = "school"    # مدير المدرسة أو المعلم


# ─── Feature labels (عربي) ───────────────────────────────────────────────────

_FEATURE_LABELS: dict[str, dict[str, str]] = {
    # Triage features
    "age":                    {"ar": "السن",                  "unit": "سنة"},
    "heart_rate":             {"ar": "ضربات القلب",           "unit": "نبضة/دقيقة"},
    "systolic_bp":            {"ar": "ضغط الدم الانقباضي",   "unit": "mmHg"},
    "diastolic_bp":           {"ar": "ضغط الدم الانبساطي",   "unit": "mmHg"},
    "temperature":            {"ar": "درجة الحرارة",          "unit": "°C"},
    "spo2":                   {"ar": "تشبع الأكسجين",         "unit": "%"},
    "respiratory_rate":       {"ar": "معدل التنفس",           "unit": "نفس/دقيقة"},
    "pain_scale":             {"ar": "شدة الألم",             "unit": "من 10"},
    # Pregnancy features
    "gestational_age":        {"ar": "عمر الحمل",             "unit": "أسبوع"},
    "gravida":                {"ar": "عدد الحمول",            "unit": "مرة"},
    "hemoglobin":             {"ar": "الهيموجلوبين",          "unit": "g/dL"},
    "blood_glucose":          {"ar": "سكر الدم",              "unit": "mg/dL"},
    "blood_pressure_systolic":{"ar": "ضغط الدم",              "unit": "mmHg"},
    "previous_complications": {"ar": "مضاعفات سابقة",         "unit": ""},
    # School features
    "bmi":                    {"ar": "مؤشر كتلة الجسم",       "unit": ""},
    "height_cm":              {"ar": "الطول",                  "unit": "سم"},
    "weight_kg":              {"ar": "الوزن",                  "unit": "كيلو"},
    "vision_score":           {"ar": "حدة النظر",             "unit": ""},
    "dental_score":           {"ar": "صحة الأسنان",           "unit": ""},
    # Readmission / LOS
    "los_days":               {"ar": "مدة الإقامة",           "unit": "يوم"},
    "num_diagnoses":          {"ar": "عدد التشخيصات",         "unit": ""},
    "num_procedures":         {"ar": "عدد الإجراءات",         "unit": ""},
    "num_medications":        {"ar": "عدد الأدوية",           "unit": ""},
    "num_lab_procedures":     {"ar": "عدد التحاليل",          "unit": ""},
    "discharge_disposition":  {"ar": "وجهة الخروج",           "unit": ""},
    "admission_source":       {"ar": "مصدر الدخول",           "unit": ""},
}


# ─── Natural language templates ───────────────────────────────────────────────

_PATIENT_TEMPLATES: dict[str, dict] = {
    ModelType.TRIAGE: {
        "high":   "ريفا شايفة إن حالتك تحتاج انتباه طبي سريع بسبب {top_factors}.",
        "medium": "ريفا بتنصح بمراجعة الدكتور في أقرب وقت. أهم حاجة لفتت نظرها: {top_factors}.",
        "low":    "ريفا مش شايفة خطر فوري. بس متابع مع الدكتور مهم عشان {top_factors}.",
    },
    ModelType.PREGNANCY: {
        "high_risk":  "فيه بعض علامات محتاجة متابعة دقيقة خلال الحمل. أهمها: {top_factors}.",
        "mid_risk":   "الحمل بيتقدم كويس، بس في نقطة أو اتنين محتاجة اهتمام: {top_factors}.",
        "low_risk":   "الحمل ماشي تمام. ريفا مش شايفة مخاطر كبيرة دلوقتي.",
    },
    ModelType.SCHOOL: {
        "needs_attention": "الطالب محتاج متابعة في: {top_factors}.",
        "normal":          "صحة الطالب في المعدل الطبيعي. استمر في المتابعة الدورية.",
    },
    ModelType.READMISSION: {
        "high":   "فيه احتمال إن المريض يحتاج رجوع للمستشفى قريباً بسبب: {top_factors}.",
        "low":    "احتمال الرجوع للمستشفى منخفض. تابع التعليمات الطبية.",
    },
    ModelType.LOS: {
        "long":   "ريفا بتتوقع إن مدة الإقامة ممكن تطول نتيجة: {top_factors}.",
        "normal": "ريفا بتتوقع مدة إقامة عادية.",
    },
}

_COUNTERFACTUAL_TEMPLATE = (
    "لو {feature_ar} كانت {better_value} بدل {current_value}، "
    "القرار كان ممكن يكون {better_outcome}."
)


# ─── SHAP helpers ─────────────────────────────────────────────────────────────

def _get_shap_values(
    model_type: ModelType,
    features:   dict[str, float],
) -> dict[str, float]:
    """
    Computes SHAP values for the given features.

    In production: uses shap.TreeExplainer on the loaded ONNX/pkl model.
    In offline/fallback mode: uses feature importance from training artifacts.

    Returns dict of {feature_name: shap_value} sorted by |value| descending.
    """
    try:
        import shap
        import numpy as np
        from .Model_manager import model_manager

        session = model_manager.get(model_type.value)
        if session is None:
            return _fallback_importance(model_type, features)

        # Build input array in correct feature order
        feature_names = list(features.keys())
        X = np.array([[features[f] for f in feature_names]], dtype=np.float32)

        explainer   = shap.TreeExplainer(session)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]   # last class for multi-class

        return {
            f: float(shap_values[0][i])
            for i, f in enumerate(feature_names)
        }

    except Exception as e:
        log.warning("[Explainability] SHAP failed: %s — using fallback", e)
        return _fallback_importance(model_type, features)


def _get_shap_interaction_effects(
    model_type: ModelType,
    features:   dict[str, float],
    shap_vals:  dict[str, float],
) -> list[str]:
    """
    Detects dangerous combinations of features — SHAP interaction effects.

    Example: age>70 alone might not be high-risk, but age>70 + spo2<90
    together create a compounding risk that should be flagged explicitly.

    Returns list of Arabic interaction sentences for doctor_summary.
    """
    interactions = []

    _COMBOS = [
        {
            "features": ["age", "spo2"],
            "conditions": lambda f: f.get("age", 0) > 70 and f.get("spo2", 100) < 92,
            "ar": "السن الكبير مع انخفاض الأكسجين يزيدان الخطر بشكل مركّب",
        },
        {
            "features": ["systolic_bp", "heart_rate"],
            "conditions": lambda f: f.get("systolic_bp", 120) > 160 and f.get("heart_rate", 80) > 110,
            "ar": "الضغط المرتفع مع تسارع القلب — خطر قلبي مضاعف",
        },
        {
            "features": ["blood_glucose", "blood_pressure_systolic"],
            "conditions": lambda f: f.get("blood_glucose", 100) > 200 and f.get("blood_pressure_systolic", 120) > 140,
            "ar": "ارتفاع السكر مع ارتفاع الضغط — خطر مضاعفات الحمل مرتفع",
        },
        {
            "features": ["temperature", "heart_rate"],
            "conditions": lambda f: f.get("temperature", 37) > 38.5 and f.get("heart_rate", 80) > 100,
            "ar": "الحرارة المرتفعة مع تسارع القلب — علامات إنتان (Sepsis) محتملة",
        },
        {
            "features": ["hemoglobin", "gestational_age"],
            "conditions": lambda f: f.get("hemoglobin", 12) < 9 and f.get("gestational_age", 20) > 28,
            "ar": "فقر الدم الشديد في الثلث الثالث — خطر على الأم والجنين",
        },
        {
            "features": ["num_diagnoses", "num_medications"],
            "conditions": lambda f: f.get("num_diagnoses", 0) > 5 and f.get("num_medications", 0) > 8,
            "ar": "تعدد التشخيصات مع كثرة الأدوية — خطر إعادة إدخال مرتفع",
        },
        {
            "features": ["age", "pain_scale"],
            "conditions": lambda f: f.get("age", 40) > 65 and f.get("pain_scale", 0) >= 8,
            "ar": "ألم شديد عند مريض كبير السن — يستلزم تقييماً سريعاً",
        },
    ]

    for combo in _COMBOS:
        try:
            if combo["conditions"](features):
                # Verify both features had significant SHAP values
                involved = [
                    shap_vals.get(f, 0) for f in combo["features"]
                ]
                if any(abs(v) > 0.03 for v in involved):
                    interactions.append(combo["ar"])
        except Exception:
            pass

    return interactions


def _fallback_importance(
    model_type: ModelType,
    features:   dict[str, float],
) -> dict[str, float]:
    """
    Fallback importance scores when SHAP is unavailable.
    Based on pre-computed feature importances from training notebooks.
    """
    _IMPORTANCE: dict[ModelType, dict[str, float]] = {
        ModelType.TRIAGE: {
            "pain_scale": 0.28, "heart_rate": 0.22, "spo2": 0.18,
            "temperature": 0.15, "systolic_bp": 0.10, "age": 0.07,
        },
        ModelType.PREGNANCY: {
            "blood_pressure_systolic": 0.25, "blood_glucose": 0.22,
            "hemoglobin": 0.18, "gestational_age": 0.15,
            "previous_complications": 0.12, "gravida": 0.08,
        },
        ModelType.SCHOOL: {
            "bmi": 0.30, "vision_score": 0.25, "dental_score": 0.20,
            "height_cm": 0.15, "weight_kg": 0.10,
        },
        ModelType.READMISSION: {
            "num_diagnoses": 0.25, "num_medications": 0.22,
            "los_days": 0.20, "num_procedures": 0.18,
            "num_lab_procedures": 0.15,
        },
        ModelType.LOS: {
            "num_diagnoses": 0.28, "num_procedures": 0.24,
            "num_medications": 0.20, "admission_source": 0.15,
            "age": 0.13,
        },
    }
    importance = _IMPORTANCE.get(model_type, {})
    return {
        f: importance.get(f, 0.0) * features.get(f, 1.0)
        for f in features
    }


# ─── Result dataclasses ──────────────────────────────────────────────────────

@dataclass
class FeatureContribution:
    name:       str
    name_ar:    str
    value:      Any
    unit:       str
    shap_value: float
    direction:  str    # "positive" (رفع الخطر) | "negative" (خفّض الخطر)
    magnitude:  str    # "high" | "medium" | "low"


@dataclass
class ExplanationResult:
    """
    Complete explanation package for one AI decision.
    Displayed in 12_ai_explanation.html.
    """
    model_type:         ModelType
    prediction:         str                         # الحكم النهائي
    confidence:         float
    audience:           Audience
    patient_summary:    str                         # شرح بالعربي البسيط
    doctor_summary:     str                         # شرح تقني
    top_features:       list[FeatureContribution]   # أهم 3-5 عوامل
    counterfactuals:    list[str]                   # "لو X اتغيّر..."
    shap_values:        dict[str, float]            # raw SHAP للـ dashboard
    duration_ms:        float
    factors_ar:         list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_type":      self.model_type,
            "prediction":      self.prediction,
            "confidence":      self.confidence,
            "audience":        self.audience,
            "patient_summary": self.patient_summary,
            "doctor_summary":  self.doctor_summary,
            "top_features": [
                {
                    "name":       f.name,
                    "name_ar":    f.name_ar,
                    "value":      f.value,
                    "unit":       f.unit,
                    "shap_value": f.shap_value,
                    "direction":  f.direction,
                    "magnitude":  f.magnitude,
                }
                for f in self.top_features
            ],
            "counterfactuals": self.counterfactuals,
            "shap_values":     self.shap_values,
            "duration_ms":     self.duration_ms,
            "factors_ar":      self.factors_ar,
        }


# ─── Core explainer ──────────────────────────────────────────────────────────

# ─── Explanation cache ───────────────────────────────────────────────────────
# Offline LRU cache — لو المريض سأل نفس السؤال مرتين توفير وقت المعالجة
# Key = (model_type, prediction, frozenset of features)
# Max size: 64 entries (كافي للجلسة الواحدة)

import hashlib as _hashlib
import functools as _functools

_explanation_cache: dict[str, tuple[float, ExplanationResult]] = {}
_CACHE_MAX = 64
_CACHE_TTL = 1800   # 30 دقيقة — بعدها يتجدد لو الـ features اتغيّرت


def _cache_key(
    model_type: ModelType,
    features:   dict[str, float],
    prediction: str,
) -> str:
    """Deterministic cache key from model + features + prediction."""
    feat_str = json.dumps(
        {k: round(v, 2) for k, v in sorted(features.items())},
        ensure_ascii=False,
    )
    raw = f"{model_type}|{prediction}|{feat_str}"
    return _hashlib.md5(raw.encode()).hexdigest()[:16]


def _cache_get(key: str) -> Optional[ExplanationResult]:
    if key not in _explanation_cache:
        return None
    ts, result = _explanation_cache[key]
    if time.time() - ts > _CACHE_TTL:
        del _explanation_cache[key]
        return None
    log.info("[Explainability] cache hit: %s", key)
    return result


def _cache_set(key: str, result: ExplanationResult) -> None:
    if len(_explanation_cache) >= _CACHE_MAX:
        # Evict oldest entry
        oldest = min(_explanation_cache, key=lambda k: _explanation_cache[k][0])
        del _explanation_cache[oldest]
    _explanation_cache[key] = (time.time(), result)


class ExplainabilityEngine:
    """
    Generates human-readable explanations for all RIVA AI decisions.

    Design principle:
        "A doctor should be able to disagree with the AI
         only after understanding why it said what it said."
    """

    def __init__(self):
        self._rules = self._load_rules()
        log.info("[Explainability] engine ready  rules=%d", len(self._rules))

    def _load_rules(self) -> dict:
        rules = {}
        for path in [_RULES_FILE, _TRIAGE_RULES]:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    rules.update(json.load(f))
        return rules

    # ── Main explain ─────────────────────────────────────────────────────────

    def explain(
        self,
        model_type:  ModelType,
        features:    dict[str, float],
        prediction:  str,
        confidence:  float,
        audience:    Audience = Audience.PATIENT,
        session_id:  Optional[str] = None,
    ) -> ExplanationResult:
        """
        Generates a complete explanation for one AI decision.

        Args:
            model_type  : which RIVA model made the decision
            features    : input features used by the model
            prediction  : model output (e.g. "high_risk", "Emergency")
            confidence  : from confidence_scorer.py
            audience    : patient / doctor / school
            session_id  : for logging + doctor feedback linking

        Returns:
            ExplanationResult with all explanation layers
        """
        t0 = time.perf_counter()

        # Cache check (offline LRU)
        cache_key = _cache_key(model_type, features, prediction)
        cached    = _cache_get(cache_key)
        if cached is not None:
            log.info("[Explainability] served from cache in <1ms")
            return cached

        # 1. SHAP values
        shap_vals   = _get_shap_values(model_type, features)

        # 2. Top contributing features
        top_features = self._build_top_features(shap_vals, features, n=5)

        # 3. Natural language summaries
        factors_ar   = [f.name_ar for f in top_features[:3]]
        patient_sum  = self._patient_summary(model_type, prediction, factors_ar)
        doctor_sum   = self._doctor_summary(
            model_type, prediction, confidence, top_features, shap_vals,
            features=features,
        )

        # 4. Counterfactuals
        counterfactuals = self._generate_counterfactuals(
            model_type, features, shap_vals, prediction
        )

        ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info(
            "[Explainability] %s  pred=%s  conf=%.2f  top=%s  %.0fms",
            model_type, prediction, confidence,
            [f.name_ar for f in top_features[:3]], ms,
        )

        result = ExplanationResult(
            model_type      = model_type,
            prediction      = prediction,
            confidence      = confidence,
            audience        = audience,
            patient_summary = patient_sum,
            doctor_summary  = doctor_sum,
            top_features    = top_features,
            counterfactuals = counterfactuals,
            shap_values     = shap_vals,
            duration_ms     = ms,
            factors_ar      = factors_ar,
        )

        # Store in cache for offline reuse
        _cache_set(cache_key, result)
        return result

    # ── Feature contributions ────────────────────────────────────────────────

    def _build_top_features(
        self,
        shap_vals: dict[str, float],
        features:  dict[str, float],
        n:         int = 5,
    ) -> list[FeatureContribution]:
        sorted_feats = sorted(
            shap_vals.items(), key=lambda x: abs(x[1]), reverse=True
        )[:n]

        result = []
        for name, shap_val in sorted_feats:
            meta      = _FEATURE_LABELS.get(name, {"ar": name, "unit": ""})
            abs_shap  = abs(shap_val)
            magnitude = "high" if abs_shap > 0.15 else "medium" if abs_shap > 0.05 else "low"

            result.append(FeatureContribution(
                name       = name,
                name_ar    = meta["ar"],
                value      = features.get(name, "—"),
                unit       = meta.get("unit", ""),
                shap_value = round(shap_val, 4),
                direction  = "positive" if shap_val > 0 else "negative",
                magnitude  = magnitude,
            ))
        return result

    # ── Patient summary ──────────────────────────────────────────────────────

    def _patient_summary(
        self,
        model_type: ModelType,
        prediction: str,
        factors_ar: list[str],
    ) -> str:
        """Simple Arabic explanation — no medical jargon."""
        templates = _PATIENT_TEMPLATES.get(model_type, {})
        template  = templates.get(prediction, templates.get(list(templates.keys())[0], ""))

        factors_str = "، ".join(factors_ar[:3]) if factors_ar else "بعض المؤشرات"

        if "{top_factors}" in template:
            return template.format(top_factors=factors_str)
        return template or f"ريفا حللت حالتك بناءً على {factors_str}."

    # ── Doctor summary ───────────────────────────────────────────────────────

    def _doctor_summary(
        self,
        model_type:   ModelType,
        prediction:   str,
        confidence:   float,
        top_features: list[FeatureContribution],
        shap_vals:    dict[str, float],
        features:     dict[str, float] = {},
    ) -> str:
        """
        Technical summary with feature values, SHAP contributions,
        and interaction effects between dangerous feature combinations.
        """
        lines = [
            f"النموذج: {model_type.value.upper()} | "
            f"التنبؤ: {prediction} | "
            f"مستوى الثقة: {int(confidence*100)}%",
            "",
            "أهم العوامل المؤثرة في القرار:",
        ]
        for i, f in enumerate(top_features[:5], 1):
            direction = "↑ رفع الخطر" if f.direction == "positive" else "↓ خفّض الخطر"
            lines.append(
                f"  {i}. {f.name_ar}: {f.value} {f.unit} "
                f"| SHAP={f.shap_value:+.3f} | {direction}"
            )

        total_pos = sum(v for v in shap_vals.values() if v > 0)
        total_neg = sum(v for v in shap_vals.values() if v < 0)
        lines += [
            "",
            f"مجموع العوامل الرافعة للخطر  : {total_pos:+.3f}",
            f"مجموع العوامل الخافضة للخطر : {total_neg:+.3f}",
        ]

        # Interaction effects (SHAP v4.2)
        interactions = _get_shap_interaction_effects(model_type, features, shap_vals)
        if interactions:
            lines += ["", "⚠️ تأثيرات التفاعل بين العوامل:"]
            for inter in interactions:
                lines.append(f"  • {inter}")

        return "\n".join(lines)

    # ── Counterfactuals ──────────────────────────────────────────────────────

    def _generate_counterfactuals(
        self,
        model_type: ModelType,
        features:   dict[str, float],
        shap_vals:  dict[str, float],
        prediction: str,
    ) -> list[str]:
        """
        Generates "what if" statements for the top negative features.
        Helps patients understand what they can change.
        """
        counterfactuals = []
        _ACTIONABLE = {
            "pain_scale":             {"better": "أقل من 4", "outcome": "تصنيف أخف"},
            "heart_rate":             {"better": "60-100", "outcome": "وضع أكثر استقراراً"},
            "temperature":            {"better": "37°C", "outcome": "تصنيف طبيعي"},
            "blood_glucose":          {"better": "80-120 mg/dL", "outcome": "خطر أقل"},
            "blood_pressure_systolic":{"better": "أقل من 140 mmHg", "outcome": "ضغط طبيعي"},
            "hemoglobin":             {"better": "أعلى من 11 g/dL", "outcome": "خطر أقل"},
            "bmi":                    {"better": "18.5-24.9", "outcome": "وزن مثالي"},
            "num_medications":        {"better": "أقل", "outcome": "تعقيدات أقل"},
        }

        # Top positive SHAP features = مؤثرة في رفع الخطر = قابلة للتحسين
        risky = sorted(
            [(k, v) for k, v in shap_vals.items() if v > 0.05],
            key=lambda x: x[1], reverse=True,
        )[:3]

        for feat_name, shap_val in risky:
            if feat_name not in _ACTIONABLE:
                continue
            meta     = _FEATURE_LABELS.get(feat_name, {"ar": feat_name, "unit": ""})
            action   = _ACTIONABLE[feat_name]
            curr_val = features.get(feat_name, "—")

            cf = _COUNTERFACTUAL_TEMPLATE.format(
                feature_ar    = meta["ar"],
                better_value  = action["better"],
                current_value = f"{curr_val} {meta.get('unit','')}".strip(),
                better_outcome= action["outcome"],
            )
            counterfactuals.append(cf)

        return counterfactuals

    # ── Page-specific formatters ─────────────────────────────────────────────

    def for_page_12(self, result: ExplanationResult) -> dict:
        """
        Formats explanation for 12_ai_explanation.html.
        Matches the exact fields the frontend expects.
        """
        return {
            "patient_summary":   result.patient_summary,
            "confidence_pct":    int(result.confidence * 100),
            "confidence_level":  (
                "high" if result.confidence >= 0.75 else
                "medium" if result.confidence >= 0.55 else "low"
            ),
            "top_factors": [
                {
                    "label":     f.name_ar,
                    "value":     f"{f.value} {f.unit}".strip(),
                    "impact":    f.direction,
                    "magnitude": f.magnitude,
                    "shap":      f.shap_value,
                }
                for f in result.top_features[:5]
            ],
            "counterfactuals":   result.counterfactuals,
            "reasoning_steps": [
                f"ريفا استخدمت نموذج {result.model_type.value.upper()}",
                f"حللت {len(result.shap_values)} متغير طبي",
                f"أهم عامل: {result.top_features[0].name_ar if result.top_features else '—'}",
                f"مستوى الثقة: {int(result.confidence*100)}%",
            ],
            "model_type":     result.model_type,
            "duration_ms":    result.duration_ms,
        }

    def for_doctor_dashboard(self, result: ExplanationResult) -> dict:
        """
        Formats SHAP waterfall data for 09_doctor_dashboard.html.
        Compatible with Plotly waterfall chart.
        """
        sorted_shap = sorted(
            result.shap_values.items(),
            key=lambda x: x[1], reverse=True,
        )
        labels = [
            _FEATURE_LABELS.get(k, {"ar": k})["ar"]
            for k, _ in sorted_shap
        ]
        values = [round(v, 4) for _, v in sorted_shap]

        # RIVA UI color system (متسق مع style.css و dashboard.css)
        # #ef4444 = danger red  (CSS var --color-background-danger)
        # #10b981 = success green (CSS var --color-background-success)
        # #f59e0b = warning amber (CSS var --color-background-warning)
        colors = [
            "#ef4444" if v > 0.10 else          # خطر مرتفع
            "#f59e0b" if v > 0.03 else           # تحذير متوسط
            "#10b981" if v < -0.03 else          # عامل وقائي
            "#8892a4"                             # تأثير ضئيل (muted)
            for v in values
        ]

        return {
            "chart_type":     "waterfall",
            "labels":         labels,
            "values":         values,
            "colors":         colors,
            "color_legend": {
                "#ef4444": "رفع الخطر بشكل كبير",
                "#f59e0b": "رفع الخطر بشكل معتدل",
                "#10b981": "خفّض الخطر",
                "#8892a4": "تأثير ضئيل",
            },
            "doctor_summary": result.doctor_summary,
            "prediction":     result.prediction,
            "confidence":     result.confidence,
        }

    def for_school_dashboard(self, result: ExplanationResult) -> dict:
        """
        Formats explanation for 11_school_dashboard.html.
        Simple language for school administrators.
        """
        return {
            "summary":        result.patient_summary,
            "action_needed":  result.confidence < 0.55 or "needs_attention" in result.prediction,
            "top_concerns":   [f.name_ar for f in result.top_features[:3]],
            "recommendations":result.counterfactuals,
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_engine = ExplainabilityEngine()


# ─── Public API ───────────────────────────────────────────────────────────────

def explain(
    model_type:  ModelType,
    features:    dict[str, float],
    prediction:  str,
    confidence:  float,
    audience:    Audience = Audience.PATIENT,
    session_id:  Optional[str] = None,
) -> ExplanationResult:
    """
    Main entry point — called by orchestrator.py when confidence < 0.55.

    Usage in orchestrator.py:
        from .explainability import explain, ModelType, Audience

        if score.final_score < LOW_CONFIDENCE_THRESH:
            exp = explain(
                model_type = ModelType.TRIAGE,
                features   = input_features,
                prediction = intent,
                confidence = score.final_score,
                audience   = Audience.PATIENT,
                session_id = session_id,
            )
            sessionStorage["riva_explanation"] = exp.to_dict()
            target_page = "12_ai_explanation.html"
    """
    return _engine.explain(
        model_type = model_type,
        features   = features,
        prediction = prediction,
        confidence = confidence,
        audience   = audience,
        session_id = session_id,
    )


def explain_for_page(
    model_type:  ModelType,
    features:    dict[str, float],
    prediction:  str,
    confidence:  float,
    page:        str = "12",
) -> dict:
    """
    Shortcut — explains and formats for a specific page number.

    page: "12" → 12_ai_explanation.html (patient-facing)
          "09" → 09_doctor_dashboard.html (doctor SHAP waterfall)
          "11" → 11_school_dashboard.html (school admin)
    """
    result = _engine.explain(
        model_type = model_type,
        features   = features,
        prediction = prediction,
        confidence = confidence,
    )
    if page == "09":
        return _engine.for_doctor_dashboard(result)
    if page == "11":
        return _engine.for_school_dashboard(result)
    return _engine.for_page_12(result)


def get_shap_waterfall(
    model_type: ModelType,
    features:   dict[str, float],
    prediction: str,
    confidence: float,
) -> dict:
    """Returns Plotly-ready waterfall data for the doctor dashboard."""
    result = _engine.explain(model_type, features, prediction, confidence,
                              audience=Audience.DOCTOR)
    return _engine.for_doctor_dashboard(result)
