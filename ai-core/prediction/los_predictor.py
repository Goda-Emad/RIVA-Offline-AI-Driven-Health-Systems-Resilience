"""
los_predictor.py
================
RIVA Health Platform — Length of Stay Predictor
------------------------------------------------
يتوقع مدة إقامة المريض في المستشفى.

النموذج: XGBoost مدرّب على MIMIC-III
الدقة:   MAE 3.23 يوم (من prediction_goals.md)
الملف:   ai-core/models/los/los_final_xgb_tuned_20260317_174240.pkl

التحسينات على الكود الأصلي:
    1. ربط مع unified_predictor  — نفس paths + numpy inference
    2. ربط مع explanation_generator — explain_los() مباشرة
    3. ربط مع orchestrator       — target_page = 14_los_dashboard.html
    4. ربط مع confidence_scorer  — confidence يغذّي الـ scorer
    5. MAE-based confidence interval بدل الحسبة اليدوية
    6. numpy بدل reshape — متسق مع باقي المنظومة
    7. Graceful fallback — rule-based لو الموديل مش موجود
    8. Platt-consistent thresholds — متسقة مع unified_predictor

Author : GODA EMAD
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("riva.prediction.los_predictor")

# ─── Paths ───────────────────────────────────────────────────────────────────

_BASE       = Path(__file__).parent.parent
_MODEL_PATH = _BASE / "models/los/los_final_xgb_tuned_20260317_174240.pkl"
_FEATS_PATH = _BASE / "models/los/feature_names_los_20260317_174240.json"

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_LOS      = 30.0
MODEL_MAE    = 3.23    # from training artifacts
LOS_LONG     = 10.0
LOS_MEDIUM   = 5.0
LOS_SHORT    = 3.0

# ─── Category metadata (متسق مع unified_predictor) ───────────────────────────

_CATEGORY_META: dict[str, dict] = {
    "long": {
        "ar":          "طويلة جداً",
        "color":       "#ef4444",
        "target_page": "14_los_dashboard.html",
        "action":      "تجهيز جناح طويل الإقامة + متابعة يومية",
    },
    "medium": {
        "ar":          "متوسطة",
        "color":       "#f59e0b",
        "target_page": "14_los_dashboard.html",
        "action":      "تحضير خطة خروج منظمة + تثقيف المريض",
    },
    "short": {
        "ar":          "قصيرة",
        "color":       "#10b981",
        "target_page": "05_history.html",
        "action":      "تجهيز تقرير الخروج + موعد متابعة خارجي",
    },
}

# ─── Default features (matches MIMIC-III schema) ─────────────────────────────

_DEFAULT_FEATURES = [
    "CHF", "ARRHYTHMIA", "VALVULAR", "PULMONARY", "PVD",
    "HYPERTENSION", "PARALYSIS", "NEUROLOGICAL", "HYPOTHYROID",
    "RENAL", "LIVER", "ULCER", "AIDS",
    "lab_count", "lab_mean", "had_outpatient_labs",
    "heart_rate_mean", "sbp_mean", "dbp_mean",
    "resp_rate_mean", "temperature_mean", "spo2_mean",
]

# ─── Rule-based fallback ─────────────────────────────────────────────────────

_FALLBACK_WEIGHTS: dict[str, float] = {
    "CHF":         2.0,
    "RENAL":       2.5,
    "NEUROLOGICAL":2.8,
    "LIVER":       2.2,
    "AIDS":        3.0,
    "PARALYSIS":   2.5,
    "ARRHYTHMIA":  1.5,
    "HYPERTENSION":0.8,
    "PULMONARY":   1.8,
}


def _fallback_predict(features: dict) -> float:
    """Rule-based LOS estimate when model is unavailable."""
    base = 3.5
    for condition, weight in _FALLBACK_WEIGHTS.items():
        if features.get(condition, 0):
            base += weight
    if features.get("lab_count", 0) > 100:
        base += 1.5
    return min(round(base, 1), MAX_LOS)


# ─── Core predictor ───────────────────────────────────────────────────────────

class LOSPredictor:
    """
    Length of Stay predictor integrated with RIVA.

    Integrated with:
        - unified_predictor.py       : same model path + numpy inference
        - explanation_generator.py   : explain_los() for Arabic explanation
        - orchestrator.py            : target_page routing
        - confidence_scorer.py       : confidence feeds scorer
        - 14_los_dashboard.html      : prediction display
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model        = None
        self._feature_names= _DEFAULT_FEATURES
        self._model_mae    = MODEL_MAE

        self._try_load(model_path or _MODEL_PATH)

    def _try_load(self, path: Path) -> bool:
        try:
            import joblib, json
            self._model = joblib.load(str(path))

            feat_path = _FEATS_PATH
            if feat_path.exists():
                with open(feat_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._feature_names = data.get("features", _DEFAULT_FEATURES)
                self._model_mae     = data.get("mae", MODEL_MAE)

            log.info(
                "[LOSPredictor] loaded  features=%d  MAE=%.2f",
                len(self._feature_names), self._model_mae,
            )
            return True
        except Exception as e:
            log.warning("[LOSPredictor] model load failed: %s — using fallback", e)
            return False

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        features:    dict,
        session_id:  Optional[str] = None,
        for_doctor:  bool = False,
    ) -> dict:
        """
        Predicts Length of Stay.

        Args:
            features   : clinical features dict (MIMIC-III schema)
            session_id : for logging
            for_doctor : include technical details

        Returns:
            {
                days                : predicted LOS (float)
                category            : "short" | "medium" | "long"
                category_ar         : بالعربي
                color               : RIVA UI hex
                confidence_interval : {lower, upper}
                confidence          : 0.0-1.0
                explanation         : Arabic summary
                reasons             : contributing factors
                target_page         : من الـ 17 صفحة
                action              : الخطوة التالية
                model_mae           : MAE for display
                shap_ready          : features for explainability.py
                offline             : True
            }
        """
        # ── Inference ──────────────────────────────────────────────────────
        if self.is_loaded():
            try:
                X = np.array(
                    [[self._safe_val(features.get(f)) for f in self._feature_names]],
                    dtype=np.float32,
                )
                log_days = float(self._model.predict(X)[0])
                days     = float(np.expm1(log_days))   # model outputs log1p(days)
            except Exception as e:
                log.warning("[LOSPredictor] inference error: %s — fallback", e)
                days = _fallback_predict(features)
        else:
            days = _fallback_predict(features)

        days = min(round(days, 1), MAX_LOS)

        # ── Category ───────────────────────────────────────────────────────
        category = (
            "long"   if days >= LOS_LONG   else
            "medium" if days >= LOS_MEDIUM else
            "short"
        )
        meta = _CATEGORY_META[category]

        # ── MAE-based confidence interval ──────────────────────────────────
        ci = {
            "lower": round(max(0.0, days - self._model_mae), 2),
            "upper": round(days + self._model_mae, 2),
        }

        # ── Confidence ─────────────────────────────────────────────────────
        feature_count = sum(1 for v in features.values() if v)
        confidence    = round(min(0.70 + feature_count * 0.01, 0.95), 2)

        # ── Contributing reasons ────────────────────────────────────────────
        reasons = self._extract_reasons(features)

        # ── Explanation ────────────────────────────────────────────────────
        try:
            from .explanation_generator import explain_los
            exp = explain_los(
                days       = days,
                features   = features,
                los_mae    = self._model_mae,
                session_id = session_id,
                for_doctor = for_doctor,
            )
            explanation = exp["summary"]
        except Exception:
            explanation = (
                f"مدة الإقامة المتوقعة {days} يوم — إقامة {meta['ar']}"
                + (f" — بسبب: {', '.join(reasons[:2])}" if reasons else "")
            )

        # ── Doctor technical note ───────────────────────────────────────────
        doctor_note = None
        if for_doctor:
            doctor_note = (
                f"Predicted: {days}d | MAE={self._model_mae}d | "
                f"CI=[{ci['lower']},{ci['upper']}] | "
                f"Model: {'XGBoost' if self.is_loaded() else 'rule-based fallback'}"
            )

        log.info(
            "[LOSPredictor] session=%s  days=%.1f  cat=%s  conf=%.2f",
            session_id, days, category, confidence,
        )

        return {
            "days":                days,
            "category":            category,
            "category_ar":         meta["ar"],
            "color":               meta["color"],
            "confidence_interval": ci,
            "confidence":          confidence,
            "explanation":         explanation,
            "reasons":             reasons,
            "target_page":         meta["target_page"],
            "action":              meta["action"],
            "model_mae":           self._model_mae,
            "model_used":          "xgboost" if self.is_loaded() else "fallback",
            "doctor_note":         doctor_note,
            "shap_ready":          {f: self._safe_val(features.get(f))
                                    for f in self._feature_names},
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "offline":             True,
        }

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(self, patients: list[dict]) -> list[dict]:
        """
        Vectorised batch prediction — sorted by days descending.
        Used in 14_los_dashboard.html for ward-level LOS planning.
        """
        if not patients:
            return []

        if self.is_loaded():
            try:
                X = np.array(
                    [[self._safe_val(p.get(f)) for f in self._feature_names]
                     for p in patients],
                    dtype=np.float32,
                )
                log_days_arr = self._model.predict(X)
                days_arr     = np.expm1(log_days_arr)
            except Exception as e:
                log.warning("[LOSPredictor] batch error: %s — fallback", e)
                days_arr = np.array([_fallback_predict(p) for p in patients])
        else:
            days_arr = np.array([_fallback_predict(p) for p in patients])

        results = []
        for i, (p, raw_days) in enumerate(zip(patients, days_arr)):
            days     = min(round(float(raw_days), 1), MAX_LOS)
            category = ("long" if days >= LOS_LONG else
                        "medium" if days >= LOS_MEDIUM else "short")
            meta     = _CATEGORY_META[category]
            results.append({
                "patient_id":  p.get("patient_id", f"patient_{i}"),
                "days":        days,
                "category":    category,
                "category_ar": meta["ar"],
                "color":       meta["color"],
                "target_page": meta["target_page"],
            })

        return sorted(results, key=lambda x: x["days"], reverse=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_val(val) -> float:
        if val is None:
            return 0.0
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _extract_reasons(self, features: dict) -> list[str]:
        _REASON_LABELS: dict[str, str] = {
            "CHF":         "قصور القلب",
            "RENAL":       "مشاكل كلوية",
            "NEUROLOGICAL":"أمراض عصبية",
            "LIVER":       "مشاكل الكبد",
            "AIDS":        "نقص المناعة",
            "PARALYSIS":   "شلل",
            "ARRHYTHMIA":  "اضطراب النبض",
            "PULMONARY":   "أمراض رئوية",
        }
        reasons = [
            _REASON_LABELS[k]
            for k in _REASON_LABELS
            if features.get(k, 0)
        ]
        if features.get("lab_count", 0) > 100:
            reasons.append("كثرة التحاليل المخبرية")
        return reasons[:4]

    @property
    def status(self) -> dict:
        return {
            "loaded":      self.is_loaded(),
            "model_path":  str(_MODEL_PATH),
            "features":    len(self._feature_names),
            "model_mae":   self._model_mae,
            "max_los":     MAX_LOS,
            "thresholds":  {"long": LOS_LONG, "medium": LOS_MEDIUM, "short": LOS_SHORT},
            "target_pages":  {k: v["target_page"] for k, v in _CATEGORY_META.items()},
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_predictor = LOSPredictor()


# ─── Public API ───────────────────────────────────────────────────────────────

def predict_los(
    features:   dict,
    session_id: Optional[str] = None,
    for_doctor: bool = False,
) -> dict:
    """
    Main entry point.

    Usage in orchestrator.py (intent=LOS):
        from .los_predictor import predict_los

        result = predict_los(
            features   = patient_clinical_features,
            session_id = session_id,
        )
        target_page = result["target_page"]  # 14_los_dashboard.html

        # Feed into confidence_scorer
        score = compute_confidence(
            intent            = "LOS",
            model_logit_score = result["confidence"],
        )

        # Feed into explainability
        explanation = explain(
            model_type = ModelType.LOS,
            features   = result["shap_ready"],
            prediction = result["category"],
            confidence = result["confidence"],
        )
    """
    return _predictor.predict(features, session_id=session_id, for_doctor=for_doctor)


def predict_los_batch(patients: list[dict]) -> list[dict]:
    """Batch LOS prediction sorted by days (longest first)."""
    return _predictor.predict_batch(patients)


def get_status() -> dict:
    return _predictor.status
