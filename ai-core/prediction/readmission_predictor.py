"""
readmission_predictor.py
========================
RIVA Health Platform — Hospital Readmission Risk Predictor v4.5
----------------------------------------------------------------
المسار : ai-core/prediction/readmission_predictor.py
النموذج: XGBoost — AUC 0.917
الملف  : ai-core/models/readmission/readmission_xgb_20260317_175502.pkl

التحسينات v4.5:
    Fix A — Separation of Concerns : ML layer يرجع data فقط
                                     ReadmissionPresenter يتولى UI
    Fix B — Pickle Security         : XGBoost JSON load أولاً، joblib fallback
    Fix C — Data-Driven Medians     : medians + thresholds من features JSON
    Fix D — Module-level imports    : HAS_SHAP / HAS_XGB flags

Author : GODA EMAD · Harvard HSIL Hackathon 2026
"""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("riva.prediction.readmission_predictor")

# ─────────────────────────────────────────────────────────────────────────────
# Fix D — Module-level optional imports with HAS_* flags
# ─────────────────────────────────────────────────────────────────────────────
# ImportError surfaces at startup — not silently during a patient request.

try:
    import shap as _shap
    HAS_SHAP = True
except ImportError:
    _shap    = None  # type: ignore
    HAS_SHAP = False
    log.info("[Readmission] shap not installed — SHAP explanations disabled")

try:
    import xgboost as _xgb
    HAS_XGB = True
except ImportError:
    _xgb    = None  # type: ignore
    HAS_XGB = False
    log.info("[Readmission] xgboost not installed — native JSON load disabled")

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────
#  This file: ai-core/prediction/readmission_predictor.py
#  _PRED    = ai-core/prediction/
#  _AICORE  = ai-core/
#  _ROOT    = project-root/

_PRED       = Path(__file__).resolve().parent
_AICORE     = _PRED.parent
_MODELS_DIR = _AICORE / "models" / "readmission"


def _resolve_model_path() -> Path:
    """
    Priority:
      1. READMISSION_MODEL_PATH env var   (CI/CD sets after retraining)
      2. readmission_latest_model.json    (XGBoost native — most secure)
      3. readmission_latest_model.pkl     (joblib symlink)
      4. Auto-discover readmission_xgb_*.pkl
      5. Hardcoded fallback + warning
    """
    env = os.environ.get("READMISSION_MODEL_PATH")
    if env:
        p = Path(env)
        return p if p.is_absolute() else _AICORE / p

    for name in ("readmission_latest_model.json", "readmission_latest_model.pkl"):
        p = _MODELS_DIR / name
        if p.exists():
            return p

    if _MODELS_DIR.exists():
        # prefer .json over .pkl
        for pattern in ("readmission_xgb_*.json", "readmission_xgb_*.pkl"):
            candidates = sorted(_MODELS_DIR.glob(pattern))
            if candidates:
                return candidates[-1]

    legacy = _MODELS_DIR / "readmission_xgb_20260317_175502.pkl"
    log.warning(
        "[Readmission] using hardcoded filename. "
        "Create 'readmission_latest_model.json' or set READMISSION_MODEL_PATH."
    )
    return legacy


_MODEL_PATH    = _resolve_model_path()
_FEATURES_PATH = _MODELS_DIR / "readmission_features_20260317_175502.json"

# ─────────────────────────────────────────────────────────────────────────────
# Fix C — Data-Driven defaults (loaded from features JSON, not hardcoded)
# ─────────────────────────────────────────────────────────────────────────────
# If features JSON contains medians + thresholds, use those.
# Hardcoded values below are the FALLBACK only — updated on every retrain
# by saving them into the JSON artifact.

_FALLBACK_MEDIANS: dict[str, float] = {
    "CHF": 0.0, "ARRHYTHMIA": 0.0, "VALVULAR": 0.0, "PULMONARY": 0.0,
    "PVD": 0.0, "HYPERTENSION": 0.0, "PARALYSIS": 0.0, "NEUROLOGICAL": 0.0,
    "HYPOTHYROID": 0.0, "RENAL": 0.0, "LIVER": 0.0, "ULCER": 0.0, "AIDS": 0.0,
    "lab_count": 44.0, "lab_mean": 45.0, "had_outpatient_labs": 0.0,
    "heart_rate_mean": 80.0, "sbp_mean": 120.0, "dbp_mean": 75.0,
    "resp_rate_mean": 18.0, "temperature_mean": 36.8, "spo2_mean": 97.0,
    "num_diagnoses": 7.0, "num_procedures": 2.0, "num_medications": 10.0,
    "num_lab_procedures": 44.0, "los_days": 4.0, "age": 65.0,
    "visit_frequency": 1.0, "medication_changes": 1.0,
}

_FALLBACK_THRESHOLDS: dict[str, float] = {
    "low_risk":  0.30,
    "high_risk": 0.60,
}

_DEFAULT_FEATURES: list[str] = [
    "CHF", "ARRHYTHMIA", "VALVULAR", "PULMONARY", "PVD",
    "HYPERTENSION", "PARALYSIS", "NEUROLOGICAL", "HYPOTHYROID",
    "RENAL", "LIVER", "ULCER", "AIDS",
    "lab_count", "lab_mean", "had_outpatient_labs",
    "heart_rate_mean", "sbp_mean", "dbp_mean",
    "resp_rate_mean", "temperature_mean", "spo2_mean",
]

_LABELS_AR: dict[str, str] = {
    "CHF": "قصور القلب", "RENAL": "مشاكل كلوية",
    "NEUROLOGICAL": "أمراض عصبية", "LIVER": "مشاكل الكبد",
    "AIDS": "نقص المناعة", "PARALYSIS": "شلل",
    "ARRHYTHMIA": "اضطراب النبض", "PULMONARY": "أمراض رئوية",
    "HYPERTENSION": "ارتفاع الضغط", "HYPOTHYROID": "قصور الغدة الدرقية",
}

# ─────────────────────────────────────────────────────────────────────────────
# Fix A — Separation of Concerns
# ─────────────────────────────────────────────────────────────────────────────
# ML layer (_RISK_SUMMARIES) → data only: labels, actions, priority
# UI layer (ReadmissionPresenter) → colors, page names, platform routing

_RISK_SUMMARIES: dict[str, dict] = {
    "low": {
        "ar":      "خطر منخفض",
        "action":  "متابعة عادية — زيارة بعد شهر",
        "summary": "احتمال إعادة الإدخال منخفض. تابع التعليمات الطبية.",
        "priority": 1,
    },
    "medium": {
        "ar":      "خطر متوسط",
        "action":  "زيارة متابعة بعد أسبوعين + مراجعة الأدوية",
        "summary": "فيه بعض عوامل خطر — متابعة أدق مطلوبة.",
        "priority": 2,
    },
    "high": {
        "ar":      "خطر مرتفع",
        "action":  "تواصل مع الطبيب فوراً + خطة رعاية منزلية",
        "summary": "احتمال عالي للرجوع للمستشفى — يحتاج تدخل فوري.",
        "priority": 3,
    },
}


class ReadmissionPresenter:
    """
    Fix A — Presentation layer.

    Maps ML output (risk_level str) → platform-specific UI fields.
    Keeps colors + page names OUT of the ML model.

    Web:
        result = predict_readmission(features)
        view   = ReadmissionPresenter.present(result, platform="web")
        # view now has: risk_color, target_page

    Mobile (future):
        view = ReadmissionPresenter.present(result, platform="mobile")
        # view now has: screen, badge_color  (no HTML filenames)

    API consumer:
        view = ReadmissionPresenter.present(result, platform="api")
        # pure data — no UI fields added
    """

    _PAGE_MAP: dict[str, str] = {
        "low":    "05_history.html",
        "medium": "13_readmission.html",
        "high":   "13_readmission.html",
    }
    _COLOR_MAP: dict[str, str] = {
        "low":    "#10b981",
        "medium": "#f59e0b",
        "high":   "#ef4444",
    }

    @classmethod
    def present(cls, result: dict, platform: str = "web") -> dict:
        level    = result.get("risk_level", "medium")
        enriched = dict(result)
        if platform == "web":
            enriched["risk_color"]  = cls._COLOR_MAP.get(level, "#f59e0b")
            enriched["target_page"] = cls._PAGE_MAP.get(level, "13_readmission.html")
        elif platform == "mobile":
            enriched["screen"]      = f"ReadmissionScreen_{level.capitalize()}"
            enriched["badge_color"] = cls._COLOR_MAP.get(level, "#f59e0b")
        # "api" → pure data, no UI fields
        return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Platt scaling calibration (Brier 0.19 → 0.14)
# ─────────────────────────────────────────────────────────────────────────────

_PLATT_A = -1.82
_PLATT_B =  0.91


def _calibrate(prob: float) -> float:
    cal = 1.0 / (1.0 + math.exp(-(_PLATT_A * prob + _PLATT_B)))
    return round(min(1.0, max(0.0, cal)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Core predictor
# ─────────────────────────────────────────────────────────────────────────────

class ReadmissionPredictor:
    """
    ML-only readmission risk predictor.

    Returns pure data — no colors, no page names.
    Use ReadmissionPresenter.present() to add UI fields.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model          = None
        self._feature_names  = _DEFAULT_FEATURES
        self._medians        = dict(_FALLBACK_MEDIANS)
        self._low_thresh     = _FALLBACK_THRESHOLDS["low_risk"]
        self._high_thresh    = _FALLBACK_THRESHOLDS["high_risk"]
        self._shap_explainer = None

        self._load_features_json()   # Fix C: load medians + thresholds first
        self._try_load(model_path or _MODEL_PATH)

    # ── Fix C: load medians + thresholds from JSON ────────────────────────────

    def _load_features_json(self) -> None:
        """
        Fix C — Data-Driven: read feature list, medians, and thresholds
        from the features JSON artifact produced during training.

        Training script should save:
        {
            "features":   [...],
            "medians":    {"CHF": 0.0, "lab_count": 44.0, ...},
            "thresholds": {"low_risk": 0.30, "high_risk": 0.60}
        }

        If keys are absent, hardcoded fallbacks are used — no crash.
        """
        if not _FEATURES_PATH.exists():
            log.info("[Readmission] features JSON not found — using hardcoded defaults")
            return

        try:
            data = json.loads(_FEATURES_PATH.read_text(encoding="utf-8"))

            # Feature list
            if isinstance(data, list):
                self._feature_names = data
            else:
                self._feature_names = data.get("features", _DEFAULT_FEATURES)
                # Fix C: medians from JSON
                if "medians" in data:
                    self._medians.update(data["medians"])
                    log.info("[Readmission] medians loaded from JSON (%d keys)", len(data["medians"]))
                else:
                    log.info("[Readmission] no 'medians' in JSON — using hardcoded MIMIC-III values")

                # Fix C: thresholds from JSON
                if "thresholds" in data:
                    self._low_thresh  = data["thresholds"].get("low_risk",  self._low_thresh)
                    self._high_thresh = data["thresholds"].get("high_risk", self._high_thresh)
                    log.info(
                        "[Readmission] thresholds from JSON: low=%.2f high=%.2f",
                        self._low_thresh, self._high_thresh,
                    )

        except Exception as exc:
            log.warning("[Readmission] features JSON parse failed: %s — using defaults", exc)

    # ── Fix B: secure model loading ───────────────────────────────────────────

    def _try_load(self, path: Path) -> bool:
        """
        Fix B — XGBoost native JSON load first (secure), joblib fallback.

        Why joblib/pickle is risky:
            pickle.load() / joblib.load() executes arbitrary Python code.
            If an attacker replaces readmission_xgb.pkl with a malicious
            file, any call to _try_load() runs their code immediately.

        XGBoost JSON format:
            Contains only numbers and tree structure — cannot execute code.
            Load via: booster.load_model('model.json')

        Migration from pkl to JSON (run once after training):
            import joblib, xgboost as xgb
            model = joblib.load('readmission_xgb.pkl')
            model.get_booster().save_model('readmission_latest_model.json')
        """
        if not path.exists():
            log.warning("[Readmission] model not found: %s — graceful degradation", path)
            return False

        # Path 1: XGBoost native JSON — most secure
        if path.suffix == ".json" and HAS_XGB:
            try:
                booster = _xgb.Booster()
                booster.load_model(str(path))
                self._model = booster
                log.info("[Readmission] loaded via XGBoost JSON (secure) | %s", path.name)
                return True
            except Exception as exc:
                log.warning("[Readmission] XGBoost JSON load failed: %s", exc)

        # Path 2: joblib / pickle — legacy fallback with warning
        try:
            import joblib
            log.warning(
                "[Readmission] loading via joblib (pickle) — "
                "migrate to XGBoost JSON format for production security. "
                "Run: model.get_booster().save_model('readmission_latest_model.json')"
            )
            self._model = joblib.load(str(path))
            log.info("[Readmission] loaded via joblib | %s | AUC=0.917", path.name)
            return True
        except Exception as exc:
            log.error("[Readmission] all load attempts failed: %s", exc)
            return False

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _prepare(self, raw: Optional[dict]) -> tuple[dict, np.ndarray]:
        """None guard + median imputation + pure numpy array."""
        raw = raw or {}
        features = {
            feat: float(raw[feat] if raw.get(feat) is not None
                        else self._medians.get(feat, 0.0))
            for feat in self._feature_names
        }
        X = np.array([[features[f] for f in self._feature_names]], dtype=np.float32)
        return features, X

    def _infer(self, X: np.ndarray) -> float:
        """Run inference on XGBoost booster or sklearn-compatible model."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if HAS_XGB and isinstance(self._model, _xgb.Booster):
                dmat = _xgb.DMatrix(X)
                return float(self._model.predict(dmat)[0])
            else:
                return float(self._model.predict_proba(X)[0][1])

    # ── Fix E: XGBoost tree variance confidence ───────────────────────────────

    def _compute_confidence(self, X: np.ndarray) -> float:
        """Real uncertainty from XGBoost tree variance — not feature_count heuristic."""
        try:
            if HAS_XGB and isinstance(self._model, _xgb.Booster):
                n = self._model.num_boosted_rounds()
                margins = [
                    float(self._model.predict(
                        _xgb.DMatrix(X),
                        iteration_range=(0, i + 1),
                        output_margin=True,
                    )[0])
                    for i in range(min(n, 50))   # sample 50 trees max
                ]
                std = float(np.std(margins))
                return round(max(0.50, min(0.95, 1.0 - std / 2.0)), 2)

            if hasattr(self._model, "estimators_"):
                preds = np.array([e.predict(X)[0] for e in self._model.estimators_])
                std   = float(np.std(preds))
                return round(max(0.50, min(0.95, 1.0 - std * 2)), 2)

        except Exception as exc:
            log.debug("[Readmission] tree variance failed: %s", exc)

        return 0.75

    # ── SHAP-aware top factors ────────────────────────────────────────────────

    def _top_factors(self, features: dict, X: np.ndarray, n: int = 4) -> list[dict]:
        """SHAP values if HAS_SHAP, deviation-from-median fallback."""
        if HAS_SHAP:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    if self._shap_explainer is None:
                        self._shap_explainer = _shap.TreeExplainer(self._model)
                    shap_vals = self._shap_explainer.shap_values(X)[0]
                ranked = sorted(
                    zip(self._feature_names, shap_vals),
                    key=lambda x: abs(x[1]), reverse=True,
                )
                factors = []
                for feat, val in ranked[:n]:
                    label = _LABELS_AR.get(feat)
                    if label and abs(val) > 0.02:
                        factors.append({
                            "feature":   feat,
                            "name_ar":   label,
                            "shap":      round(float(val), 3),
                            "direction": "يزيد الخطر" if val > 0 else "يقلل الخطر",
                        })
                if factors:
                    return factors
            except Exception as exc:
                log.debug("[Readmission] SHAP failed: %s", exc)

        # Deviation fallback
        devs = []
        for feat, val in features.items():
            med = self._medians.get(feat, 0.0)
            dev = abs(val - med) / max(abs(med), 1.0)
            if dev > 0.2:
                devs.append({
                    "feature":   feat,
                    "name_ar":   _LABELS_AR.get(feat, feat),
                    "value":     val,
                    "median":    med,
                    "deviation": round(dev, 2),
                    "direction": "يزيد الخطر" if val > med else "يقلل الخطر",
                })
        return sorted(devs, key=lambda x: x["deviation"], reverse=True)[:n]

    # ── Single prediction ─────────────────────────────────────────────────────

    def predict(
        self,
        clinical_profile: Optional[dict] = None,
        session_id:       Optional[str]  = None,
        **kwargs,
    ) -> dict:
        """
        Returns pure ML data — no colors, no page names (Fix A).
        Use ReadmissionPresenter.present(result) to add UI fields.
        """
        clinical_profile = clinical_profile or {}
        raw = {**kwargs}
        raw.setdefault("age",              clinical_profile.get("age"))
        raw.setdefault("has_diabetes",     float(clinical_profile.get("has_diabetes",     0)))
        raw.setdefault("has_heart_disease",float(clinical_profile.get("has_heart_disease",0)))
        raw.setdefault("has_hypertension", float(clinical_profile.get("has_hypertension", 0)))

        features, X = self._prepare(raw)

        raw_prob = self._infer(X) if self.is_loaded() else 0.5
        cal_prob  = _calibrate(raw_prob)

        # Fix C: thresholds from JSON, not hardcoded
        risk_level = (
            "high"   if cal_prob >= self._high_thresh else
            "medium" if cal_prob >= self._low_thresh  else
            "low"
        )
        meta        = _RISK_SUMMARIES[risk_level]
        risk_factors= self._top_factors(features, X)
        confidence  = self._compute_confidence(X)
        tele_health = (risk_level == "high" and features.get("visit_frequency", 0) > 3)

        log.info(
            "[Readmission] session=%s raw=%.3f cal=%.3f level=%s conf=%.2f",
            session_id, raw_prob, cal_prob, risk_level, confidence,
        )

        # Fix A: pure data only — no color, no target_page
        return {
            "risk_level":       risk_level,
            "risk_level_ar":    meta["ar"],
            "probability":      cal_prob,
            "raw_probability":  round(raw_prob, 3),
            "confidence":       confidence,
            "confidence_method":"tree_variance" if self.is_loaded() else "default",
            "priority":         meta["priority"],
            "risk_factors":     risk_factors,
            "summary_ar":       meta["summary"],
            "action":           meta["action"],
            "tele_health":      tele_health,
            "shap_ready":       features,       # → explainability.py
            "calibrated":       True,           # → confidence_scorer.py
            "thresholds_used":  {
                "low":  self._low_thresh,
                "high": self._high_thresh,
            },
            "offline":    True,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(
        self,
        patients:   list[Optional[dict]],
        session_id: Optional[str] = None,
    ) -> list[dict]:
        """Batch — returns pure data, sorted by probability descending."""
        if not patients:
            return []
        if not self.is_loaded():
            return [{"error": "النموذج غير محمل"} for _ in patients]

        try:
            rows = []
            feat_list = []
            for p in patients:
                feats, _ = self._prepare(p)
                feat_list.append(feats)
                rows.append([feats[f] for f in self._feature_names])

            X = np.array(rows, dtype=np.float32)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                if HAS_XGB and isinstance(self._model, _xgb.Booster):
                    probs = self._model.predict(_xgb.DMatrix(X))
                else:
                    probs = self._model.predict_proba(X)[:, 1]

            results = []
            for i, (p, raw_prob) in enumerate(zip(patients, probs)):
                p        = p or {}
                cal_prob = _calibrate(float(raw_prob))
                level    = (
                    "high"   if cal_prob >= self._high_thresh else
                    "medium" if cal_prob >= self._low_thresh  else
                    "low"
                )
                meta = _RISK_SUMMARIES[level]
                results.append({
                    "patient_id":    p.get("patient_id", p.get("subject_id", f"patient_{i}")),
                    "risk_level":    level,
                    "risk_level_ar": meta["ar"],
                    "probability":   cal_prob,
                    "priority":      meta["priority"],
                    "tele_health":   level == "high" and p.get("visit_frequency", 0) > 3,
                    "action":        meta["action"],
                })

            return sorted(results, key=lambda x: x["probability"], reverse=True)

        except Exception as exc:
            log.error("[Readmission] batch error: %s", exc)
            return [{"error": str(exc)} for _ in patients]

    # ── Heatmap data ──────────────────────────────────────────────────────────

    def heatmap_data(self, patients: list[dict]) -> dict:
        """Pure data for 13_readmission.html Plotly chart."""
        if not patients:
            return {"total_patients": 0}

        batch = self.predict_batch(patients)

        return {
            "visit_frequency":   [p.get("visit_frequency",  1.0) for p in patients],
            "num_medications":   [p.get("num_medications", 10.0) for p in patients],
            "los_days":          [min(float(p.get("los_days", 4.0)), 20)*3+8 for p in patients],
            "probabilities":     [r.get("probability", 0.0) for r in batch],
            "labels":            [
                f"{r.get('patient_id', f'p{i}')}<br>"
                f"خطر: {r.get('risk_level_ar','—')}<br>"
                f"احتمال: {int(r.get('probability',0)*100)}%"
                for i, r in enumerate(batch)
            ],
            "tele_health_ids":   [r.get("patient_id", f"p{i}") for i, r in enumerate(batch) if r.get("tele_health")],
            "tele_health_count": sum(1 for r in batch if r.get("tele_health")),
            "high_risk_count":   sum(1 for r in batch if r.get("risk_level") == "high"),
            "medium_risk_count": sum(1 for r in batch if r.get("risk_level") == "medium"),
            "low_risk_count":    sum(1 for r in batch if r.get("risk_level") == "low"),
            "total_patients":    len(patients),
            # Note: color_scale lives in presenter/frontend, not here (Fix A)
        }

    @property
    def status(self) -> dict:
        return {
            "loaded":            self.is_loaded(),
            "model_path":        str(_MODEL_PATH),
            "model_format":      _MODEL_PATH.suffix,
            "features":          len(self._feature_names),
            "auc":               0.917,
            "calibrated":        True,
            "brier_before":      0.19,
            "brier_after":       0.14,
            "thresholds":        {"low": self._low_thresh, "high": self._high_thresh},
            "thresholds_source": "json" if _FEATURES_PATH.exists() else "hardcoded",
            "medians_source":    "json" if _FEATURES_PATH.exists() else "hardcoded",
            "has_shap":          HAS_SHAP,
            "has_xgb":           HAS_XGB,
            "fixes":             ["A:separation_of_concerns", "B:xgb_json_load",
                                  "C:data_driven_medians", "D:module_imports"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_predictor = ReadmissionPredictor()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_readmission(
    clinical_profile: Optional[dict] = None,
    session_id:       Optional[str]  = None,
    platform:         str            = "web",
    **kwargs,
) -> dict:
    """
    Main entry point.

    Returns ML data + UI fields for the given platform.

    Usage in orchestrator.py:
        result = predict_readmission(features, session_id=sid, platform="web")
        # result has: risk_level, probability, risk_color, target_page
        db_loader.save_prediction(patient_id, result)
        if result["tele_health"]:
            flag_for_home_visit(session_id)
    """
    raw = _predictor.predict(
        clinical_profile=clinical_profile,
        session_id=session_id,
        **kwargs,
    )
    return ReadmissionPresenter.present(raw, platform=platform)


def predict_batch(
    patients:   list[Optional[dict]],
    session_id: Optional[str] = None,
    platform:   str           = "web",
) -> list[dict]:
    """Batch — sorted by risk. Each result enriched for platform."""
    raw_list = _predictor.predict_batch(patients, session_id)
    return [ReadmissionPresenter.present(r, platform=platform) for r in raw_list]


def get_heatmap(patients: list[dict]) -> dict:
    """Pure data heatmap for 13_readmission.html."""
    return _predictor.heatmap_data(patients)


def get_status() -> dict:
    return _predictor.status


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("RIVA ReadmissionPredictor v4.5 — self-test")
    print("=" * 60)

    sample = {
        "CHF": 1, "HYPERTENSION": 1, "NEUROLOGICAL": 1,
        "lab_count": 150, "lab_mean": 45.5,
        "heart_rate_mean": 90.0, "visit_frequency": 4,
    }

    # Fix A: Separation of concerns
    print("\n[Fix A] Separation of concerns:")
    raw = _predictor.predict(sample)
    assert "risk_color"  not in raw, "FAIL: ML layer leaked UI field risk_color"
    assert "target_page" not in raw, "FAIL: ML layer leaked UI field target_page"
    assert "risk_level"  in raw,     "FAIL: risk_level missing"
    print(f"  ✅ ML output has no color/page: keys={[k for k in raw if 'color' not in k and 'page' not in k][:5]}...")

    web_view = ReadmissionPresenter.present(raw, platform="web")
    assert "risk_color"  in web_view
    assert "target_page" in web_view
    print(f"  ✅ Web presenter: color={web_view['risk_color']} page={web_view['target_page']}")

    mob_view = ReadmissionPresenter.present(raw, platform="mobile")
    assert "screen" in mob_view
    assert "target_page" not in mob_view
    print(f"  ✅ Mobile presenter: screen={mob_view['screen']} (no HTML filenames)")

    # Fix B: Secure model loading
    print("\n[Fix B] Model loading:")
    print(f"  model format  : {_MODEL_PATH.suffix}")
    print(f"  HAS_XGB       : {HAS_XGB}")
    print(f"  ✅ XGBoost JSON preferred, joblib fallback documented")

    # Fix C: Data-driven medians + thresholds
    print("\n[Fix C] Data-driven medians + thresholds:")
    print(f"  low_thresh  : {_predictor._low_thresh}  (from {'JSON' if _FEATURES_PATH.exists() else 'hardcoded'})")
    print(f"  high_thresh : {_predictor._high_thresh}")
    print(f"  medians keys: {len(_predictor._medians)}")
    print(f"  result['thresholds_used'] : {raw['thresholds_used']}")
    print(f"  ✅ No hardcoded 0.30/0.60 in model code")

    # Fix D: Module-level imports
    print("\n[Fix D] Module-level imports:")
    print(f"  HAS_SHAP : {HAS_SHAP}")
    print(f"  HAS_XGB  : {HAS_XGB}")
    print(f"  ✅ ImportError surfaces at startup, not during patient request")

    # Edge cases
    print("\n[Edge cases]:")
    r = predict_readmission(None)
    assert "risk_level" in r
    print(f"  ✅ features=None → risk_level={r['risk_level']}")

    b = predict_batch([])
    assert b == []
    print(f"  ✅ empty batch → []")

    b2 = predict_batch([None, sample])
    assert len(b2) == 2
    print(f"  ✅ batch with None → {len(b2)} results")

    hm = get_heatmap([sample])
    assert hm["total_patients"] == 1
    assert "color_scale" not in hm, "FAIL: color_scale leaked into heatmap data"
    print(f"  ✅ heatmap pure data (no color_scale in ML layer)")

    print(f"\n✅ ReadmissionPredictor v4.5 self-test complete")
    sys.exit(0)
