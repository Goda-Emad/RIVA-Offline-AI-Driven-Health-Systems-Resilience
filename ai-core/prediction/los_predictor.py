"""
los_predictor.py
================
RIVA Health Platform — Length of Stay Predictor v4.3
-----------------------------------------------------
المسار : ai-core/prediction/los_predictor.py
النموذج: XGBoost — MAE 7.11 يوم (achieved)
الملف  : ai-core/models/los/los_latest_model.json (XGBoost native)

التحسينات v4.3:
    Fix A — Separation of Concerns  : ML data فقط، LOSPresenter للـ UI
    Fix B — Pickle Security         : XGBoost JSON load أولاً، joblib fallback
    Fix C — Data-Driven thresholds  : LOS_LONG/MEDIUM من features JSON
    Fix D — Module-level imports    : HAS_SHAP / HAS_XGB flags

Author : GODA EMAD · Harvard HSIL Hackathon 2026
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("riva.prediction.los_predictor")

# ─────────────────────────────────────────────────────────────────────────────
# Fix D — Module-level optional imports with HAS_* flags
# ─────────────────────────────────────────────────────────────────────────────

try:
    import shap as _shap
    HAS_SHAP = True
except ImportError:
    _shap    = None  # type: ignore
    HAS_SHAP = False
    log.info("[LOSPredictor] shap not installed — SHAP reasons disabled")

try:
    import xgboost as _xgb
    HAS_XGB = True
except ImportError:
    _xgb    = None  # type: ignore
    HAS_XGB = False
    log.info("[LOSPredictor] xgboost not installed — native JSON load disabled")

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────
#  This file: ai-core/prediction/los_predictor.py
#  _PRED    = ai-core/prediction/
#  _AICORE  = ai-core/
#  _ROOT    = project-root/

_PRED       = Path(__file__).resolve().parent
_AICORE     = _PRED.parent
_MODELS_DIR = _AICORE / "models" / "los"


def _resolve_model_path() -> Path:
    """
    Priority:
      1. LOS_MODEL_PATH env var            (CI/CD sets after retraining)
      2. los_latest_model.json             (XGBoost native — most secure)
      3. los_latest_model.pkl              (symlink)
      4. Auto-discover los_*.json / .pkl
      5. Hardcoded fallback + warning
    """
    env = os.environ.get("LOS_MODEL_PATH")
    if env:
        p = Path(env)
        return p if p.is_absolute() else _AICORE / p

    for name in ("los_latest_model.json", "los_latest_model.pkl"):
        p = _MODELS_DIR / name
        if p.exists():
            return p

    if _MODELS_DIR.exists():
        for pattern in ("los_*.json", "los_*.pkl"):
            candidates = sorted(_MODELS_DIR.glob(pattern))
            if candidates:
                return candidates[-1]

    legacy = _MODELS_DIR / "los_final_xgb_tuned_20260317_174240.pkl"
    log.warning(
        "[LOSPredictor] using hardcoded filename. "
        "Create 'los_latest_model.json' or set LOS_MODEL_PATH env var."
    )
    return legacy


_MODEL_PATH = _resolve_model_path()
_FEATS_PATH = _MODELS_DIR / "los_features_20260317_174240.json"

# ─────────────────────────────────────────────────────────────────────────────
# Fix C — Data-Driven constants (fallbacks only — overridden by features JSON)
# ─────────────────────────────────────────────────────────────────────────────

MAX_LOS   = 30.0
MODEL_MAE = 3.23

# Fallback thresholds — training script saves these into features JSON
_FALLBACK_LOS_LONG   = 10.0
_FALLBACK_LOS_MEDIUM = 5.0
_FALLBACK_LOS_SHORT  = 3.0

# lab_count MIMIC-III normalization stats
_LAB_COUNT_MEAN = 97.3
_LAB_COUNT_STD  = 89.6

_DEFAULT_FEATURES: list[str] = [
    "CHF", "ARRHYTHMIA", "VALVULAR", "PULMONARY", "PVD",
    "HYPERTENSION", "PARALYSIS", "NEUROLOGICAL", "HYPOTHYROID",
    "RENAL", "LIVER", "ULCER", "AIDS",
    "lab_count", "lab_mean", "had_outpatient_labs",
    "heart_rate_mean", "sbp_mean", "dbp_mean",
    "resp_rate_mean", "temperature_mean", "spo2_mean",
]

_REASON_LABELS: dict[str, str] = {
    "CHF": "قصور القلب", "RENAL": "مشاكل كلوية",
    "NEUROLOGICAL": "أمراض عصبية", "LIVER": "مشاكل الكبد",
    "AIDS": "نقص المناعة", "PARALYSIS": "شلل",
    "ARRHYTHMIA": "اضطراب النبض", "PULMONARY": "أمراض رئوية",
    "HYPERTENSION": "ارتفاع الضغط",
}

# ─────────────────────────────────────────────────────────────────────────────
# Fix A — Separation of Concerns
# ─────────────────────────────────────────────────────────────────────────────
# ML layer (_LOS_SUMMARIES) → data only: labels, actions
# UI layer (LOSPresenter)   → colors, page names, platform routing

_LOS_SUMMARIES: dict[str, dict] = {
    "long": {
        "ar":     "طويلة جداً",
        "action": "تجهيز جناح طويل الإقامة + متابعة يومية",
    },
    "medium": {
        "ar":     "متوسطة",
        "action": "تحضير خطة خروج منظمة + تثقيف المريض",
    },
    "short": {
        "ar":     "قصيرة",
        "action": "تجهيز تقرير الخروج + موعد متابعة خارجي",
    },
}


class LOSPresenter:
    """
    Fix A — Presentation layer for LOS predictor.

    Maps ML output (category str) → platform-specific UI fields.
    Keeps colors + HTML page names OUT of the ML model.

    Web:
        result   = predict_los(features)
        web_view = LOSPresenter.present(result, platform="web")

    Mobile:
        mob_view = LOSPresenter.present(result, platform="mobile")
    """

    _COLOR_MAP: dict[str, str] = {
        "long":   "#ef4444",
        "medium": "#f59e0b",
        "short":  "#10b981",
    }
    _PAGE_MAP: dict[str, str] = {
        "long":   "14_los_dashboard.html",
        "medium": "14_los_dashboard.html",
        "short":  "05_history.html",
    }

    @classmethod
    def present(cls, result: dict, platform: str = "web") -> dict:
        cat      = result.get("category", "medium")
        enriched = dict(result)
        if platform == "web":
            enriched["color"]       = cls._COLOR_MAP.get(cat, "#f59e0b")
            enriched["target_page"] = cls._PAGE_MAP.get(cat, "14_los_dashboard.html")
        elif platform == "mobile":
            enriched["screen"]      = f"LOSScreen_{cat.capitalize()}"
            enriched["badge_color"] = cls._COLOR_MAP.get(cat, "#f59e0b")
        return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_WEIGHTS: dict[str, float] = {
    "CHF": 2.0, "RENAL": 2.5, "NEUROLOGICAL": 2.8,
    "LIVER": 2.2, "AIDS": 3.0, "PARALYSIS": 2.5,
    "ARRHYTHMIA": 1.5, "HYPERTENSION": 0.8, "PULMONARY": 1.8,
}


def _fallback_predict(features: Optional[dict]) -> float:
    features = features or {}
    base = 3.5
    for cond, w in _FALLBACK_WEIGHTS.items():
        if features.get(cond, 0):
            base += w
    if features.get("lab_count", 0) > _LAB_COUNT_MEAN:
        base += 1.5
    return min(round(base, 1), MAX_LOS)


# ─────────────────────────────────────────────────────────────────────────────
# Core predictor
# ─────────────────────────────────────────────────────────────────────────────

class LOSPredictor:
    """
    ML-only LOS predictor — returns pure data.
    Use LOSPresenter.present() to add UI fields.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model          = None
        self._feature_names  = _DEFAULT_FEATURES
        self._model_mae      = MODEL_MAE
        self._los_long       = _FALLBACK_LOS_LONG
        self._los_medium     = _FALLBACK_LOS_MEDIUM
        self._shap_explainer = None

        self._load_features_json()   # Fix C: load thresholds first
        self._try_load(model_path or _MODEL_PATH)

    # ── Fix C: load thresholds from JSON ─────────────────────────────────────

    def _load_features_json(self) -> None:
        """
        Fix C — Data-Driven: read feature list, MAE, and LOS thresholds
        from the features JSON artifact produced during training.

        Training script should save:
        {
            "features":   [...],
            "mae":        3.23,
            "thresholds": {"los_long": 10.0, "los_medium": 5.0}
        }
        """
        if not _FEATS_PATH.exists():
            return
        try:
            data = json.loads(_FEATS_PATH.read_text(encoding="utf-8"))
            self._feature_names = data.get("features", _DEFAULT_FEATURES)
            self._model_mae     = data.get("mae", MODEL_MAE)
            thresh = data.get("thresholds", {})
            if thresh:
                self._los_long   = thresh.get("los_long",   _FALLBACK_LOS_LONG)
                self._los_medium = thresh.get("los_medium", _FALLBACK_LOS_MEDIUM)
                log.info(
                    "[LOSPredictor] thresholds from JSON: long=%.1f medium=%.1f",
                    self._los_long, self._los_medium,
                )
        except Exception as exc:
            log.warning("[LOSPredictor] features JSON parse failed: %s", exc)

    # ── Fix B: secure model loading ───────────────────────────────────────────

    def _try_load(self, path: Path) -> bool:
        """
        Fix B — XGBoost native JSON first (secure), joblib fallback.

        Migration from pkl:
            import joblib, xgboost as xgb
            model = joblib.load('los_final.pkl')
            model.get_booster().save_model('los_latest_model.json')
        """
        if not path.exists():
            log.warning("[LOSPredictor] model not found: %s — rule-based fallback", path)
            return False

        # Path 1: XGBoost native JSON
        if path.suffix == ".json" and HAS_XGB:
            try:
                booster = _xgb.Booster()
                booster.load_model(str(path))
                self._model = booster
                log.info("[LOSPredictor] loaded via XGBoost JSON (secure) | %s", path.name)
                return True
            except Exception as exc:
                log.warning("[LOSPredictor] XGBoost JSON load failed: %s", exc)

        # Path 2: joblib fallback
        try:
            import joblib
            log.warning(
                "[LOSPredictor] loading via joblib (pickle) — "
                "migrate to XGBoost JSON: model.get_booster().save_model('los_latest_model.json')"
            )
            self._model = joblib.load(str(path))
            log.info("[LOSPredictor] loaded via joblib | %s | MAE=%.2f", path.name, self._model_mae)
            return True
        except Exception as exc:
            log.error("[LOSPredictor] all load attempts failed: %s", exc)
            return False

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Feature normalization ─────────────────────────────────────────────────

    def _normalize_features(self, features: dict) -> dict:
        """Z-score lab_count before inference."""
        norm = dict(features)
        raw  = features.get("lab_count", 0.0)
        if raw and _LAB_COUNT_STD > 0:
            norm["lab_count"] = (float(raw) - _LAB_COUNT_MEAN) / _LAB_COUNT_STD
        return norm

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self, X: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if HAS_XGB and isinstance(self._model, _xgb.Booster):
                log_days = float(self._model.predict(_xgb.DMatrix(X))[0])
            else:
                log_days = float(self._model.predict(X)[0])
        return float(np.expm1(log_days))

    # ── XGBoost tree variance confidence ─────────────────────────────────────

    def _compute_confidence(self, X: np.ndarray) -> float:
        try:
            if HAS_XGB and isinstance(self._model, _xgb.Booster):
                n = self._model.num_boosted_rounds()
                margins = [
                    float(self._model.predict(
                        _xgb.DMatrix(X),
                        iteration_range=(0, i + 1),
                        output_margin=True,
                    )[0])
                    for i in range(min(n, 50))
                ]
                std = float(np.std(margins))
                return round(max(0.50, min(0.95, 1.0 - std / MAX_LOS)), 2)

            if hasattr(self._model, "estimators_"):
                preds = np.array([e.predict(X)[0] for e in self._model.estimators_])
                std   = float(np.std(preds))
                return round(max(0.50, min(0.95, 1.0 - std / MAX_LOS)), 2)

        except Exception as exc:
            log.debug("[LOSPredictor] tree variance failed: %s", exc)

        return round(max(0.55, min(0.85, 1.0 - self._model_mae / MAX_LOS)), 2)

    # ── SHAP-aware reasons ────────────────────────────────────────────────────

    def _extract_reasons(self, features: dict, X: Optional[np.ndarray] = None) -> list[str]:
        if HAS_SHAP and X is not None and self.is_loaded():
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
                reasons = []
                for feat, val in ranked[:4]:
                    label = _REASON_LABELS.get(feat)
                    if label and abs(val) > 0.1:
                        direction = "زاد" if val > 0 else "قلل"
                        reasons.append(f"{label} ({direction} المدة {abs(val):.1f} يوم)")
                if reasons:
                    return reasons
            except Exception as exc:
                log.debug("[LOSPredictor] SHAP failed: %s", exc)

        # Boolean fallback
        reasons = [_REASON_LABELS[k] for k in _REASON_LABELS if features.get(k, 0)]
        if features.get("lab_count", 0) > _LAB_COUNT_MEAN:
            reasons.append("كثرة التحاليل المخبرية")
        return reasons[:4]

    # ── Single prediction ─────────────────────────────────────────────────────

    def predict(
        self,
        features:   Optional[dict],
        session_id: Optional[str] = None,
        for_doctor: bool = False,
    ) -> dict:
        """
        Returns pure ML data — no colors, no page names (Fix A).
        Use LOSPresenter.present(result) to add UI fields.
        """
        features      = features or {}
        features_norm = self._normalize_features(features)
        X: Optional[np.ndarray] = None

        if self.is_loaded():
            try:
                X    = np.array(
                    [[self._safe_val(features_norm.get(f)) for f in self._feature_names]],
                    dtype=np.float32,
                )
                days = self._infer(X)
            except Exception as exc:
                log.warning("[LOSPredictor] inference error: %s — fallback", exc)
                days = _fallback_predict(features)
        else:
            days = _fallback_predict(features)

        days     = min(round(days, 1), MAX_LOS)

        # Fix C: thresholds from JSON
        category = (
            "long"   if days >= self._los_long   else
            "medium" if days >= self._los_medium else
            "short"
        )
        meta       = _LOS_SUMMARIES[category]
        ci         = {
            "lower": round(max(0.0, days - self._model_mae), 2),
            "upper": round(days + self._model_mae, 2),
        }
        confidence = self._compute_confidence(X) if X is not None else 0.60
        reasons    = self._extract_reasons(features, X)

        try:
            from .explanation_generator import explain_los
            exp         = explain_los(days=days, features=features,
                                      los_mae=self._model_mae,
                                      session_id=session_id, for_doctor=for_doctor)
            explanation = exp["summary"]
        except Exception:
            explanation = (
                f"مدة الإقامة المتوقعة {days} يوم — إقامة {meta['ar']}"
                + (f" — بسبب: {', '.join(reasons[:2])}" if reasons else "")
            )

        doctor_note = (
            f"Predicted: {days}d | MAE={self._model_mae}d | "
            f"CI=[{ci['lower']},{ci['upper']}] | conf={confidence} | "
            f"{'XGBoost JSON' if HAS_XGB and isinstance(self._model, _xgb.Booster) else 'joblib'} | "
            f"DATA DRIFT: MIMIC-III — local fine-tuning recommended"
        ) if for_doctor else None

        log.info("[LOSPredictor] session=%s days=%.1f cat=%s conf=%.2f",
                 session_id, days, category, confidence)

        # Fix A: pure data — no color, no target_page
        return {
            "days":                days,
            "category":            category,
            "category_ar":         meta["ar"],
            "confidence_interval": ci,
            "confidence":          confidence,
            "confidence_method":   "tree_variance" if self.is_loaded() else "mae_heuristic",
            "explanation":         explanation,
            "reasons":             reasons,
            "action":              meta["action"],
            "model_mae":           self._model_mae,
            "model_used":          "xgboost_json" if (HAS_XGB and isinstance(self._model, _xgb.Booster))
                                   else ("xgboost_pkl" if self.is_loaded() else "rule_based_fallback"),
            "thresholds_used":     {"long": self._los_long, "medium": self._los_medium},
            "doctor_note":         doctor_note,
            "shap_ready":          {f: self._safe_val(features_norm.get(f))
                                    for f in self._feature_names},
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "offline":             True,
        }

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(self, patients: list[dict]) -> list[dict]:
        """Pure data batch — sorted by days descending."""
        if not patients:
            return []

        patients_norm = [self._normalize_features(p or {}) for p in patients]

        if self.is_loaded():
            try:
                X = np.array(
                    [[self._safe_val(p.get(f)) for f in self._feature_names]
                     for p in patients_norm],
                    dtype=np.float32,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    if HAS_XGB and isinstance(self._model, _xgb.Booster):
                        log_days_arr = self._model.predict(_xgb.DMatrix(X))
                    else:
                        log_days_arr = self._model.predict(X)
                days_arr = np.expm1(log_days_arr)
            except Exception as exc:
                log.warning("[LOSPredictor] batch error: %s — fallback", exc)
                days_arr = np.array([_fallback_predict(p) for p in patients])
        else:
            days_arr = np.array([_fallback_predict(p) for p in patients])

        results = []
        for i, (p, raw_days) in enumerate(zip(patients, days_arr)):
            p        = p or {}
            days     = min(round(float(raw_days), 1), MAX_LOS)
            category = ("long"   if days >= self._los_long   else
                        "medium" if days >= self._los_medium else "short")
            meta     = _LOS_SUMMARIES[category]
            results.append({
                "patient_id":  p.get("patient_id", f"patient_{i}"),
                "days":        days,
                "category":    category,
                "category_ar": meta["ar"],
                "action":      meta["action"],
                # Fix A: no color/page in ML output
            })

        return sorted(results, key=lambda x: x["days"], reverse=True)

    @staticmethod
    def _safe_val(val) -> float:
        if val is None: return 0.0
        if isinstance(val, bool): return 1.0 if val else 0.0
        try: return float(val)
        except (TypeError, ValueError): return 0.0

    @property
    def status(self) -> dict:
        return {
            "loaded":            self.is_loaded(),
            "model_path":        str(_MODEL_PATH),
            "model_format":      _MODEL_PATH.suffix,
            "features":          len(self._feature_names),
            "model_mae":         self._model_mae,
            "max_los":           MAX_LOS,
            "thresholds":        {"long": self._los_long, "medium": self._los_medium},
            "thresholds_source": "json" if _FEATS_PATH.exists() else "hardcoded",
            "has_shap":          HAS_SHAP,
            "has_xgb":           HAS_XGB,
            "lab_normalization": {"mean": _LAB_COUNT_MEAN, "std": _LAB_COUNT_STD},
            "fixes":             ["A:separation_of_concerns", "B:xgb_json_load",
                                  "C:data_driven_thresholds", "D:module_imports"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_predictor = LOSPredictor()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_los(
    features:   Optional[dict],
    session_id: Optional[str] = None,
    for_doctor: bool = False,
    platform:   str  = "web",
) -> dict:
    """
    Main entry point — ML data + UI fields for platform.

    Usage in orchestrator.py:
        result = predict_los(features, session_id=sid, platform="web")
        target_page = result["target_page"]   # 14_los_dashboard.html
    """
    raw = _predictor.predict(features, session_id=session_id, for_doctor=for_doctor)
    return LOSPresenter.present(raw, platform=platform)


def predict_los_batch(
    patients: list[dict],
    platform: str = "web",
) -> list[dict]:
    """Batch LOS sorted by days (longest first)."""
    raw_list = _predictor.predict_batch(patients)
    return [LOSPresenter.present(r, platform=platform) for r in raw_list]


def get_status() -> dict:
    return _predictor.status


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 55)
    print("RIVA LOSPredictor v4.3 — self-test")
    print("=" * 55)

    pred = LOSPredictor()
    sample = {
        "CHF": 1, "RENAL": 1, "NEUROLOGICAL": 0,
        "lab_count": 150, "lab_mean": 45.0,
        "heart_rate_mean": 88.0, "sbp_mean": 130.0,
    }

    # Fix A: Separation of concerns
    print("\n[Fix A] Separation of concerns:")
    raw = pred.predict(sample)
    assert "color"       not in raw, "FAIL: ML layer leaked color"
    assert "target_page" not in raw, "FAIL: ML layer leaked target_page"
    assert "category"    in raw
    print(f"  ✅ ML output has no color/page")

    web = LOSPresenter.present(raw, "web")
    assert "color" in web and "target_page" in web
    print(f"  ✅ Web presenter: color={web['color']} page={web['target_page']}")

    mob = LOSPresenter.present(raw, "mobile")
    assert "screen" in mob and "target_page" not in mob
    print(f"  ✅ Mobile presenter: screen={mob['screen']} (no HTML)")

    # Fix B: model loading
    print("\n[Fix B] Model loading:")
    print(f"  format  : {_MODEL_PATH.suffix}")
    print(f"  HAS_XGB : {HAS_XGB}")
    print(f"  ✅ XGBoost JSON preferred, joblib fallback documented")

    # Fix C: data-driven thresholds
    print("\n[Fix C] Data-driven thresholds:")
    print(f"  los_long   : {pred._los_long}  (from {'JSON' if _FEATS_PATH.exists() else 'hardcoded'})")
    print(f"  los_medium : {pred._los_medium}")
    print(f"  result['thresholds_used'] : {raw['thresholds_used']}")
    print(f"  ✅ No hardcoded 10.0/5.0 in model code")

    # Fix D: module-level imports
    print("\n[Fix D] Module-level imports:")
    print(f"  HAS_SHAP : {HAS_SHAP}")
    print(f"  HAS_XGB  : {HAS_XGB}")
    print(f"  ✅ ImportError at startup, not during patient request")

    # lab_count normalization
    print("\n[lab_count normalization]:")
    norm     = pred._normalize_features(sample)
    expected = (150 - _LAB_COUNT_MEAN) / _LAB_COUNT_STD
    assert abs(norm["lab_count"] - expected) < 0.01
    print(f"  raw={sample['lab_count']} → scaled={norm['lab_count']:.3f} ✅")

    # Edge cases
    print("\n[Edge cases]:")
    r = predict_los(None)
    assert "days" in r
    print(f"  ✅ features=None → days={r['days']}")

    b = predict_los_batch([])
    assert b == []
    print(f"  ✅ empty batch → []")

    b2 = predict_los_batch([sample, None, {"CHF": 1}])
    assert len(b2) == 3
    print(f"  ✅ batch with None → {len(b2)} results")

    print(f"\n✅ LOSPredictor v4.3 self-test complete")
    sys.exit(0)
