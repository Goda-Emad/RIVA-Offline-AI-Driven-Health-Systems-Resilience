"""
__init__.py
===========
RIVA Health Platform v4.0 — Prediction Package
-----------------------------------------------
المسار: ai-core/prediction/__init__.py

يعرّف الـ public API لحزمة prediction ويربطها بباقي المنظومة:
    - los_predictor.py          → predict_los()
    - readmission_predictor.py  → predict_readmission()
    - explanation_generator.py  → explain()
    - feature_engineering.py    → engineer_features()

Author : GODA EMAD · Harvard HSIL Hackathon 2026
"""

from __future__ import annotations

import logging

log = logging.getLogger("riva.prediction")

# ─────────────────────────────────────────────────────────────────────────────
# LOS Predictor
# ─────────────────────────────────────────────────────────────────────────────
try:
    from .los_predictor import (
        predict_los,
        predict_los_batch,
        get_status as get_los_status,
        LOSPresenter,
        LOSPredictor,
    )
    _HAS_LOS = True
except Exception as _e:
    log.warning("prediction: los_predictor unavailable — %s", _e)
    _HAS_LOS = False

# ─────────────────────────────────────────────────────────────────────────────
# Readmission Predictor
# ─────────────────────────────────────────────────────────────────────────────
try:
    from .readmission_predictor import (
        predict_readmission,
        predict_batch as predict_readmission_batch,
        get_heatmap,
        get_status as get_readmission_status,
        ReadmissionPresenter,
        ReadmissionPredictor,
    )
    _HAS_READMISSION = True
except Exception as _e:
    log.warning("prediction: readmission_predictor unavailable — %s", _e)
    _HAS_READMISSION = False

# ─────────────────────────────────────────────────────────────────────────────
# Explanation Generator
# ─────────────────────────────────────────────────────────────────────────────
try:
    from .explanation_generator import explain
    _HAS_EXPLAIN = True
except Exception as _e:
    log.warning("prediction: explanation_generator unavailable — %s", _e)
    _HAS_EXPLAIN = False

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
try:
    from .feature_engineering import engineer_features
    _HAS_FEATURES = True
except Exception as _e:
    log.warning("prediction: feature_engineering unavailable — %s", _e)
    _HAS_FEATURES = False

# ─────────────────────────────────────────────────────────────────────────────
# Package metadata
# ─────────────────────────────────────────────────────────────────────────────

__version__ = "4.0.0"
__author__  = "GODA EMAD"

__all__ = [
    # LOS
    "predict_los",
    "predict_los_batch",
    "get_los_status",
    "LOSPresenter",
    "LOSPredictor",
    # Readmission
    "predict_readmission",
    "predict_readmission_batch",
    "get_heatmap",
    "get_readmission_status",
    "ReadmissionPresenter",
    "ReadmissionPredictor",
    # Explanation
    "explain",
    # Features
    "engineer_features",
    # Health
    "get_prediction_health",
]


def get_prediction_health() -> dict:
    """
    Health check for all prediction modules.
    Called by FastAPI /health endpoint.

    Usage:
        from ai_core.prediction import get_prediction_health
        health = get_prediction_health()
    """
    return {
        "package":     "riva.prediction",
        "version":     __version__,
        "los":         _HAS_LOS,
        "readmission": _HAS_READMISSION,
        "explanation": _HAS_EXPLAIN,
        "features":    _HAS_FEATURES,
        "all_ready":   all([_HAS_LOS, _HAS_READMISSION]),
    }


log.info(
    "riva.prediction loaded | los=%s readmission=%s explain=%s features=%s",
    _HAS_LOS, _HAS_READMISSION, _HAS_EXPLAIN, _HAS_FEATURES,
)
