"""
ai-core/prediction/__init__.py
================================
RIVA Health Platform — Prediction Module
"""

from .explanation_generator import (
    explain_readmission,
    explain_los,
    explain_combined,
)
from .feature_engineering import (
    process,
    get_stats as get_feature_stats,
    PatientFeatures,
    SeverityLevel,
)
from .los_predictor import (
    predict_los,
    predict_los_batch,
    get_status as get_los_status,
)
from .readmission_predictor import (
    predict_readmission,
    predict_batch as predict_readmission_batch,
    get_heatmap as get_readmission_heatmap,
    get_status as get_readmission_status,
)

__all__ = [
    # Explanation
    "explain_readmission",
    "explain_los",
    "explain_combined",
    # Feature engineering
    "process",
    "get_feature_stats",
    "PatientFeatures",
    "SeverityLevel",
    # LOS
    "predict_los",
    "predict_los_batch",
    "get_los_status",
    # Readmission
    "predict_readmission",
    "predict_readmission_batch",
    "get_readmission_heatmap",
    "get_readmission_status",
]
