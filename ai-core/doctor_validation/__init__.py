"""
ai-core/doctor_validation/__init__.py
=======================================
RIVA Health Platform — Doctor Validation Module
"""

from .clinical_override_log import (
    log_override,
    update_outcome,
    get_summary  as get_override_summary,
    get_by_patient,
    get_by_session,
    verify_audit_log,
    OverrideReason,
    Severity,
    Outcome,
)
from .doctor_feedback_handler import (
    submit_rating,
    submit_correction,
    submit_validation,
    submit_flag,
    get_summary  as get_feedback_summary,
    get_signals,
    get_feedback,
    FeedbackType,
    FlagReason,
    RetrainingPriority,
)

__all__ = [
    # Clinical override log
    "log_override",
    "update_outcome",
    "get_override_summary",
    "get_by_patient",
    "get_by_session",
    "verify_audit_log",
    "OverrideReason",
    "Severity",
    "Outcome",
    # Doctor feedback handler
    "submit_rating",
    "submit_correction",
    "submit_validation",
    "submit_flag",
    "get_feedback_summary",
    "get_signals",
    "get_feedback",
    "FeedbackType",
    "FlagReason",
    "RetrainingPriority",
]
