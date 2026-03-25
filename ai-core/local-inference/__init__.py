"""
ai-core/local-inference/__init__.py
====================================
RIVA Health Platform — Local Inference Module
"""

from .Model_manager       import model_manager, ModelManager
from .ambiguity_handler   import handle_ambiguity, clear_session_slots
from .confidence_scorer   import compute_confidence, is_low_confidence, get_thresholds
from .data_compressor     import compress, decompress, compress_for_qr, decompress_from_qr
from .drug_interaction    import check_interaction, check_food_interaction, get_heatmap, normalize
from .explainability      import explain, explain_for_page, get_shap_waterfall, ModelType, Audience
from .history_analyzer    import analyze_history, get_profile_updates, get_readmission_features
from .pregnancy_risk      import predict_pregnancy_risk, predict_batch as predict_pregnancy_batch
from .readmission_predictor import predict_readmission, predict_batch as predict_readmission_batch, get_heatmap as get_readmission_heatmap
from .school_health       import analyze_student, analyze_class, get_heatmap as get_school_heatmap
from .triage_classifier   import classify_triage, classify_batch
from .triage_engine       import decide, get_target_page
from .unified_predictor   import predict, predict_batch, get_statistics
from .prescription_gen    import generate_prescription, verify_prescription, check_ai_drug
from .medical_rag         import *

__all__ = [
    # Model manager
    "model_manager", "ModelManager",
    # Ambiguity
    "handle_ambiguity", "clear_session_slots",
    # Confidence
    "compute_confidence", "is_low_confidence", "get_thresholds",
    # Compression
    "compress", "decompress", "compress_for_qr", "decompress_from_qr",
    # Drug interaction
    "check_interaction", "check_food_interaction", "get_heatmap", "normalize",
    # Explainability
    "explain", "explain_for_page", "get_shap_waterfall", "ModelType", "Audience",
    # History
    "analyze_history", "get_profile_updates", "get_readmission_features",
    # Pregnancy
    "predict_pregnancy_risk", "predict_pregnancy_batch",
    # Readmission
    "predict_readmission", "predict_readmission_batch", "get_readmission_heatmap",
    # School
    "analyze_student", "analyze_class", "get_school_heatmap",
    # Triage
    "classify_triage", "classify_batch", "decide", "get_target_page",
    # Unified
    "predict", "predict_batch", "get_statistics",
    # Prescription
    "generate_prescription", "verify_prescription", "check_ai_drug",
    # Medical RAG
    "medical_rag",
]
