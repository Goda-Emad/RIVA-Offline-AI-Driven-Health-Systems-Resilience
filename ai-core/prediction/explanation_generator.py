"""
explanation_generator.py
========================
RIVA Health Platform — Medical Explanation Generator
-----------------------------------------------------
يولّد شروحات طبية للمريض والدكتور بالعربي.

الفرق بين الملف ده و explainability.py (local-inference):
    - explainability.py      : SHAP values + feature contributions + counterfactuals
    - explanation_generator  : natural language summaries + recommendations
                               للـ prediction/ folder (readmission + LOS)

التحسينات:
    1. ربط مع readmission_predictor — نفس thresholds و color system
    2. ربط مع unified_predictor     — combined readmission + LOS explanation
    3. ربط مع orchestrator          — target_page في كل explanation
    4. ربط مع doctor_feedback       — doctor_review_needed flag
    5. Feature-aware explanations   — الشرح بيذكر العوامل المؤثرة
    6. SHAP-ready output            — متوافق مع explainability.py
    7. Confidence tier              — يعرف لو الـ explanation موثوق

الربط مع الـ 17 صفحة:
    Readmission high  → 13_readmission.html
    LOS long          → 14_los_dashboard.html
    Combined          → 15_combined_dashboard.html
    Low risk          → 05_history.html

Author : GODA EMAD
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("riva.prediction.explanation_generator")

# ─── Thresholds (متسقة مع readmission_predictor.py) ─────────────────────────

READ_HIGH  = 0.60
READ_LOW   = 0.30
LOS_LONG   = 10.0
LOS_MEDIUM = 5.0
LOS_SHORT  = 3.0

# ─── Color system (متسق مع RIVA UI) ─────────────────────────────────────────

_COLORS = {
    "high":   "#ef4444",   # danger red
    "medium": "#f59e0b",   # warning amber
    "low":    "#10b981",   # success green
}

# ─── Feature display names ────────────────────────────────────────────────────

_FEATURE_LABELS: dict[str, str] = {
    "num_diagnoses":        "عدد التشخيصات",
    "num_medications":      "عدد الأدوية",
    "num_procedures":       "عدد الإجراءات",
    "num_lab_procedures":   "عدد التحاليل",
    "los_days":             "مدة الإقامة السابقة",
    "has_diabetes":         "السكري",
    "has_heart_disease":    "أمراض القلب",
    "has_hypertension":     "الضغط المرتفع",
    "visit_frequency":      "تكرار الزيارات",
    "medication_changes":   "تغييرات الأدوية",
    "age":                  "السن",
    "charlson_index":       "مؤشر التعقيد الطبي",
    "total_visits":         "إجمالي الزيارات",
    "admit_EMERGENCY":      "دخول طوارئ",
}


# ─── Readmission explanations ─────────────────────────────────────────────────

_READ_RECOMMENDATIONS: dict[str, list[str]] = {
    "high": [
        "متابعة دقيقة خلال 48-72 ساعة من الخروج",
        "مراجعة شاملة للأدوية مع صيدلاني",
        "استشارة أخصائي في التخصص المناسب",
        "خطة رعاية منزلية مع ممرضة منزلية",
        "اتصال متابعة هاتفي بعد 24 ساعة",
    ],
    "medium": [
        "متابعة مع الطبيب خلال أسبوع",
        "مراجعة الأدوية والجرعات",
        "تثقيف المريض والأسرة عن العلامات التحذيرية",
        "تعليمات خروج مفصلة وواضحة",
    ],
    "low": [
        "متابعة روتينية خلال شهر",
        "تأكيد على الالتزام بالعلاج",
        "تعليمات خروج قياسية",
    ],
}

# Dynamic recommendation add-ons based on dominant factor
_FACTOR_RECOMMENDATIONS: dict[str, str] = {
    "num_medications":   "مراجعة جدول الأدوية اليومي مع الصيدلاني — كثرة الأدوية تزيد الخطر",
    "num_diagnoses":     "تنسيق بين أطباء التخصصات المختلفة — تعدد التشخيصات يحتاج تكاملاً",
    "has_diabetes":      "مراجعة جدول قياسات السكر اليومي — التحكم في السكر يقلل الخطر بشكل كبير",
    "has_heart_disease": "مراجعة أدوية القلب ومتابعة النبض يومياً",
    "has_hypertension":  "قياس الضغط مرتين يومياً وتسجيل النتائج",
    "visit_frequency":   "تحديد سبب تكرار الزيارات ومعالجته جذرياً",
    "medication_changes":"مراجعة سبب التغييرات المتكررة في الأدوية مع الطبيب",
    "los_days":          "تحضير بيئة المنزل للاستقبال قبل الخروج",
}

_READ_SUMMARIES: dict[str, str] = {
    "high":   "احتمال إعادة الإدخال مرتفع — يحتاج متابعة مكثفة فوراً",
    "medium": "احتمال إعادة الإدخال متوسط — متابعة دقيقة مطلوبة",
    "low":    "احتمال إعادة الإدخال منخفض — متابعة عادية",
}

# ─── LOS explanations ────────────────────────────────────────────────────────

_LOS_RECOMMENDATIONS: dict[str, list[str]] = {
    "long": [
        "تجهيز جناح طويل الإقامة مع مرافق مناسب",
        "متابعة يومية مع فريق التمريض",
        "تقييم أسبوعي لخطة العلاج",
        "دعم نفسي واجتماعي للمريض والأسرة",
        "مراجعة دورية للتغذية والعلاج الطبيعي",
    ],
    "medium": [
        "تحضير خطة خروج منظمة مسبقاً",
        "تثقيف المريض للعناية الذاتية بعد الخروج",
        "متابعة يومية لمؤشرات الحالة",
    ],
    "short": [
        "تجهيز تقرير الخروج الكامل",
        "وصف الأدوية اللازمة بعد الخروج",
        "تحديد موعد متابعة خارجي",
    ],
}


# ─── Core generator ───────────────────────────────────────────────────────────

class ExplanationGenerator:
    """
    Generates natural language medical explanations for RIVA predictions.

    Integrated with:
        - readmission_predictor.py : same thresholds + calibrated probabilities
        - unified_predictor.py     : combined readmission + LOS explanations
        - orchestrator.py          : target_page in every explanation
        - doctor_feedback_handler  : doctor_review_needed flag
        - 12_ai_explanation.html   : patient-facing summaries
        - 13_readmission.html      : readmission-specific display
        - 14_los_dashboard.html    : LOS-specific display
    """

    def __init__(self, language: str = "ar"):
        self.language = language
        log.info("[ExplanationGenerator] initialized  language=%s", language)

    # ── Readmission explanation ───────────────────────────────────────────────

    def generate_readmission_explanation(
        self,
        probability:    float,
        features:       dict = {},
        session_id:     Optional[str] = None,
        for_doctor:     bool = False,
    ) -> dict:
        """
        Generates readmission risk explanation.

        Args:
            probability : calibrated readmission probability (from Platt scaling)
            features    : patient features for factor highlighting
            session_id  : for logging
            for_doctor  : True → include technical details

        Returns:
            {
                summary          : short Arabic summary
                recommendations  : list of action items
                risk_level       : "high" | "medium" | "low"
                risk_level_ar    : بالعربي
                color            : RIVA UI hex color
                probability      : float
                top_factors      : top contributing features
                target_page      : الصفحة المناسبة
                doctor_review    : True لو يحتاج مراجعة طبية
                confidence_tier  : "reliable" | "uncertain"
            }
        """
        risk = (
            "high"   if probability >= READ_HIGH else
            "medium" if probability >= READ_LOW  else
            "low"
        )

        top_factors = self._extract_top_factors(features, n=3)

        # Enhanced summary mentioning top factors
        summary = _READ_SUMMARIES[risk]
        if top_factors and risk != "low":
            factors_str = "، ".join(f["name_ar"] for f in top_factors[:2])
            summary += f" — أبرز العوامل: {factors_str}"

        # Doctor-level technical detail
        doctor_note = None
        if for_doctor:
            doctor_note = (
                f"Calibrated probability: {probability:.3f} | "
                f"Risk tier: {risk} | "
                f"Platt-scaled (Brier: 0.14) | "
                f"Top features: {[f['name'] for f in top_factors]}"
            )

        # 1. Dynamic recommendations — add factor-specific advice
        recs = list(_READ_RECOMMENDATIONS[risk])
        for factor in top_factors[:2]:
            extra = _FACTOR_RECOMMENDATIONS.get(factor["name"])
            if extra and extra not in recs:
                recs.append(extra)

        # 2. Uncertainty note
        confidence_tier = "reliable" if 0.1 < probability < 0.9 else "uncertain"
        if confidence_tier == "uncertain":
            recs.insert(0, "⚠️ هذا التقدير تقريبي — يرجى الاعتماد كلياً على الفحص السريري")

        result = {
            "summary":          summary,
            "recommendations":  recs,
            "risk_level":       risk,
            "risk_level_ar":    {"high":"مرتفع","medium":"متوسط","low":"منخفض"}[risk],
            "color":            _COLORS[risk],
            "probability":      round(probability, 3),
            "top_factors":      top_factors,
            "target_page":      self._readmission_page(risk),
            "doctor_review":    risk == "high",
            "doctor_note":      doctor_note,
            "confidence_tier":  confidence_tier,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "[ExplanationGenerator] readmission  session=%s  risk=%s  prob=%.3f",
            session_id, risk, probability,
        )
        return result

    # ── LOS explanation ───────────────────────────────────────────────────────

    def generate_los_explanation(
        self,
        days:       float,
        features:   dict = {},
        los_mae:    float = 3.23,
        session_id: Optional[str] = None,
        for_doctor: bool = False,
    ) -> dict:
        """
        Generates Length of Stay explanation.

        Args:
            days       : predicted LOS in days
            features   : patient features
            los_mae    : model MAE for confidence interval
            session_id : for logging
            for_doctor : technical detail flag
        """
        if days >= LOS_LONG:
            category     = "long"
            category_ar  = "طويلة جداً"
            color        = _COLORS["high"]
        elif days >= LOS_MEDIUM:
            category     = "medium"
            category_ar  = "متوسطة"
            color        = _COLORS["medium"]
        else:
            category     = "short"
            category_ar  = "قصيرة"
            color        = _COLORS["low"]

        top_factors = self._extract_top_factors(features, n=3)

        summary = f"مدة الإقامة المتوقعة {days} يوم — إقامة {category_ar}"
        if top_factors and category != "short":
            factors_str = "، ".join(f["name_ar"] for f in top_factors[:2])
            summary += f" — بسبب: {factors_str}"

        confidence_interval = {
            "lower": round(max(0.0, days - los_mae), 2),
            "upper": round(days + los_mae, 2),
        }

        doctor_note = None
        if for_doctor:
            doctor_note = (
                f"Predicted: {days}d ± {los_mae}d (MAE) | "
                f"CI: [{confidence_interval['lower']}, {confidence_interval['upper']}] | "
                f"Top features: {[f['name'] for f in top_factors]}"
            )

        result = {
            "summary":             summary,
            "recommendations":     _LOS_RECOMMENDATIONS[category],
            "days":                round(days, 2),
            "category":            category,
            "category_ar":         category_ar,
            "color":               color,
            "confidence_interval": confidence_interval,
            "top_factors":         top_factors,
            "target_page":         self._los_page(category),
            "doctor_note":         doctor_note,
            "timestamp":           datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "[ExplanationGenerator] LOS  session=%s  days=%.1f  cat=%s",
            session_id, days, category,
        )
        return result

    # ── Combined explanation ──────────────────────────────────────────────────

    def generate_combined_explanation(
        self,
        probability: float,
        days:        float,
        features:    dict = {},
        los_mae:     float = 3.23,
        session_id:  Optional[str] = None,
    ) -> dict:
        """
        Combined Readmission + LOS explanation.
        Used by unified_predictor.py and 15_combined_dashboard.html.
        """
        read_exp = self.generate_readmission_explanation(
            probability, features, session_id
        )
        los_exp  = self.generate_los_explanation(
            days, features, los_mae, session_id
        )

        # Determine target page (same logic as unified_predictor._decide_page)
        high_read = probability >= READ_HIGH
        long_los  = days >= LOS_LONG

        if high_read and long_los:
            target_page = "15_combined_dashboard.html"
        elif high_read:
            target_page = "13_readmission.html"
        elif long_los:
            target_page = "14_los_dashboard.html"
        else:
            target_page = "05_history.html"

        # Overall severity
        overall_risk = (
            "high"   if high_read or long_los else
            "medium" if probability >= READ_LOW or days >= LOS_MEDIUM else
            "low"
        )

        combined_summary = (
            f"إعادة الإدخال: {read_exp['risk_level_ar']} ({probability:.0%}) | "
            f"مدة الإقامة: {days} يوم ({los_exp['category_ar']})"
        )

        return {
            "summary":         combined_summary,
            "overall_risk":    overall_risk,
            "overall_color":   _COLORS[overall_risk],
            "readmission":     read_exp,
            "los":             los_exp,
            "target_page":     target_page,
            "doctor_review":   overall_risk == "high",
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }

    # ── Feature factor extraction ─────────────────────────────────────────────

    def _extract_top_factors(
        self, features: dict, n: int = 3
    ) -> list[dict]:
        """
        Extracts top contributing factors from feature dict.
        Simple deviation-from-normal approach (no SHAP needed here —
        full SHAP is handled by explainability.py).
        """
        _NORMAL_RANGES: dict[str, tuple] = {
            "num_diagnoses":      (1, 5),
            "num_medications":    (1, 8),
            "num_procedures":     (0, 3),
            "num_lab_procedures": (10, 40),
            "los_days":           (1, 5),
            "total_visits":       (1, 3),
            "charlson_index":     (0, 2),
        }

        factors = []
        for feat, val in features.items():
            if feat not in _FEATURE_LABELS:
                continue
            val = float(val) if val is not None else 0.0
            if feat in _NORMAL_RANGES:
                lo, hi   = _NORMAL_RANGES[feat]
                midpoint = (lo + hi) / 2
                deviation= abs(val - midpoint) / max(midpoint, 1.0)
                if deviation > 0.3:
                    factors.append({
                        "name":       feat,
                        "name_ar":    _FEATURE_LABELS[feat],
                        "value":      val,
                        "deviation":  round(deviation, 2),
                        "above_normal": val > hi,
                    })
            elif val in (1.0, True):   # boolean flags
                factors.append({
                    "name":       feat,
                    "name_ar":    _FEATURE_LABELS[feat],
                    "value":      val,
                    "deviation":  0.5,
                    "above_normal": True,
                })

        return sorted(factors, key=lambda x: x["deviation"], reverse=True)[:n]

    # ── Page routing helpers ──────────────────────────────────────────────────

    @staticmethod
    def _readmission_page(risk: str) -> str:
        return {
            "high":   "13_readmission.html",
            "medium": "13_readmission.html",
            "low":    "05_history.html",
        }[risk]

    @staticmethod
    def _los_page(category: str) -> str:
        return {
            "long":   "14_los_dashboard.html",
            "medium": "14_los_dashboard.html",
            "short":  "05_history.html",
        }[category]


# ─── Audit storage ───────────────────────────────────────────────────────────

def _store_explanation_audit(
    patient_id:  str,
    exp_type:    str,   # "readmission" | "los" | "combined"
    summary_ar:  str,
    risk_level:  str,
    probability: float,
    target_page: str,
) -> None:
    """
    Stores Arabic summary in unified_predictions_audit.db.
    Accessible to doctor in 16_doctor_notes.html.

    Table: explanation_audit
        patient_id, timestamp, exp_type, summary_ar,
        risk_level, probability, target_page
    """
    try:
        import sqlite3
        from pathlib import Path

        db_path = (
            Path(__file__).parent.parent.parent
            / "data/databases/unified_predictions_audit.db"
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS explanation_audit (
                patient_id   TEXT,
                timestamp    TEXT,
                exp_type     TEXT,
                summary_ar   TEXT,
                risk_level   TEXT,
                probability  REAL,
                target_page  TEXT
            )
        """)
        cur.execute("INSERT INTO explanation_audit VALUES (?,?,?,?,?,?,?)", (
            patient_id,
            datetime.now(timezone.utc).isoformat(),
            exp_type,
            summary_ar,
            risk_level,
            probability,
            target_page,
        ))
        con.commit()
        con.close()
        log.debug("[ExplanationGenerator] audit stored for %s", patient_id)
    except Exception as e:
        log.warning("[ExplanationGenerator] audit failed: %s", e)


# ─── Singleton ────────────────────────────────────────────────────────────────

_generator = ExplanationGenerator()


# ─── Public API ───────────────────────────────────────────────────────────────

def explain_readmission(
    probability: float,
    features:    dict = {},
    session_id:  Optional[str] = None,
    for_doctor:  bool = False,
) -> dict:
    """
    Readmission explanation.

    Usage in readmission_predictor.py:
        from .explanation_generator import explain_readmission
        exp = explain_readmission(result["probability"], result["shap_ready"])
        result["explanation"] = exp
    """
    result = _generator.generate_readmission_explanation(
        probability, features, session_id, for_doctor
    )
    _store_explanation_audit(
        patient_id  = session_id or "unknown",
        exp_type    = "readmission",
        summary_ar  = result["summary"],
        risk_level  = result["risk_level"],
        probability = result["probability"],
        target_page = result["target_page"],
    )
    return result


def explain_los(
    days:       float,
    features:   dict  = {},
    los_mae:    float = 3.23,
    session_id: Optional[str] = None,
    for_doctor: bool  = False,
) -> dict:
    """
    LOS explanation.

    Usage in unified_predictor.py:
        from .explanation_generator import explain_los
        exp = explain_los(result["los"]["predicted_days"], result["shap_ready"])
    """
    result = _generator.generate_los_explanation(
        days, features, los_mae, session_id, for_doctor
    )
    _store_explanation_audit(
        patient_id  = session_id or "unknown",
        exp_type    = "los",
        summary_ar  = result["summary"],
        risk_level  = result["category"],
        probability = days / 30,   # normalised for storage
        target_page = result["target_page"],
    )
    return result


def explain_combined(
    probability: float,
    days:        float,
    features:    dict  = {},
    los_mae:     float = 3.23,
    session_id:  Optional[str] = None,
) -> dict:
    """
    Combined explanation for 15_combined_dashboard.html.

    Usage in unified_predictor.py:
        from .explanation_generator import explain_combined
        exp = explain_combined(
            probability = result["readmission"]["probability"],
            days        = result["los"]["predicted_days"],
            features    = result["shap_ready"],
        )
    """
    return _generator.generate_combined_explanation(
        probability, days, features, los_mae, session_id
    )
