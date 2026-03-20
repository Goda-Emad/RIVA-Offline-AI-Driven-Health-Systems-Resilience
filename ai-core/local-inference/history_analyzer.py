"""
history_analyzer.py
====================
RIVA Health Platform — Patient Medical History Analyzer
--------------------------------------------------------
يحلل التاريخ الطبي للمريض ويستخرج أنماط مهمة لدعم القرار الطبي.

الفائدة في RIVA:
    - قبل الفرز: يزوّد triage_engine بسياق تاريخ المريض
    - قبل الشات: يزوّد chat.py بالـ clinical_profile المكتمل
    - صفحة 05_history.html: يعرض التاريخ بشكل مرئي منظم
    - صفحة 09_doctor_dashboard: يعطي الدكتور ملخص سريع

ما بيحلله:
    1. Chronic conditions  — الأمراض المزمنة (سكر، ضغط، قلب...)
    2. Visit patterns      — تكرار الزيارات وأسبابها
    3. Medication history  — الأدوية السابقة والحالية
    4. Risk trajectory     — هل الحالة بتتحسن أو بتتدهور؟
    5. Alert patterns      — أعراض تتكرر وتحتاج انتباه

الربط مع المنظومة:
    - triage_engine.py      : يستدعيه قبل التصنيف
    - chat.py               : يحدّث الـ clinical_profile
    - readmission_predictor : يمد features من التاريخ
    - 05_history.html       : يعرض النتيجة

Author : GODA EMAD
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.local_inference.history_analyzer")

# ─── Paths ───────────────────────────────────────────────────────────────────

_BASE        = Path(__file__).parent.parent
_RISK_FACTORS= _BASE / "business-intelligence/medical-content/readmission_risk_factors.json"

# ─── Chronic condition keywords ───────────────────────────────────────────────

_CHRONIC_CONDITIONS: dict[str, list[str]] = {
    "diabetes": [
        # رسمي
        "سكر", "سكري", "ديابتيس", "diabetes", "metformin",
        "insulin", "glibenclamide", "هيموجلوبين سكري", "hba1c",
        # أعراض مبكرة — NLP expansion v4.3
        # المريض مش مشخص رسمي بس بيوصف الأعراض
        "عطش شديد", "عطشان أوي", "بيشرب مية كتير",
        "تبول كتير", "بيتبول كتير", "صحي كتير بالليل",
        "تنميل", "تنميل في الأرجل", "تنميل في اليدين",
        "خدر", "حرقان في الأرجل", "جروح مش بتعدي",
        "تعب من غير سبب", "وزن نازل من غير رجيم",
        "نظر بيتعشّى", "عينيه ضعفت فجأة",
    ],
    "hypertension": [
        # رسمي
        "ضغط", "ضغط مرتفع", "hypertension", "captopril",
        "amlodipine", "furosemide", "blood pressure",
        # أعراض مبكرة — NLP expansion v4.3
        "صداع في الرقبة", "صداع في القفا", "دوخة مفاجئة",
        "طنين في الأذن", "احمرار في الوجه", "ضربان في الرأس",
        "ضيقة في الصدر مع صداع",
    ],
    "heart_disease": [
        # رسمي
        "قلب", "جلطة", "ذبحة", "heart", "digoxin", "amiodarone",
        "warfarin", "cardiac", "coronary", "angina",
        # أعراض مبكرة — NLP expansion v4.3
        "ضربان غير منتظم", "قلبي بيدق بسرعة", "خفقان",
        "ضيقة في الصدر مع الجهد", "تعب من المشي",
        "إجهاد سريع", "تورم في القدمين",
    ],
    "respiratory": [
        # رسمي
        "ربو", "asthma", "copd", "انتفاخ رئة", "ضيقة", "inhaler",
        "salbutamol", "budesonide",
        # أعراض مبكرة — NLP expansion v4.3
        "سعال مزمن", "كحة مستمرة", "صافرة في الصدر",
        "صعوبة في التنفس بالليل", "بيصحى من النوم من الزهاق",
    ],
    "kidney_disease": [
        # رسمي
        "كلى", "فشل كلوي", "kidney", "renal", "creatinine",
        "dialysis", "غسيل كلى",
        # أعراض مبكرة — NLP expansion v4.3
        "تورم في الوجه الصبح", "بول رغوي", "ألم في الظهر أسفل",
        "بول قليل مع تورم",
    ],
    "liver_disease": [
        # رسمي
        "كبد", "تليف", "liver", "hepatitis", "cirrhosis",
        "التهاب كبدي",
        # أعراض مبكرة — NLP expansion v4.3
        "اصفرار في العين", "اصفرار الجلد", "يرقان",
        "بطن منتفخة", "حكة في الجسم كله",
    ],
    "thyroid": [
        # رسمي
        "غدة درقية", "thyroid", "levothyroxine", "hypothyroid",
        "hyperthyroid",
        # أعراض مبكرة — NLP expansion v4.3
        "وزن بيزيد من غير أكل زيادة", "وزن نازل من غير سبب",
        "إحساس بالبرد دايماً", "شعر بيطلع", "تعب وخمول شديد",
        "عصبية وقلق زيادة عن اللازم",
    ],
    "anemia": [
        # رسمي
        "أنيميا", "فقر دم", "anemia", "hemoglobin < 10",
        "iron deficiency", "نقص حديد",
        # أعراض مبكرة — NLP expansion v4.3
        "شحوب", "وجهه أبيض", "تعب من أي مجهود بسيط",
        "دوخة لما بيقوم", "قلبه بيدق بسرعة من غير سبب",
        "أظافر هشة", "شهية للأكل غريب (طفلة أو تراب)",
    ],
}

# ─── Risk escalation indicators ───────────────────────────────────────────────

_ESCALATION_KEYWORDS = [
    "تدهور", "ازداد سوءاً", "مش بيتحسن", "worse", "deteriorating",
    "أسوأ", "زاد الوجع", "ارتفع السكر", "ارتفع الضغط",
    "نزيف جديد", "رجع تاني", "مش رادّ",
]

_IMPROVEMENT_KEYWORDS = [
    "تحسّن", "أحسن", "improved", "better", "نزل السكر",
    "انتظم الضغط", "بيتحسن", "كويس أكتر",
]

# ─── Visit reason categories ──────────────────────────────────────────────────

# ─── Vitals trend analysis ───────────────────────────────────────────────────

_VITAL_KEYS = {
    "systolic_bp":  {"ar": "ضغط الدم الانقباضي", "normal_max": 140, "unit": "mmHg"},
    "diastolic_bp": {"ar": "ضغط الدم الانبساطي", "normal_max": 90,  "unit": "mmHg"},
    "heart_rate":   {"ar": "ضربات القلب",          "normal_max": 100, "unit": "نبضة/دقيقة"},
    "temperature":  {"ar": "الحرارة",               "normal_max": 37.5,"unit": "°C"},
    "blood_glucose":{"ar": "سكر الدم",              "normal_max": 126, "unit": "mg/dL"},
    "spo2":         {"ar": "تشبع الأكسجين",          "normal_min": 95,  "unit": "%"},
    "weight_kg":    {"ar": "الوزن",                  "normal_max": None,"unit": "كيلو"},
}


def _compute_vital_trends(visits: list) -> list[str]:
    """
    Computes linear slope for each vital sign across visits.
    Raises an alert if a vital is trending toward danger
    even if current value is still in normal range.

    Algorithm:
        slope = (last_value - first_value) / num_visits
        if slope > threshold AND still_normal → early warning
        if slope > threshold AND already_abnormal → escalation alert
    """
    alerts = []

    for vital_key, meta in _VITAL_KEYS.items():
        # Collect (visit_index, value) pairs
        series = []
        for i, visit in enumerate(visits):
            val = visit.vitals.get(vital_key)
            if val is not None:
                try:
                    series.append((i, float(val)))
                except (ValueError, TypeError):
                    pass

        if len(series) < 3:   # نحتاج على الأقل 3 نقاط للـ trend
            continue

        # Linear slope (simple rise/run)
        x_vals   = [s[0] for s in series]
        y_vals   = [s[1] for s in series]
        n        = len(series)
        x_mean   = sum(x_vals) / n
        y_mean   = sum(y_vals) / n
        num      = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        den      = sum((x - x_mean) ** 2 for x in x_vals)
        slope    = num / den if den != 0 else 0

        last_val = y_vals[-1]
        first_val= y_vals[0]
        change   = last_val - first_val

        # Upward trend for vitals with max threshold
        if "normal_max" in meta and meta["normal_max"]:
            norm_max = meta["normal_max"]
            if slope > 0 and change > (norm_max * 0.05):   # >5% rise
                if last_val < norm_max:
                    alerts.append(
                        f"{meta['ar']} في ارتفاع تدريجي "
                        f"({first_val:.0f}→{last_val:.0f} {meta['unit']}) "
                        f"— لا يزال طبيعياً لكن الاتجاه مقلق"
                    )
                else:
                    alerts.append(
                        f"{meta['ar']} مرتفع ومستمر في الارتفاع "
                        f"({first_val:.0f}→{last_val:.0f} {meta['unit']})"
                    )

        # Downward trend for vitals with min threshold (spo2)
        if "normal_min" in meta and meta["normal_min"]:
            norm_min = meta["normal_min"]
            if slope < 0 and abs(change) > 2:   # >2% drop
                if last_val > norm_min:
                    alerts.append(
                        f"{meta['ar']} في انخفاض تدريجي "
                        f"({first_val:.0f}→{last_val:.0f}{meta['unit']}) "
                        f"— لا يزال طبيعياً لكن يحتاج متابعة"
                    )
                else:
                    alerts.append(
                        f"{meta['ar']} منخفض ومستمر في الانخفاض "
                        f"({first_val:.0f}→{last_val:.0f}{meta['unit']})"
                    )

    return alerts


_VISIT_CATEGORIES: dict[str, list[str]] = {
    "emergency":   ["طوارئ", "emergency", "حادة", "مفاجئ", "فجأة"],
    "followup":    ["متابعة", "follow", "روتيني", "مراجعة"],
    "medication":  ["دواء", "روشتة", "prescription", "refill"],
    "pregnancy":   ["حمل", "كشف حمل", "prenatal", "maternity"],
    "school":      ["مدرسة", "school health", "كشف مدرسي"],
    "lab":         ["تحليل", "أشعة", "lab", "x-ray", "مختبر"],
}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class VisitRecord:
    date:       str
    reason:     str
    diagnosis:  str
    medications:list[str] = field(default_factory=list)
    vitals:     dict      = field(default_factory=dict)
    notes:      str       = ""


@dataclass
class HistoryAnalysisResult:
    """
    Complete history analysis — feeds into clinical_profile + risk models.
    """
    patient_id_hash:      str
    analysis_date:        str
    total_visits:         int
    visit_span_days:      int                  # مدة التاريخ الطبي

    # Conditions
    detected_conditions:  list[str]            # الأمراض المزمنة المكتشفة
    condition_confidence: dict[str, float]     # ثقة الكشف لكل مرض

    # Trajectory
    trajectory:           str                  # "improving" | "stable" | "declining"
    trajectory_evidence:  list[str]            # الأدلة

    # Medications
    current_medications:  list[str]
    discontinued_meds:    list[str]
    medication_changes:   int                  # عدد تغييرات الدواء

    # Visit patterns
    visit_frequency:      float                # زيارة/شهر
    frequent_complaints:  list[str]            # الأعراض الأكثر تكراراً
    visit_categories:     dict[str, int]       # توزيع الزيارات

    # Risk signals
    readmission_signals:  list[str]            # مؤشرات إعادة الإدخال
    alert_patterns:       list[str]            # أنماط تستحق انتباه

    # For clinical_profile update
    profile_updates:      dict                 # يُدمج في session["metadata"]

    # Summary
    summary_ar:           str                  # ملخص بالعربي للدكتور
    duration_ms:          float

    def to_dict(self) -> dict:
        return {
            "patient_id_hash":     self.patient_id_hash,
            "analysis_date":       self.analysis_date,
            "total_visits":        self.total_visits,
            "visit_span_days":     self.visit_span_days,
            "detected_conditions": self.detected_conditions,
            "condition_confidence":self.condition_confidence,
            "trajectory":          self.trajectory,
            "trajectory_evidence": self.trajectory_evidence,
            "current_medications": self.current_medications,
            "discontinued_meds":   self.discontinued_meds,
            "medication_changes":  self.medication_changes,
            "visit_frequency":     self.visit_frequency,
            "frequent_complaints": self.frequent_complaints,
            "visit_categories":    self.visit_categories,
            "readmission_signals": self.readmission_signals,
            "alert_patterns":      self.alert_patterns,
            "profile_updates":     self.profile_updates,
            "summary_ar":          self.summary_ar,
            "duration_ms":         self.duration_ms,
        }


# ─── Core analyzer ───────────────────────────────────────────────────────────

class HistoryAnalyzer:
    """
    Analyzes patient visit history to extract clinically meaningful patterns.

    Input: list of VisitRecord objects (loaded from patients.encrypted)
    Output: HistoryAnalysisResult with conditions, trajectory, risk signals
    """

    def __init__(self):
        self._risk_factors = self._load_risk_factors()
        log.info("[HistoryAnalyzer] ready")

    def _load_risk_factors(self) -> dict:
        if _RISK_FACTORS.exists():
            with open(_RISK_FACTORS, encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ── Main analyze ─────────────────────────────────────────────────────────

    def analyze(
        self,
        patient_id_hash: str,
        visits:          list[VisitRecord],
        current_session: Optional[dict] = None,
    ) -> HistoryAnalysisResult:
        """
        Full history analysis pipeline.

        Args:
            patient_id_hash : anonymised patient ID
            visits          : sorted list of VisitRecord (oldest first)
            current_session : current chat session data (merged into result)
        """
        t0 = time.perf_counter()

        if not visits:
            return self._empty_result(patient_id_hash)

        # 1. Chronic conditions
        conditions, confidence = self._detect_conditions(visits)

        # 2. Trajectory
        trajectory, evidence = self._assess_trajectory(visits)

        # 3. Medications
        current_meds, discontinued, changes = self._analyze_medications(visits)

        # 4. Visit patterns
        frequency, categories, complaints = self._visit_patterns(visits)

        # 5. Risk signals
        readmission_signals = self._readmission_signals(
            visits, conditions, changes
        )
        alert_patterns = self._detect_alert_patterns(visits, complaints)

        # 6. Visit span
        span_days = self._visit_span(visits)

        # 7. Clinical profile updates (for session["metadata"])
        profile_updates = self._build_profile_updates(
            conditions, current_meds, visits
        )

        # 8. Arabic summary
        summary = self._build_summary(
            visits, conditions, trajectory,
            readmission_signals, frequency,
        )

        ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info(
            "[HistoryAnalyzer] patient=%s  visits=%d  conditions=%s  "
            "trajectory=%s  %.0fms",
            patient_id_hash[:8], len(visits),
            conditions, trajectory, ms,
        )

        return HistoryAnalysisResult(
            patient_id_hash      = patient_id_hash,
            analysis_date        = datetime.now(timezone.utc).isoformat(),
            total_visits         = len(visits),
            visit_span_days      = span_days,
            detected_conditions  = conditions,
            condition_confidence = confidence,
            trajectory           = trajectory,
            trajectory_evidence  = evidence,
            current_medications  = current_meds,
            discontinued_meds    = discontinued,
            medication_changes   = changes,
            visit_frequency      = frequency,
            frequent_complaints  = complaints,
            visit_categories     = categories,
            readmission_signals  = readmission_signals,
            alert_patterns       = alert_patterns,
            profile_updates      = profile_updates,
            summary_ar           = summary,
            duration_ms          = ms,
        )

    # ── 1. Chronic conditions ─────────────────────────────────────────────────

    def _detect_conditions(
        self, visits: list[VisitRecord]
    ) -> tuple[list[str], dict[str, float]]:
        """
        Detects chronic conditions from visit text using keyword matching.
        Returns (conditions_list, confidence_dict).

        Confidence = mentions / total_visits (more mentions = more confident).
        """
        hits: dict[str, int] = {c: 0 for c in _CHRONIC_CONDITIONS}
        total = len(visits)

        for visit in visits:
            text = " ".join([
                visit.reason, visit.diagnosis, visit.notes,
                " ".join(visit.medications),
            ]).lower()

            for condition, keywords in _CHRONIC_CONDITIONS.items():
                if any(kw.lower() in text for kw in keywords):
                    hits[condition] += 1

        conditions  = [c for c, h in hits.items() if h > 0]
        confidence  = {
            c: round(min(1.0, hits[c] / max(total * 0.3, 1)), 2)
            for c in conditions
        }

        return sorted(conditions, key=lambda c: confidence[c], reverse=True), confidence

    # ── 2. Trajectory ─────────────────────────────────────────────────────────

    def _assess_trajectory(
        self, visits: list[VisitRecord]
    ) -> tuple[str, list[str]]:
        """
        Assesses whether the patient's health is improving, stable, or declining.
        Looks at the last 3 visits for trend signals.
        """
        recent_text = " ".join([
            f"{v.reason} {v.notes}" for v in visits[-3:]
        ]).lower()

        esc_hits  = sum(1 for kw in _ESCALATION_KEYWORDS if kw.lower() in recent_text)
        imp_hits  = sum(1 for kw in _IMPROVEMENT_KEYWORDS if kw.lower() in recent_text)

        evidence = []

        if esc_hits > imp_hits:
            trajectory = "declining"
            evidence   = [kw for kw in _ESCALATION_KEYWORDS if kw.lower() in recent_text][:3]
        elif imp_hits > 0:
            trajectory = "improving"
            evidence   = [kw for kw in _IMPROVEMENT_KEYWORDS if kw.lower() in recent_text][:3]
        else:
            trajectory = "stable"
            evidence   = ["لا توجد إشارات تدهور أو تحسن واضحة"]

        return trajectory, evidence

    # ── 3. Medications ────────────────────────────────────────────────────────

    def _analyze_medications(
        self, visits: list[VisitRecord]
    ) -> tuple[list[str], list[str], int]:
        """
        Extracts current and discontinued medications.
        Counts medication changes as a readmission risk factor.
        """
        all_meds_by_visit = [set(v.medications) for v in visits]

        if not all_meds_by_visit:
            return [], [], 0

        current      = list(all_meds_by_visit[-1])
        all_ever     = set().union(*all_meds_by_visit)
        discontinued = list(all_ever - set(current))

        # Count changes between consecutive visits
        changes = sum(
            1 for i in range(1, len(all_meds_by_visit))
            if all_meds_by_visit[i] != all_meds_by_visit[i - 1]
        )

        return (
            sorted(current),
            sorted(discontinued),
            changes,
        )

    # ── 4. Visit patterns ─────────────────────────────────────────────────────

    def _visit_patterns(
        self, visits: list[VisitRecord]
    ) -> tuple[float, dict[str, int], list[str]]:
        """
        Computes visit frequency, category distribution, and top complaints.
        """
        span_days = self._visit_span(visits)
        frequency = round(
            len(visits) / max(span_days / 30, 1), 2
        )   # visits per month

        # Category distribution
        categories: dict[str, int] = {c: 0 for c in _VISIT_CATEGORIES}
        all_reasons: list[str]      = []

        for visit in visits:
            text = (visit.reason + " " + visit.notes).lower()
            all_reasons.append(visit.reason)
            for cat, keywords in _VISIT_CATEGORIES.items():
                if any(kw.lower() in text for kw in keywords):
                    categories[cat] += 1
                    break

        # Top complaints
        reason_counter = Counter(all_reasons)
        top_complaints = [r for r, _ in reason_counter.most_common(5) if r]

        return frequency, categories, top_complaints

    # ── 5. Readmission signals ────────────────────────────────────────────────

    def _readmission_signals(
        self,
        visits:     list[VisitRecord],
        conditions: list[str],
        med_changes:int,
    ) -> list[str]:
        """
        Identifies readmission risk signals.
        Feeds directly into readmission_predictor.py.
        """
        signals = []

        # High visit frequency
        span = self._visit_span(visits)
        freq = len(visits) / max(span / 30, 1)
        if freq > 3:
            signals.append(f"تردد عالي على المستشفى ({freq:.1f} زيارة/شهر)")

        # Multiple chronic conditions
        if len(conditions) >= 3:
            signals.append(f"أمراض مزمنة متعددة ({len(conditions)} حالة)")

        # High medication changes
        if med_changes >= 3:
            signals.append(f"تغييرات متكررة في الأدوية ({med_changes} مرة)")

        # Declining trajectory
        if visits:
            _, traj_evidence = self._assess_trajectory(visits)
            recent_text = " ".join([v.reason + " " + v.notes for v in visits[-2:]]).lower()
            if any(kw.lower() in recent_text for kw in _ESCALATION_KEYWORDS):
                signals.append("تدهور في آخر الزيارات")

        # Emergency visits in history
        emergency_visits = sum(
            1 for v in visits
            if any(kw in (v.reason + v.notes).lower() for kw in ["طوارئ", "emergency"])
        )
        if emergency_visits >= 2:
            signals.append(f"سبق إدخال الطوارئ {emergency_visits} مرات")

        return signals

    # ── 2. Medication adherence proxy ────────────────────────────────────────

    def _medication_adherence(
        self, visits: list[VisitRecord]
    ) -> Optional[str]:
        """
        Estimates medication adherence from prescription refill patterns.

        If a patient requests refills at irregular intervals (e.g. every
        45 days instead of every 30), it suggests they skip doses or
        stop taking medication without telling the doctor.

        Returns an alert string if non-adherence is suspected, else None.
        """
        refill_visits = [
            v for v in visits
            if any(kw in (v.reason + v.notes).lower()
                   for kw in ["refill", "روشتة", "دواء تاني", "نفد الدواء",
                               "خلص الدواء", "prescription"])
        ]

        if len(refill_visits) < 3:
            return None   # not enough data

        # Parse dates of refill visits
        dates = [
            self._parse_date(v.date)
            for v in refill_visits
            if self._parse_date(v.date)
        ]
        if len(dates) < 3:
            return None

        # Compute intervals between refills
        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        avg_interval = sum(intervals) / len(intervals)
        max_interval = max(intervals)
        min_interval = min(intervals)
        variance     = max_interval - min_interval

        # High variance = irregular refills = possible non-adherence
        if variance > 20 and avg_interval > 35:
            return (
                f"انتظام الدواء غير منتظم — فترات صرف تتراوح بين "
                f"{min_interval} و{max_interval} يوم "
                f"(متوسط {avg_interval:.0f} يوم). "
                f"قد يتخطى المريض جرعات."
            )
        return None

    # ── 6. Alert patterns ────────────────────────────────────────────────────

    def _detect_alert_patterns(
        self,
        visits:    list[VisitRecord],
        complaints:list[str],
    ) -> list[str]:
        """
        Detects recurring symptoms that need medical attention.
        """
        alerts = []
        reason_counter = Counter(v.reason for v in visits)

        # Recurring same complaint
        for reason, count in reason_counter.most_common(3):
            if count >= 3 and reason:
                alerts.append(f"'{reason}' تكرّرت {count} مرات — تحتاج تقييم")

        # Long gap then sudden visits
        if len(visits) >= 2:
            dates = [self._parse_date(v.date) for v in visits if self._parse_date(v.date)]
            if len(dates) >= 2:
                gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                max_gap  = max(gaps) if gaps else 0
                last_gap = gaps[-1] if gaps else 0
                if max_gap > 180 and last_gap < 30:
                    alerts.append("فترة انقطاع طويلة ثم زيارات متتالية — قد تدل على تدهور مفاجئ")

        return alerts

    # ── 7. Clinical profile updates ───────────────────────────────────────────

    def _build_profile_updates(
        self,
        conditions:   list[str],
        current_meds: list[str],
        visits:       list[VisitRecord],
    ) -> dict:
        """
        Builds updates to merge into session["metadata"] (clinical_profile).
        These updates improve confidence_scorer and chat responses.
        """
        return {
            "has_diabetes":      "diabetes"      in conditions,
            "is_pregnant":       "pregnancy"     in [v.reason.lower() for v in visits[-2:]],
            "has_hypertension":  "hypertension"  in conditions,
            "has_heart_disease": "heart_disease" in conditions,
            "current_medications": current_meds,
            "chronic_conditions":  conditions,
        }

    # ── 8. Arabic summary ────────────────────────────────────────────────────

    def _build_summary(
        self,
        visits:             list[VisitRecord],
        conditions:         list[str],
        trajectory:         str,
        readmission_signals:list[str],
        frequency:          float,
    ) -> str:
        """
        Generates a concise Arabic summary for 09_doctor_dashboard.html.
        """
        _TRAJ_AR = {
            "improving": "الحالة في تحسن",
            "stable":    "الحالة مستقرة",
            "declining": "يُلاحظ تدهور في الحالة",
        }
        _COND_AR = {
            "diabetes":      "سكري",
            "hypertension":  "ضغط مرتفع",
            "heart_disease": "أمراض قلب",
            "respiratory":   "أمراض تنفسية",
            "kidney_disease":"أمراض كلى",
            "liver_disease": "أمراض كبد",
            "thyroid":       "غدة درقية",
            "anemia":        "أنيميا",
        }

        cond_ar   = [_COND_AR.get(c, c) for c in conditions[:3]]
        traj_ar   = _TRAJ_AR.get(trajectory, trajectory)
        span_days = self._visit_span(visits)

        lines = [
            f"سجل طبي: {len(visits)} زيارة على مدار {span_days} يوم "
            f"({frequency:.1f} زيارة/شهر).",
        ]

        if cond_ar:
            lines.append(f"الأمراض المزمنة: {', '.join(cond_ar)}.")

        lines.append(f"مسار الحالة: {traj_ar}.")

        if readmission_signals:
            lines.append(
                f"مؤشرات خطر إعادة الإدخال: {'; '.join(readmission_signals[:2])}."
            )

        return " ".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _visit_span(self, visits: list[VisitRecord]) -> int:
        if len(visits) < 2:
            return 0
        dates = [self._parse_date(v.date) for v in visits if self._parse_date(v.date)]
        if len(dates) < 2:
            return 0
        return (dates[-1] - dates[0]).days

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def _empty_result(self, patient_id_hash: str) -> HistoryAnalysisResult:
        return HistoryAnalysisResult(
            patient_id_hash      = patient_id_hash,
            analysis_date        = datetime.now(timezone.utc).isoformat(),
            total_visits         = 0,
            visit_span_days      = 0,
            detected_conditions  = [],
            condition_confidence = {},
            trajectory           = "stable",
            trajectory_evidence  = [],
            current_medications  = [],
            discontinued_meds    = [],
            medication_changes   = 0,
            visit_frequency      = 0.0,
            frequent_complaints  = [],
            visit_categories     = {},
            readmission_signals  = [],
            alert_patterns       = [],
            profile_updates      = {},
            summary_ar           = "لا يوجد تاريخ طبي مسجل.",
            duration_ms          = 0.0,
        )


# ─── Singleton ────────────────────────────────────────────────────────────────

_analyzer = HistoryAnalyzer()


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_history(
    patient_id_hash: str,
    visits:          list[dict],
    current_session: Optional[dict] = None,
) -> dict:
    """
    Main entry point.

    Usage in triage_engine.py:
        from .history_analyzer import analyze_history

        history = analyze_history(
            patient_id_hash = session["metadata"]["patient_id_hash"],
            visits          = db.get_visits(patient_id_hash),
            current_session = session,
        )
        # Merge profile updates into session
        session["metadata"].update(history["profile_updates"])
        # Use readmission signals as extra features
        triage_features["prev_admissions"] = len(history["readmission_signals"])

    Usage in chat.py:
        history = analyze_history(pid_hash, visits)
        if history["trajectory"] == "declining":
            system_prompt += "\\nالمريض في حالة تدهور — كن أكثر حذراً."
    """
    visit_records = [
        VisitRecord(
            date        = v.get("date", ""),
            reason      = v.get("reason", ""),
            diagnosis   = v.get("diagnosis", ""),
            medications = v.get("medications", []),
            vitals      = v.get("vitals", {}),
            notes       = v.get("notes", ""),
        )
        for v in visits
    ]

    result = _analyzer.analyze(patient_id_hash, visit_records, current_session)
    return result.to_dict()


def get_profile_updates(
    patient_id_hash: str,
    visits:          list[dict],
) -> dict:
    """
    Quick version — returns only profile_updates for session["metadata"].
    Used when full analysis isn't needed.
    """
    result = analyze_history(patient_id_hash, visits)
    return result.get("profile_updates", {})


def get_readmission_features(
    patient_id_hash: str,
    visits:          list[dict],
) -> dict:
    """
    Returns features formatted for readmission_predictor.py.
    """
    result = analyze_history(patient_id_hash, visits)
    return {
        "num_prior_visits":      result["total_visits"],
        "visit_frequency":       result["visit_frequency"],
        "num_conditions":        len(result["detected_conditions"]),
        "medication_changes":    result["medication_changes"],
        "trajectory_declining":  int(result["trajectory"] == "declining"),
        "readmission_signals":   len(result["readmission_signals"]),
        "visit_span_days":       result["visit_span_days"],
    }
