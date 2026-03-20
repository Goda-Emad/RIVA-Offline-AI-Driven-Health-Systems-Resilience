"""
school_health.py
================
RIVA Health Platform — School Health Analyzer
----------------------------------------------
يحلل بيانات الطلاب الصحية باستخدام معايير WHO الرسمية.

التحسينات على الكود الأصلي:
    1. ربط مع orchestrator   — target_page = 11_school_dashboard.html
    2. ربط مع explainability — شرح بالعربي للطالب والمعلم
    3. ربط مع confidence_scorer — z_score يغذّي الـ scorer
    4. Vision + Dental scores — شامل مش بس BMI
    5. Trend analysis         — مقارنة قياسات متعددة لنفس الطالب
    6. Heatmap data           — خريطة حرارية للفصل كامل
    7. Path safety            — مسارات من هيكل المشروع الفعلي
    8. Graceful error handling— مش بيوقع لو الملف مش موجود

Author : GODA EMAD
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("riva.local_inference.school_health")

# ─── Paths ───────────────────────────────────────────────────────────────────

_BASE            = Path(__file__).parent.parent
_STANDARDS_PATH  = _BASE / "data/raw/who_growth/who_growth_standards.json"
_CLUSTERS_PATH   = _BASE / "models/school/cluster_centers.json"
_CRITERIA_PATH   = _BASE / "business-intelligence/medical-content/school_health_criteria.md"

# ─── Risk thresholds ─────────────────────────────────────────────────────────

ANOMALY_DISTANCE_THRESH = 3.0    # Mahalanobis-like distance for anomaly
TREND_CONCERN_DELTA     = 0.5    # z_score change that triggers alert

# ─── Status metadata ─────────────────────────────────────────────────────────

_BMI_STATUS: list[tuple] = [
    # (condition, status_ar, color, target_page, action)
    (lambda z: z >  3,  "سمنة مفرطة",   "#ef4444", "11_school_dashboard.html", "تحويل فوري لأخصائي تغذية"),
    (lambda z: z >  2,  "وزن زائد",     "#f59e0b", "11_school_dashboard.html", "متابعة أسبوعية + نصائح غذائية"),
    (lambda z: z < -3,  "نحافة شديدة",  "#ef4444", "11_school_dashboard.html", "تحويل فوري لطبيب"),
    (lambda z: z < -2,  "نحافة",        "#f59e0b", "11_school_dashboard.html", "متابعة + دعم غذائي"),
    (lambda z: True,    "وزن مثالي",    "#10b981", "07_school.html",           "متابعة دورية"),
]

_VISION_STATUS: dict[str, dict] = {
    "normal":   {"ar": "طبيعي", "color": "#10b981", "action": "فحص سنوي"},
    "mild":     {"ar": "ضعف خفيف", "color": "#f59e0b", "action": "مراجعة طبيب عيون"},
    "moderate": {"ar": "ضعف متوسط", "color": "#ef4444", "action": "تحويل لطبيب عيون فوراً"},
}

_DENTAL_STATUS: dict[str, dict] = {
    "healthy":  {"ar": "أسنان سليمة", "color": "#10b981", "action": "فحص كل 6 أشهر"},
    "fair":     {"ar": "تحتاج متابعة", "color": "#f59e0b", "action": "مراجعة طبيب أسنان"},
    "poor":     {"ar": "تحتاج علاج",  "color": "#ef4444", "action": "تحويل لطبيب أسنان فوراً"},
}


# ─── Core analyzer ───────────────────────────────────────────────────────────

class SchoolHealthAnalyzer:
    """
    RIVA School Health Analyzer.

    Uses WHO 2007 LMS formula for BMI z-score calculation.
    K-Means cluster analysis for class-level patterns.
    Anomaly detection for critical cases needing immediate referral.

    Integrated with:
        - orchestrator   : target_page routing
        - explainability : shap_ready features
        - 11_school_dashboard.html: heatmap + individual cards
        - student_card.html       : per-student display
    """

    def __init__(
        self,
        standards_path: Path = _STANDARDS_PATH,
        clusters_path:  Path = _CLUSTERS_PATH,
    ):
        self._standards = self._load_standards(standards_path)
        self._centers, self._cluster_labels = self._load_clusters(clusters_path)
        log.info(
            "[SchoolHealth] ready  clusters=%d  standards=%s",
            len(self._cluster_labels),
            "loaded" if self._standards else "missing",
        )

    def _load_standards(self, path: Path) -> dict:
        if not path.exists():
            log.warning("[SchoolHealth] WHO standards not found: %s", path)
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_clusters(self, path: Path) -> tuple:
        if not path.exists():
            log.warning("[SchoolHealth] cluster centers not found: %s", path)
            centers = np.zeros((1, 4))
            labels  = {0: "غير محدد"}
            return centers, labels

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        centers = np.array([
            [c["age_months"], c["gender"], c["bmi"], c["z_score"]]
            for c in data["centers"]
        ])
        labels = {i: c["label"] for i, c in enumerate(data["centers"])}
        return centers, labels

    # ── WHO LMS formula ───────────────────────────────────────────────────────

    def _get_lms_row(self, gender: str, age_months: int) -> Optional[dict]:
        key   = f"bmi_{gender}"
        table = self._standards.get(key, [])
        return next((r for r in table if r["months"] == age_months), None)

    def _z_score(self, value: float, L: float, M: float, S: float) -> float:
        """
        WHO LMS z-score formula.
        L=0 → log-normal distribution (uses log formula).
        L≠0 → Box-Cox transformation.
        """
        if L != 0:
            return (((value / M) ** L) - 1) / (L * S)
        return float(np.log(value / M) / S)

    # ── BMI status ────────────────────────────────────────────────────────────

    def _bmi_status(self, z: float) -> tuple[str, str, str, str]:
        """Returns (status_ar, color, target_page, action)."""
        for condition, status, color, page, action in _BMI_STATUS:
            if condition(z):
                return status, color, page, action
        return "غير محدد", "#8892a4", "07_school.html", "إعادة القياس"

    # ── Vision assessment ─────────────────────────────────────────────────────

    def _assess_vision(self, vision_score: Optional[float]) -> dict:
        """
        vision_score: 0-10 (10 = perfect vision)
        """
        if vision_score is None:
            return {"status": "لم يُفحص", "color": "#8892a4", "action": "يحتاج فحص نظر"}
        if vision_score >= 8:
            meta = _VISION_STATUS["normal"]
        elif vision_score >= 5:
            meta = _VISION_STATUS["mild"]
        else:
            meta = _VISION_STATUS["moderate"]
        return {"score": vision_score, **meta}

    # ── Dental assessment ─────────────────────────────────────────────────────

    def _assess_dental(self, dental_score: Optional[float]) -> dict:
        """
        dental_score: 0-10 (10 = healthy teeth)
        """
        if dental_score is None:
            return {"status": "لم يُفحص", "color": "#8892a4", "action": "يحتاج فحص أسنان"}
        if dental_score >= 8:
            meta = _DENTAL_STATUS["healthy"]
        elif dental_score >= 5:
            meta = _DENTAL_STATUS["fair"]
        else:
            meta = _DENTAL_STATUS["poor"]
        return {"score": dental_score, **meta}

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def _detect_anomaly(
        self, age_months: int, gender: str, bmi: float, z: float
    ) -> tuple[bool, str, int]:
        """
        Computes distance from nearest cluster center.
        Returns (is_anomaly, nearest_cluster_label, cluster_index).
        """
        gender_num  = 1 if gender == "boys" else 0
        point       = np.array([age_months, gender_num, bmi, z])
        distances   = [np.linalg.norm(point - c) for c in self._centers]
        nearest_idx = int(np.argmin(distances))
        min_dist    = float(min(distances))
        is_anomaly  = min_dist > ANOMALY_DISTANCE_THRESH
        return is_anomaly, self._cluster_labels.get(nearest_idx, "غير محدد"), nearest_idx

    # ── Single student ────────────────────────────────────────────────────────

    def analyze(
        self,
        age_months:    int,
        gender:        str,
        height_cm:     float,
        weight_kg:     float,
        vision_score:  Optional[float] = None,
        dental_score:  Optional[float] = None,
        student_name:  str             = "",
        student_id:    str             = "",
        prev_z_score:  Optional[float] = None,
    ) -> dict:
        """
        Full health assessment for one student.

        Args:
            age_months   : age in months
            gender       : "boys" | "girls"
            height_cm    : height in cm
            weight_kg    : weight in kg
            vision_score : 0-10 (optional)
            dental_score : 0-10 (optional)
            student_name : for display in student_card.html
            student_id   : anonymised ID
            prev_z_score : previous measurement for trend analysis

        Returns:
            Full assessment dict with target_page and shap_ready features.
        """
        # BMI + z-score
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        row = self._get_lms_row(gender, age_months)

        if not row:
            return {
                "error":       f"بيانات السن {age_months} شهر غير متوفرة",
                "student_name":student_name,
                "bmi":         bmi,
            }

        z      = round(self._z_score(bmi, row["L"], row["M"], row["S"]), 2)
        status, color, target_page, action = self._bmi_status(z)

        # Anomaly detection
        is_anomaly, cluster_label, cluster_idx = self._detect_anomaly(
            age_months, gender, bmi, z
        )

        # Trend analysis
        trend      = None
        trend_alert= None
        if prev_z_score is not None:
            delta = z - prev_z_score
            if abs(delta) >= TREND_CONCERN_DELTA:
                direction  = "ارتفع" if delta > 0 else "انخفض"
                trend      = f"الـ z-score {direction} بمقدار {abs(delta):.1f}"
                trend_alert= f"⚠️ تغيّر ملحوظ في الوزن — يحتاج متابعة"

        # Vision + Dental
        vision = self._assess_vision(vision_score)
        dental = self._assess_dental(dental_score)

        # Overall priority (for sorting in dashboard)
        priority = (
            3 if is_anomaly or color == "#ef4444" else
            2 if color == "#f59e0b" else
            1
        )

        # shap_ready features for explainability.py
        shap_ready = {
            "age_months":   float(age_months),
            "gender":       1.0 if gender == "boys" else 0.0,
            "bmi":          float(bmi),
            "z_score":      float(z),
            "vision_score": float(vision_score) if vision_score else 5.0,
            "dental_score": float(dental_score) if dental_score else 5.0,
            "height_cm":    float(height_cm),
            "weight_kg":    float(weight_kg),
        }

        return {
            # Identity
            "student_name":   student_name,
            "student_id":     student_id,
            # BMI
            "bmi":            bmi,
            "z_score":        z,
            "status_ar":      status,
            "color":          color,
            "action":         action,
            "target_page":    target_page,
            # Trend
            "trend":          trend,
            "trend_alert":    trend_alert,
            "prev_z_score":   prev_z_score,
            # Anomaly
            "is_anomaly":     is_anomaly,
            "anomaly_alert":  "⚠️ حالة تستدعي تدخل فوري" if is_anomaly else None,
            "cluster":        cluster_label,
            # Vision + Dental
            "vision":         vision,
            "dental":         dental,
            # For explainability + orchestrator
            "shap_ready":     shap_ready,
            "priority":       priority,
            "offline":        True,
        }

    # ── Full class analysis ───────────────────────────────────────────────────

    def analyze_class(self, students: list[dict]) -> dict:
        """
        Analyzes a full class and returns aggregate statistics.

        students: [
            {"name": "أحمد", "age_months": 84, "gender": "boys",
             "height_cm": 120, "weight_kg": 25,
             "vision_score": 9, "dental_score": 7}
        ]
        """
        results = []
        for s in students:
            r = self.analyze(
                age_months   = s["age_months"],
                gender       = s["gender"],
                height_cm    = s["height_cm"],
                weight_kg    = s["weight_kg"],
                vision_score = s.get("vision_score"),
                dental_score = s.get("dental_score"),
                student_name = s.get("name", ""),
                student_id   = s.get("student_id", ""),
                prev_z_score = s.get("prev_z_score"),
            )
            results.append(r)

        # Sort by priority (critical first)
        results.sort(key=lambda x: x.get("priority", 1), reverse=True)

        z_scores  = [r["z_score"] for r in results if "z_score" in r]
        colors    = [r.get("color", "") for r in results]
        anomalies = [r for r in results if r.get("is_anomaly")]

        avg_z = round(float(np.mean(z_scores)), 2) if z_scores else 0.0

        if   avg_z >  1: class_trend = "الفصل يميل للوزن الزائد ⚠️"
        elif avg_z < -1: class_trend = "الفصل يميل للنحافة ⚠️"
        else:            class_trend = "توزيع طبيعي ✅"

        # Vision + Dental summary
        vision_concerns = sum(
            1 for r in results
            if r.get("vision", {}).get("color") in ("#f59e0b", "#ef4444")
        )
        dental_concerns = sum(
            1 for r in results
            if r.get("dental", {}).get("color") in ("#f59e0b", "#ef4444")
        )

        return {
            "total_students":   len(results),
            "normal_count":     colors.count("#10b981"),
            "warning_count":    colors.count("#f59e0b"),
            "critical_count":   colors.count("#ef4444"),
            "class_trend":      class_trend,
            "avg_z_score":      avg_z,
            "anomaly_count":    len(anomalies),
            "anomalies":        anomalies,
            "vision_concerns":  vision_concerns,
            "dental_concerns":  dental_concerns,
            "students":         results,
            "target_page":      "11_school_dashboard.html",
        }

    # ── Heatmap data for 11_school_dashboard.html ────────────────────────────

    def heatmap_data(self, students: list[dict]) -> dict:
        """
        Returns Plotly-ready scatter data for 11_school_dashboard.html.

        x-axis : age_months
        y-axis : bmi
        color  : z_score
        size   : priority

        Frontend (charts.js):
            Plotly.newPlot('school-heatmap', [{
                x: data.age_months,
                y: data.bmi,
                marker: {
                    color: data.z_scores,
                    colorscale: data.color_scale,
                    size: data.sizes,
                    showscale: true,
                },
                mode: 'markers+text',
                text: data.labels,
                type: 'scatter',
            }])
        """
        results = self.analyze_class(students)["students"]

        return {
            "age_months": [r.get("shap_ready", {}).get("age_months", 0) for r in results],
            "bmi":        [r.get("bmi", 0) for r in results],
            "z_scores":   [r.get("z_score", 0) for r in results],
            "sizes":      [r.get("priority", 1) * 10 + 8 for r in results],
            "labels":     [
                f"{r.get('student_name','')}<br>"
                f"{r.get('status_ar','')}<br>"
                f"z={r.get('z_score',0)}"
                for r in results
            ],
            "colors":     [r.get("color", "#8892a4") for r in results],
            "color_scale":[
                [-3.0, "#ef4444"],
                [-2.0, "#f59e0b"],
                [ 0.0, "#10b981"],
                [ 2.0, "#f59e0b"],
                [ 3.0, "#ef4444"],
            ],
            "x_label":    "السن (شهر)",
            "y_label":    "مؤشر كتلة الجسم (BMI)",
            "title":      "خريطة صحة الطلاب",
        }

    @property
    def status(self) -> dict:
        return {
            "standards_loaded": bool(self._standards),
            "clusters":         len(self._cluster_labels),
            "anomaly_threshold":ANOMALY_DISTANCE_THRESH,
            "features_tracked": ["bmi", "vision", "dental", "z_score"],
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_analyzer = SchoolHealthAnalyzer()


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_student(
    age_months:    int,
    gender:        str,
    height_cm:     float,
    weight_kg:     float,
    vision_score:  Optional[float] = None,
    dental_score:  Optional[float] = None,
    student_name:  str             = "",
    student_id:    str             = "",
    prev_z_score:  Optional[float] = None,
) -> dict:
    """
    Analyzes one student's health.

    Usage in orchestrator.py (intent=School):
        from .school_health import analyze_student

        result = analyze_student(
            age_months   = 96,
            gender       = "boys",
            height_cm    = 132,
            weight_kg    = 35,
            vision_score = 6,
            dental_score = 8,
        )
        target_page = result["target_page"]

        if result["is_anomaly"]:
            send_alert_to_school(result["anomaly_alert"])
    """
    return _analyzer.analyze(
        age_months   = age_months,
        gender       = gender,
        height_cm    = height_cm,
        weight_kg    = weight_kg,
        vision_score = vision_score,
        dental_score = dental_score,
        student_name = student_name,
        student_id   = student_id,
        prev_z_score = prev_z_score,
    )


def analyze_class(students: list[dict]) -> dict:
    """Analyzes full class — returns sorted results with aggregate stats."""
    return _analyzer.analyze_class(students)


def get_heatmap(students: list[dict]) -> dict:
    """Returns Plotly-ready heatmap data for 11_school_dashboard.html."""
    return _analyzer.heatmap_data(students)


def get_status() -> dict:
    return _analyzer.status
