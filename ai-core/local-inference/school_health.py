
import json
import numpy as np
from typing import Union

class SchoolHealthAnalyzer:
    """
    RIVA - School Health Analyzer
    معادلة LMS الرسمية من WHO 2007
    + Cluster Analysis للفصل كامل
    + Anomaly Detection للحالات الحرجة
    Author: GODA EMAD
    """
    def __init__(self,
                 standards_path: str = "data-storage/raw/who_growth/who_growth_standards.json",
                 clusters_path:  str = "ai-core/models/school/cluster_centers.json"):

        with open(standards_path, encoding="utf-8") as f:
            self.data = json.load(f)

        with open(clusters_path, encoding="utf-8") as f:
            cluster_data = json.load(f)
            self.centers = np.array([
                [c["age_months"], c["gender"], c["bmi"], c["z_score"]]
                for c in cluster_data["centers"]
            ])
            self.cluster_labels = {
                i: c["label"] for i, c in enumerate(cluster_data["centers"])
            }

    # ── LMS ─────────────────────────────────────
    def _get_row(self, gender: str, age_months: int) -> dict:
        table = self.data[f"bmi_{gender}"]
        return next((r for r in table if r["months"] == age_months), None)

    def _z_score(self, value: float, L: float, M: float, S: float) -> float:
        if L != 0:
            return (((value / M) ** L) - 1) / (L * S)
        return np.log(value / M) / S

    # ── تشخيص طالب واحد ─────────────────────────
    def analyze(self, age_months: int, gender: str,
                height_cm: float, weight_kg: float) -> dict:

        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        row = self._get_row(gender, age_months)
        if not row:
            return {"error": f"بيانات السن {age_months} شهر غير متوفرة"}

        z = round(self._z_score(bmi, row["L"], row["M"], row["S"]), 2)

        if   z >  3: status, color = "سمنة مفرطة 🔴",  "red"
        elif z >  2: status, color = "وزن زائد 🟡",    "yellow"
        elif z < -3: status, color = "نحافة شديدة 🔴", "red"
        elif z < -2: status, color = "نحافة 🟡",        "yellow"
        else:        status, color = "وزن مثالي ✅",    "green"

        # Anomaly Detection
        gender_num = 1 if gender == "boys" else 0
        point = np.array([age_months, gender_num, bmi, z])
        distances = [np.linalg.norm(point - c) for c in self.centers]
        nearest_cluster = int(np.argmin(distances))
        min_distance = round(float(min(distances)), 2)
        is_anomaly = min_distance > 3.0

        return {
            "bmi":            bmi,
            "z_score":        z,
            "status":         status,
            "color":          color,
            "recommendation": "متابعة دورية" if color == "green" else "تحويل للمختص",
            "cluster":        self.cluster_labels[nearest_cluster],
            "distance":       min_distance,
            "is_anomaly":     is_anomaly,
            "anomaly_alert":  "⚠️ حالة تستدعي تدخل فوري" if is_anomaly else None,
        }

    # ── تحليل فصل كامل ──────────────────────────
    def analyze_class(self, students: list[dict]) -> dict:
        """
        students: [{"name": "أحمد", "age_months": 84,
                    "gender": "boys", "height_cm": 120, "weight_kg": 25}]
        """
        results = []
        for s in students:
            r = self.analyze(s["age_months"], s["gender"],
                             s["height_cm"], s["weight_kg"])
            r["name"] = s.get("name", "")
            results.append(r)

        # إحصائيات الفصل
        colors   = [r["color"] for r in results]
        clusters = [r["cluster"] for r in results]
        anomalies = [r for r in results if r.get("is_anomaly")]

        # مركز ثقل الفصل
        avg_z = round(float(np.mean([r["z_score"] for r in results])), 2)
        if   avg_z >  1: class_trend = "يميل للوزن الزائد ⚠️"
        elif avg_z < -1: class_trend = "يميل للنحافة ⚠️"
        else:            class_trend = "توزيع طبيعي ✅"

        return {
            "total_students":  len(results),
            "normal_count":    colors.count("green"),
            "warning_count":   colors.count("yellow"),
            "critical_count":  colors.count("red"),
            "class_trend":     class_trend,
            "avg_z_score":     avg_z,
            "anomalies":       anomalies,
            "anomaly_count":   len(anomalies),
            "students":        results,
        }
