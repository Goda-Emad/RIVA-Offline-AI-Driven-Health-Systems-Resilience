
import json
import numpy as np
from typing import Union

class SchoolHealthAnalyzer:
    """
    RIVA - School Health Analyzer
    معادلة LMS الرسمية من WHO 2007
    Author: GODA EMAD
    """
    def __init__(self, standards_path: str = "data-storage/raw/who_growth/who_growth_standards.json"):
        with open(standards_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def _get_row(self, gender: str, age_months: int) -> dict:
        table = self.data[f"bmi_{gender}"]
        return next((r for r in table if r["months"] == age_months), None)

    def _z_score(self, value: float, L: float, M: float, S: float) -> float:
        if L != 0:
            return (((value / M) ** L) - 1) / (L * S)
        return np.log(value / M) / S

    def analyze(self, age_months: int, gender: str,
                height_cm: float, weight_kg: float) -> dict:
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        row = self._get_row(gender, age_months)
        if not row:
            return {"error": f"بيانات السن {age_months} شهر غير متوفرة"}

        z = round(self._z_score(bmi, row["L"], row["M"], row["S"]), 2)

        if   z >  3: status, color = "سمنة مفرطة 🔴",   "red"
        elif z >  2: status, color = "وزن زائد 🟡",     "yellow"
        elif z < -3: status, color = "نحافة شديدة 🔴",  "red"
        elif z < -2: status, color = "نحافة 🟡",         "yellow"
        else:        status, color = "وزن مثالي ✅",     "green"

        return {
            "bmi":            bmi,
            "z_score":        z,
            "status":         status,
            "color":          color,
            "recommendation": "متابعة دورية" if color == "green" else "تحويل للمختص",
            "age_months":     age_months,
            "gender":         gender,
        }
