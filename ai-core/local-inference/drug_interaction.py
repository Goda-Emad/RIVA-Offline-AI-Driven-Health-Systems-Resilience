"""
drug_interaction.py
===================
RIVA Health Platform — Drug Interaction Checker
------------------------------------------------
يكشف التفاعلات الدوائية الخطيرة بين الأدوية.

التحسينات على الكود الأصلي:
    1. ربط مع clinical_profile  — يسحب أدوية المريض من الـ session تلقائياً
    2. ربط مع hallucination guard — لو الـ chat اقترح دواء يتحقق منه فوراً
    3. ربط مع confidence_scorer  — التفاعل الخطير يخفّض الـ confidence
    4. Arabic drug names          — يفهم "باراسيتامول"، "ميتفورمين" بالعربي
    5. Pregnancy safety layer     — تحذير إضافي للحوامل (FDA Category)
    6. httpx بدل requests         — متسق مع باقي المنظومة

Author : GODA EMAD
"""

from __future__ import annotations

import csv
import difflib
import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.local_inference.drug_interaction")

# ─── Paths ───────────────────────────────────────────────────────────────────

_BASE            = Path(__file__).parent.parent
_MEDICINES_CSV   = _BASE / "data/raw/drug_bank/medicines_list.csv"
_INTERACTIONS_CSV= _BASE / "data/raw/drug_bank/drug_interactions.csv"
_EXTENDED_JSON   = _BASE / "business-intelligence/medical-content/drug_conflicts.json"

# ─── Brand → Generic map ─────────────────────────────────────────────────────

BRAND_TO_GENERIC: dict[str, str] = {
    # Analgesics
    "panadol": "paracetamol", "paramol": "paracetamol",
    "adol":    "paracetamol", "cataflam": "diclofenac",
    "voltaren":"diclofenac",  "brufen":   "ibuprofen",
    "advil":   "ibuprofen",   "nurofen":  "ibuprofen",
    "aspocid": "aspirin",
    # Antibiotics
    "amoxil":  "amoxicillin", "augmentin":"amoxicillin",
    "ciproxin":"ciprofloxacin","flagyl":  "metronidazole",
    "zithromax":"azithromycin",
    # Cardiovascular
    "coumadin":"warfarin",    "marevan":  "warfarin",
    "capoten": "captopril",   "norvasc":  "amlodipine",
    "zocor":   "simvastatin", "crestor":  "rosuvastatin",
    "lanoxin": "digoxin",     "cordarone":"amiodarone",
    "lasix":   "furosemide",
    # Diabetes
    "glucophage":"metformin", "daonil":  "glibenclamide",
    "amaryl":  "glimepiride",
    # GI / Thyroid
    "nexium":  "omeprazole",  "losec":   "omeprazole",
    "synthroid":"levothyroxine","eltroxin":"levothyroxine",
    # Psych
    "valium":  "diazepam",    "xanax":   "alprazolam",
}

# ─── Arabic → Generic map (تحسين جديد) ───────────────────────────────────────

ARABIC_TO_GENERIC: dict[str, str] = {
    "باراسيتامول": "paracetamol",
    "بنادول":       "paracetamol",
    "ديكلوفيناك":  "diclofenac",
    "ايبوبروفين":  "ibuprofen",
    "أسبرين":       "aspirin",
    "أموكسيسيلين": "amoxicillin",
    "أوجمنتين":    "amoxicillin",
    "سيبروفلوكساسين":"ciprofloxacin",
    "ميترونيدازول": "metronidazole",
    "فلاجيل":      "metronidazole",
    "أزيثروميسين": "azithromycin",
    "وارفارين":    "warfarin",
    "ديجوكسين":    "digoxin",
    "أميودارون":   "amiodarone",
    "ميتفورمين":   "metformin",
    "جلوكوفاج":    "metformin",
    "أوميبرازول":  "omeprazole",
    "ليفوثيروكسين":"levothyroxine",
    "ديازيبام":    "diazepam",
    "إيبوبروفين":  "ibuprofen",
    "كابتوبريل":   "captopril",
    "أملوديبين":   "amlodipine",
    "سيمفاستاتين": "simvastatin",
    "فوروسيميد":   "furosemide",
    "إنسولين":     "insulin",
    "كورتيزون":    "cortisone",
    "ديكساميثازون":"dexamethasone",
}

# ─── Pregnancy FDA categories ─────────────────────────────────────────────────
# Category X = absolutely contraindicated in pregnancy

_PREGNANCY_RISK: dict[str, dict] = {
    "warfarin":         {"category": "X", "ar": "ممنوع في الحمل — يسبب تشوهات"},
    "methotrexate":     {"category": "X", "ar": "ممنوع في الحمل — سام للجنين"},
    "isotretinoin":     {"category": "X", "ar": "ممنوع في الحمل — تشوهات خطيرة"},
    "ibuprofen":        {"category": "D", "ar": "تجنّب في الثلث الثالث من الحمل"},
    "aspirin":          {"category": "D", "ar": "تجنّب الجرعات العالية في الحمل"},
    "diazepam":         {"category": "D", "ar": "خطر الإدمان عند الجنين"},
    "alprazolam":       {"category": "D", "ar": "خطر الإدمان عند الجنين"},
    "metformin":        {"category": "B", "ar": "آمن نسبياً بإشراف طبي"},
    "amoxicillin":      {"category": "B", "ar": "آمن للاستخدام في الحمل"},
    "paracetamol":      {"category": "B", "ar": "المسكن المفضّل في الحمل"},
    "levothyroxine":    {"category": "A", "ar": "آمن وضروري في الحمل"},
}

# ─── Local interactions DB ───────────────────────────────────────────────────

LOCAL_INTERACTIONS: dict[tuple[str, str], dict] = {
    ("warfarin",    "aspirin"):       {"severity":"high",   "effect_ar":"خطر نزيف داخلي شديد"},
    ("warfarin",    "ibuprofen"):     {"severity":"high",   "effect_ar":"زيادة خطر النزيف"},
    ("warfarin",    "paracetamol"):   {"severity":"high",   "effect_ar":"تعزيز مفعول وارفارين"},
    ("metformin",   "alcohol"):       {"severity":"high",   "effect_ar":"خطر الحماض اللاكتيكي"},
    ("digoxin",     "amiodarone"):    {"severity":"high",   "effect_ar":"سمية قلبية خطيرة"},
    ("simvastatin", "amlodipine"):    {"severity":"high",   "effect_ar":"تلف عضلي (رابدومايوليسيس)"},
    ("ciprofloxacin","warfarin"):     {"severity":"high",   "effect_ar":"تعزيز مفعول وارفارين بشدة"},
    ("aspirin",     "ibuprofen"):     {"severity":"medium", "effect_ar":"قرحة معدة وتقليل فعالية الأسبرين"},
    ("metformin",   "glibenclamide"): {"severity":"medium", "effect_ar":"هبوط سكر الدم"},
    ("captopril",   "potassium"):     {"severity":"medium", "effect_ar":"ارتفاع خطير في البوتاسيوم"},
    ("omeprazole",  "clopidogrel"):   {"severity":"medium", "effect_ar":"تقليل فعالية الكلوبيدوجريل"},
    ("levothyroxine","calcium"):      {"severity":"low",    "effect_ar":"تقليل امتصاص الثيروكسين"},
    ("calcium",     "iron"):          {"severity":"low",    "effect_ar":"تقليل امتصاص الحديد"},
    ("insulin",     "alcohol"):       {"severity":"high",   "effect_ar":"هبوط سكر شديد"},
    ("metformin",   "contrast"):      {"severity":"high",   "effect_ar":"خطر الحماض قبل صبغة الأشعة"},
    ("dexamethasone","insulin"):      {"severity":"medium", "effect_ar":"رفع السكر — قد يحتاج زيادة الإنسولين"},
}

SEVERITY_LABELS = {
    "high":   "خطير جداً ⛔",
    "medium": "تحذير ⚠️",
    "low":    "ملاحظة ℹ️",
}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# ─── Confidence impact ───────────────────────────────────────────────────────
# لو الـ chat اقترح دواء وفيه تفاعل خطير → نخفّض الـ confidence

SEVERITY_CONFIDENCE_PENALTY = {
    "high":   0.40,
    "medium": 0.20,
    "low":    0.05,
}


# ─── Name normalisation ──────────────────────────────────────────────────────

def _load_medicines_csv() -> None:
    if not _MEDICINES_CSV.exists():
        log.warning("[DrugChecker] medicines CSV not found: %s", _MEDICINES_CSV)
        return
    with open(_MEDICINES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            brand   = row["brand_name"].lower().strip()
            generic = row["generic_name"].lower().strip()
            BRAND_TO_GENERIC[brand] = generic
    log.info("[DrugChecker] medicines loaded: %d names", len(BRAND_TO_GENERIC))


_load_medicines_csv()

_ALL_KNOWN = (
    list(BRAND_TO_GENERIC.keys()) +
    list(set(BRAND_TO_GENERIC.values())) +
    list(ARABIC_TO_GENERIC.keys())
)


def _fuzzy_match(name: str, cutoff: float = 0.80) -> Optional[str]:
    matches = difflib.get_close_matches(name, _ALL_KNOWN, n=1, cutoff=cutoff)
    if matches:
        log.info("[DrugChecker] fuzzy: '%s' → '%s'", name, matches[0])
        return matches[0]
    return None


def smart_normalize(name: str) -> str:
    """
    Converts any drug name (brand/Arabic/misspelled) to generic English.
    Pipeline: Arabic → Brand → Fuzzy → lowercase fallback
    """
    cleaned = name.lower().strip()

    # Arabic names
    if name in ARABIC_TO_GENERIC:
        g = ARABIC_TO_GENERIC[name]
        log.info("[DrugChecker] arabic: '%s' → '%s'", name, g)
        return g

    # Brand → generic
    if cleaned in BRAND_TO_GENERIC:
        g = BRAND_TO_GENERIC[cleaned]
        log.info("[DrugChecker] brand: '%s' → '%s'", name, g)
        return g

    # Fuzzy match
    fuzzy = _fuzzy_match(cleaned)
    if fuzzy:
        g = BRAND_TO_GENERIC.get(fuzzy, ARABIC_TO_GENERIC.get(fuzzy, fuzzy))
        log.info("[DrugChecker] fuzzy+brand: '%s' → '%s'", name, g)
        return g

    return cleaned



# ─── Drug-Food interactions ───────────────────────────────────────────────────
# بسيطة تقنياً — ضخمة طبياً
# كل تفاعل اتحقق منه طبياً بالمراجع

DRUG_FOOD_INTERACTIONS: dict[str, list[dict]] = {
    "warfarin": [
        {"food": "ثوم (garlic)",       "severity": "high",   "effect_ar": "زيادة خطر النزيف"},
        {"food": "جريب فروت",          "severity": "medium", "effect_ar": "تعزيز مفعول وارفارين"},
        {"food": "خضروات ورقية كثيرة", "severity": "medium", "effect_ar": "تقليل مفعول وارفارين (فيتامين K)"},
        {"food": "كحول",               "severity": "high",   "effect_ar": "خطر نزيف شديد"},
    ],
    "simvastatin": [
        {"food": "جريب فروت",          "severity": "high",   "effect_ar": "تلف عضلي — رابدومايوليسيس"},
    ],
    "amlodipine": [
        {"food": "جريب فروت",          "severity": "medium", "effect_ar": "زيادة تركيز الدواء في الدم"},
    ],
    "metformin": [
        {"food": "كحول",               "severity": "high",   "effect_ar": "خطر الحماض اللاكتيكي"},
    ],
    "levothyroxine": [
        {"food": "قهوة",               "severity": "medium", "effect_ar": "تقليل امتصاص الثيروكسين"},
        {"food": "حليب أو كالسيوم",    "severity": "medium", "effect_ar": "تقليل امتصاص الثيروكسين"},
        {"food": "فول الصويا",         "severity": "low",    "effect_ar": "تداخل في الامتصاص"},
    ],
    "ciprofloxacin": [
        {"food": "منتجات الألبان",      "severity": "medium", "effect_ar": "تقليل امتصاص المضاد الحيوي"},
        {"food": "كافيين",             "severity": "low",    "effect_ar": "زيادة تأثير الكافيين"},
    ],
    "digoxin": [
        {"food": "عرق السوس",          "severity": "high",   "effect_ar": "اضطراب القلب"},
        {"food": "نخالة",              "severity": "low",    "effect_ar": "تقليل امتصاص الديجوكسين"},
    ],
    "captopril": [
        {"food": "جريب فروت",          "severity": "medium", "effect_ar": "زيادة مفعول دواء الضغط"},
        {"food": "أطعمة غنية بالبوتاسيوم","severity":"medium","effect_ar":"ارتفاع البوتاسيوم في الدم"},
    ],
    "aspirin": [
        {"food": "كحول",               "severity": "high",   "effect_ar": "زيادة خطر نزيف المعدة"},
    ],
    "insulin": [
        {"food": "كحول",               "severity": "high",   "effect_ar": "هبوط سكر مفاجئ"},
    ],
}

# ─── Core checker ─────────────────────────────────────────────────────────────

class DrugInteractionChecker:

    def __init__(self) -> None:
        self.db = LOCAL_INTERACTIONS.copy()
        self._load_csv()
        self._load_extended()
        log.info("[DrugChecker] ready — %d interactions", len(self.db))

    def _load_csv(self) -> None:
        if not _INTERACTIONS_CSV.exists():
            return
        count = 0
        with open(_INTERACTIONS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["drug_a"].lower().strip(), row["drug_b"].lower().strip())
                self.db[key] = {
                    "severity":  row["severity"],
                    "effect_ar": row["effect_ar"],
                }
                count += 1
        log.info("[DrugChecker] CSV loaded: %d interactions", count)

    def _load_extended(self) -> None:
        if not _EXTENDED_JSON.exists():
            return
        with open(_EXTENDED_JSON, encoding="utf-8") as f:
            extra = json.load(f)
        for k, v in extra.items():
            parts = k.split("_")
            if len(parts) == 2:
                self.db[(parts[0], parts[1])] = v
        log.info("[DrugChecker] extended DB loaded")

    # ── Offline check ────────────────────────────────────────────────────────

    def check_offline(
        self,
        new_drug:      str,
        current_drugs: list[str],
    ) -> list[dict]:
        new_g  = smart_normalize(new_drug)
        alerts: list[dict] = []

        for drug in current_drugs:
            ex_g        = smart_normalize(drug)
            interaction = (
                self.db.get((new_g, ex_g)) or
                self.db.get((ex_g, new_g))
            )
            if interaction:
                alerts.append({
                    "drug_a":            new_drug,
                    "drug_b":            drug,
                    "drug_a_generic":    new_g,
                    "drug_b_generic":    ex_g,
                    "severity":          SEVERITY_LABELS[interaction["severity"]],
                    "severity_code":     interaction["severity"],
                    "effect_ar":         interaction["effect_ar"],
                    "confidence_penalty":SEVERITY_CONFIDENCE_PENALTY.get(
                                            interaction["severity"], 0.0),
                    "source":            "local",
                    "offline":           True,
                })
                log.warning(
                    "[DrugChecker] INTERACTION: %s + %s | %s",
                    new_drug, drug, interaction["severity"].upper(),
                )

        if not alerts:
            log.info("[DrugChecker] SAFE: %s", new_drug)

        return sorted(alerts, key=lambda x: SEVERITY_ORDER.get(x["severity_code"], 2))

    # ── Online check (OpenFDA) ────────────────────────────────────────────────

    def check_online(self, new_drug: str) -> list[dict]:
        generic = smart_normalize(new_drug)
        try:
            import httpx   # متسق مع باقي المنظومة
            url  = (
                f"https://api.fda.gov/drug/label.json"
                f"?search=openfda.generic_name:{generic}&limit=1"
            )
            resp    = httpx.get(url, timeout=3.0)
            results = resp.json().get("results", [])
            if not results:
                return []
            text = results[0].get("drug_interactions", [""])[0]
            if not text:
                return []
            log.info("[DrugChecker] OpenFDA: found interactions for %s", new_drug)
            return [{
                "drug_a":             new_drug,
                "drug_b":             "see_fda_text",
                "severity":           SEVERITY_LABELS["medium"],
                "severity_code":      "medium",
                "effect_ar":          text[:300] + "…",
                "confidence_penalty": SEVERITY_CONFIDENCE_PENALTY["medium"],
                "source":             "fda_online",
                "offline":            False,
            }]
        except Exception as e:
            log.warning("[DrugChecker] online check failed: %s", e)
            return []

    # ── Pregnancy safety ─────────────────────────────────────────────────────

    def pregnancy_check(
        self,
        new_drug:    str,
        is_pregnant: bool = False,
    ) -> Optional[dict]:
        """
        Extra layer for pregnant patients.
        Checks FDA pregnancy category and returns a warning if risky.

        Called automatically by check() if clinical_profile.is_pregnant=True.
        """
        if not is_pregnant:
            return None

        generic = smart_normalize(new_drug)
        risk    = _PREGNANCY_RISK.get(generic)
        if not risk:
            return None

        cat = risk["category"]
        if cat in ("X", "D"):
            return {
                "drug":               new_drug,
                "fda_category":       cat,
                "severity_code":      "high" if cat == "X" else "medium",
                "severity":           SEVERITY_LABELS["high" if cat == "X" else "medium"],
                "effect_ar":          risk["ar"],
                "confidence_penalty": 0.40 if cat == "X" else 0.20,
                "pregnancy_warning":  True,
            }
        return None

    # ── Main check ───────────────────────────────────────────────────────────

    def check(
        self,
        new_drug:         str,
        current_drugs:    list[str],
        clinical_profile: dict = {},
    ) -> dict:
        """
        Full drug interaction check.

        Args:
            new_drug         : الدواء الجديد (brand أو generic أو عربي)
            current_drugs    : الأدوية الحالية للمريض
            clinical_profile : من session metadata
                               تلقائياً يشيل is_pregnant, has_diabetes...

        Returns:
            {
                alerts            : list of interaction alerts
                pregnancy_alert   : warning if pregnant + risky drug
                is_safe           : bool
                confidence_impact : max confidence penalty to apply
                source_mode       : "offline" | "hybrid"
            }
        """
        log.info("[DrugChecker] check: '%s' with %s", new_drug, current_drugs)

        # Offline interactions
        alerts    = self.check_offline(new_drug, current_drugs)

        # Online enrichment (non-blocking)
        online: list[dict] = []
        try:
            online = self.check_online(new_drug)
        except Exception:
            pass

        all_alerts = alerts + online

        # Pregnancy safety layer
        is_pregnant      = clinical_profile.get("is_pregnant", False)
        pregnancy_alert  = self.pregnancy_check(new_drug, is_pregnant)

        # Max confidence penalty (لو في تفاعل خطير → confidence ينزل)
        penalties        = [a["confidence_penalty"] for a in all_alerts]
        if pregnancy_alert:
            penalties.append(pregnancy_alert["confidence_penalty"])
        max_penalty      = max(penalties) if penalties else 0.0

        is_safe = (
            not any(a["severity_code"] == "high" for a in all_alerts)
            and (pregnancy_alert is None or pregnancy_alert["severity_code"] != "high")
        )

        # Drug-food warnings
        food_warnings = self.check_food(new_drug)

        result = {
            "new_drug":          new_drug,
            "new_drug_generic":  smart_normalize(new_drug),
            "checked_against":   current_drugs,
            "alerts":            all_alerts,
            "pregnancy_alert":   pregnancy_alert,
            "food_warnings":     food_warnings,
            "is_safe":           is_safe,
            "confidence_impact": round(max_penalty, 2),
            "source_mode":       "hybrid" if online else "offline",
            "alert_count":       len(all_alerts),
            "food_alert_count":  len(food_warnings),
            "offline":           True,
        }

        log.info(
            "[DrugChecker] %s | alerts=%d | penalty=%.2f",
            "SAFE" if is_safe else "DANGER",
            len(all_alerts),
            max_penalty,
        )
        return result

    def is_safe(
        self,
        new_drug:      str,
        current_drugs: list[str],
        clinical_profile: dict = {},
    ) -> bool:
        return self.check(new_drug, current_drugs, clinical_profile)["is_safe"]

    # ── Drug-Food check ─────────────────────────────────────────────────────

    def check_food(self, drug: str) -> list[dict]:
        """
        Returns food interactions for a drug.
        Simple technically — huge medically.

        Usage in prescription_gen.py:
            food_warnings = checker.check_food("warfarin")
            # [{food: "ثوم", severity: "high", effect_ar: "..."}]
        """
        generic = smart_normalize(drug)
        items   = DRUG_FOOD_INTERACTIONS.get(generic, [])
        if items:
            log.info("[DrugChecker] food interactions for '%s': %d", drug, len(items))
        return sorted(items, key=lambda x: SEVERITY_ORDER.get(x["severity"], 2))

    # ── Heatmap data ─────────────────────────────────────────────────────────

    def heatmap_data(self, drug_list: list[str]) -> dict:
        """
        Builds a drug-drug interaction matrix for the Doctor Dashboard.
        Used to render a Heatmap in 09_doctor_dashboard.html via Plotly.

        Returns:
            {
                "drugs"  : ["warfarin", "aspirin", ...],
                "matrix" : [[0,2,0,...], [2,0,1,...], ...],
                "legend" : {0: "آمن", 1: "تحذير", 2: "خطير"}
            }

        Frontend usage (charts.js):
            const heatmap = await fetch('/drug/heatmap', {body: {drugs: [...]}})
            Plotly.newPlot('heatmap-div', [{
                z: heatmap.matrix,
                x: heatmap.drugs,
                y: heatmap.drugs,
                type: 'heatmap',
                colorscale: [[0,'#10b981'],[0.5,'#f59e0b'],[1,'#ef4444']]
            }])
        """
        generics = [smart_normalize(d) for d in drug_list]
        n        = len(generics)
        matrix   = [[0] * n for _ in range(n)]
        severity_score = {"high": 2, "medium": 1, "low": 0}

        for i in range(n):
            for j in range(i + 1, n):
                interaction = (
                    self.db.get((generics[i], generics[j])) or
                    self.db.get((generics[j], generics[i]))
                )
                if interaction:
                    score = severity_score.get(interaction["severity"], 0)
                    matrix[i][j] = score
                    matrix[j][i] = score

        return {
            "drugs":  generics,
            "matrix": matrix,
            "legend": {0: "آمن ✓", 1: "تحذير ⚠️", 2: "خطير ⛔"},
            "title":  "خريطة التفاعلات الدوائية",
        }

    @property
    def stats(self) -> dict:
        return {
            "total_interactions":      len(self.db),
            "arabic_names":            len(ARABIC_TO_GENERIC),
            "brand_names":             len(BRAND_TO_GENERIC),
            "pregnancy_tracked":       len(_PREGNANCY_RISK),
            "drug_food_tracked":       len(DRUG_FOOD_INTERACTIONS),
            "food_interaction_pairs":  sum(
                len(v) for v in DRUG_FOOD_INTERACTIONS.values()
            ),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_checker = DrugInteractionChecker()


# ─── Public API ───────────────────────────────────────────────────────────────

def check_interaction(
    new_drug:         str,
    current_drugs:    list[str],
    clinical_profile: dict = {},
) -> dict:
    """
    Main entry point — used by prescription_gen.py and chat.py hallucination guard.

    Usage in prescription_gen.py:
        result = check_interaction(
            new_drug         = "warfarin",
            current_drugs    = session["medications"],
            clinical_profile = session["metadata"],
        )
        if not result["is_safe"]:
            confidence -= result["confidence_impact"]
            return alert_doctor(result["alerts"])
    """
    return _checker.check(new_drug, current_drugs, clinical_profile)


def is_safe(
    new_drug:         str,
    current_drugs:    list[str],
    clinical_profile: dict = {},
) -> bool:
    return _checker.is_safe(new_drug, current_drugs, clinical_profile)


def normalize(name: str) -> str:
    """Normalises any drug name to generic English."""
    return smart_normalize(name)


def check_food_interaction(drug: str) -> list[dict]:
    """Returns food interactions for a drug."""
    return _checker.check_food(drug)


def get_heatmap(drug_list: list[str]) -> dict:
    """
    Returns interaction matrix for Doctor Dashboard Plotly heatmap.

    Usage in 09_doctor_dashboard.html:
        const data = await fetch('/drug/heatmap', {body: {drugs: patientMeds}})
        Plotly.newPlot('heatmap', [{z: data.matrix, x: data.drugs,
            y: data.drugs, type: 'heatmap',
            colorscale: [[0,'#10b981'],[0.5,'#f59e0b'],[1,'#ef4444']]}])
    """
    return _checker.heatmap_data(drug_list)


def get_stats() -> dict:
    return _checker.stats
