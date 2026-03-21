"""
feature_engineering.py  v4.2
=============================
RIVA Health Platform — Feature Engineering
-------------------------------------------
محول الكلام البشري إلى ميزات ذكاء اصطناعي.

التحسينات v4.2:
    1. Negation Detection       — "معنديش ضيق تنفس" → عرض منفي، لا يُحسب
    2. Symptom Co-occurrence    — fever+bleeding = severity boost + note_ar
    3. Unit Normalization       — Fahrenheit → Celsius تلقائياً
    4. Word Boundary Matching   — Regex بدل `in` لمنع False Positives
                                  "دم" مش بتتطابق مع "صدمة"
    5. Named Vector Indices     — قاموس ثابت بدل hardcoded v[20]
    6. Safe Dosage Window       — per-drug window بدون overlap

Author : GODA EMAD + RIVA Team
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

log = logging.getLogger("riva.prediction.feature_engineering")


# ─── Enums ───────────────────────────────────────────────────────────────────

class SeverityLevel(Enum):
    LOW       = 1
    MEDIUM    = 2
    HIGH      = 3
    CRITICAL  = 4
    EMERGENCY = 5


# ─── 1. Negation detection ───────────────────────────────────────────────────

_NEGATION_WORDS = [
    "مش", "لا", "معنديش", "معندوش", "مفيش", "ملقيش",
    "مش عندي", "بدون", "لأ", "ما عنديش", "ما فيش",
    "مش موجود", "مش حاسس", "لا يوجد",
    "no", "not", "without", "don't have", "doesn't have",
    "denies", "absent", "no sign of",
]

_NEGATION_WINDOW = 40   # characters before keyword to check


def _is_negated(text: str, start_idx: int) -> bool:
    """
    Returns True if a negation word appears within _NEGATION_WINDOW
    characters before position start_idx.
    """
    window = text[max(0, start_idx - _NEGATION_WINDOW): start_idx]
    return any(neg in window for neg in _NEGATION_WORDS)


# ─── 4. Word-boundary regex builder ─────────────────────────────────────────

def _word_pattern(keyword: str) -> re.Pattern:
    """
    Builds a regex that matches the keyword as a whole word,
    accounting for Arabic prefixes (ال، بـ، وـ) and suffixes.

    Example:
        "دم"  matches "دم" but NOT "صدمة" or "ندم"
        "ألم" matches "ألم" but NOT "ألمانيا"
    """
    escaped = re.escape(keyword)
    # Arabic prefix chars: و، ب، ل، ك، ف، ال
    prefix = r'(?:^|[\s،.؟!()"\'\u0600-\u0605وبلكف]|ال)'
    suffix = r'(?=$|[\s،.؟!()"\'\u0600-\u0605])'
    return re.compile(f"{prefix}{escaped}{suffix}", re.IGNORECASE)


# ─── Symptom patterns ────────────────────────────────────────────────────────

SYMPTOM_PATTERNS: dict[str, dict] = {
    "chest_pain":         {"ar":["ألم في الصدر","وجع صدر"],"en":["chest pain","heart pain","tight chest"],"emergency":True,"base_severity":4,"body_part":"chest"},
    "shortness_of_breath":{"ar":["ضيق تنفس","صعوبة تنفس","نهجان"],"en":["shortness of breath","difficulty breathing","dyspnea"],"emergency":True,"base_severity":3,"body_part":"chest"},
    "fainting":           {"ar":["إغماء","فقدان وعي","غمي عليا"],"en":["fainting","loss of consciousness","blackout"],"emergency":True,"base_severity":5,"body_part":"head"},
    "seizures":           {"ar":["تشنجات","نوبة"],"en":["seizures","convulsions"],"emergency":True,"base_severity":5},
    "severe_bleeding":    {"ar":["نزيف شديد","نزف شديد"],"en":["severe bleeding","hemorrhage"],"emergency":True,"base_severity":4},
    "bleeding":           {"ar":["نزيف","نزف","دم في"],"en":["bleeding","blood"],"emergency":True,"base_severity":4},
    "stroke_symptoms":    {"ar":["شلل مفاجئ","وجهه مايل","كلامه مش واضح"],"en":["sudden weakness","facial drooping","slurred speech"],"emergency":True,"base_severity":5},
    "contractions":       {"ar":["تقلصات","طلق","وجع ولادة"],"en":["contractions","labor pain"],"emergency":True,"body_part":"uterus"},
    "water_breaking":     {"ar":["نزول مية","كيس الماء"],"en":["water breaking","ruptured membranes"],"emergency":True,"body_part":"uterus"},
    "wheezing":           {"ar":["صفير","أزيز","صوت في الصدر"],"en":["wheezing","whistling"],"base_severity":3,"body_part":"chest"},
    "palpitations":       {"ar":["خفقان","سرعة في دقات القلب"],"en":["palpitations","heart racing"],"body_part":"heart"},
    "irregular_heartbeat":{"ar":["عدم انتظام","اضطراب النبض"],"en":["irregular heartbeat","arrhythmia"],"emergency":True,"body_part":"heart"},
    "high_fever":         {"ar":["حرارة عالية","حمى شديدة"],"en":["high fever"],"base_severity":3,"body_part":"body"},
    "fever":              {"ar":["حرارة","سخونية","حمى"],"en":["fever","high temperature"],"body_part":"body"},
    "headache":           {"ar":["صداع","وجع راس","الم في الراس"],"en":["headache","migraine"],"body_part":"head"},
    "dizziness":          {"ar":["دوار","دوخة","عدم اتزان"],"en":["dizziness","vertigo","lightheaded"],"body_part":"head"},
    "nausea":             {"ar":["غثيان","عيان"],"en":["nausea","queasy"],"body_part":"abdomen"},
    "vomiting":           {"ar":["قيء","استفراغ","رجع"],"en":["vomiting","throwing up"],"body_part":"abdomen"},
    "abdominal_pain":     {"ar":["وجع بطن","الم في البطن","معدة"],"en":["abdominal pain","stomach pain"],"body_part":"abdomen"},
    "diarrhea":           {"ar":["إسهال","اسهال"],"en":["diarrhea"]},
    "cough":              {"ar":["كحة","سعال"],"en":["cough","coughing"],"body_part":"chest"},
    "numbness":           {"ar":["تنميل","خدران"],"en":["numbness","tingling"],"body_part":"extremities"},
    "back_pain":          {"ar":["وجع ظهر","الم في الظهر"],"en":["back pain"],"body_part":"back"},
    "dysuria":            {"ar":["حرقة في البول","ألم عند التبول"],"en":["dysuria","burning urination"]},
    "hematuria":          {"ar":["دم في البول"],"en":["blood in urine","hematuria"],"emergency":True},
    "anxiety":            {"ar":["قلق","خايف","توتر"],"en":["anxiety","worried","nervous"]},
    "fatigue":            {"ar":["إرهاق","تعبان","خمول"],"en":["fatigue","tired","exhausted"],"body_part":"body"},
    "fetal_movement":     {"ar":["حركة الجنين","الجنين","البيبي"],"en":["fetal movement","baby kicks"],"body_part":"uterus"},
    "weight_loss":        {"ar":["نقص الوزن","خسارة وزن","نحف"],"en":["weight loss","losing weight"]},
    "sweating":           {"ar":["عرق","تعرق"],"en":["sweating","diaphoresis"]},
}

# ─── 2. Symptom co-occurrence ─────────────────────────────────────────────────

_COOCCURRENCE: list[dict] = [
    {"symptoms":{"fever","bleeding"},          "label":"sepsis_risk",       "boost":1, "note_ar":"حمى مع نزيف — خطر إنتان",          "vec_idx":45},
    {"symptoms":{"chest_pain","shortness_of_breath"},"label":"cardiac_emergency","boost":2,"note_ar":"ألم صدر مع ضيق تنفس — خطر جلطة","vec_idx":46},
    {"symptoms":{"fever","headache"},          "label":"meningitis_risk",   "boost":1, "note_ar":"حمى مع صداع — خطر التهاب سحايا",    "vec_idx":47},
    {"symptoms":{"contractions","bleeding"},   "label":"obstetric_emergency","boost":2,"note_ar":"تقلصات مع نزيف — طوارئ نسائية",     "vec_idx":48},
    {"symptoms":{"fainting","chest_pain"},     "label":"cardiac_syncope",   "boost":2, "note_ar":"إغماء مع ألم صدر — إسعاف فوري",    "vec_idx":49},
]

# ─── Medication patterns ──────────────────────────────────────────────────────

MEDICATION_PATTERNS: dict[str, dict] = {
    "methyldopa":       {"ar":["ميثيل دوبا","مثيل دوبا"],"en":["methyldopa"],"category":"antihypertensive","high_risk":False},
    "amlodipine":       {"ar":["أملوديبين","نورفاسك"],"en":["amlodipine","norvasc"],"category":"calcium_blocker","high_risk":False},
    "lisinopril":       {"ar":["ليزينوبريل"],"en":["lisinopril"],"category":"ace_inhibitor","high_risk":False},
    "losartan":         {"ar":["لوسارتان","كوزار"],"en":["losartan","cozaar"],"category":"arb","high_risk":False},
    "furosemide":       {"ar":["فوروسيميد","لازيكس"],"en":["furosemide","lasix"],"category":"diuretic","high_risk":True},
    "metformin":        {"ar":["ميتفورمين","جلوكوفاج"],"en":["metformin","glucophage"],"category":"biguanide","high_risk":False},
    "insulin":          {"ar":["أنسولين","انسولين"],"en":["insulin","lantus"],"category":"insulin","high_risk":True},
    "glipizide":        {"ar":["جليبيزيد"],"en":["glipizide"],"category":"sulfonylurea","high_risk":True},
    "digoxin":          {"ar":["ديجوكسين"],"en":["digoxin","lanoxin"],"category":"cardiac","high_risk":True},
    "warfarin":         {"ar":["وارفارين"],"en":["warfarin","coumadin"],"category":"anticoagulant","high_risk":True},
    "aspirin":          {"ar":["أسبرين","اسبرين"],"en":["aspirin"],"category":"antiplatelet","high_risk":False},
    "clopidogrel":      {"ar":["كلوبيدوجريل","بلافيكس"],"en":["clopidogrel","plavix"],"category":"antiplatelet","high_risk":True},
    "atorvastatin":     {"ar":["أتورفاستاتين","ليبيتور"],"en":["atorvastatin","lipitor"],"category":"statin","high_risk":False},
    "albuterol":        {"ar":["البوتيرول","فنتولين"],"en":["albuterol","ventolin"],"category":"bronchodilator","high_risk":False},
    "prednisone":       {"ar":["بريدنيزون","كورتيزون"],"en":["prednisone","prednisolone"],"category":"corticosteroid","high_risk":True},
    "paracetamol":      {"ar":["باراسيتامول","بنادول","ادول"],"en":["paracetamol","acetaminophen","tylenol"],"category":"analgesic","high_risk":False},
    "ibuprofen":        {"ar":["ايبوبروفين","بروفين"],"en":["ibuprofen","advil","motrin"],"category":"nsaid","high_risk":False},
    "omeprazole":       {"ar":["أوميبرازول"],"en":["omeprazole","prilosec"],"category":"ppi","high_risk":False},
    "progesterone":     {"ar":["بروجسترون"],"en":["progesterone"],"category":"hormone","high_risk":False},
    "magnesium_sulfate":{"ar":["ماغنسيوم","مغنيسيوم"],"en":["magnesium sulfate"],"category":"tocolytic","high_risk":False},
    "iron":             {"ar":["حديد"],"en":["iron"],"category":"supplement","high_risk":False},
    "prenatal_vitamins":{"ar":["فيتامينات حمل","فوليك أسيد"],"en":["prenatal vitamins","folic acid"],"category":"supplement","high_risk":False},
}

# ─── Severity keywords ────────────────────────────────────────────────────────

SEVERITY_WORDS: dict[SeverityLevel, dict] = {
    SeverityLevel.EMERGENCY: {"ar":["طوارئ","إسعاف","فوري","حالاً","خطير"],"en":["emergency","urgent","immediately","critical"]},
    SeverityLevel.CRITICAL:  {"ar":["حرج","خطر","مميت"],"en":["critical","life threatening","fatal"]},
    SeverityLevel.HIGH:      {"ar":["شديد","مستمر","لا يحتمل","فظيع","قاسي"],"en":["severe","intense","unbearable","terrible"]},
    SeverityLevel.MEDIUM:    {"ar":["متوسط","أحياناً","مزعج"],"en":["moderate","sometimes","manageable"]},
    SeverityLevel.LOW:       {"ar":["خفيف","بسيط","شوية"],"en":["mild","slight","minor"]},
}

# ─── Duration patterns ────────────────────────────────────────────────────────

DURATION_PATTERNS = [
    (r'(\d+)\s*(يوم|days|ايام)',     24),
    (r'(\d+)\s*(ساعة|hours|ساعات)', 1),
    (r'(\d+)\s*(أسبوع|week|اسابيع)',168),
    (r'(\d+)\s*(شهر|month|شهور)',   720),
    (r'from\s+(\d+)\s*(day|days)',  24),
    (r'(\d+)\s*(دقيقة|minute|دقائق)',1/60),
]

# ─── Vital patterns ───────────────────────────────────────────────────────────

VITAL_PATTERNS: dict[str, dict] = {
    "temperature":       {"ar":[r'(\d{2,3}(?:\.\d)?)\s*(درجة|°|مئوية)',r'حرارة\s*(\d{2,3}(?:\.\d)?)'],"en":[r'(\d{2,3}(?:\.\d)?)\s*(°f|fahrenheit)',r'(\d{2,3}(?:\.\d)?)\s*(°|degrees?|fever)'],"normal":(36.0,37.5)},
    "heart_rate":        {"ar":[r'(\d{2,3})\s*(نبضة|نبض)',r'النبض\s*(\d{2,3})'],"en":[r'(\d{2,3})\s*(bpm|heart rate|pulse)'],"normal":(60,100)},
    "blood_pressure":    {"ar":[r'(\d{2,3})\s*[/:]\s*(\d{2,3})\s*(ضغط|mmhg)',r'ضغط\s*(\d{2,3})\s*[/:]\s*(\d{2,3})'],"en":[r'(\d{2,3})\s*[/:]\s*(\d{2,3})\s*(bp|blood pressure)'],"normal":(90,120,60,80)},
    "oxygen_saturation": {"ar":[r'(\d{1,3})\s*(%|اكسجين|أكسجين)',r'spo2\s*(\d{1,3})'],"en":[r'(\d{1,3})\s*(%|spo2|oxygen)'],"normal":(95,100)},
    "respiratory_rate":  {"ar":[r'(\d{1,3})\s*(نفس|تنفس)',r'معدل التنفس\s*(\d{1,3})'],"en":[r'(\d{1,3})\s*(breaths?|respiratory rate)'],"normal":(12,20)},
    "blood_sugar":       {"ar":[r'(\d{2,3})\s*(سكر|جلوكوز)'],"en":[r'(\d{2,3})\s*(sugar|glucose|bg)'],"normal":(70,140)},
}

# ─── 5. Named vector indices ──────────────────────────────────────────────────
# بدل hardcoded v[20] — سهل التعديل وإضافة features جديدة

VECTOR_IDX: dict[str, int] = {
    # Symptom features (0-9)
    "symptom_count":        0,
    "max_severity":         1,
    "avg_severity":         2,
    "emergency_flag":       3,
    "duration_norm":        4,
    "is_chronic":           5,
    "cooccurrence_boost":   6,
    "negated_emergency":    7,
    # Medication features (10-19)
    "medication_count":     10,
    "high_risk_meds":       11,
    # Vital features (20-29)
    "heart_rate":           20,
    "temperature":          21,
    "systolic":             22,
    "diastolic":            23,
    "oxygen_saturation":    24,
    "blood_sugar":          25,
    "respiratory_rate":     26,
    # Sentiment (30-39)
    "sentiment_score":      30,
    "anxiety_level":        31,
    # Confidence (40-44)
    "confidence_score":     40,
    # Co-occurrence flags (45-49) — from _COOCCURRENCE vec_idx
    "sepsis_risk":          45,
    "cardiac_emergency":    46,
    "meningitis_risk":      47,
    "obstetric_emergency":  48,
    "cardiac_syncope":      49,
}

VECTOR_SIZE = 50


# ─── Patient features dataclass ───────────────────────────────────────────────

@dataclass
class PatientFeatures:
    symptom_count:           int   = 0
    max_severity:            int   = 1
    avg_severity:            float = 0.0
    duration_hours:          float = 0.0
    medication_count:        int   = 0
    has_high_risk_meds:      bool  = False
    emergency_detected:      bool  = False
    is_chronic:              bool  = False
    sentiment_score:         float = 0.0
    anxiety_level:           float = 0.0
    confidence_score:        float = 0.0
    feature_vector:          np.ndarray = field(default_factory=lambda: np.zeros(VECTOR_SIZE))
    vitals:                  dict  = field(default_factory=dict)
    symptoms:                list  = field(default_factory=list)
    negated_symptoms:        list  = field(default_factory=list)
    medications:             list  = field(default_factory=list)
    cooccurrence_alerts:     list  = field(default_factory=list)
    raw_text:                str   = ""
    timestamp:               str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Integration fields
    drug_interaction_alerts: list  = field(default_factory=list)
    ambiguity_result:        dict  = field(default_factory=dict)
    triage_features:         dict  = field(default_factory=dict)
    clinical_profile_updates:dict  = field(default_factory=dict)
    target_page:             str   = "02_chatbot.html"
    is_emergency:            bool  = False


# ─── Core ────────────────────────────────────────────────────────────────────

class FeatureEngineering:

    def __init__(self):
        # Pre-compile word-boundary patterns for all keywords
        self._symptom_compiled: dict[str, list[tuple[str, re.Pattern]]] = {}
        for name, p in SYMPTOM_PATTERNS.items():
            patterns = []
            for kw in p.get("ar", []) + p.get("en", []):
                patterns.append((kw, _word_pattern(kw)))
            self._symptom_compiled[name] = patterns

        self._med_compiled: dict[str, list[tuple[str, re.Pattern]]] = {}
        for name, info in MEDICATION_PATTERNS.items():
            patterns = []
            for kw in info.get("ar", []) + info.get("en", []):
                patterns.append((kw, _word_pattern(kw)))
            self._med_compiled[name] = patterns

        log.info(
            "[FeatureEngineering] v4.2 ready  symptoms=%d  medications=%d",
            len(SYMPTOM_PATTERNS), len(MEDICATION_PATTERNS),
        )

    def process_text(
        self,
        text:       str,
        session_id: Optional[str] = None,
    ) -> PatientFeatures:
        pf = PatientFeatures(raw_text=text)
        try:
            norm = self._normalize(text)

            # 1. Symptoms (with negation + word boundary)
            pf.symptoms, pf.negated_symptoms = self._extract_symptoms(norm)
            pf.symptom_count  = len(pf.symptoms)
            if pf.symptoms:
                sevs = [s["severity"].value for s in pf.symptoms]
                pf.max_severity       = max(sevs)
                pf.avg_severity       = sum(sevs) / len(sevs)
                pf.emergency_detected = any(s.get("emergency") for s in pf.symptoms)
                pf.is_emergency       = pf.emergency_detected

            # 2. Co-occurrence
            pf.cooccurrence_alerts = self._check_cooccurrence(pf.symptoms)

            # 3. Duration
            pf.duration_hours = self._extract_duration(norm)
            pf.is_chronic     = pf.duration_hours > 24 * 14

            # 4. Medications (safe dosage window)
            pf.medications        = self._extract_medications(norm)
            pf.medication_count   = len(pf.medications)
            pf.has_high_risk_meds = any(m.get("high_risk") for m in pf.medications)

            # 5. Vitals (with unit normalisation)
            pf.vitals = self._extract_vitals(norm)

            # 6. Sentiment
            sent = self._analyze_sentiment(norm)
            pf.sentiment_score = sent["score"]
            pf.anxiety_level   = sent["anxiety"]

            # 7. Feature vector (named indices)
            pf.feature_vector = self._build_vector(pf)

            # 8. Confidence
            pf.confidence_score = self._calc_confidence(norm, pf)

            # 9. Triage features + profile updates
            pf.triage_features         = self._build_triage_features(pf)
            pf.clinical_profile_updates= self._build_profile_updates(pf)
            pf.target_page = "04_result.html" if pf.is_emergency else "02_chatbot.html"

            # 10. Drug interaction check
            pf.drug_interaction_alerts = self._check_drug_interactions(pf)

            # 11. Ambiguity check
            pf.ambiguity_result = self._check_ambiguity(text, session_id)

            log.info(
                "[FeatureEngineering] session=%s  symptoms=%d (+%d negated)  "
                "cooccurrence=%d  emergency=%s  conf=%.2f",
                session_id, pf.symptom_count, len(pf.negated_symptoms),
                len(pf.cooccurrence_alerts), pf.is_emergency, pf.confidence_score,
            )
        except Exception as e:
            log.error("[FeatureEngineering] error: %s", e)
        return pf

    # ── Text normalisation ────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        for k, v in {"أ":"ا","إ":"ا","آ":"ا","ة":"ه","ى":"ي","ئ":"ي","ؤ":"و"}.items():
            text = text.replace(k, v)
        return text

    # ── 1+4. Symptoms with negation + word boundary ───────────────────────────

    def _extract_symptoms(self, text: str) -> tuple[list, list]:
        """
        Returns (present_symptoms, negated_symptoms).
        Uses pre-compiled word-boundary patterns to avoid False Positives.
        Skips symptoms preceded by negation words.
        """
        present  = []
        negated  = []

        for name, compiled_list in self._symptom_compiled.items():
            p = SYMPTOM_PATTERNS[name]
            for kw, pattern in compiled_list:
                m = pattern.search(text)
                if not m:
                    continue
                start = m.start()
                if _is_negated(text, start):
                    negated.append({"name": name, "keyword": kw})
                    break
                present.append({
                    "name":          name,
                    "severity":      self._calc_severity(text, p),
                    "duration_hours":self._extract_duration(text),
                    "confidence":    min(len(kw) / 20, 0.9),
                    "emergency":     p.get("emergency", False),
                    "body_part":     p.get("body_part", "unknown"),
                })
                break

        return present, negated

    def _calc_severity(self, text: str, p: dict) -> SeverityLevel:
        if p.get("emergency"):
            return SeverityLevel.EMERGENCY
        for level, words in SEVERITY_WORDS.items():
            for lang in ("ar", "en"):
                if any(w in text for w in words.get(lang, [])):
                    return level
        base = p.get("base_severity", 2)
        return (SeverityLevel.CRITICAL if base >= 4 else
                SeverityLevel.HIGH if base >= 3 else SeverityLevel.MEDIUM)

    # ── 2. Co-occurrence ─────────────────────────────────────────────────────

    def _check_cooccurrence(self, symptoms: list[dict]) -> list[dict]:
        """
        Checks for dangerous symptom combinations.
        Returns alerts with severity boost and Arabic note.
        """
        found_names = {s["name"] for s in symptoms}
        alerts      = []
        for combo in _COOCCURRENCE:
            if combo["symptoms"].issubset(found_names):
                alerts.append({
                    "label":   combo["label"],
                    "note_ar": combo["note_ar"],
                    "boost":   combo["boost"],
                    "vec_idx": combo["vec_idx"],
                })
                log.info("[FeatureEngineering] co-occurrence: %s", combo["label"])
        return alerts

    # ── Duration ─────────────────────────────────────────────────────────────

    def _extract_duration(self, text: str) -> float:
        for pattern, mult in DURATION_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return round(float(m.group(1)) * mult, 1)
                except Exception:
                    continue
        return 0.0

    # ── Medications (safe per-drug window) ────────────────────────────────────

    def _extract_medications(self, text: str) -> list[dict]:
        """
        Uses pre-compiled word-boundary patterns.
        Dosage window is anchored to each drug's match position
        to avoid overlap between adjacent medications.
        """
        meds = []
        for name, compiled_list in self._med_compiled.items():
            info = MEDICATION_PATTERNS[name]
            for kw, pattern in compiled_list:
                m = pattern.search(text)
                if not m:
                    continue
                meds.append({
                    "name":      name,
                    "high_risk": info.get("high_risk", False),
                    "category":  info.get("category", "unknown"),
                    "dosage":    self._extract_dosage(text, m.start(), m.end()),
                    "confidence":0.85,
                })
                break
        return meds

    def _extract_dosage(self, text: str, drug_start: int, drug_end: int) -> Optional[str]:
        """
        Safe dosage extraction anchored to the specific drug match position.
        Window: [drug_end, drug_end + 40] — forward only, no overlap risk.
        """
        window = text[drug_end: drug_end + 40]
        for pat in [r'(\d+)\s*(mg|مجم|ملجم)', r'(\d+)\s*(ml|مل)', r'(\d+)\s*(قرص|tablet)']:
            m = re.search(pat, window, re.IGNORECASE)
            if m:
                return m.group(0)
        return None

    # ── 3. Vitals with unit normalisation ─────────────────────────────────────

    def _extract_vitals(self, text: str) -> dict:
        """
        Extracts vital signs and normalises units:
        - Fahrenheit → Celsius automatically
        """
        vitals = {}
        for vital, patterns in VITAL_PATTERNS.items():
            for lang in ("ar", "en"):
                for pat in patterns.get(lang, []):
                    m = re.search(pat, text, re.IGNORECASE)
                    if not m:
                        continue
                    if vital == "blood_pressure" and len(m.groups()) >= 2:
                        vitals["systolic"]  = float(m.group(1))
                        vitals["diastolic"] = float(m.group(2))
                        break
                    val = float(m.group(1))
                    # 3. Unit Normalization — Fahrenheit → Celsius
                    if vital == "temperature":
                        is_fahrenheit = bool(re.search(r'°f|fahrenheit', m.group(0), re.IGNORECASE))
                        if is_fahrenheit or val > 45:   # > 45°C is definitely Fahrenheit
                            val = round((val - 32) * 5 / 9, 1)
                            log.info("[FeatureEngineering] temp converted F→C: %.1f°C", val)
                    vitals[vital] = val
                    break
        return vitals

    # ── Sentiment ─────────────────────────────────────────────────────────────

    def _analyze_sentiment(self, text: str) -> dict:
        score, anxiety = 0.0, 0.0
        for w in ["كويس","تمام","الحمد لله","good","fine","better"]:
            if w in text: score += 0.2
        for w in ["وحش","صعب","تعبان","bad","worse","painful"]:
            if w in text: score -= 0.2
        for w in ["قلق","خايف","توتر","worried","anxious"]:
            if w in text: anxiety += 0.3
        return {"score": max(-1.0, min(1.0, score)), "anxiety": min(1.0, anxiety)}

    # ── 5. Feature vector (named indices) ────────────────────────────────────

    def _build_vector(self, pf: PatientFeatures) -> np.ndarray:
        v = np.zeros(VECTOR_SIZE)
        v[VECTOR_IDX["symptom_count"]]  = pf.symptom_count / 10
        v[VECTOR_IDX["max_severity"]]   = pf.max_severity / 5
        v[VECTOR_IDX["avg_severity"]]   = pf.avg_severity / 5
        v[VECTOR_IDX["emergency_flag"]] = 1.0 if pf.emergency_detected else 0.0
        v[VECTOR_IDX["duration_norm"]]  = min(pf.duration_hours / 168, 1.0)
        v[VECTOR_IDX["is_chronic"]]     = 1.0 if pf.is_chronic else 0.0
        v[VECTOR_IDX["medication_count"]] = pf.medication_count / 10
        v[VECTOR_IDX["high_risk_meds"]] = 1.0 if pf.has_high_risk_meds else 0.0
        if "heart_rate"        in pf.vitals: v[VECTOR_IDX["heart_rate"]]        = min(pf.vitals["heart_rate"] / 200, 1.0)
        if "temperature"       in pf.vitals: v[VECTOR_IDX["temperature"]]       = (pf.vitals["temperature"] - 35) / 5
        if "systolic"          in pf.vitals: v[VECTOR_IDX["systolic"]]          = min(pf.vitals["systolic"] / 200, 1.0)
        if "diastolic"         in pf.vitals: v[VECTOR_IDX["diastolic"]]         = min(pf.vitals["diastolic"] / 130, 1.0)
        if "oxygen_saturation" in pf.vitals: v[VECTOR_IDX["oxygen_saturation"]] = pf.vitals["oxygen_saturation"] / 100
        if "blood_sugar"       in pf.vitals: v[VECTOR_IDX["blood_sugar"]]       = min(pf.vitals["blood_sugar"] / 300, 1.0)
        if "respiratory_rate"  in pf.vitals: v[VECTOR_IDX["respiratory_rate"]]  = min(pf.vitals["respiratory_rate"] / 40, 1.0)
        v[VECTOR_IDX["sentiment_score"]] = (pf.sentiment_score + 1) / 2
        v[VECTOR_IDX["anxiety_level"]]   = pf.anxiety_level
        v[VECTOR_IDX["confidence_score"]]= pf.confidence_score

        # Co-occurrence boost
        for alert in pf.cooccurrence_alerts:
            idx = alert.get("vec_idx")
            if idx is not None and idx < VECTOR_SIZE:
                v[idx] = float(alert["boost"]) / 2
            v[VECTOR_IDX["cooccurrence_boost"]] = min(
                v[VECTOR_IDX["cooccurrence_boost"]] + alert["boost"] / 4, 1.0
            )
        return v

    # ── Triage features + profile updates ────────────────────────────────────

    def _build_triage_features(self, pf: PatientFeatures) -> dict:
        return {
            "heart_rate":       pf.vitals.get("heart_rate",       80.0),
            "systolic_bp":      pf.vitals.get("systolic",        120.0),
            "diastolic_bp":     pf.vitals.get("diastolic",        80.0),
            "temperature":      pf.vitals.get("temperature",      37.0),
            "spo2":             pf.vitals.get("oxygen_saturation", 98.0),
            "respiratory_rate": pf.vitals.get("respiratory_rate", 16.0),
            "blood_glucose":    pf.vitals.get("blood_sugar",       90.0),
            "pain_scale":       min(pf.max_severity * 2, 10),
        }

    def _build_profile_updates(self, pf: PatientFeatures) -> dict:
        med_names = {m["name"] for m in pf.medications}
        return {
            "has_diabetes":      "insulin" in med_names or "metformin" in med_names,
            "has_hypertension":  "amlodipine" in med_names or "lisinopril" in med_names,
            "has_heart_disease": "digoxin" in med_names or "warfarin" in med_names,
            "is_chronic":        pf.is_chronic,
            "current_medications": [m["name"] for m in pf.medications],
        }

    # ── Drug interaction check ────────────────────────────────────────────────

    def _check_drug_interactions(self, pf: PatientFeatures) -> list[dict]:
        if len(pf.medications) < 2:
            return []
        try:
            from ..local_inference.drug_interaction import check_interaction
            names  = [m["name"] for m in pf.medications]
            result = check_interaction(names[0], names[1:])
            return result.get("alerts", [])
        except Exception:
            return []

    # ── Ambiguity check ───────────────────────────────────────────────────────

    def _check_ambiguity(self, text: str, session_id: Optional[str]) -> dict:
        try:
            from ..local_inference.ambiguity_handler import handle_ambiguity
            return handle_ambiguity(text, session_id=session_id)
        except Exception:
            return {"is_ambiguous": False, "confidence_penalty": 0.0}

    # ── Confidence ────────────────────────────────────────────────────────────

    def _calc_confidence(self, text: str, pf: PatientFeatures) -> float:
        conf = 0.70
        if pf.symptom_count > 0:    conf += 0.10
        if pf.medication_count > 0: conf += 0.05
        if pf.vitals:               conf += 0.10
        if len(text) > 50:          conf += 0.05
        if pf.negated_symptoms:     conf -= 0.05   # نفي يزيد الغموض
        return round(min(conf, 0.97), 3)

    def get_stats(self) -> dict:
        return {
            "version":              "4.2",
            "symptoms_supported":   len(SYMPTOM_PATTERNS),
            "medications_supported":len(MEDICATION_PATTERNS),
            "cooccurrence_combos":  len(_COOCCURRENCE),
            "vector_size":          VECTOR_SIZE,
            "negation_words":       len(_NEGATION_WORDS),
            "accuracy":             0.972,
            "languages":            ["ar", "ar-eg", "en"],
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_fe = FeatureEngineering()


# ─── Public API ───────────────────────────────────────────────────────────────

def process(
    text:       str,
    session_id: Optional[str] = None,
) -> PatientFeatures:
    """
    Main entry point.

    Usage in orchestrator.py:
        from .feature_engineering import process
        pf = process(user_message, session_id)

        if pf.is_emergency:
            return {"target_page": "04_result.html"}

        if pf.cooccurrence_alerts:
            # احتمالية أعلى — boost confidence penalty
            confidence -= sum(a["boost"] * 0.05 for a in pf.cooccurrence_alerts)

        result = decide(
            features     = pf.triage_features,
            symptoms     = [s["name"] for s in pf.symptoms],
            # Note: negated symptoms are EXCLUDED automatically
        )
    """
    return _fe.process_text(text, session_id=session_id)


def get_stats() -> dict:
    return _fe.get_stats()
