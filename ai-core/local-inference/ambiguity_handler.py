"""
ambiguity_handler.py  v4.1
==========================
RIVA Health Platform — Ambiguity Handler
-----------------------------------------
يعالج الحالات اللي الـ AI مش متأكد من قصد المريض.

التحسينات v4.1:
    أ. Contextual Memory   — Slot Filling من الرسائل السابقة
                             "البطن" في رسالة سابقة → مش محتاج يسأل تاني
    ب. Transcription Error — نوع جديد TRANSCRIPTION_LOW_CONFIDENCE
                             لو Whisper أنتج كلمات غير مترابطة طبياً
    ج. Dynamic Severity    — أرقام 1-10 تتحوّل لفئات شدة تلقائياً
    د. Dominant Symptom    — كلمات التأكيد (أوي، بيموتني) تحدد العرض الأساسي
    هـ. Folk synonyms      — "فم المعدة"، "نفوخي"، "عضمي" → لا false ambiguity

Author : GODA EMAD
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("riva.local_inference.ambiguity")

# ─── Session slot store (Contextual Memory) ──────────────────────────────────
# key = session_id, value = dict of filled slots

_session_slots: dict[str, dict] = {}
# Structure per session:
# {
#   "location":  "بطن",
#   "severity":  "شديد",
#   "temporal":  "من امبارح",
#   "symptoms":  ["ألم", "غثيان"],
# }


def _get_slots(session_id: Optional[str]) -> dict:
    if not session_id:
        return {}
    return _session_slots.get(session_id, {})


def _update_slots(session_id: Optional[str], new_slots: dict) -> None:
    """Merges new slot values into the session's slot store."""
    if not session_id:
        return
    existing = _session_slots.setdefault(session_id, {})
    for k, v in new_slots.items():
        if v:
            existing[k] = v


# ─── Ambiguity types ─────────────────────────────────────────────────────────

class AmbiguityType(str, Enum):
    VAGUE_SYMPTOM             = "vague_symptom"
    MISSING_LOCATION          = "missing_location"
    MISSING_SEVERITY          = "missing_severity"
    MIXED_SYMPTOMS            = "mixed_symptoms"
    NEGATION_UNCLEAR          = "negation_unclear"
    DIALECT_AMBIGUOUS         = "dialect_ambiguous"
    TEMPORAL_MISSING          = "temporal_missing"
    TRANSCRIPTION_LOW_CONF    = "transcription_low_confidence"   # ← جديد v4.1
    NOT_AMBIGUOUS             = "not_ambiguous"


# ─── Vocabulary sets ──────────────────────────────────────────────────────────

_VAGUE_WORDS = {
    "تعبان", "مش كويس", "وحش", "مش تمام", "بايظ",
    "مش عارف", "حاجة", "كده", "زي كده", "مش ماشي",
}

_LOCATION_NEEDED = {
    "وجعني", "بيوجعني", "ألم", "وجع", "بيدرب",
    "بيلسع", "بيحرق", "بيشد",
}

_SEVERITY_NEEDED = {
    "حرارة", "ضغط", "سكر", "وجع", "ألم", "تورم", "إيه",
}

_NEGATION_PATTERNS = [
    r"مش\s+(كويس|تمام|عارف|قادر|حاسس|لاقي)",
    r"ملقيش(ش)?",
    r"مفيش(ش)?\s+\w+",
]

_CHRONIC_TRIGGERS = {
    "بيرجع", "متكرر", "دايماً", "كل شوية", "من فترة",
    "مستمر", "ما بيوقفش",
}

# كلمات تأكيد لتحديد العرض الأساسي
_EMPHASIS_WORDS = {
    "أوي", "جداً", "قوي", "خالص", "بيموتني",
    "شديد", "صعب", "مش قادر", "تعبني",
}

# كلمات غير طبية — مؤشر على خطأ في التفريغ الصوتي
_NON_MEDICAL_NOISE = {
    "اهاه", "ممم", "ايوه", "لأ", "يعني", "اوك",
    "تمام", "ماشي", "اوكي", "ها", "هاه",
}

# ─── Location vocabulary (شامل المرادفات الشعبية) ────────────────────────────

_LOCATION_WORDS = {
    # أجزاء الجسم الرسمية
    "صدر", "قلب", "بطن", "معدة", "ظهر", "رأس", "راس", "رجل",
    "ركبة", "يد", "كتف", "رقبة", "عنق", "أذن", "وجه", "عين",
    "أسنان", "لثة", "حلق", "صوت", "كلى", "مثانة", "جنب",
    "طحال", "كبد", "أمعاء", "مستقيم",
    # مرادفات شعبية مصرية — تمنع False Ambiguity
    "فم المعدة",    # الجزء العلوي من المعدة
    "نفوخي",        # البطن المنتفخة
    "عضمي",         # العظام بشكل عام
    "طحالي",        # منطقة الطحال
    "تحت إبطي",     # الإبط
    "في ضهري",      # الظهر
    "جنبي",         # الجنب
    "في دماغي",     # الرأس
    "في قلبي",      # الصدر / القلب
    "في معدتي",     # المعدة
    "في رجلي",      # الرجل
}

_SEVERITY_WORDS = {
    "خفيف", "بسيط", "متوسط", "شديد", "جداً", "اوي", "قوي",
    "مش قادر", "صعب", "كتير",
}

_TEMPORAL_WORDS = {
    "من", "منذ", "امتى", "امبارح", "النهارده", "أسبوع",
    "شهر", "أيام", "ساعات", "دلوقتي", "من امبارح",
    "من أسبوع", "من شهر", "من فترة",
}


# ─── Dynamic severity (أرقام 1-10) ───────────────────────────────────────────

_SEVERITY_SCALE = re.compile(r"\b([1-9]|10)\b|[١-٩]|١٠")

_SEVERITY_CATEGORIES = {
    range(1, 4):  ("خفيف",   0.0),
    range(4, 7):  ("متوسط",  0.0),
    range(7, 9):  ("شديد",   0.0),
    range(9, 11): ("شديد جداً", 0.0),
}


def _parse_severity_number(text: str) -> Optional[str]:
    """
    يحوّل الأرقام (1-10) أو العربية (١-١٠) لفئة شدة.
    مثال: "الوجع 8 من 10" → "شديد"
    """
    match = _SEVERITY_SCALE.search(text)
    if not match:
        return None

    raw = match.group()
    # تحويل الأرقام العربية
    ar_to_en = str.maketrans("١٢٣٤٥٦٧٨٩٠", "1234567890")
    num = int(raw.translate(ar_to_en))

    for r, (label, _) in _SEVERITY_CATEGORIES.items():
        if num in r:
            return label
    return None


# ─── Clarification templates ──────────────────────────────────────────────────

_CLARIFICATION_TEMPLATES: dict[AmbiguityType, dict] = {

    AmbiguityType.VAGUE_SYMPTOM: {
        "question": "ممكن توضّح أكتر؟ إيه اللي بتحس بيه بالظبط؟",
        "options":  [
            "ألم أو وجع في جزء معين",
            "تعب عام وإرهاق",
            "حرارة أو قشعريرة",
            "دوار أو غثيان",
            "ضيقة في التنفس",
        ],
    },

    AmbiguityType.MISSING_LOCATION: {
        "question": "الوجع ده فين بالظبط؟",
        "options":  [
            "الرأس أو الرقبة",
            "الصدر أو القلب",
            "البطن أو المعدة",
            "الظهر أو الكتف",
            "الأرجل أو الركبة",
            "مكان تاني",
        ],
    },

    AmbiguityType.MISSING_SEVERITY: {
        "question": "قد إيه الوجع أو الأعراض دي شديدة؟ (من 1 لـ 10)",
        "options":  [
            "1-3 خفيف — بقدر أعمل حاجاتي عادي",
            "4-6 متوسط — بيأثر على يومي",
            "7-8 شديد — صعب أتحرك أو أشتغل",
            "9-10 شديد جداً — محتاج مساعدة فوراً",
        ],
    },

    AmbiguityType.MIXED_SYMPTOMS: {
        "question": "عندك أكتر من عَرَض — إيه أكتر حاجة بتضايقك دلوقتي؟",
        "options":  None,
    },

    AmbiguityType.NEGATION_UNCLEAR: {
        "question": "لما قلت مش كويس — إيه اللي بالظبط مش تمام؟",
        "options":  [
            "جسمي مش كويس (مرض جسدي)",
            "تعبان نفسياً أو قلقان",
            "تعبان من الشغل والضغط",
            "مش عارف بالظبط",
        ],
    },

    AmbiguityType.DIALECT_AMBIGUOUS: {
        "question": "ممكن تشرح أكتر؟ عايز أفهم قصدك صح.",
        "options":  [
            "عندي أعراض جسدية",
            "محتاج معلومة طبية",
            "عايز أعرف دواء",
            "قلقان على حالة حد تاني",
        ],
    },

    AmbiguityType.TEMPORAL_MISSING: {
        "question": "الأعراض دي من امتى بالظبط؟",
        "options":  [
            "من ساعات (النهارده)",
            "من يومين لـ 3 أيام",
            "من أسبوع",
            "من أكتر من أسبوع",
            "متكررة ومجاش من أول",
        ],
    },

    AmbiguityType.TRANSCRIPTION_LOW_CONF: {
        "question": "مش فاهم كويس — ممكن تعيد تسجيل الصوت بصوت أعلى؟",
        "options":  [
            "سجّل تاني",
            "اكتب رسالتك بدل الصوت",
        ],
    },
}


# ─── Result dataclasses ──────────────────────────────────────────────────────

@dataclass
class AmbiguityResult:
    is_ambiguous:       bool
    ambiguity_type:     AmbiguityType
    confidence_penalty: float
    detected_keywords:  list[str] = field(default_factory=list)
    resolved_by_memory: bool      = False   # True لو الـ slots حلّت الغموض


@dataclass
class ClarificationResult:
    question:       str
    options:        list[str]
    ambiguity_type: AmbiguityType
    original_text:  str


# ─── Core handler ─────────────────────────────────────────────────────────────

class AmbiguityHandler:

    # ── أ. Contextual Memory ──────────────────────────────────────────────────

    def _extract_slots(self, text: str) -> dict:
        """Extracts slot values from the current message."""
        words  = set(text.split())
        slots  = {}

        loc = words & _LOCATION_WORDS
        if loc:
            slots["location"] = next(iter(loc))

        sev = words & _SEVERITY_WORDS
        if sev:
            slots["severity"] = next(iter(sev))

        # Dynamic severity from numbers
        num_sev = _parse_severity_number(text)
        if num_sev:
            slots["severity"] = num_sev

        tmp = words & _TEMPORAL_WORDS
        if tmp:
            slots["temporal"] = next(iter(tmp))

        return slots

    # ── ب. Transcription error detection ─────────────────────────────────────

    def _is_transcription_noise(self, text: str) -> bool:
        """
        Returns True if the text looks like a bad Whisper transcription:
        - More than 60% non-medical noise words
        - Very short + no medical keywords at all
        """
        words      = text.split()
        if len(words) < 2:
            return False

        noise_hits = sum(1 for w in words if w in _NON_MEDICAL_NOISE)
        noise_ratio = noise_hits / len(words)

        has_medical = bool(
            set(words) & (_LOCATION_WORDS | _SEVERITY_WORDS | _VAGUE_WORDS)
        )
        return noise_ratio > 0.6 and not has_medical

    # ── ج. Dynamic severity ───────────────────────────────────────────────────

    def _resolve_severity(self, text: str) -> Optional[str]:
        return _parse_severity_number(text)

    # ── د. Dominant symptom ───────────────────────────────────────────────────

    def _get_dominant_symptom(
        self, text: str, symptoms: list[str]
    ) -> str:
        """
        Finds the symptom closest to an emphasis word (أوي، بيموتني...).
        Falls back to symptoms[0] if no emphasis found.
        """
        if not symptoms:
            return ""

        words = text.split()
        best_symptom  = symptoms[0]
        best_distance = float("inf")

        for i, word in enumerate(words):
            if word in _EMPHASIS_WORDS:
                for symptom in symptoms:
                    # distance = position difference in the sentence
                    for j, w in enumerate(words):
                        if w == symptom:
                            dist = abs(i - j)
                            if dist < best_distance:
                                best_distance = dist
                                best_symptom  = symptom
                            break

        log.debug("[Ambiguity] dominant_symptom='%s'", best_symptom)
        return best_symptom

    # ── Main detect ───────────────────────────────────────────────────────────

    def detect(
        self,
        text:       str,
        session_id: Optional[str] = None,
    ) -> AmbiguityResult:
        """
        Analyses the message for ambiguity.
        Uses session slots to avoid asking for already-known information.
        """
        text_lower = text.strip()
        words      = set(text_lower.split())

        # Extract + persist slots from this message
        new_slots = self._extract_slots(text_lower)
        _update_slots(session_id, new_slots)
        known = _get_slots(session_id)

        # ب. Transcription noise check (first — before any other check)
        if self._is_transcription_noise(text_lower):
            return AmbiguityResult(
                is_ambiguous       = True,
                ambiguity_type     = AmbiguityType.TRANSCRIPTION_LOW_CONF,
                confidence_penalty = 0.40,
            )

        # Dynamic severity: if a number is found, no severity ambiguity
        if _parse_severity_number(text_lower):
            _update_slots(session_id, {"severity": _parse_severity_number(text_lower)})

        # 1. Vague words
        vague_hits = words & _VAGUE_WORDS
        if vague_hits and len(text.split()) <= 5:
            return AmbiguityResult(
                is_ambiguous       = True,
                ambiguity_type     = AmbiguityType.VAGUE_SYMPTOM,
                confidence_penalty = 0.25,
                detected_keywords  = list(vague_hits),
            )

        # 2. Missing location — but check session memory first (Slot Filling)
        location_hits = words & _LOCATION_NEEDED
        if location_hits and not self._has_location(text_lower):
            if known.get("location"):
                # Already known from previous turn → no ambiguity
                log.info(
                    "[Ambiguity] location resolved from memory: '%s'",
                    known["location"],
                )
                return AmbiguityResult(
                    is_ambiguous       = False,
                    ambiguity_type     = AmbiguityType.NOT_AMBIGUOUS,
                    confidence_penalty = 0.0,
                    resolved_by_memory = True,
                )
            return AmbiguityResult(
                is_ambiguous       = True,
                ambiguity_type     = AmbiguityType.MISSING_LOCATION,
                confidence_penalty = 0.20,
                detected_keywords  = list(location_hits),
            )

        # 3. Missing severity — check session memory + dynamic number
        severity_hits = words & _SEVERITY_NEEDED
        if severity_hits and not self._has_severity(text_lower):
            if known.get("severity"):
                return AmbiguityResult(
                    is_ambiguous       = False,
                    ambiguity_type     = AmbiguityType.NOT_AMBIGUOUS,
                    confidence_penalty = 0.0,
                    resolved_by_memory = True,
                )
            return AmbiguityResult(
                is_ambiguous       = True,
                ambiguity_type     = AmbiguityType.MISSING_SEVERITY,
                confidence_penalty = 0.15,
                detected_keywords  = list(severity_hits),
            )

        # 4. Negation unclear
        for pattern in _NEGATION_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                return AmbiguityResult(
                    is_ambiguous       = True,
                    ambiguity_type     = AmbiguityType.NEGATION_UNCLEAR,
                    confidence_penalty = 0.20,
                    detected_keywords  = [m.group()],
                )

        # 5. Missing temporal context
        if self._needs_temporal(text_lower) and not self._has_temporal(text_lower):
            if known.get("temporal"):
                return AmbiguityResult(
                    is_ambiguous       = False,
                    ambiguity_type     = AmbiguityType.NOT_AMBIGUOUS,
                    confidence_penalty = 0.0,
                    resolved_by_memory = True,
                )
            return AmbiguityResult(
                is_ambiguous       = True,
                ambiguity_type     = AmbiguityType.TEMPORAL_MISSING,
                confidence_penalty = 0.10,
            )

        # 6. Mixed symptoms
        if self._is_mixed_symptoms(text_lower):
            return AmbiguityResult(
                is_ambiguous       = True,
                ambiguity_type     = AmbiguityType.MIXED_SYMPTOMS,
                confidence_penalty = 0.15,
            )

        return AmbiguityResult(
            is_ambiguous       = False,
            ambiguity_type     = AmbiguityType.NOT_AMBIGUOUS,
            confidence_penalty = 0.0,
        )

    def generate_clarification(
        self,
        text:           str,
        ambiguity_type: AmbiguityType,
        symptoms_found: Optional[list[str]] = None,
    ) -> ClarificationResult:
        template = _CLARIFICATION_TEMPLATES.get(
            ambiguity_type,
            _CLARIFICATION_TEMPLATES[AmbiguityType.DIALECT_AMBIGUOUS],
        )
        question = template["question"]
        options  = template["options"]

        if ambiguity_type == AmbiguityType.MIXED_SYMPTOMS and symptoms_found:
            dominant = self._get_dominant_symptom(text, symptoms_found)
            reordered = [dominant] + [s for s in symptoms_found if s != dominant]
            options   = reordered[:4] + ["حاجة تانية"]

        if options is None:
            options = ["وضّح أكتر", "مش عارف أحدد"]

        return ClarificationResult(
            question       = question,
            options        = options,
            ambiguity_type = ambiguity_type,
            original_text  = text,
        )

    def process(
        self,
        text:       str,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Main entry point — used by chat.py and orchestrator.py

        Usage in chat.py:
            amb = handle_ambiguity(user_message, session_id)
            if amb["is_ambiguous"]:
                confidence -= amb["confidence_penalty"]
                return amb["clarification"]
        """
        result = self.detect(text, session_id=session_id)

        clarification = None
        if result.is_ambiguous:
            cl = self.generate_clarification(text, result.ambiguity_type)
            clarification = {
                "question": cl.question,
                "options":  cl.options,
            }

        return {
            "is_ambiguous":       result.is_ambiguous,
            "ambiguity_type":     result.ambiguity_type,
            "confidence_penalty": result.confidence_penalty,
            "detected_keywords":  result.detected_keywords,
            "resolved_by_memory": result.resolved_by_memory,
            "clarification":      clarification,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _has_location(self, text: str) -> bool:
        words = set(text.split())
        if words & _LOCATION_WORDS:
            return True
        # Multi-word folk synonyms
        return any(phrase in text for phrase in [
            "فم المعدة", "نفوخي", "عضمي", "تحت إبطي",
            "في ضهري", "جنبي", "في دماغي",
        ])

    def _has_severity(self, text: str) -> bool:
        if set(text.split()) & _SEVERITY_WORDS:
            return True
        return _parse_severity_number(text) is not None

    def _has_temporal(self, text: str) -> bool:
        return bool(set(text.split()) & _TEMPORAL_WORDS)

    def _needs_temporal(self, text: str) -> bool:
        return bool(set(text.split()) & _CHRONIC_TRIGGERS)

    def _is_mixed_symptoms(self, text: str) -> bool:
        return len(set(text.split()) & _LOCATION_WORDS) >= 3


# ─── Singleton ────────────────────────────────────────────────────────────────

_handler = AmbiguityHandler()


# ─── Public API ───────────────────────────────────────────────────────────────

def handle_ambiguity(
    text:       str,
    session_id: Optional[str] = None,
) -> dict:
    return _handler.process(text, session_id=session_id)


def clear_session_slots(session_id: str) -> None:
    """Call this when a session ends to free memory."""
    _session_slots.pop(session_id, None)
