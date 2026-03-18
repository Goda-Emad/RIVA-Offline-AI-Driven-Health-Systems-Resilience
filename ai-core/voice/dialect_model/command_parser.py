"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Command Parser                              ║
║           ai-core/voice/dialect_model/command_parser.py                      ║
║                                                                              ║
║  Purpose : Parse transcribed Egyptian Arabic voice/text commands into        ║
║            structured ParsedCommand objects ready for downstream AI models.  ║
║                                                                              ║
║  Pipeline: Whisper STT → [CommandParser] → Triage / Pregnancy / School /    ║
║            Readmission / LOS / Chatbot engines                               ║
║                                                                              ║
║  Author  : Goda Emad  (AI Core)                                              ║
║  Version : 2.2.0                                                             ║
║  Updated : 2026-03-18                                                        ║
║                                                                              ║
║  Changelog v2.2.0                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • FIX  Duration: added "بقالي/بقاله/بقالها" patterns — Egyptian speakers   ║
║         rarely use "من"; this caused duration_days=None on most inputs.     ║
║  • FIX  Negation: _extract_symptoms now strips negated symptoms before       ║
║         returning — prevents false positives like "مش دايخ" → دوار.        ║
║  • ADD  Lexicon: expanded _BODY_MAP (زوري, جنبي, معدتي, ركبتي, كتفي…)      ║
║         and _DIALECT_MAP (ترجيع, برجع, تقلص, شد, وخز, بردان, عرقان…)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ── Internal imports ────────────────────────────────────────────────────────
from .confidence_scorer import ConfidenceScorer
from .ambiguity_handler import AmbiguityHandler
from .sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class Intent(str, Enum):
    """High-level user intents detected from Egyptian Arabic commands."""

    # ── Clinical ──────────────────────────────────────────────────────────
    TRIAGE          = "triage"           # "عندي ألم في صدري"
    PREGNANCY       = "pregnancy"        # "أنا حامل وعندي دوخة"
    SCHOOL_HEALTH   = "school_health"    # "الطالب ده مريض"
    READMISSION     = "readmission"      # "المريض ده اتعمله عملية"
    LOS_ESTIMATE    = "los_estimate"     # "هيفضل كام يوم؟"

    # ── Navigation ────────────────────────────────────────────────────────
    VIEW_HISTORY    = "view_history"     # "شوف سجلي"
    VIEW_RESULTS    = "view_results"     # "النتيجة إيه؟"
    BOOK_APPOINTMENT= "book_appointment" # "عايز موعد"

    # ── System ────────────────────────────────────────────────────────────
    HELP            = "help"             # "إيه اللي تعمله؟"
    REPEAT          = "repeat"           # "تاني"
    CANCEL          = "cancel"           # "لأ / إلغاء"
    EMERGENCY       = "emergency"        # "نجدة / إسعاف"
    UNKNOWN         = "unknown"


class UrgencyLevel(str, Enum):
    """Clinical urgency as detected from linguistic cues."""
    EMERGENCY = "emergency"   # كلمات زي: نجدة، مش قادر أتنفس، بيموت
    HIGH      = "high"        # ألم شديد، من امبارح، متواصل
    MEDIUM    = "medium"      # من يومين، متقطع، بيتحسن وبيتعب
    LOW       = "low"         # خفيف، أحياناً، مش مزعجني


class TargetModel(str, Enum):
    """Which downstream AI model should handle this command."""
    TRIAGE_ENGINE       = "triage_engine"
    PREGNANCY_RISK      = "pregnancy_risk"
    SCHOOL_HEALTH       = "school_health"
    READMISSION         = "readmission_predictor"
    LOS_PREDICTOR       = "los_predictor"
    MEDICAL_CHATBOT     = "medical_chatbot"
    NAVIGATION_HANDLER  = "navigation_handler"
    SYSTEM_HANDLER      = "system_handler"


# ═══════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedCommand:
    """
    Fully structured output of CommandParser.

    Passed directly to the appropriate AI engine or API route.
    All fields are populated even if confidence is low — consumers
    should check `confidence` before acting.
    """

    # ── Core ──────────────────────────────────────────────────────────────
    raw_text       : str                        # Original transcription
    normalized_text: str                        # After dialect normalization
    intent         : Intent                     # Detected intent
    target_model   : TargetModel                # Routing destination
    confidence     : float                      # 0.0 – 1.0

    # ── Clinical payload ──────────────────────────────────────────────────
    symptoms       : list[str]     = field(default_factory=list)
    body_parts     : list[str]     = field(default_factory=list)
    duration_days  : int | None    = None       # Parsed duration in days
    urgency        : UrgencyLevel  = UrgencyLevel.LOW
    patient_context: dict[str, Any]= field(default_factory=dict)
    # e.g. {"is_pregnant": True, "age": 28, "is_student": False}

    # ── Sentiment & ambiguity ─────────────────────────────────────────────
    sentiment_score: float         = 0.0        # –1.0 (distress) → +1.0 (calm)
    is_ambiguous   : bool          = False
    ambiguity_reason: str          = ""
    clarification_needed: list[str]= field(default_factory=list)

    # ── Meta ──────────────────────────────────────────────────────────────
    language_detected: str         = "ar-EG"
    parse_time_ms  : float         = 0.0
    timestamp      : float         = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════
#  Lexicon — Egyptian Arabic medical vocabulary
# ═══════════════════════════════════════════════════════════════════════════

# Maps Egyptian colloquial words → normalized Arabic / clinical term
# ── Order matters: longer phrases must appear before shorter substrings ──────
_DIALECT_MAP: dict[str, str] = {
    # ── Pain / general symptoms ───────────────────────────────────────────
    "بيوجعني"          : "ألم",
    "بتوجعني"          : "ألم",
    "واجعاني"          : "ألم",
    "وجعاني"           : "ألم",
    "وجع"              : "ألم",
    "بيألمني"          : "ألم",
    "شد"               : "تقلص عضلي",          # "عندي شد في ضهري"
    "تقلص"             : "تقلص عضلي",
    "وخز"              : "وخز",                 # "حاسس بوخز"
    "تنميل"            : "تنميل",

    # ── Sickness / fatigue ────────────────────────────────────────────────
    "تعبان"            : "مريض",
    "تعبانة"           : "مريضة",
    "مش كويس"          : "مريض",
    "مش تمام"          : "مريض",
    "مش حاسس بنفسي"    : "مريض",
    "حاسس بتقل"        : "إرهاق",
    "حاسة بتقل"        : "إرهاق",
    "زهقان"            : "إرهاق",
    "تعبت"             : "إرهاق",
    "خامل"             : "إرهاق",
    "بردان"            : "قشعريرة",             # "أنا بردان وعندي حرارة"
    "بردانة"           : "قشعريرة",
    "عرقان"            : "تعرق",
    "عرقانة"           : "تعرق",

    # ── Dizziness / head ─────────────────────────────────────────────────
    "دايخ"             : "دوار",
    "دايخة"            : "دوار",
    "دوخة"             : "دوار",
    "راسي بيلف"        : "دوار",
    "الدنيا بتلف"      : "دوار",

    # ── Breathing ─────────────────────────────────────────────────────────
    "مش لاقي نفسي"     : "ضيق تنفس",
    "مش لاقية نفسي"    : "ضيق تنفس",
    "مش قادر أتنفس"    : "ضيق تنفس",
    "ضيق في نفسي"      : "ضيق تنفس",
    "صعوبة في التنفس"   : "ضيق تنفس",

    # ── Cardiac ───────────────────────────────────────────────────────────
    "تعب في قلبي"      : "ألم صدر",
    "قلبي بيدق"        : "خفقان",
    "قلبي بيرفرف"      : "خفقان",
    "قلبي بيوجعني"     : "ألم صدر",

    # ── Fever ─────────────────────────────────────────────────────────────
    "حرارة"            : "حمى",
    "سخونة"            : "حمى",
    "سخن"              : "حمى",

    # ── GI symptoms ───────────────────────────────────────────────────────
    "إسهال"            : "إسهال",
    "بطني بيوجعني"     : "ألم بطن",
    # ↓ FIX v2.2: "ترجيع/برجع" — most-used Egyptian vomiting terms
    "ترجيع"            : "قيء",
    "برجع"             : "قيء",
    "بترجع"            : "قيء",
    "عندي ترجيع"       : "قيء",
    "بقيء"             : "قيء",
    "بتقيأ"            : "قيء",
    "غثيان"            : "غثيان",
    "حاسس بغثيان"      : "غثيان",
    "حاسة بغثيان"      : "غثيان",
    "معدتي بتوجعني"    : "ألم المعدة",          # ← added v2.2
    "حرقة في معدتي"    : "حرقة معدة",

    # ── Throat / ENT (school health especially) ───────────────────────────
    # ↓ FIX v2.2: "زوري" شائعة جداً في أمراض الأطفال والمدارس
    "زوري بيوجعني"     : "التهاب حلق",
    "زوري"             : "الحلق",
    "حلقي بيوجعني"     : "التهاب حلق",
    "لوزاتي"           : "التهاب اللوزتين",

    # ── Sleep ─────────────────────────────────────────────────────────────
    "مش عارف أنام"     : "أرق",
    "مش قادر أنام"     : "أرق",

    # ── Duration (Egyptian patterns) ─────────────────────────────────────
    "من امبارح"        : "من يوم واحد",
    "من إمبارح"        : "من يوم واحد",
    "النهارده"         : "اليوم",
    "دلوقتي"           : "الآن",
    "من شوية"          : "منذ ساعات",
    "من الصبح"         : "منذ ساعات",
    "من الفجر"         : "منذ ساعات",

    # ── Pregnancy ────────────────────────────────────────────────────────
    "حامل"             : "حمل",

    # ── Emergency cues ───────────────────────────────────────────────────
    "نجدة"             : "طوارئ",
    "إسعاف"            : "طوارئ",
    "بيموت"            : "طوارئ",
    "بتموت"            : "طوارئ",
}

# Body parts — Egyptian slang → clinical Arabic
# ↓ v2.2: expanded with زوري، جنبي، معدتي، ركبتي، كتفي، فخدي، صدري الأمامي
_BODY_MAP: dict[str, str] = {
    # Trunk
    "صدري"      : "الصدر",
    "بطني"      : "البطن",
    "ضهري"      : "الظهر",
    "جنبي"      : "الخاصرة",          # ← FIX v2.2 — شائع في مغص الكلى
    "كتفي"      : "الكتف",
    "خصري"      : "أسفل الظهر",

    # GI
    "معدتي"     : "المعدة",            # ← FIX v2.2
    "كبدي"      : "الكبد",
    "كليتي"     : "الكلية",

    # Head / neck
    "راسي"      : "الرأس",
    "رقبتي"     : "الرقبة",
    "زوري"      : "الحلق",             # ← FIX v2.2 — شائع جداً في المدارس
    "ودني"      : "الأذن",
    "عيني"      : "العين",
    "أنفي"      : "الأنف",
    "سناني"     : "الأسنان",
    "لسني"      : "اللسان",

    # Limbs
    "إيدي"      : "الذراع",
    "رجلي"      : "الساق",
    "ركبتي"     : "الركبة",            # ← FIX v2.2
    "فخدي"      : "الفخذ",
    "كعبي"      : "الكعب",
    "أصابعي"    : "الأصابع",

    # Cardiac
    "قلبي"      : "القلب",
}

# Intent keyword patterns  {pattern: (Intent, confidence_boost)}
_INTENT_PATTERNS: list[tuple[re.Pattern, Intent, float]] = [
    # Emergency — highest priority
    (re.compile(r"نجدة|إسعاف|بيموت|بتموت|مش قادر أتنفس", re.UNICODE), Intent.EMERGENCY,    0.40),

    # Triage
    (re.compile(r"عندي|بيوجعني|بتوجعني|تعبان|ألم|وجع|دايخ|حرارة|سخونة", re.UNICODE), Intent.TRIAGE, 0.20),

    # Pregnancy
    (re.compile(r"حامل|حمل|أسبوع الحمل|الحمل|موعد الولادة|جنين", re.UNICODE), Intent.PREGNANCY, 0.30),

    # School health
    (re.compile(r"طالب|تلميذ|مدرسة|فصل|ولد في المدرسة", re.UNICODE), Intent.SCHOOL_HEALTH, 0.30),

    # Readmission / hospital
    (re.compile(r"اتعمله عملية|خرج من المستشفى|إعادة|رجع تاني|تنويم", re.UNICODE), Intent.READMISSION, 0.35),

    # LOS
    (re.compile(r"كام يوم|هيفضل قد إيه|مدة الإقامة", re.UNICODE), Intent.LOS_ESTIMATE, 0.35),

    # History
    (re.compile(r"سجل|تاريخ|الزيارات|اللي فات", re.UNICODE), Intent.VIEW_HISTORY, 0.25),

    # Appointment
    (re.compile(r"موعد|حجز|عايز أحجز", re.UNICODE), Intent.BOOK_APPOINTMENT, 0.25),

    # Help
    (re.compile(r"ساعدني|إيه اللي|تعمل إيه|مش فاهم", re.UNICODE), Intent.HELP, 0.20),

    # Cancel
    (re.compile(r"^(لأ|إلغاء|مش عايز|وقف)$", re.UNICODE), Intent.CANCEL, 0.45),

    # Repeat
    (re.compile(r"^(تاني|كرر|إعادة|مش سمعت)$", re.UNICODE), Intent.REPEAT, 0.45),
]

# ── Duration extraction ───────────────────────────────────────────────────
#
# FIX v2.2: Egyptian speakers overwhelmingly use "بقالي/بقاله/بقالها" rather
# than "من".  Both anchors are now captured by a single pattern so either
# phrasing resolves correctly to duration_days.
#
# Supported examples
# ──────────────────
#   "بقالي 3 أيام"          → 3
#   "بقاله أسبوعين"         → 14
#   "بقالها شهر"            → 30
#   "من 3 أيام"             → 3
#   "من أسبوع"              → 7
#   "من يومين"              → 2
#   "منذ أسبوعين"           → 14

_DURATION_PATTERN = re.compile(
    r"(?:بقال[يهاة]\s+|من\s+|منذ\s+)"        # anchor: بقالي / من / منذ
    r"(?:"
        r"(?P<num>\d+)\s*(?P<unit>يوم|أيام|أسبوع|أسابيع|شهر|شهور)"
        r"|(?P<word>يومين|أسبوعين|شهرين)"
    r")",
    re.UNICODE,
)

_DURATION_WORD_MAP: dict[str, int] = {
    "يومين"    : 2,
    "أسبوعين"  : 14,
    "شهرين"    : 60,
}
_DURATION_UNIT_MAP: dict[str, int] = {
    "يوم"      : 1,
    "أيام"     : 1,
    "أسبوع"    : 7,
    "أسابيع"   : 7,
    "شهر"      : 30,
    "شهور"     : 30,
}

# ── Negation detection ────────────────────────────────────────────────────
#
# FIX v2.2: String-matching without negation awareness caused false-positive
# symptoms, e.g. "معنديش سخونة" was extracted as حمى.
#
# Strategy: before checking for a symptom term, scan a small left-context
# window (≤ 4 tokens) for any negation marker.  If found, the symptom is
# marked as ABSENT and excluded from the returned list.
#
# Egyptian Arabic negation markers covered
# ─────────────────────────────────────────
#   Proclitic  : "مش"   ("مش دايخ")
#   Circumfix  : "ما…ش" ("ماعنديش", "مابيوجعنيش")  — handled by prefix list
#   Standalone : "معنديش", "مفيش", "بدون", "لا يوجد", "ولا"

_NEGATION_PREFIXES: frozenset[str] = frozenset({
    "مش", "مو", "ليس", "لا", "ولا",
    "معنديش", "معنداش", "مفيش",
    "بدون", "من غير",
    "ما",          # prefix of ما…ش circumfix
})

# Window size (in whitespace-separated tokens) to look left of a symptom term
_NEGATION_WINDOW: int = 4

# ── Urgency keywords ──────────────────────────────────────────────────────
_URGENCY_EMERGENCY = re.compile(r"نجدة|إسعاف|بيموت|بتموت|مش قادر أتنفس|ضيق تنفس شديد")
_URGENCY_HIGH      = re.compile(r"ألم شديد|من امبارح|متواصل|طول الليل|مستمر")
_URGENCY_MEDIUM    = re.compile(r"من يومين|من أيام|متقطع|أحياناً|بيتحسن وبيرجع")


# ═══════════════════════════════════════════════════════════════════════════
#  CommandParser
# ═══════════════════════════════════════════════════════════════════════════

class CommandParser:
    """
    Parses Egyptian Arabic medical commands into structured ParsedCommand objects.

    Designed for offline use — no external API calls.  All processing happens
    in-memory using regex lexicons and injected ML helpers.

    Usage
    -----
    >>> parser = CommandParser()
    >>> cmd = parser.parse("عندي ألم في صدري من امبارح")
    >>> cmd.intent
    <Intent.TRIAGE: 'triage'>
    >>> cmd.urgency
    <UrgencyLevel.HIGH: 'high'>
    >>> cmd.symptoms
    ['ألم', 'الصدر']
    """

    def __init__(
        self,
        confidence_scorer : ConfidenceScorer  | None = None,
        ambiguity_handler : AmbiguityHandler  | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        min_confidence    : float = 0.35,
    ) -> None:
        self._confidence  = confidence_scorer  or ConfidenceScorer()
        self._ambiguity   = ambiguity_handler  or AmbiguityHandler()
        self._sentiment   = sentiment_analyzer or SentimentAnalyzer()
        self._min_conf    = min_confidence

        logger.info(
            "CommandParser initialised (min_confidence=%.2f, offline=True)",
            min_confidence,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def parse(self, raw_text: str, patient_context: dict[str, Any] | None = None) -> ParsedCommand:
        """
        Parse a raw Egyptian Arabic string into a ParsedCommand.

        Parameters
        ----------
        raw_text        : Transcribed or typed user input.
        patient_context : Optional dict with known patient metadata,
                          e.g. {"is_pregnant": True, "age": 28}.

        Returns
        -------
        ParsedCommand — fully populated, never raises.
        """
        t_start = time.perf_counter()
        context = patient_context or {}

        # 1. Normalise text
        normalised = self._normalise(raw_text)

        # 2. Detect intent
        intent, intent_conf = self._detect_intent(normalised, context)

        # 3. Extract clinical entities
        symptoms   = self._extract_symptoms(normalised)
        body_parts = self._extract_body_parts(normalised)
        duration   = self._extract_duration(normalised)
        urgency    = self._detect_urgency(normalised, intent)

        # 4. Sentiment
        sentiment = self._sentiment.score(normalised)

        # 5. Ambiguity
        is_ambiguous, reason, clarifications = self._ambiguity.check(
            text=normalised,
            intent=intent,
            symptoms=symptoms,
            confidence=intent_conf,
        )

        # 6. Final confidence (weighted)
        confidence = self._confidence.compute(
            intent_confidence = intent_conf,
            symptoms_found    = len(symptoms),
            has_duration      = duration is not None,
            is_ambiguous      = is_ambiguous,
            urgency           = urgency,
        )

        # 7. Route to target model
        target_model = self._route(intent, context)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug(
            "parse() done | intent=%s urgency=%s conf=%.2f symptoms=%s time=%.1fms",
            intent.value, urgency.value, confidence, symptoms, elapsed_ms,
        )

        return ParsedCommand(
            raw_text            = raw_text,
            normalized_text     = normalised,
            intent              = intent,
            target_model        = target_model,
            confidence          = confidence,
            symptoms            = symptoms,
            body_parts          = body_parts,
            duration_days       = duration,
            urgency             = urgency,
            patient_context     = context,
            sentiment_score     = sentiment,
            is_ambiguous        = is_ambiguous,
            ambiguity_reason    = reason,
            clarification_needed= clarifications,
            parse_time_ms       = elapsed_ms,
        )

    def batch_parse(
        self,
        texts   : list[str],
        contexts: list[dict[str, Any]] | None = None,
    ) -> list[ParsedCommand]:
        """Parse a list of commands (e.g. conversation history)."""
        ctxs = contexts or [{}] * len(texts)
        return [self.parse(t, c) for t, c in zip(texts, ctxs)]

    # ── Private helpers ──────────────────────────────────────────────────

    def _normalise(self, text: str) -> str:
        """
        Lowercase, strip diacritics, and map Egyptian dialect words
        to their normalised clinical Arabic equivalents.
        """
        text = text.strip()

        # Remove Arabic diacritics (tashkeel)
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

        # Apply dialect map (longest match first to avoid partial replacements)
        for colloquial, standard in sorted(_DIALECT_MAP.items(), key=lambda x: -len(x[0])):
            text = text.replace(colloquial, standard)

        # Collapse multiple spaces
        text = re.sub(r"\s{2,}", " ", text)

        return text

    def _detect_intent(
        self,
        text   : str,
        context: dict[str, Any],
    ) -> tuple[Intent, float]:
        """
        Score all intent patterns and return the highest-confidence match.
        Context (e.g. known pregnancy) can boost relevant intents.
        """
        scores: dict[Intent, float] = {i: 0.0 for i in Intent}
        scores[Intent.UNKNOWN] = self._min_conf  # default floor

        for pattern, intent, boost in _INTENT_PATTERNS:
            if pattern.search(text):
                scores[intent] = min(1.0, scores[intent] + boost)

        # Context boosts
        if context.get("is_pregnant"):
            scores[Intent.PREGNANCY] = min(1.0, scores[Intent.PREGNANCY] + 0.25)
        if context.get("is_student"):
            scores[Intent.SCHOOL_HEALTH] = min(1.0, scores[Intent.SCHOOL_HEALTH] + 0.20)
        if context.get("post_discharge"):
            scores[Intent.READMISSION] = min(1.0, scores[Intent.READMISSION] + 0.25)

        best_intent = max(scores, key=scores.__getitem__)
        return best_intent, scores[best_intent]

    def _extract_symptoms(self, text: str) -> list[str]:
        """
        Return a deduplicated list of confirmed (non-negated) clinical symptoms.

        FIX v2.2 — Negation-aware extraction
        ──────────────────────────────────────
        Before adding a symptom to the result, we inspect a left-context window
        of `_NEGATION_WINDOW` tokens.  If any negation marker is found there,
        the symptom is ABSENT and is silently dropped.

        Examples
        --------
        "عندي كحة بس معنديش سخونة"  → ['سعال']          (حمى excluded)
        "مش دايخ وعندي ألم في صدري" → ['ألم الصدر']     (دوار excluded)
        "بقيء وعندي حرارة"           → ['قيء', 'حمى']   (both confirmed)
        """
        tokens = text.split()
        found: list[str] = []

        symptom_terms: dict[str, str] = {
            # clinical term → canonical name (same key=value means term IS the name)
            "ألم"          : "ألم",
            "دوار"         : "دوار",
            "حمى"          : "حمى",
            "إسهال"        : "إسهال",
            "قيء"          : "قيء",
            "غثيان"        : "غثيان",
            "ضيق تنفس"     : "ضيق تنفس",
            "خفقان"        : "خفقان",
            "إرهاق"        : "إرهاق",
            "أرق"          : "أرق",
            "صداع"         : "صداع",
            "سعال"         : "سعال",
            "كحة"          : "سعال",       # Egyptian → clinical
            "التهاب"       : "التهاب",
            "تورم"         : "تورم",
            "نزيف"         : "نزيف",
            "حرقة"         : "حرقة معدة",
            "تنميل"        : "تنميل",
            "وخز"          : "وخز",
            "تقلص عضلي"    : "تقلص عضلي",
            "قشعريرة"      : "قشعريرة",
            "تعرق"         : "تعرق",
            "التهاب حلق"   : "التهاب حلق",
            "التهاب اللوزتين": "التهاب اللوزتين",
        }

        def _is_negated(term: str) -> bool:
            """
            Return True if `term` is preceded by a negation marker within
            the left-context window of _NEGATION_WINDOW tokens.

            Also catches Arabic circumfix negation ما…ش by checking if the
            token immediately before the term ends with 'ش'.
            """
            # Find position of term in token list (search all occurrences)
            term_tokens = term.split()
            term_len    = len(term_tokens)

            for i, tok in enumerate(tokens):
                # Check if tokens[i : i+term_len] match the term
                if tokens[i : i + term_len] == term_tokens:
                    window_start = max(0, i - _NEGATION_WINDOW)
                    left_window  = tokens[window_start : i]

                    for w in left_window:
                        if w in _NEGATION_PREFIXES:
                            return True
                        # ما…ش circumfix: token ending in 'ش' and starting 'ما'
                        if w.startswith("ما") and w.endswith("ش"):
                            return True
            return False

        # ── Check standalone symptom terms ────────────────────────────────
        for raw_term, canonical in symptom_terms.items():
            if raw_term in text:
                if not _is_negated(raw_term) and canonical not in found:
                    found.append(canonical)

        # ── Check body-part pain mentions (e.g. "ألم الصدر") ─────────────
        for colloquial, clinical in _BODY_MAP.items():
            if colloquial in text and "ألم" in text:
                combined = f"ألم {clinical}"
                if not _is_negated(colloquial) and combined not in found:
                    found.append(combined)

        return found

    def _extract_body_parts(self, text: str) -> list[str]:
        """Return clinical names of body parts mentioned in text."""
        found = []
        for colloquial, clinical in _BODY_MAP.items():
            if colloquial in text and clinical not in found:
                found.append(clinical)
        return found

    def _extract_duration(self, text: str) -> int | None:
        """
        Extract symptom duration as number of days.

        FIX v2.2: now handles "بقالي/بقاله/بقالها" in addition to "من/منذ"
        because Egyptian speakers rarely use "من" for duration.

        Examples
        --------
        "بقالي 3 أيام"    → 3
        "بقاله أسبوعين"   → 14
        "بقالها شهر"      → 30
        "من 3 أيام"       → 3
        "من أسبوعين"      → 14
        "من يوم واحد"     → 1   (after dialect normalisation of "من امبارح")
        """
        m = _DURATION_PATTERN.search(text)
        if not m:
            return None
        if m.group("word"):
            return _DURATION_WORD_MAP.get(m.group("word"))
        num  = int(m.group("num"))
        unit = m.group("unit")
        return num * _DURATION_UNIT_MAP.get(unit, 1)

    def _detect_urgency(self, text: str, intent: Intent) -> UrgencyLevel:
        """Map textual cues and intent to a UrgencyLevel."""
        if intent == Intent.EMERGENCY or _URGENCY_EMERGENCY.search(text):
            return UrgencyLevel.EMERGENCY
        if _URGENCY_HIGH.search(text):
            return UrgencyLevel.HIGH
        if _URGENCY_MEDIUM.search(text):
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _route(self, intent: Intent, context: dict[str, Any]) -> TargetModel:
        """Map intent + context to the appropriate downstream TargetModel."""
        routing: dict[Intent, TargetModel] = {
            Intent.TRIAGE          : TargetModel.TRIAGE_ENGINE,
            Intent.PREGNANCY       : TargetModel.PREGNANCY_RISK,
            Intent.SCHOOL_HEALTH   : TargetModel.SCHOOL_HEALTH,
            Intent.READMISSION     : TargetModel.READMISSION,
            Intent.LOS_ESTIMATE    : TargetModel.LOS_PREDICTOR,
            Intent.EMERGENCY       : TargetModel.TRIAGE_ENGINE,
            Intent.VIEW_HISTORY    : TargetModel.NAVIGATION_HANDLER,
            Intent.VIEW_RESULTS    : TargetModel.NAVIGATION_HANDLER,
            Intent.BOOK_APPOINTMENT: TargetModel.NAVIGATION_HANDLER,
            Intent.HELP            : TargetModel.SYSTEM_HANDLER,
            Intent.REPEAT          : TargetModel.SYSTEM_HANDLER,
            Intent.CANCEL          : TargetModel.SYSTEM_HANDLER,
            Intent.UNKNOWN         : TargetModel.MEDICAL_CHATBOT,
        }
        return routing.get(intent, TargetModel.MEDICAL_CHATBOT)


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level convenience function
# ═══════════════════════════════════════════════════════════════════════════

_default_parser: CommandParser | None = None


def parse_command(
    text   : str,
    context: dict[str, Any] | None = None,
) -> ParsedCommand:
    """
    Module-level shortcut — uses a cached default CommandParser instance.

    Suitable for single-call usage without managing parser lifecycle.

    >>> from ai_core.voice.dialect_model.command_parser import parse_command
    >>> cmd = parse_command("عندي ألم في صدري من امبارح")
    """
    global _default_parser
    if _default_parser is None:
        _default_parser = CommandParser()
    return _default_parser.parse(text, context)
