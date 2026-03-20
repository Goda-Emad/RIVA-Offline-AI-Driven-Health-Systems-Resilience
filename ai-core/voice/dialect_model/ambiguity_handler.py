"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Ambiguity Handler                           ║
║           ai-core/voice/dialect_model/ambiguity_handler.py                   ║
║                                                                              ║
║  Purpose : Detect ambiguous or underspecified Egyptian Arabic medical        ║
║            commands and generate targeted clarification questions in         ║
║            natural Egyptian Arabic.                                          ║
║                                                                              ║
║  Pipeline position                                                           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  CommandParser                                                               ║
║      ↓  (text, intent, symptoms, confidence)                                 ║
║  AmbiguityHandler.check()                                                    ║
║      ↓                                                                       ║
║  (is_ambiguous, reason, clarification_questions)                             ║
║      ↓                                                                       ║
║  ParsedCommand.is_ambiguous  →  API route decides: act / ask / fallback     ║
║                                                                              ║
║  Ambiguity types detected                                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. LOW_CONFIDENCE   — intent score below threshold                          ║
║  2. NO_SYMPTOMS      — intent=TRIAGE but no symptoms extracted               ║
║  3. MISSING_LOCATION — pain mentioned but no body part                       ║
║  4. MISSING_DURATION — chronic complaint with no time reference              ║
║  5. CONFLICTING      — contradictory signals (e.g. "كويس بس بيموت")         ║
║  6. VAGUE_COMPLAINT  — generic distress with no clinical specifics           ║
║  7. INTENT_MISMATCH  — text doesn't match detected intent                    ║
║                                                                              ║
║  Author  : Goda Emad  (AI Core)                                              ║
║  Version : 1.1.0                                                             ║
║  Updated : 2026-03-18                                                        ║
║                                                                              ║
║  Changelog v1.1.0                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • FIX  Text normalisation: strip Arabic/ASCII punctuation before tokenise.  ║
║  • FIX  Substring matching: all keyword checks use `any(kw in text)` so     ║
║         multi-word phrases ("الحمد لله") and punctuated tokens ("كويس،")   ║
║         are correctly detected — set intersection missed both cases.         ║
║  • CHG  All private checks now receive clean_text alongside tokens.         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class AmbiguityType(str, Enum):
    """Classifies the reason a command was flagged as ambiguous."""
    LOW_CONFIDENCE   = "low_confidence"
    NO_SYMPTOMS      = "no_symptoms"
    MISSING_LOCATION = "missing_location"
    MISSING_DURATION = "missing_duration"
    CONFLICTING      = "conflicting"
    VAGUE_COMPLAINT  = "vague_complaint"
    INTENT_MISMATCH  = "intent_mismatch"
    NONE             = "none"


# ═══════════════════════════════════════════════════════════════════════════
#  Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AmbiguityResult:
    """
    Full output of AmbiguityHandler.check().

    Stored in ParsedCommand and surfaced in /explain API endpoint.
    """
    is_ambiguous          : bool
    ambiguity_type        : AmbiguityType
    reason                : str               # human-readable English reason
    clarification_questions: list[str]        # Egyptian Arabic questions for the user
    severity              : float             # 0.0 = clear  →  1.0 = completely ambiguous
    confidence_penalty    : float             # how much to subtract from ParsedCommand.confidence

    def as_dict(self) -> dict:
        return {
            "is_ambiguous"           : self.is_ambiguous,
            "ambiguity_type"         : self.ambiguity_type.value,
            "reason"                 : self.reason,
            "clarification_questions": self.clarification_questions,
            "severity"               : round(self.severity, 3),
            "confidence_penalty"     : round(self.confidence_penalty, 3),
        }

    @classmethod
    def clear(cls) -> "AmbiguityResult":
        """Return a result indicating no ambiguity."""
        return cls(
            is_ambiguous           = False,
            ambiguity_type         = AmbiguityType.NONE,
            reason                 = "",
            clarification_questions= [],
            severity               = 0.0,
            confidence_penalty     = 0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Clarification question bank — Egyptian Arabic
# ═══════════════════════════════════════════════════════════════════════════

# Questions are grouped by AmbiguityType.
# Each entry is a list; the handler picks the most relevant one(s).
_QUESTIONS: dict[AmbiguityType, list[str]] = {

    AmbiguityType.NO_SYMPTOMS: [
        "ممكن تقولي أكتر؟ عندك ألم فين بالظبط؟",
        "إيه اللي بتحس بيه بالضبط؟",
        "تقدر توصف الشعور ده أكتر؟",
    ],

    AmbiguityType.MISSING_LOCATION: [
        "الألم ده فين بالظبط؟ في صدرك، بطنك، راسك؟",
        "في أنهي جزء من جسمك بتحس بالوجع؟",
        "ممكن تحدد مكان الألم؟",
    ],

    AmbiguityType.MISSING_DURATION: [
        "بقالك قد إيه حاسس بكده؟",
        "الشعور ده من امتى بالتحديد؟",
        "بقالك كام يوم أو أسبوع كده؟",
    ],

    AmbiguityType.CONFLICTING: [
        "لقيت في كلامك حاجات متناقضة — ممكن توضح أكتر؟",
        "قلت إنك كويس وفي نفس الوقت عندك ألم — إيه اللي بتحس بيه بالظبط دلوقتي؟",
        "في تناقض في اللي قلته — هل حالتك تحسنت ولا لسه فيه مشكلة؟",
    ],

    AmbiguityType.VAGUE_COMPLAINT: [
        "تعبان إزاي بالظبط؟ في ألم، دوار، أم حاجة تانية؟",
        "ممكن توصف إحساسك أكتر — مثلاً ألم، ضعف، حرارة؟",
        "عشان أساعدك صح، قولي أكتر عن اللي بتحس بيه.",
    ],

    AmbiguityType.LOW_CONFIDENCE: [
        "مش فاهم كويس — ممكن تعيد الكلام بطريقة تانية؟",
        "ممكن توضح أكتر؟ مش فاهم قصدك بالظبط.",
        "تقدر تقول نفس الكلام بأسلوب تاني؟",
    ],

    AmbiguityType.INTENT_MISMATCH: [
        "عايز أتأكد — هل عندك شكوى صحية ولا عايز حاجة تانية؟",
        "هل بتتكلم عن حالة صحية ولا حاجة تانية؟",
        "ممكن تقولي إيه اللي محتاجه بالظبط؟",
    ],
}

# Pain-related keywords — used to detect MISSING_LOCATION
_PAIN_KEYWORDS: frozenset[str] = frozenset({
    "ألم", "وجع", "بيوجعني", "بتوجعني", "وجعاني", "حرقة",
    "شد", "تقلص", "وخز", "تنميل", "ضغط",
})

# Positive/reassurance keywords — used to detect CONFLICTING signals
_POSITIVE_KEYWORDS: frozenset[str] = frozenset({
    "كويس", "كويسة", "تمام", "أحسن", "اتحسن", "مطمئن",
    "الحمد لله", "حمدلله", "بخير",
})

# Generic distress words that lack clinical specificity → VAGUE_COMPLAINT
_VAGUE_KEYWORDS: frozenset[str] = frozenset({
    "تعبان", "تعبانة", "مش كويس", "مش تمام", "مش حاسس بنفسي",
    "زهقان", "خامل", "إرهاق",
})

# Duration indicator keywords — presence means duration was communicated
_DURATION_KEYWORDS: frozenset[str] = frozenset({
    "بقالي", "بقاله", "بقالها",
    "من", "منذ", "امبارح", "إمبارح", "النهارده", "الصبح",
    "يومين", "أسبوعين", "شهرين",
    "يوم", "أيام", "أسبوع", "شهر",
})

# Intents that require body-part information for confident triage
_LOCATION_REQUIRED_INTENTS: frozenset[str] = frozenset({
    "triage", "emergency",
})

# Intents that are inherently chronic and need duration context
_DURATION_IMPORTANT_INTENTS: frozenset[str] = frozenset({
    "triage", "readmission", "los_estimate",
})


# ═══════════════════════════════════════════════════════════════════════════
#  AmbiguityHandler
# ═══════════════════════════════════════════════════════════════════════════

class AmbiguityHandler:
    """
    Stateless ambiguity detector — instantiate once, call check() many times.

    Runs a priority-ordered pipeline of checks.  The first check that fires
    is the one reported (highest-severity wins).  All checks are fast
    rule-based operations — no ML model required.

    Parameters
    ----------
    min_confidence_for_no_check : float
        If intent confidence ≥ this, skip most checks (default 0.75).
    max_questions : int
        Maximum clarification questions to return (default 2).
    severe_threshold : float
        Severity ≥ this is flagged as "severe ambiguity" (default 0.65).
    """

    def __init__(
        self,
        min_confidence_for_no_check: float = 0.75,
        max_questions              : int   = 2,
        severe_threshold           : float = 0.65,
    ) -> None:
        self._min_conf       = min_confidence_for_no_check
        self._max_q          = max_questions
        self._severe_thr     = severe_threshold

        logger.info(
            "AmbiguityHandler ready | min_conf=%.2f max_questions=%d severe_thr=%.2f",
            min_confidence_for_no_check, max_questions, severe_threshold,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def check(
        self,
        text      : str,
        intent    : str,
        symptoms  : list[str],
        confidence: float,
        body_parts: list[str] | None         = None,
        duration_days: int | None            = None,
        context   : dict[str, Any] | None    = None,
    ) -> tuple[bool, str, list[str]]:
        """
        Check a parsed command for ambiguity.

        Parameters
        ----------
        text        : Normalised command text.
        intent      : Detected intent string (e.g. "triage", "pregnancy").
        symptoms    : List of confirmed symptoms from CommandParser.
        confidence  : Intent confidence score [0, 1].
        body_parts  : Detected body-part tokens (optional).
        duration_days: Parsed duration in days (None if not detected).
        context     : Optional patient context dict.

        Returns
        -------
        (is_ambiguous, reason, clarification_questions)

        This signature matches what CommandParser expects:
            is_ambiguous, reason, clarifications = self._ambiguity.check(...)
        """
        result = self.check_full(
            text=text, intent=intent, symptoms=symptoms,
            confidence=confidence, body_parts=body_parts,
            duration_days=duration_days, context=context,
        )
        return result.is_ambiguous, result.reason, result.clarification_questions

    def check_full(
        self,
        text      : str,
        intent    : str,
        symptoms  : list[str],
        confidence: float,
        body_parts: list[str] | None         = None,
        duration_days: int | None            = None,
        context   : dict[str, Any] | None    = None,
    ) -> AmbiguityResult:
        """
        Full ambiguity check — returns AmbiguityResult with complete breakdown.

        Use this in the /explain API or in unit tests that need severity/penalty.

        FIX v1.1 — Text normalisation before keyword matching
        ───────────────────────────────────────────────────────
        Two bugs were fixed:
        1. Multi-word keywords ("الحمد لله") never matched via set intersection
           because split() produces single tokens.  Fix: use substring search
           on clean_text for all keyword lookups.
        2. Punctuation attached to words ("كويس،") broke token matching.
           Fix: strip Arabic + ASCII punctuation before splitting.

        Both fixes are applied uniformly — clean_text is passed to every
        private check that does keyword lookup.
        """
        parts      = body_parts or []
        ctx        = context    or {}

        # FIX v1.1: strip punctuation before tokenising so "كويس،" → "كويس"
        clean_text = re.sub(
            r"[،؛؟!-/:-@[-`{-~]",
            " ", text,
        )
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        tokens     = set(clean_text.split())

        # ── Fast-path: high confidence + symptoms present → clear ─────────
        if confidence >= self._min_conf and symptoms:
            logger.debug("AmbiguityHandler: fast-path clear (conf=%.2f)", confidence)
            return AmbiguityResult.clear()

        # ── Priority-ordered checks ───────────────────────────────────────
        # Each check returns an AmbiguityResult or None (didn't fire).
        # Order matters: more severe / more actionable checks come first.
        # clean_text is passed alongside tokens to enable substring matching
        # for multi-word keywords (e.g. "الحمد لله").
        checks = [
            self._check_conflicting(clean_text, tokens, symptoms),
            self._check_low_confidence(confidence),
            self._check_no_symptoms(intent, symptoms, clean_text, tokens),
            self._check_vague_complaint(intent, symptoms, clean_text, tokens),
            self._check_missing_location(intent, symptoms, parts, clean_text, tokens),
            self._check_missing_duration(intent, duration_days, clean_text, tokens, symptoms),
            self._check_intent_mismatch(intent, confidence, symptoms),
        ]

        for result in checks:
            if result is not None:
                logger.debug(
                    "AmbiguityHandler | type=%s severity=%.2f reason=%s",
                    result.ambiguity_type.value, result.severity, result.reason,
                )
                return result

        return AmbiguityResult.clear()

    # ── Private checks ────────────────────────────────────────────────────

    def _check_conflicting(
        self,
        text    : str,
        tokens  : set[str],
        symptoms: list[str],
    ) -> AmbiguityResult | None:
        """
        Detect contradictory signals.

        FIX v1.1: uses substring search (any kw in text) so multi-word
        keywords like "الحمد لله" are caught — set intersection missed them.

        Example: "أنا كويس بس بيموت"         → CONFLICTING
        Example: "الحمد لله تمام بس عندي ألم" → CONFLICTING
        Example: "كويس،"                      → کویس matches after punct strip
        """
        # FIX: exclude negated positives — "مش كويس" contains "كويس" but is NOT positive.
        # We check that the keyword is not immediately preceded by a negation marker.
        _NEG_PREFIXES = ("مش ", "مو ", "لا ", "ليس ", "معنديش ", "مفيش ")
        has_positive = any(
            kw in text and not any(text[max(0, text.index(kw)-5):text.index(kw)].strip().endswith(neg.strip())
                                   for neg in _NEG_PREFIXES)
            for kw in _POSITIVE_KEYWORDS
            if kw in text
        )
        has_negative = bool(symptoms) or any(kw in text for kw in _PAIN_KEYWORDS)

        if not (has_positive and has_negative):
            return None

        return AmbiguityResult(
            is_ambiguous           = True,
            ambiguity_type         = AmbiguityType.CONFLICTING,
            reason                 = "Contradictory signals: positive reassurance + negative symptoms",
            clarification_questions= self._pick_questions(AmbiguityType.CONFLICTING),
            severity               = 0.70,
            confidence_penalty     = 0.20,
        )

    def _check_low_confidence(self, confidence: float) -> AmbiguityResult | None:
        """Flag when intent confidence is below a safe threshold."""
        if confidence >= 0.40:
            return None

        severity = max(0.0, 1.0 - confidence * 2)   # 0.0 conf → 1.0 severity

        return AmbiguityResult(
            is_ambiguous           = True,
            ambiguity_type         = AmbiguityType.LOW_CONFIDENCE,
            reason                 = f"Intent confidence too low ({confidence:.2f} < 0.40)",
            clarification_questions= self._pick_questions(AmbiguityType.LOW_CONFIDENCE),
            severity               = round(severity, 3),
            confidence_penalty     = 0.15,
        )

    def _check_no_symptoms(
        self,
        intent  : str,
        symptoms: list[str],
        text    : str,
        tokens  : set[str],
    ) -> AmbiguityResult | None:
        """
        For triage/emergency intents with no extracted symptoms — too vague
        to route to a clinical model safely.

        FIX v1.1: uses substring search for pain keyword fallback so
        "ألم،" (with punctuation) still passes the pain-presence check.
        """
        if intent not in ("triage", "emergency"):
            return None
        if symptoms:
            return None
        # Allow if text itself contains direct pain keywords (substring match)
        if any(kw in text for kw in _PAIN_KEYWORDS):
            return None

        return AmbiguityResult(
            is_ambiguous           = True,
            ambiguity_type         = AmbiguityType.NO_SYMPTOMS,
            reason                 = "Triage intent detected but no symptoms extracted",
            clarification_questions= self._pick_questions(AmbiguityType.NO_SYMPTOMS),
            severity               = 0.75,
            confidence_penalty     = 0.25,
        )

    def _check_missing_location(
        self,
        intent    : str,
        symptoms  : list[str],
        body_parts: list[str],
        text      : str,
        tokens    : set[str],
    ) -> AmbiguityResult | None:
        """
        Pain is mentioned but no body part is specified.
        Critical for triage — "عندي ألم" needs a location for safe routing.

        FIX v1.1: substring search catches "ألم،" and multi-word pain terms.
        """
        if intent not in _LOCATION_REQUIRED_INTENTS:
            return None

        pain_present = (
            any(kw in text for kw in _PAIN_KEYWORDS)
            or any("ألم" in s or "وجع" in s for s in symptoms)
        )

        if not pain_present:
            return None
        if body_parts:
            return None

        return AmbiguityResult(
            is_ambiguous           = True,
            ambiguity_type         = AmbiguityType.MISSING_LOCATION,
            reason                 = "Pain reported but no body location specified",
            clarification_questions= self._pick_questions(AmbiguityType.MISSING_LOCATION),
            severity               = 0.55,
            confidence_penalty     = 0.10,
        )

    def _check_missing_duration(
        self,
        intent       : str,
        duration_days: int | None,
        text         : str,
        tokens       : set[str],
        symptoms     : list[str],
    ) -> AmbiguityResult | None:
        """
        Chronic-relevant intent with no duration hint.
        Duration is important for readmission risk and LOS estimation.
        Less critical for emergency (no duration needed in حالة طوارئ).

        FIX v1.1: substring search for duration keywords so "امبارح،"
        (with punctuation) still counts as a duration hint.
        """
        if intent not in _DURATION_IMPORTANT_INTENTS:
            return None
        if duration_days is not None:
            return None
        # Substring search catches "بقالي..." and punctuation-attached words
        if any(kw in text for kw in _DURATION_KEYWORDS):
            return None
        # Only flag if there are symptoms (otherwise handled by NO_SYMPTOMS)
        if not symptoms:
            return None

        return AmbiguityResult(
            is_ambiguous           = True,
            ambiguity_type         = AmbiguityType.MISSING_DURATION,
            reason                 = "Symptoms present but no duration information",
            clarification_questions= self._pick_questions(AmbiguityType.MISSING_DURATION),
            severity               = 0.35,
            confidence_penalty     = 0.05,
        )

    def _check_vague_complaint(
        self,
        intent  : str,
        symptoms: list[str],
        text    : str,
        tokens  : set[str],
    ) -> AmbiguityResult | None:
        """
        Generic distress words without any specific clinical symptom.
        Example: "أنا تعبان" → vague.  "أنا تعبان وعندي حمى" → specific enough.

        FIX v1.1: substring search catches multi-word vague phrases like
        "مش كويس" and "مش حاسس بنفسي" which set intersection missed.
        """
        if intent not in ("triage", "unknown"):
            return None

        has_vague    = any(kw in text for kw in _VAGUE_KEYWORDS)
        has_specific = bool(symptoms) and any(
            s not in _VAGUE_KEYWORDS for s in symptoms
        )

        if not has_vague:
            return None
        if has_specific:
            return None

        return AmbiguityResult(
            is_ambiguous           = True,
            ambiguity_type         = AmbiguityType.VAGUE_COMPLAINT,
            reason                 = "Only vague distress words — no specific clinical symptoms",
            clarification_questions= self._pick_questions(AmbiguityType.VAGUE_COMPLAINT),
            severity               = 0.50,
            confidence_penalty     = 0.10,
        )

    def _check_intent_mismatch(
        self,
        intent    : str,
        confidence: float,
        symptoms  : list[str],
    ) -> AmbiguityResult | None:
        """
        Intent was detected but with low-medium confidence and no supporting symptoms.
        Suggests the text may have been misclassified.
        """
        if intent in ("unknown", "help", "cancel", "repeat"):
            return None
        if confidence >= 0.55:
            return None
        if symptoms:
            return None

        return AmbiguityResult(
            is_ambiguous           = True,
            ambiguity_type         = AmbiguityType.INTENT_MISMATCH,
            reason                 = f"Intent '{intent}' detected with low confidence ({confidence:.2f}) and no symptoms",
            clarification_questions= self._pick_questions(AmbiguityType.INTENT_MISMATCH),
            severity               = 0.45,
            confidence_penalty     = 0.10,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _pick_questions(self, atype: AmbiguityType) -> list[str]:
        """Return up to max_questions from the question bank for this type."""
        bank = _QUESTIONS.get(atype, [])
        return bank[: self._max_q]

    @property
    def severe_threshold(self) -> float:
        """Severity threshold above which ambiguity is flagged as severe."""
        return self._severe_thr

    def is_severe(self, result: AmbiguityResult) -> bool:
        """Return True if the ambiguity result qualifies as severe."""
        return result.is_ambiguous and result.severity >= self._severe_thr
