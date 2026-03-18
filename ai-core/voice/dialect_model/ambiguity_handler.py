"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform v4.0 — Ambiguity Handler                     ║
║           ai-core/voice/dialect_model/ambiguity_handler.py                  ║
║                                                                              ║
║  Detects and resolves ambiguous medical input from Egyptian Arabic patients: ║
║  • Low-confidence STT output → targeted clarification questions             ║
║  • Ambiguous symptom descriptions → structured disambiguation               ║
║  • Mixed intent signals → ranked intent resolution                          ║
║  • Missing critical medical info → smart follow-up prompts                  ║
║  • Arabizi (Arabic in Latin script) detection & normalization               ║
║                                                                              ║
║  Harvard HSIL Hackathon 2026                                                 ║
║  Maintainer: GODA EMAD                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("riva.ambiguity_handler")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_HERE            = Path(__file__).parent
AMBIGUITY_MAP    = _HERE.parent.parent.parent / "business-intelligence" / "mapping" / "ambiguity_map.json"
INTENT_MAP       = _HERE.parent.parent.parent / "business-intelligence" / "mapping" / "intent_mapping.json"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LOW_CONFIDENCE_THRESHOLD: float = 0.72    # below → ask for clarification
AMBIGUOUS_INTENT_DELTA:   float = 0.15    # top-2 confidences within delta → ambiguous
MAX_CLARIFICATION_ROUNDS: int   = 2       # max follow-up questions per session
MIN_TEXT_LENGTH:          int   = 3       # chars — shorter than this is too vague


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ─────────────────────────────────────────────────────────────────────────────

class AmbiguityType(Enum):
    LOW_STT_CONFIDENCE  = "low_stt_confidence"   # Whisper wasn't sure
    VAGUE_SYMPTOM       = "vague_symptom"         # "مش كويس" / "تعبان"
    MIXED_INTENT        = "mixed_intent"          # triage + pregnancy together
    MISSING_BODY_PART   = "missing_body_part"     # pain mentioned, location unknown
    MISSING_DURATION    = "missing_duration"      # severity known, timing unknown
    ARABIZI_INPUT       = "arabizi_input"         # 3ndi waga3 fi batni
    TOO_SHORT           = "too_short"             # 1-2 words, not enough context
    NONE                = "none"                  # no ambiguity detected


class ResolutionStrategy(Enum):
    CLARIFY         = "clarify"      # ask follow-up question
    ASSUME_WORST    = "assume_worst" # escalate to triage (safety-first)
    PICK_TOP_INTENT = "pick_top"     # use highest-confidence intent
    NORMALIZE       = "normalize"    # fix Arabizi → Arabic and retry
    REJECT          = "reject"       # too ambiguous, ask to repeat


@dataclass
class AmbiguitySignal:
    """Detected ambiguity with its type, score and suggested resolution."""
    ambiguity_type:    AmbiguityType
    confidence:        float          # 0.0 = definite ambiguity, 1.0 = clear
    description:       str            # human-readable explanation
    strategy:          ResolutionStrategy
    clarification_q:   Optional[str] = None   # Arabic question to ask patient
    normalized_text:   Optional[str] = None   # Arabizi-fixed text if applicable


@dataclass
class AmbiguityResult:
    """
    Output of AmbiguityHandler.check().
    If is_ambiguous is False, text and intent_hint are ready to use directly.
    """
    is_ambiguous:     bool
    original_text:    str
    resolved_text:    str                        # possibly normalized
    signals:          list[AmbiguitySignal]
    primary_signal:   Optional[AmbiguitySignal] # highest priority
    clarification_q:  Optional[str]             # question to show patient
    intent_hint:      Optional[str]             # best-guess intent if resolvable
    rounds_used:      int = 0

    def to_dict(self) -> dict:
        return {
            "is_ambiguous":    self.is_ambiguous,
            "original_text":   self.original_text,
            "resolved_text":   self.resolved_text,
            "clarification_q": self.clarification_q,
            "intent_hint":     self.intent_hint,
            "signals": [
                {
                    "type":     s.ambiguity_type.value,
                    "confidence": s.confidence,
                    "strategy": s.strategy.value,
                    "description": s.description,
                }
                for s in self.signals
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Arabizi Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class ArabiziNormalizer:
    """
    Converts Arabizi (Arabic written in Latin letters) to Arabic script.
    Common in Egyptian young patients texting/speaking casually.

    Examples:
        "3ndi waga3"     → "عندي وجع"
        "el doktor"      → "الدكتور"
        "ana 7amil"      → "أنا حامل"
    """

    # Core Arabizi → Arabic character mapping (Egyptian conventions)
    _CHAR_MAP: dict[str, str] = {
        "3":  "ع",   # 3ain
        "7":  "ح",   # ha
        "2":  "أ",   # hamza / alef
        "5":  "خ",   # kha
        "9":  "ص",   # sad
        "6":  "ط",   # ta
        "8":  "ق",   # qaf (sometimes)
        "4":  "ذ",   # dhal (less common)
        "gh": "غ",
        "kh": "خ",
        "sh": "ش",
        "th": "ث",
        "dh": "ذ",
    }

    # Common full-word Arabizi medical phrases → Arabic
    _WORD_MAP: dict[str, str] = {
        "3ndi":       "عندي",
        "waga3":      "وجع",
        "waja3":      "وجع",
        "sokhan":     "سخونة",
        "skhona":     "سخونة",
        "so5on":      "سخونة",
        "da7ia":      "دوخة",
        "daw5a":      "دوخة",
        "9oda3":      "صداع",
        "sodaa":      "صداع",
        "7amil":      "حامل",
        "7amel":      "حامل",
        "el":         "ال",
        "al":         "ال",
        "doktor":     "دكتور",
        "doctor":     "دكتور",
        "mostasfaa":  "مستشفى",
        "mostashfa":  "مستشفى",
        "dawa":       "دواء",
        "dawaa":      "دواء",
        "ta7lil":     "تحليل",
        "ashaa":      "أشعة",
        "3amalia":    "عملية",
        "3amalya":    "عملية",
        "batni":      "بطني",
        "rassi":      "رأسي",
        "sadri":      "صدري",
        "dahri":      "ظهري",
        "rigly":      "رجلي",
        "eddi":       "يدي",
        "ana":        "أنا",
        "mesh":       "مش",
        "msh":        "مش",
        "kwayis":     "كويس",
        "quayis":     "كويس",
        "tb3an":      "طبعاً",
        "mn":         "من",
        "fi":         "في",
        "3la":        "على",
        "b3d":        "بعد",
        "mb3d":       "مبعد",
        "lama":       "لما",
        "lamma":      "لما",
        "momken":     "ممكن",
        "3awz":       "عايز",
        "3ayez":      "عايز",
        "tab3an":     "طبعاً",
    }

    @classmethod
    def is_arabizi(cls, text: str) -> bool:
        """
        Detect if text is likely Arabizi (Latin-script Arabic).
        Heuristic: mostly Latin chars but contains Arabizi digits (3, 7, 2…)
        or known Arabizi words.
        """
        if not text:
            return False
        latin_chars  = sum(1 for c in text if c.isascii() and c.isalpha())
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_alpha  = latin_chars + arabic_chars + 1  # +1 avoids div-zero

        is_mostly_latin = latin_chars / total_alpha > 0.6
        has_arabizi_num = any(c in text for c in "3725968")
        has_arabizi_word = any(w in text.lower().split() for w in cls._WORD_MAP)
        return is_mostly_latin and (has_arabizi_num or has_arabizi_word)

    @classmethod
    def normalize(cls, text: str) -> str:
        """Convert Arabizi text to Arabic script."""
        words    = text.lower().split()
        result   = []

        for word in words:
            # Full-word lookup first (fastest, most accurate)
            if word in cls._WORD_MAP:
                result.append(cls._WORD_MAP[word])
                continue

            # Character-level substitution
            converted = word
            for lat, ar in cls._CHAR_MAP.items():
                converted = converted.replace(lat, ar)
            result.append(converted)

        normalized = " ".join(result)
        logger.debug("Arabizi normalized: '%s' → '%s'", text, normalized)
        return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Vague Symptom Detector
# ─────────────────────────────────────────────────────────────────────────────

class VagueSymptomDetector:
    """
    Identifies overly vague symptom descriptions that need clarification.
    Egyptian patients often say "مش كويس" or "تعبان" without details.
    """

    # Vague phrases that trigger clarification
    _VAGUE_PHRASES: set[str] = {
        "مش كويس", "مش كويسة", "تعبان", "تعبانة",
        "وجع", "بيوجعني", "ألم", "مريض", "مريضة",
        "حاسس بإيه", "مش عارف", "صاحي", "إحساس غريب",
        "مش نفسي", "عيان", "عيانة", "مضايق", "مش عارف أوصف",
    }

    # High-risk vague phrases → escalate immediately (safety-first)
    _HIGH_RISK_VAGUE: set[str] = {
        "مش قادر أتنفس", "ضيق تنفس", "مش شايف",
        "بيغمى عليا", "خدر", "شلل", "سقطت",
    }

    # Clarification questions keyed by detected vagueness
    _CLARIFICATION_MAP: dict[str, str] = {
        "وجع":         "فين بالظبط الوجع؟ (رأس / بطن / صدر / ظهر / رجل)",
        "تعبان":       "إيه اللي بتحس بيه تحديداً؟ (سخونة / صداع / وجع / دوخة)",
        "مش كويس":    "ممكن توصفلي أكتر؟ الإحساس ده من إمتى؟",
        "ألم":         "الألم ده شديد ولا خفيف؟ وفين بالظبط؟",
        "مريض":        "عندك سخونة؟ وجع فين؟ من إمتى؟",
        "default":     "ممكن توصفلي أكتر؟ إيه اللي بتحس بيه بالظبط؟",
    }

    def check(self, text: str) -> Optional[AmbiguitySignal]:
        """Returns AmbiguitySignal if text is vague, None otherwise."""
        text_lower = text.strip()

        # High-risk vague → escalate, don't clarify
        for phrase in self._HIGH_RISK_VAGUE:
            if phrase in text_lower:
                return AmbiguitySignal(
                    ambiguity_type=AmbiguityType.VAGUE_SYMPTOM,
                    confidence=0.2,
                    description=f"عبارة عالية الخطورة غامضة: '{phrase}'",
                    strategy=ResolutionStrategy.ASSUME_WORST,
                    clarification_q=None,  # escalate directly
                )

        # Check vague phrases
        for phrase in self._VAGUE_PHRASES:
            if phrase in text_lower and len(text_lower.split()) <= 4:
                q = self._CLARIFICATION_MAP.get(
                    phrase, self._CLARIFICATION_MAP["default"]
                )
                return AmbiguitySignal(
                    ambiguity_type=AmbiguityType.VAGUE_SYMPTOM,
                    confidence=0.35,
                    description=f"وصف غامض جداً: '{phrase}'",
                    strategy=ResolutionStrategy.CLARIFY,
                    clarification_q=q,
                )

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Missing Info Detector
# ─────────────────────────────────────────────────────────────────────────────

class MissingInfoDetector:
    """
    Identifies when critical medical information is present but incomplete.
    E.g. patient mentions pain but no location, or mentions duration vaguely.
    """

    _PAIN_KEYWORDS: set[str] = {
        "وجع", "ألم", "بيوجعني", "بيتألم", "حرقان", "ضغط",
    }
    _BODY_PARTS: set[str] = {
        "رأس", "بطن", "صدر", "ظهر", "رجل", "يد", "عين", "أذن",
        "رقبة", "ركبة", "كتف", "فم", "معدة", "قلب", "كلى",
    }
    _DURATION_WORDS: set[str] = {
        "من", "منذ", "امبارح", "النهارده", "أمس", "أسبوع",
        "ساعة", "يوم", "شهر", "دلوقتي", "فجأة",
    }
    _SEVERITY_WORDS: set[str] = {
        "شديد", "خفيف", "متوسط", "قوي", "قليل", "جداً",
        "٧", "٨", "٩", "١٠", "7", "8", "9", "10",
    }

    def check(self, text: str) -> list[AmbiguitySignal]:
        """Returns list of missing-info signals (may be empty)."""
        signals: list[AmbiguitySignal] = []
        has_pain     = any(kw in text for kw in self._PAIN_KEYWORDS)
        has_location = any(bp in text for bp in self._BODY_PARTS)
        has_duration = any(dw in text for dw in self._DURATION_WORDS)

        if has_pain and not has_location:
            signals.append(AmbiguitySignal(
                ambiguity_type=AmbiguityType.MISSING_BODY_PART,
                confidence=0.45,
                description="ذكر الألم بدون تحديد المكان",
                strategy=ResolutionStrategy.CLARIFY,
                clarification_q="الوجع ده فين بالظبط؟ (رأس / بطن / صدر / ظهر / رجل)",
            ))

        if has_pain and not has_duration:
            signals.append(AmbiguitySignal(
                ambiguity_type=AmbiguityType.MISSING_DURATION,
                confidence=0.55,
                description="ذكر الألم بدون تحديد المدة",
                strategy=ResolutionStrategy.CLARIFY,
                clarification_q="الوجع ده من إمتى؟ (ساعات / أيام / أسبوع؟)",
            ))

        return signals


# ─────────────────────────────────────────────────────────────────────────────
# Intent Conflict Detector
# ─────────────────────────────────────────────────────────────────────────────

class IntentConflictDetector:
    """
    Detects when multiple intents are competing (mixed signals).
    E.g. patient mentions both pregnancy concerns and drug interaction.
    Resolution: pick safer/higher-priority intent or ask to confirm.
    """

    # Intent priority (higher = more urgent)
    _INTENT_PRIORITY: dict[str, int] = {
        "emergency_alert":       10,
        "triage_intent":          9,
        "pregnancy_concern":      8,
        "medication_query":       7,
        "test_result_query":      6,
        "doctor_followup":        5,
        "symptom_description":    4,
        "appointment_request":    3,
        "school_health_report":   2,
        "nutrition_advice":       1,
        "general_inquiry":        0,
    }

    def check(
        self,
        intent_scores: dict[str, float],
    ) -> Optional[AmbiguitySignal]:
        """
        Check if top-2 intents are too close (within AMBIGUOUS_INTENT_DELTA).
        Returns AmbiguitySignal with the higher-priority intent as hint.
        """
        if not intent_scores or len(intent_scores) < 2:
            return None

        sorted_intents = sorted(
            intent_scores.items(), key=lambda x: x[1], reverse=True
        )
        top1_name, top1_score = sorted_intents[0]
        top2_name, top2_score = sorted_intents[1]

        delta = abs(top1_score - top2_score)
        if delta > AMBIGUOUS_INTENT_DELTA:
            return None  # clear winner

        # Resolve by priority
        p1 = self._INTENT_PRIORITY.get(top1_name, 0)
        p2 = self._INTENT_PRIORITY.get(top2_name, 0)
        winner = top1_name if p1 >= p2 else top2_name

        return AmbiguitySignal(
            ambiguity_type=AmbiguityType.MIXED_INTENT,
            confidence=delta,
            description=(
                f"تعارض بين '{top1_name}' ({top1_score:.2f}) "
                f"و '{top2_name}' ({top2_score:.2f}) — δ={delta:.2f}"
            ),
            strategy=ResolutionStrategy.PICK_TOP_INTENT,
            clarification_q=(
                f"هل طلبك أقرب لـ '{self._intent_label(top1_name)}' "
                f"أم '{self._intent_label(top2_name)}'؟"
            ),
        )

    @staticmethod
    def _intent_label(intent: str) -> str:
        """Human-readable Arabic label for intent names."""
        labels = {
            "triage_intent":       "فرز طبي / طوارئ",
            "pregnancy_concern":   "صحة الحمل",
            "medication_query":    "استفسار دواء",
            "symptom_description": "وصف أعراض",
            "appointment_request": "حجز موعد",
            "emergency_alert":     "حالة طارئة",
            "test_result_query":   "نتيجة تحليل",
            "doctor_followup":     "متابعة مع دكتور",
            "school_health_report":"صحة مدرسية",
            "nutrition_advice":    "نصيحة غذائية",
            "general_inquiry":     "استفسار عام",
        }
        return labels.get(intent, intent)


# ─────────────────────────────────────────────────────────────────────────────
# Main AmbiguityHandler
# ─────────────────────────────────────────────────────────────────────────────

class AmbiguityHandler:
    """
    Central orchestrator for all ambiguity detection and resolution in RIVA.

    Checks in order:
    1. Arabizi detection → normalize and re-evaluate
    2. Too-short input → ask to repeat
    3. Low STT confidence → ask to repeat or rephrase
    4. Vague symptom → targeted clarification question
    5. Missing critical info → specific follow-up question
    6. Mixed intent → resolve by priority

    Usage:
        handler = AmbiguityHandler()
        result  = handler.check(text, stt_confidence=0.65)
        if result.is_ambiguous:
            show_to_patient(result.clarification_q)
        else:
            pass_to_triage(result.resolved_text)
    """

    def __init__(
        self,
        ambiguity_map_path: Path = AMBIGUITY_MAP,
        intent_map_path: Path    = INTENT_MAP,
    ) -> None:
        self._vague    = VagueSymptomDetector()
        self._missing  = MissingInfoDetector()
        self._conflict = IntentConflictDetector()
        self._normalizer = ArabiziNormalizer()
        self._extra_map: dict = {}
        self._load_maps(ambiguity_map_path, intent_map_path)

    # ── Public API ──────────────────────────────────────────────────────────

    def check(
        self,
        text: str,
        stt_confidence: float = 1.0,
        intent_scores: Optional[dict[str, float]] = None,
        rounds_used: int = 0,
    ) -> AmbiguityResult:
        """
        Full ambiguity check pipeline.

        Args:
            text:           Raw transcribed text from STT
            stt_confidence: Overall confidence from SpeechToText (0–1)
            intent_scores:  Dict of {intent_name: score} from chatbot model
            rounds_used:    How many clarification rounds already used

        Returns:
            AmbiguityResult — check is_ambiguous before proceeding
        """
        signals: list[AmbiguitySignal] = []
        resolved_text = text.strip()

        # ── 0. Empty / too short ────────────────────────────────────────
        if not resolved_text or len(resolved_text) < MIN_TEXT_LENGTH:
            sig = AmbiguitySignal(
                ambiguity_type=AmbiguityType.TOO_SHORT,
                confidence=0.0,
                description="النص قصير جداً أو فارغ",
                strategy=ResolutionStrategy.REJECT,
                clarification_q="عذراً، ما سمعتكش كويس. ممكن تعيد الكلام؟",
            )
            return self._build_result(
                text, resolved_text, [sig], rounds_used
            )

        # ── 1. Arabizi detection & normalization ────────────────────────
        if ArabiziNormalizer.is_arabizi(resolved_text):
            normalized = ArabiziNormalizer.normalize(resolved_text)
            sig = AmbiguitySignal(
                ambiguity_type=AmbiguityType.ARABIZI_INPUT,
                confidence=0.6,
                description=f"تم اكتشاف Arabizi وتحويله: '{resolved_text}' → '{normalized}'",
                strategy=ResolutionStrategy.NORMALIZE,
                normalized_text=normalized,
            )
            signals.append(sig)
            resolved_text = normalized
            logger.info("Arabizi normalized: '%s' → '%s'", text, normalized)

        # ── 2. Low STT confidence ───────────────────────────────────────
        if stt_confidence < LOW_CONFIDENCE_THRESHOLD:
            sig = AmbiguitySignal(
                ambiguity_type=AmbiguityType.LOW_STT_CONFIDENCE,
                confidence=stt_confidence,
                description=f"ثقة الـ STT منخفضة: {stt_confidence:.2f} < {LOW_CONFIDENCE_THRESHOLD}",
                strategy=ResolutionStrategy.CLARIFY,
                clarification_q=(
                    "معذرة، ما فهمتش كويس. ممكن تعيد الكلام ببطء أكتر؟"
                    if stt_confidence < 0.50 else
                    f"هل قصدك: «{resolved_text}»؟"
                ),
            )
            signals.append(sig)

        # ── 3. Vague symptom ────────────────────────────────────────────
        vague_sig = self._vague.check(resolved_text)
        if vague_sig:
            signals.append(vague_sig)
            # High-risk vague → stop here, escalate immediately
            if vague_sig.strategy == ResolutionStrategy.ASSUME_WORST:
                return self._build_result(
                    text, resolved_text, signals, rounds_used
                )

        # ── 4. Missing critical info ────────────────────────────────────
        missing_sigs = self._missing.check(resolved_text)
        signals.extend(missing_sigs)

        # ── 5. Intent conflict ──────────────────────────────────────────
        if intent_scores:
            conflict_sig = self._conflict.check(intent_scores)
            if conflict_sig:
                signals.append(conflict_sig)

        # ── 6. Max rounds reached → pick best guess ─────────────────────
        if rounds_used >= MAX_CLARIFICATION_ROUNDS and signals:
            for sig in signals:
                sig.strategy = ResolutionStrategy.PICK_TOP_INTENT
            logger.info(
                "Max clarification rounds reached (%d) — forcing best-guess resolution",
                rounds_used,
            )

        return self._build_result(text, resolved_text, signals, rounds_used)

    def generate_clarification(
        self,
        result: AmbiguityResult,
        patient_name: Optional[str] = None,
    ) -> str:
        """
        Format a patient-friendly clarification question in Egyptian Arabic.
        Adds a polite greeting if patient_name is provided.
        """
        if not result.is_ambiguous or not result.clarification_q:
            return ""

        prefix = f"يا {patient_name}، " if patient_name else ""
        return f"{prefix}{result.clarification_q}"

    # ── Private Helpers ─────────────────────────────────────────────────────

    def _build_result(
        self,
        original: str,
        resolved: str,
        signals: list[AmbiguitySignal],
        rounds_used: int,
    ) -> AmbiguityResult:
        """Build AmbiguityResult from collected signals."""
        if not signals:
            return AmbiguityResult(
                is_ambiguous=False,
                original_text=original,
                resolved_text=resolved,
                signals=[],
                primary_signal=None,
                clarification_q=None,
                intent_hint=None,
                rounds_used=rounds_used,
            )

        # Pick primary signal: ASSUME_WORST > REJECT > CLARIFY > others
        priority_order = [
            ResolutionStrategy.ASSUME_WORST,
            ResolutionStrategy.REJECT,
            ResolutionStrategy.CLARIFY,
            ResolutionStrategy.NORMALIZE,
            ResolutionStrategy.PICK_TOP_INTENT,
        ]
        primary = min(
            signals,
            key=lambda s: priority_order.index(s.strategy)
            if s.strategy in priority_order else len(priority_order),
        )

        # Determine if truly ambiguous (NORMALIZE alone is not blocking)
        blocking_strategies = {
            ResolutionStrategy.CLARIFY,
            ResolutionStrategy.ASSUME_WORST,
            ResolutionStrategy.REJECT,
        }
        is_ambiguous = any(s.strategy in blocking_strategies for s in signals)

        # Intent hint from highest-priority intent if available
        intent_hint: Optional[str] = None
        for sig in signals:
            if sig.ambiguity_type == AmbiguityType.MIXED_INTENT:
                # Extract winner from description
                intent_hint = sig.description.split("'")[1] if "'" in sig.description else None
                break

        return AmbiguityResult(
            is_ambiguous=is_ambiguous,
            original_text=original,
            resolved_text=resolved,
            signals=signals,
            primary_signal=primary,
            clarification_q=primary.clarification_q if is_ambiguous else None,
            intent_hint=intent_hint,
            rounds_used=rounds_used,
        )

    def _load_maps(
        self,
        ambiguity_path: Path,
        intent_path: Path,
    ) -> None:
        """Load external ambiguity/intent maps if available."""
        for path, label in [
            (ambiguity_path, "ambiguity_map"),
            (intent_path, "intent_map"),
        ]:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self._extra_map.update(data)
                    logger.info("Loaded %s from %s", label, path)
                except Exception as exc:
                    logger.warning("Could not load %s: %s", label, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton for FastAPI
# ─────────────────────────────────────────────────────────────────────────────

_handler_instance: Optional[AmbiguityHandler] = None


def get_ambiguity_handler() -> AmbiguityHandler:
    """
    Shared AmbiguityHandler for FastAPI dependency injection.

    Usage in routes:
        from ai_core.voice.dialect_model.ambiguity_handler import get_ambiguity_handler

        @router.post("/chat")
        async def chat_endpoint(
            body: ChatRequest,
            handler: AmbiguityHandler = Depends(get_ambiguity_handler),
        ):
            result = handler.check(body.text, stt_confidence=body.confidence)
            if result.is_ambiguous:
                return {"clarification": result.clarification_q}
            ...
    """
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = AmbiguityHandler()
    return _handler_instance


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    handler = AmbiguityHandler()

    print("=" * 60)
    print("RIVA AmbiguityHandler — self-test")
    print("=" * 60)

    test_cases = [
        # (text, stt_confidence, intent_scores, description)
        ("",                      1.0,  None,                                "Empty input"),
        ("آه",                    1.0,  None,                                "Too short"),
        ("3ndi waga3 fi batni",   0.90, None,                                "Arabizi input"),
        ("تعبان",                  1.0,  None,                                "Vague symptom (short)"),
        ("بيوجعني من امبارح",       1.0,  None,                                "Pain, missing location"),
        ("عندي وجع",               0.45, None,                                "Low STT + vague"),
        ("مش قادر أتنفس",          1.0,  None,                                "High-risk vague → escalate"),
        ("عندي سخونة وكحة من 3 أيام", 0.88,
         {"triage_intent": 0.55, "pregnancy_concern": 0.52},               "Mixed intent (close scores)"),
        ("عندي صداع شديد في الرأس من امبارح", 0.91, None,                   "Clear — no ambiguity"),
    ]

    for text, conf, intents, desc in test_cases:
        result = handler.check(text, stt_confidence=conf, intent_scores=intents)
        status = "🔴 AMBIGUOUS" if result.is_ambiguous else "🟢 CLEAR"
        print(f"\n[{status}] {desc}")
        print(f"  Input    : '{text[:45]}'")
        print(f"  Resolved : '{result.resolved_text[:45]}'")
        if result.is_ambiguous:
            sig = result.primary_signal
            print(f"  Type     : {sig.ambiguity_type.value}")
            print(f"  Strategy : {sig.strategy.value}")
            print(f"  Question : {result.clarification_q}")

    # Arabizi normalizer direct test
    print("\n" + "─" * 40)
    print("Arabizi normalizer:")
    arabizi_samples = [
        "3ndi waga3 fi batni",
        "ana 7amil w 3ndi daghet",
        "el doctor 2al 3andi sokhan",
    ]
    for s in arabizi_samples:
        print(f"  '{s}' → '{ArabiziNormalizer.normalize(s)}'")

    print("\n✅ AmbiguityHandler self-test complete")
    sys.exit(0)
