"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Confidence Scorer                           ║
║           ai-core/voice/dialect_model/confidence_scorer.py                   ║
║                                                                              ║
║  Purpose : Compute a single normalised confidence score (0.0 – 1.0) for     ║
║            every ParsedCommand, combining intent confidence, clinical         ║
║            evidence strength, and penalty signals.                           ║
║                                                                              ║
║  Consumer: CommandParser._confidence.compute(...)                            ║
║            The score is stored in ParsedCommand.confidence and is used by   ║
║            downstream engines to decide whether to act, ask for             ║
║            clarification, or escalate to the medical chatbot.               ║
║                                                                              ║
║  Design principles                                                           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Fully offline — no network calls, no ML model, pure arithmetic.          ║
║  • Deterministic — same inputs always produce the same score.               ║
║  • Transparent — every component is logged at DEBUG level so engineers      ║
║    can trace exactly why a score came out as it did.                        ║
║  • Safe defaults — ambiguous or symptom-free inputs score LOW, never HIGH.  ║
║                                                                              ║
║  Score anatomy (weights sum to 1.0)                                          ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║   Component              Weight   Range    Direction                         ║
║   ─────────────────────  ──────   ──────   ─────────                        ║
║   intent_confidence       0.45    0–1      higher = better                  ║
║   symptom_evidence        0.25    0–1      more symptoms = better           ║
║   duration_bonus          0.10    0/1      present = +0.10                  ║
║   urgency_adjustment      0.10    −/+      EMERGENCY/HIGH boost             ║
║   ambiguity_penalty       0.10    0/−      ambiguous = −0.10 or −0.20       ║
║                                                                              ║
║  Author  : Goda Emad  (AI Core)                                              ║
║  Version : 1.0.0                                                             ║
║  Updated : 2026-03-18                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Thresholds — used by consumers to decide downstream action
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceThreshold:
    """
    Named thresholds for confidence-gated routing decisions.

    Usage in downstream engines
    ---------------------------
    if cmd.confidence >= ConfidenceThreshold.ACT:
        run_model(cmd)
    elif cmd.confidence >= ConfidenceThreshold.CLARIFY:
        ask_clarification(cmd)
    else:
        fallback_to_chatbot(cmd)
    """
    ACT      : float = 0.70   # Act directly — confidence is high enough
    CLARIFY  : float = 0.45   # Ask one clarifying question before acting
    FALLBACK : float = 0.30   # Route to medical chatbot for open conversation
    # Below FALLBACK → log as unrecognised, return generic help message


# ═══════════════════════════════════════════════════════════════════════════
#  Score breakdown dataclass  (returned alongside the scalar for traceability)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScoreBreakdown:
    """
    Detailed breakdown of how the final confidence score was computed.

    Stored in logs and optionally surfaced in the /explain API endpoint
    so clinicians can understand why the system acted with a given confidence.
    """
    intent_component    : float   # weighted intent confidence contribution
    symptom_component   : float   # weighted symptom evidence contribution
    duration_component  : float   # flat bonus if duration was parsed
    urgency_component   : float   # urgency-based adjustment (can be negative)
    ambiguity_penalty   : float   # penalty applied for ambiguity (≤ 0)
    raw_sum             : float   # sum before clamping
    final_score         : float   # clamped to [MIN_SCORE, 1.0]

    def as_dict(self) -> dict[str, float]:
        return {
            "intent_component"  : round(self.intent_component,  3),
            "symptom_component" : round(self.symptom_component, 3),
            "duration_component": round(self.duration_component,3),
            "urgency_component" : round(self.urgency_component, 3),
            "ambiguity_penalty" : round(self.ambiguity_penalty, 3),
            "raw_sum"           : round(self.raw_sum,           3),
            "final_score"       : round(self.final_score,       3),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  ConfidenceScorer
# ═══════════════════════════════════════════════════════════════════════════

# Import here to avoid a circular import — UrgencyLevel is defined in the
# same package but ConfidenceScorer must remain independently importable.
try:
    from .command_parser import UrgencyLevel  # noqa: F401  (re-exported)
except ImportError:
    # Standalone / testing context — define a minimal replica so this module
    # remains importable without the full package installed.
    from enum import Enum  # type: ignore[assignment]

    class UrgencyLevel(str, Enum):  # type: ignore[no-redef]
        """Replica used only when running outside the RIVA package."""
        EMERGENCY = "emergency"
        HIGH      = "high"
        MEDIUM    = "medium"
        LOW       = "low"


class ConfidenceScorer:
    """
    Stateless scorer — instantiate once, call compute() many times.

    All weights and bounds are configurable via constructor arguments so that
    future fine-tuning does not require code changes.

    Parameters
    ----------
    w_intent   : Weight for the intent confidence component  (default 0.45)
    w_symptom  : Weight for the symptom evidence component   (default 0.25)
    w_duration : Flat bonus weight if duration was detected  (default 0.10)
    w_urgency  : Weight for urgency adjustment               (default 0.10)
    w_ambiguity: Weight for ambiguity penalty                (default 0.10)
    max_symptoms_for_full_score : Number of symptoms that yields full symptom
                                  component — extra symptoms are ignored
                                  (default 3)
    min_score  : Hard floor — score never drops below this   (default 0.10)
    """

    def __init__(
        self,
        w_intent                   : float = 0.45,
        w_symptom                  : float = 0.25,
        w_duration                 : float = 0.10,
        w_urgency                  : float = 0.10,
        w_ambiguity                : float = 0.10,
        max_symptoms_for_full_score: int   = 3,
        min_score                  : float = 0.10,
    ) -> None:
        # Validate weights sum to 1.0 within floating-point tolerance
        total = w_intent + w_symptom + w_duration + w_urgency + w_ambiguity
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"ConfidenceScorer weights must sum to 1.0, got {total:.6f}"
            )

        self._w_intent    = w_intent
        self._w_symptom   = w_symptom
        self._w_duration  = w_duration
        self._w_urgency   = w_urgency
        self._w_ambiguity = w_ambiguity
        self._max_sym     = max_symptoms_for_full_score
        self._min_score   = min_score

        logger.info(
            "ConfidenceScorer ready | weights=(intent=%.2f sym=%.2f dur=%.2f "
            "urg=%.2f amb=%.2f) max_symptoms=%d min_score=%.2f",
            w_intent, w_symptom, w_duration, w_urgency, w_ambiguity,
            max_symptoms_for_full_score, min_score,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def compute(
        self,
        intent_confidence : float,
        symptoms_found    : int,
        has_duration      : bool,
        is_ambiguous      : bool,
        urgency           : "UrgencyLevel",
        *,
        ambiguity_severe  : bool = False,
    ) -> float:
        """
        Compute and return the final confidence score.

        Parameters
        ----------
        intent_confidence : Raw confidence returned by intent detection (0–1).
        symptoms_found    : Number of confirmed (non-negated) symptoms extracted.
        has_duration      : True if duration_days was successfully parsed.
        is_ambiguous      : True if AmbiguityHandler flagged the command.
        urgency           : UrgencyLevel from CommandParser._detect_urgency().
        ambiguity_severe  : True for severe ambiguity (double penalty).

        Returns
        -------
        float clamped to [min_score, 1.0].
        """
        breakdown = self._compute_breakdown(
            intent_confidence = intent_confidence,
            symptoms_found    = symptoms_found,
            has_duration      = has_duration,
            is_ambiguous      = is_ambiguous,
            urgency           = urgency,
            ambiguity_severe  = ambiguity_severe,
        )

        logger.debug(
            "ConfidenceScorer | %s",
            " | ".join(f"{k}={v:.3f}" for k, v in breakdown.as_dict().items()),
        )

        return breakdown.final_score

    def compute_with_breakdown(
        self,
        intent_confidence : float,
        symptoms_found    : int,
        has_duration      : bool,
        is_ambiguous      : bool,
        urgency           : "UrgencyLevel",
        *,
        ambiguity_severe  : bool = False,
    ) -> tuple[float, ScoreBreakdown]:
        """
        Same as compute() but also returns the full ScoreBreakdown.

        Use this variant in the /explain API endpoint or in unit tests
        that need to assert on individual components.

        Returns
        -------
        (final_score, ScoreBreakdown)
        """
        breakdown = self._compute_breakdown(
            intent_confidence = intent_confidence,
            symptoms_found    = symptoms_found,
            has_duration      = has_duration,
            is_ambiguous      = is_ambiguous,
            urgency           = urgency,
            ambiguity_severe  = ambiguity_severe,
        )
        return breakdown.final_score, breakdown

    def label(self, score: float) -> str:
        """
        Return a human-readable Arabic label for a confidence score.

        Used in the /explain page and in doctor dashboard badges.

        Examples
        --------
        >>> scorer.label(0.82)
        'عالية'
        >>> scorer.label(0.50)
        'متوسطة'
        >>> scorer.label(0.28)
        'منخفضة'
        """
        if score >= ConfidenceThreshold.ACT:
            return "عالية"
        if score >= ConfidenceThreshold.CLARIFY:
            return "متوسطة"
        if score >= ConfidenceThreshold.FALLBACK:
            return "منخفضة"
        return "غير محددة"

    def routing_decision(self, score: float) -> str:
        """
        Return the routing action name for a given score.

        Returns
        -------
        One of: "act" | "clarify" | "fallback" | "unrecognised"
        """
        if score >= ConfidenceThreshold.ACT:
            return "act"
        if score >= ConfidenceThreshold.CLARIFY:
            return "clarify"
        if score >= ConfidenceThreshold.FALLBACK:
            return "fallback"
        return "unrecognised"

    # ── Private helpers ──────────────────────────────────────────────────

    def _compute_breakdown(
        self,
        intent_confidence : float,
        symptoms_found    : int,
        has_duration      : bool,
        is_ambiguous      : bool,
        urgency           : "UrgencyLevel",
        ambiguity_severe  : bool,
    ) -> ScoreBreakdown:
        """Core arithmetic — returns a full ScoreBreakdown."""

        # ── 1. Intent component ───────────────────────────────────────────
        # Clamp raw intent confidence to [0, 1] before weighting.
        intent_clamped   = max(0.0, min(1.0, intent_confidence))
        intent_component = self._w_intent * intent_clamped

        # ── 2. Symptom evidence component ────────────────────────────────
        # Saturates at max_symptoms_for_full_score — extra symptoms beyond
        # that threshold do not keep inflating the score (avoids gaming).
        symptom_ratio     = min(symptoms_found, self._max_sym) / self._max_sym
        symptom_component = self._w_symptom * symptom_ratio

        # ── 3. Duration bonus ─────────────────────────────────────────────
        # A parsed duration (e.g. "بقالي 3 أيام") is strong evidence that
        # the speaker is describing a real clinical situation.
        duration_component = self._w_duration if has_duration else 0.0

        # ── 4. Urgency adjustment ─────────────────────────────────────────
        # EMERGENCY and HIGH urgency increase confidence — if someone says
        # "نجدة" or "ألم شديد" the intent is unambiguous, so we boost.
        # MEDIUM is neutral.  LOW gets a small penalty (vague complaint).
        urgency_adjustments: dict[str, float] = {
            UrgencyLevel.EMERGENCY : +1.0,   # full urgency weight
            UrgencyLevel.HIGH      : +0.70,
            UrgencyLevel.MEDIUM    : +0.30,
            UrgencyLevel.LOW       : -0.20,  # vague — slight reduction
        }
        urgency_factor    = urgency_adjustments.get(urgency, 0.0)
        urgency_component = self._w_urgency * urgency_factor

        # ── 5. Ambiguity penalty ──────────────────────────────────────────
        # is_ambiguous   → −1× weight
        # ambiguity_severe → −2× weight (e.g. conflicting symptoms + no body part)
        if ambiguity_severe:
            penalty_factor = -2.0
        elif is_ambiguous:
            penalty_factor = -1.0
        else:
            penalty_factor = 0.0
        ambiguity_penalty = self._w_ambiguity * penalty_factor

        # ── 6. Aggregate ──────────────────────────────────────────────────
        raw_sum     = (
            intent_component
            + symptom_component
            + duration_component
            + urgency_component
            + ambiguity_penalty
        )
        final_score = max(self._min_score, min(1.0, raw_sum))

        return ScoreBreakdown(
            intent_component   = intent_component,
            symptom_component  = symptom_component,
            duration_component = duration_component,
            urgency_component  = urgency_component,
            ambiguity_penalty  = ambiguity_penalty,
            raw_sum            = raw_sum,
            final_score        = final_score,
        )
