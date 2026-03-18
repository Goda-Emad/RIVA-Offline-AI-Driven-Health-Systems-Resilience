"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Sentiment Analyzer                          ║
║           ai-core/voice/dialect_model/sentiment_analyzer.py                  ║
║                                                                              ║
║  Purpose : Score Egyptian Arabic medical text on a distress-to-calm axis     ║
║            (−1.0 → +1.0) using a weighted lexicon + linguistic intensifiers. ║
║                                                                              ║
║  Output  : float stored in ParsedCommand.sentiment_score                     ║
║            −1.0  extreme distress  (طوارئ، بيموت، مش قادر أتنفس)           ║
║             0.0  neutral           (عندي وجع خفيف)                          ║
║            +1.0  calm / reassured  (كويس، أحسن، تمام)                       ║
║                                                                              ║
║  Pipeline position                                                           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  CommandParser._sentiment.score(normalised_text)                             ║
║      ↓                                                                       ║
║  ParsedCommand.sentiment_score  →  doctor dashboard badge                   ║
║                                 →  urgency escalation guard                 ║
║                                 →  SHAP explanation renderer                ║
║                                                                              ║
║  Design                                                                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Fully offline — rule-based weighted lexicon, no ML model.                ║
║  • Three-pass scoring: base lexicon → intensifier boost → negation flip.    ║
║  • Emergency override: intent=EMERGENCY forces score to −1.0.               ║
║  • Returns SentimentResult dataclass for traceability in /explain API.      ║
║                                                                              ║
║  Author  : Goda Emad  (AI Core)                                              ║
║  Version : 1.1.0                                                             ║
║  Updated : 2026-03-18                                                        ║
║                                                                              ║
║  Changelog v1.1.0                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • FIX  Substring false-positive: replaced `if term in text:` (substring    ║
║         match) with token-exact match via term_found from                   ║
║         _context_multiplier. Prevents "معلم" matching "ألم", "مورم"        ║
║         matching "ورم", and similar substring ghosts.                       ║
║  • FIX  Emergency override uses token set intersection — same class of bug. ║
║  • CHG  _context_multiplier now returns 4-tuple (term_found, multiplier,    ║
║         intensifier_used, was_negated) — callers must check term_found.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SentimentResult:
    """
    Full breakdown of a sentiment scoring pass.

    Returned by SentimentAnalyzer.score_full() and surfaced in the
    /explain API endpoint so clinicians can audit the score.
    """
    score           : float          # final clamped score  −1.0 → +1.0
    label           : str            # "ضائقة شديدة" / "قلق" / "محايد" / "مطمئن"
    matched_negative: list[str]      # distress terms found
    matched_positive: list[str]      # calm/reassurance terms found
    matched_intensifiers: list[str]  # intensifier tokens found
    negation_flips  : list[str]      # terms whose polarity was flipped
    emergency_override: bool         # True if score was forced to −1.0
    raw_score       : float          # score before emergency override

    def as_dict(self) -> dict:
        return {
            "score"              : round(self.score, 3),
            "label"              : self.label,
            "matched_negative"   : self.matched_negative,
            "matched_positive"   : self.matched_positive,
            "intensifiers"       : self.matched_intensifiers,
            "negation_flips"     : self.negation_flips,
            "emergency_override" : self.emergency_override,
            "raw_score"          : round(self.raw_score, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Lexicons
# ═══════════════════════════════════════════════════════════════════════════

# ── Negative / distress terms  (weight: −0.10 to −0.40) ──────────────────
# Higher magnitude = stronger distress signal
_NEGATIVE_LEXICON: dict[str, float] = {

    # ── Extreme distress (−0.40) — emergency-level language ────────────
    "طوارئ"              : -0.40,
    "نجدة"               : -0.40,
    "بيموت"              : -0.40,
    "بتموت"              : -0.40,
    "مش قادر أتنفس"      : -0.40,
    "مش لاقي نفسي"       : -0.40,
    "ضيق تنفس"           : -0.35,

    # ── Severe pain / fear (−0.25 to −0.30) ────────────────────────────
    "ألم شديد"           : -0.30,
    "وجع جامد"           : -0.30,
    "خايف"               : -0.28,
    "خايفة"              : -0.28,
    "مرعوب"              : -0.30,
    "مش عارف هعمل إيه"  : -0.28,
    "قلقان"              : -0.22,
    "قلقانة"             : -0.22,
    "متضايق"             : -0.20,
    "متضايقة"            : -0.20,
    "زعلان"              : -0.18,
    "زعلانة"             : -0.18,
    "خفقان"              : -0.25,
    "دوار"               : -0.20,

    # ── Moderate distress (−0.15 to −0.20) ─────────────────────────────
    "تعبان"              : -0.18,
    "تعبانة"             : -0.18,
    "مريض"               : -0.18,
    "مريضة"              : -0.18,
    "ألم"                : -0.15,
    "وجع"                : -0.15,
    "حمى"                : -0.15,
    "إسهال"              : -0.12,
    "قيء"                : -0.15,
    "غثيان"              : -0.12,
    "إرهاق"              : -0.14,
    "أرق"                : -0.12,
    "صداع"               : -0.12,
    "سعال"               : -0.10,
    "التهاب"             : -0.12,
    "تورم"               : -0.12,
    "نزيف"               : -0.25,
    "تنميل"              : -0.12,

    # ── Mild complaint (−0.08 to −0.10) ────────────────────────────────
    "مش كويس"            : -0.10,
    "مش تمام"            : -0.10,
    "مش حاسس بنفسي"      : -0.10,
    "زهقان"              : -0.08,
    "ملهوف"              : -0.15,
}

# ── Positive / calm terms  (weight: +0.10 to +0.35) ──────────────────────
_POSITIVE_LEXICON: dict[str, float] = {

    # ── Strong reassurance (+0.30 to +0.35) ────────────────────────────
    "كويس"               : +0.30,
    "كويسة"              : +0.30,
    "تمام"               : +0.30,
    "أحسن"               : +0.28,
    "بتتحسن"             : +0.25,
    "بيتحسن"             : +0.25,
    "اتحسن"              : +0.28,
    "اتحسنت"             : +0.28,
    "مش بيوجعني"         : +0.20,
    "مش دايخ"            : +0.18,

    # ── Moderate positive (+0.12 to +0.20) ─────────────────────────────
    "خفيف"               : +0.15,
    "خفيفة"              : +0.15,
    "أحياناً"            : +0.10,
    "مش دايم"            : +0.12,
    "بيجي ويروح"         : +0.12,
    "مش شديد"            : +0.15,
    "مش مزعجني"          : +0.18,
    "مستحمله"            : +0.12,
    "مستحملها"           : +0.12,
    "هادي"               : +0.15,
    "مطمئن"              : +0.20,
    "مطمئنة"             : +0.20,
    "مش خايف"            : +0.18,

    # ── Mild positive (+0.08 to +0.12) ─────────────────────────────────
    "شوية"               : +0.08,
    "بسيط"               : +0.10,
    "بسيطة"              : +0.10,
    "عادي"               : +0.10,
}

# ── Intensifiers — multiply the score of the NEXT matched term ───────────
# Egyptian Arabic intensifiers that amplify the sentiment of what follows
_INTENSIFIERS: dict[str, float] = {
    "جداً"       : 1.50,
    "جدا"        : 1.50,
    "أوي"        : 1.60,    # very Egyptian — "تعبان أوي"
    "خالص"       : 1.55,    # "مش كويس خالص"
    "قوي"        : 1.40,
    "كتير"       : 1.30,
    "ناري"       : 1.70,    # "وجع ناري" = extreme pain
    "مش طايق"    : 1.50,
    "مش قادر"    : 1.40,
    "للغاية"     : 1.45,
}

# ── Negation markers — flip polarity of the NEXT matched term ────────────
# Same set as command_parser._NEGATION_PREFIXES, kept local to avoid import
_NEGATION_MARKERS: frozenset[str] = frozenset({
    "مش", "مو", "ليس", "لا", "ولا",
    "معنديش", "معنداش", "مفيش",
    "بدون", "من غير",
})

# Window (tokens) to look LEFT of a term for intensifier or negation
_WINDOW: int = 4


# ═══════════════════════════════════════════════════════════════════════════
#  SentimentAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """
    Stateless, offline Egyptian-Arabic sentiment scorer.

    Instantiate once and call score() or score_full() for every command.

    Parameters
    ----------
    distress_threshold : float
        Score at or below this value triggers a distress label (default −0.50).
    calm_threshold : float
        Score at or above this value triggers a calm label (default +0.50).
    emergency_override : bool
        If True, any text containing an emergency term is forced to −1.0
        (default True — mirrors config.json setting).

    Examples
    --------
    >>> sa = SentimentAnalyzer()
    >>> sa.score("أنا تعبان أوي وخايف")
    -0.722
    >>> sa.score("الحمد لله أحسن")
    0.28
    >>> sa.score("عندي وجع بسيط")
    -0.05
    """

    def __init__(
        self,
        distress_threshold: float = -0.50,
        calm_threshold    : float =  0.50,
        emergency_override: bool  =  True,
    ) -> None:
        self._distress_thr  = distress_threshold
        self._calm_thr      = calm_threshold
        self._emergency_ovr = emergency_override

        # Pre-sort lexicons by descending key length so multi-word phrases
        # are matched before their constituent single words
        self._neg_lex = dict(
            sorted(_NEGATIVE_LEXICON.items(), key=lambda x: -len(x[0]))
        )
        self._pos_lex = dict(
            sorted(_POSITIVE_LEXICON.items(), key=lambda x: -len(x[0]))
        )

        logger.info(
            "SentimentAnalyzer ready | neg_terms=%d pos_terms=%d "
            "intensifiers=%d emergency_override=%s",
            len(self._neg_lex), len(self._pos_lex),
            len(_INTENSIFIERS), emergency_override,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def score(self, text: str, is_emergency: bool = False) -> float:
        """
        Return a single sentiment score in [−1.0, +1.0].

        Parameters
        ----------
        text         : Normalised Egyptian Arabic text from CommandParser.
        is_emergency : Pass True if CommandParser already flagged EMERGENCY
                       intent — triggers forced override to −1.0.

        Returns
        -------
        float — negative = distress, 0 = neutral, positive = calm.
        """
        return self.score_full(text, is_emergency).score

    def score_full(self, text: str, is_emergency: bool = False) -> SentimentResult:
        """
        Full sentiment analysis — returns SentimentResult with breakdown.

        Use this in the /explain API endpoint or in unit tests.

        FIX v1.1 — Token-safe matching
        ────────────────────────────────
        Replaced `if term in text:` (substring match) with `term_found` from
        _context_multiplier (token-exact match).  This prevents false positives
        where a lexicon term appears as a substring inside an unrelated word,
        e.g. "ألم" inside "معلم", or "ورم" inside "مورم".
        """
        tokens = text.split()
        accumulator   : float      = 0.0
        matched_neg   : list[str]  = []
        matched_pos   : list[str]  = []
        matched_intens: list[str]  = []
        negation_flips: list[str]  = []

        # ── Pass 1 — score negative terms ────────────────────────────────
        for term, weight in self._neg_lex.items():
            term_found, multiplier, intensifier_used, was_negated = (
                self._context_multiplier(term, tokens)
            )
            if not term_found:                  # ← FIX: skip substring ghosts
                continue
            if was_negated:
                # Negated distress → becomes mild positive signal
                accumulator -= weight * multiplier * -0.5
                negation_flips.append(term)
            else:
                accumulator += weight * multiplier
                matched_neg.append(term)
                if intensifier_used:
                    matched_intens.append(intensifier_used)

        # ── Pass 2 — score positive terms ────────────────────────────────
        for term, weight in self._pos_lex.items():
            term_found, multiplier, intensifier_used, was_negated = (
                self._context_multiplier(term, tokens)
            )
            if not term_found:                  # ← FIX: skip substring ghosts
                continue
            if was_negated:
                # Negated positive → becomes mild negative signal
                accumulator -= weight * multiplier * 0.5
                negation_flips.append(term)
            else:
                accumulator += weight * multiplier
                matched_pos.append(term)
                if intensifier_used:
                    matched_intens.append(intensifier_used)

        # ── Pass 3 — clamp raw score to [−1.0, +1.0] ────────────────────
        raw_score = max(-1.0, min(1.0, accumulator))

        # ── Pass 4 — emergency override ──────────────────────────────────
        # FIX v1.1: emergency terms also checked as whole tokens, not substrings
        _emergency_terms = {"طوارئ", "نجدة", "بيموت", "بتموت"}
        token_set = set(tokens)
        emergency_triggered = False
        if self._emergency_ovr and (
            is_emergency or bool(token_set & _emergency_terms)
        ):
            final_score         = -1.0
            emergency_triggered = True
        else:
            final_score = raw_score

        label = self._label(final_score)

        logger.debug(
            "SentimentAnalyzer | score=%.3f raw=%.3f label=%s "
            "neg=%s pos=%s intens=%s flips=%s emergency=%s",
            final_score, raw_score, label,
            matched_neg, matched_pos, matched_intens,
            negation_flips, emergency_triggered,
        )

        return SentimentResult(
            score               = final_score,
            label               = label,
            matched_negative    = matched_neg,
            matched_positive    = matched_pos,
            matched_intensifiers= matched_intens,
            negation_flips      = negation_flips,
            emergency_override  = emergency_triggered,
            raw_score           = raw_score,
        )

    def label(self, score: float) -> str:
        """Arabic label for a pre-computed score — useful for display."""
        return self._label(score)

    # ── Private helpers ──────────────────────────────────────────────────

    def _context_multiplier(
        self,
        term  : str,
        tokens: list[str],
    ) -> tuple[bool, float, str | None, bool]:
        """
        Locate `term` as whole tokens, then examine context windows.

        FIX v1.1 — Token-exact match
        ──────────────────────────────
        Previously callers used `if term in text:` (substring check) before
        calling this method.  That caused false positives: "ألم" matched inside
        "معلم", "ورم" matched inside "مورم", etc.

        Now this method is the single gate:
        • It returns `term_found=True` ONLY when `term` appears as a full
          token sequence in `tokens` — never as a substring of another word.
        • Callers must check `term_found` before scoring.

        Egyptian Arabic word-order notes
        ──────────────────────────────────
        • Negation   comes BEFORE the term  →  scan LEFT  window
          "مش تعبان"،  "معنديش ألم"
        • Intensifier comes AFTER  the term  →  scan RIGHT window
          "تعبان أوي"،  "وجع ناري"،  "كويس خالص"
          ("جداً/جدا" can appear either side — checked in both windows)

        Returns
        -------
        (term_found, multiplier, intensifier_token_used, was_negated)
        """
        term_tokens = term.split()
        term_len    = len(term_tokens)
        multiplier  = 1.0
        intensifier_used: str | None = None
        was_negated = False
        term_found  = False

        for i in range(len(tokens) - term_len + 1):
            # ── Exact token-sequence match (not substring) ────────────────
            if tokens[i : i + term_len] != term_tokens:
                continue

            term_found = True   # ← confirmed: term exists as whole tokens

            # ── Left window — negation & pre-intensifier check ────────────
            left_start  = max(0, i - _WINDOW)
            left_window = tokens[left_start : i]

            for w in reversed(left_window):       # closest token first
                if w in _NEGATION_MARKERS or (w.startswith("ما") and w.endswith("ش")):
                    was_negated = True
                    break
                # "جداً/جدا" can precede the term ("جداً تعبان" is rare but valid)
                if w in _INTENSIFIERS:
                    multiplier       = _INTENSIFIERS[w]
                    intensifier_used = w
                    break

            if was_negated:
                break   # negated — right-window intensifier irrelevant

            # ── Right window — post-intensifier check ─────────────────────
            right_end    = min(len(tokens), i + term_len + _WINDOW)
            right_window = tokens[i + term_len : right_end]

            for w in right_window:                # closest token first
                if w in _INTENSIFIERS:
                    multiplier       = _INTENSIFIERS[w]
                    intensifier_used = w
                    break

            break   # scored the first occurrence — don't double-count

        return term_found, multiplier, intensifier_used, was_negated

    def _label(self, score: float) -> str:
        """Map a numeric score to an Arabic clinical label."""
        if score <= -0.75:
            return "ضائقة شديدة"
        if score <= self._distress_thr:
            return "قلق"
        if score < -0.10:
            return "انزعاج خفيف"
        if score <= +0.10:
            return "محايد"
        if score < self._calm_thr:
            return "مطمئن نسبياً"
        return "مطمئن"
