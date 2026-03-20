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
║  Design — Hybrid ONNX + Lexicon (v2.0)                                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Primary  : ONNX INT8 model (model_int8.onnx) — understands context,        ║
║             negation, and complex Egyptian Arabic sentences.                 ║
║  Fallback : Rule-based weighted lexicon — zero-latency, works offline,      ║
║             always available even without the trained model.                ║
║                                                                              ║
║  Scoring pipeline                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║   text → Emergency trigger check (lexicon, instant)                         ║
║       ↓ no emergency                                                         ║
║   ONNX model available? → Yes → ONNX score (context-aware)                  ║
║                         → No  → Lexicon score (rule-based fallback)         ║
║       ↓                                                                      ║
║   Blend: 0.70 × ONNX + 0.30 × Lexicon  (when both available)               ║
║       ↓                                                                      ║
║   SentimentResult (score, label, breakdown, engine_used)                    ║
║                                                                              ║
║  Author  : Goda Emad  (AI Core)                                              ║
║  Version : 2.0.0                                                             ║
║  Updated : 2026-03-18                                                        ║
║                                                                              ║
║  Changelog v2.0.0                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • ADD  ONNX inference engine — loads model_int8.onnx via onnxruntime.      ║
║  • ADD  Hybrid blending: 70% ONNX + 30% Lexicon when model is loaded.      ║
║  • ADD  SentimentResult.engine_used field — "onnx" | "lexicon" | "hybrid". ║
║  • ADD  load_model() / unload_model() for explicit lifecycle control.       ║
║  • CHG  score_full() now returns engine_used in the result dict.            ║
║  • KEEP All v1.1 lexicon fixes (token-exact match, negation, intensifiers). ║
║                                                                              ║
║  Changelog v2.1.0                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • FIX 1 Negation+Intensifier: "مش تعبان أوي" now scores both flags.       ║
║  • FIX 2 Punctuation: strip before tokenise — "تعبان،" matches "تعبان".    ║
║  • FIX 3 Multi-occurrence: all term repeats scored, not just first.         ║
║  • FIX 4 ONNX threads: configurable via onnx_threads param + env var.      ║
║  • FIX 5 Negation weight: heavy terms (weight<-0.30) negated → near 0.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional ONNX Runtime — graceful degradation if not installed ─────────
try:
    import numpy as np
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    logger.warning(
        "onnxruntime not installed — SentimentAnalyzer will use lexicon-only mode.\n"
        "  pip install onnxruntime"
    )


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
    score               : float       # final clamped score  −1.0 → +1.0
    label               : str         # "ضائقة شديدة" / "قلق" / "محايد" / "مطمئن"
    matched_negative    : list[str]   # distress terms found (lexicon)
    matched_positive    : list[str]   # calm/reassurance terms found (lexicon)
    matched_intensifiers: list[str]   # intensifier tokens found
    negation_flips      : list[str]   # terms whose polarity was flipped
    emergency_override  : bool        # True if score was forced to −1.0
    raw_score           : float       # score before emergency override & blending
    engine_used         : str         # "onnx" | "lexicon" | "hybrid" | "emergency"
    onnx_score          : float       # ONNX raw score (0.0 if not used)
    lexicon_score       : float       # Lexicon raw score (always computed)

    def as_dict(self) -> dict:
        return {
            "score"              : round(self.score, 3),
            "label"              : self.label,
            "engine_used"        : self.engine_used,
            "onnx_score"         : round(self.onnx_score, 3),
            "lexicon_score"      : round(self.lexicon_score, 3),
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

# Default ONNX model path — mirrors ai-core/models/chatbot/model_int8.onnx
_DEFAULT_ONNX_PATH = Path(__file__).resolve().parent.parent.parent /     "models" / "chatbot" / "model_int8.onnx"

# Blend weights: how much to trust ONNX vs Lexicon when both are available
_ONNX_BLEND_WEIGHT    = 0.70
_LEXICON_BLEND_WEIGHT = 0.30


class SentimentAnalyzer:
    """
    Hybrid Egyptian-Arabic sentiment scorer — ONNX model + Lexicon fallback.

    Scoring strategy
    ────────────────
    1. Emergency triggers checked instantly via lexicon (no model needed).
    2. If ONNX model loaded:
         hybrid_score = 0.70 × onnx_score + 0.30 × lexicon_score
    3. If no ONNX model:
         score = lexicon_score  (pure rule-based, always offline)

    Parameters
    ----------
    distress_threshold : float
        Score at or below this triggers a distress label (default −0.50).
    calm_threshold : float
        Score at or above this triggers a calm label (default +0.50).
    emergency_override : bool
        If True, emergency terms force score to −1.0 (default True).
    onnx_model_path : Path | None
        Path to model_int8.onnx. If None, uses default path.
        Pass False to disable ONNX entirely (lexicon-only mode).
    onnx_blend : float
        Weight given to ONNX score in hybrid mode (default 0.70).

    Examples
    --------
    >>> sa = SentimentAnalyzer()                    # auto-loads ONNX if available
    >>> sa = SentimentAnalyzer(onnx_model_path=False)  # lexicon-only mode
    >>> sa.score("أنا تعبان أوي وخايف")
    -0.65
    >>> sa.score("الحمد لله أحسن")
    0.72
    """

    def __init__(
        self,
        distress_threshold: float        = -0.50,
        calm_threshold    : float        =  0.50,
        emergency_override: bool         =  True,
        onnx_model_path   : Path | None | bool = None,
        onnx_blend        : float        =  _ONNX_BLEND_WEIGHT,
        onnx_threads      : int | None   =  None,
    ) -> None:
        # FIX v2.1: ONNX thread count — env var > constructor arg > default 2
        # Allows tuning per deployment (Docker/K8s CPU limits, Vercel, etc.)
        import os
        self._onnx_threads = (
            onnx_threads
            or int(os.environ.get("RIVA_ONNX_THREADS", 2))
        )
        self._distress_thr  = distress_threshold
        self._calm_thr      = calm_threshold
        self._emergency_ovr = emergency_override
        self._onnx_blend    = onnx_blend
        self._ort_session   = None         # loaded lazily or at init
        self._tokenizer     = None

        # Pre-sort lexicons — multi-word phrases matched before single words
        self._neg_lex = dict(
            sorted(_NEGATIVE_LEXICON.items(), key=lambda x: -len(x[0]))
        )
        self._pos_lex = dict(
            sorted(_POSITIVE_LEXICON.items(), key=lambda x: -len(x[0]))
        )

        # Load ONNX model
        if onnx_model_path is False:
            # Explicitly disabled — lexicon-only mode
            logger.info("SentimentAnalyzer: ONNX disabled — lexicon-only mode")
        else:
            model_path = Path(onnx_model_path) if onnx_model_path else _DEFAULT_ONNX_PATH
            self._try_load_onnx(model_path)

        logger.info(
            "SentimentAnalyzer ready | engine=%s neg_terms=%d pos_terms=%d",
            "hybrid" if self._ort_session else "lexicon",
            len(self._neg_lex), len(self._pos_lex),
        )

    # ── ONNX lifecycle ───────────────────────────────────────────────────

    def load_model(self, model_path: Path | None = None) -> bool:
        """
        Explicitly load (or reload) the ONNX model.
        Returns True if loaded successfully.
        """
        path = Path(model_path) if model_path else _DEFAULT_ONNX_PATH
        return self._try_load_onnx(path)

    def unload_model(self) -> None:
        """Release the ONNX session from memory (frees RAM on low-memory devices)."""
        self._ort_session = None
        self._tokenizer   = None
        logger.info("SentimentAnalyzer: ONNX model unloaded")

    @property
    def has_onnx_model(self) -> bool:
        """True if ONNX model is currently loaded."""
        return self._ort_session is not None

    # ── Public API ───────────────────────────────────────────────────────

    def score(self, text: str, is_emergency: bool = False) -> float:
        """Return a single sentiment score in [−1.0, +1.0]."""
        return self.score_full(text, is_emergency).score

    def score_full(self, text: str, is_emergency: bool = False) -> SentimentResult:
        """
        Hybrid sentiment analysis — ONNX + Lexicon with full breakdown.

        Pipeline
        ────────
        1. Emergency check  — instant lexicon trigger, stops pipeline if hit
        2. Lexicon score    — always computed (needed for hybrid blend + fallback)
        3. ONNX score       — computed if model loaded (context-aware)
        4. Blend            — 0.70 × ONNX + 0.30 × Lexicon (or lexicon-only)
        5. Clamp & label
        """
        # FIX v2.1: strip punctuation before tokenising
        # "تعبان، ومش" → "تعبان ومش" so exact-token match works on all inputs
        # FIX 2: remove ALL punctuation including Arabic ، ؟ ؛ and ASCII ,!?.;
        clean_text = re.sub(r"[\u060C\u061B\u061F\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]", " ", text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        tokens = clean_text.split()

        # ── Step 1: Emergency override (lexicon, instant) ─────────────────
        _emergency_terms = {"طوارئ", "نجدة", "بيموت", "بتموت"}
        token_set = set(tokens)

        # FIX 5: check if emergency term is negated before overriding
        # "مش بيموت" / "الحمدلله مش بيموت" → NOT an emergency
        def _emergency_is_negated(term: str) -> bool:
            if term not in token_set:
                return False
            idx = next((i for i, t in enumerate(tokens) if t == term), -1)
            if idx < 0:
                return False
            window = tokens[max(0, idx - _WINDOW) : idx]
            return any(
                w in _NEGATION_MARKERS or (w.startswith("ما") and w.endswith("ش"))
                for w in window
            )

        unnegated_emergency = any(
            t for t in (_emergency_terms & token_set)
            if not _emergency_is_negated(t)
        )

        if self._emergency_ovr and (is_emergency or unnegated_emergency):
            return SentimentResult(
                score               = -1.0,
                label               = "ضائقة شديدة",
                matched_negative    = list(token_set & _emergency_terms),
                matched_positive    = [],
                matched_intensifiers= [],
                negation_flips      = [],
                emergency_override  = True,
                raw_score           = -1.0,
                engine_used         = "emergency",
                onnx_score          = 0.0,
                lexicon_score       = -1.0,
            )

        # ── Step 2: Lexicon score (always computed) ────────────────────────
        lex_result  = self._lexicon_score(clean_text, tokens)   # FIX 2: use clean_text
        lexicon_raw = lex_result["raw_score"]

        # ── Step 3: ONNX score (if model loaded) ──────────────────────────
        onnx_raw    = 0.0
        engine_used = "lexicon"

        if self._ort_session is not None:
            try:
                onnx_raw    = self._onnx_score(text)
                engine_used = "hybrid"
            except Exception as exc:
                logger.warning("ONNX inference failed — falling back to lexicon: %s", exc)

        # ── Step 4: Blend ─────────────────────────────────────────────────
        if engine_used == "hybrid":
            raw_score = (
                self._onnx_blend       * onnx_raw
                + (1 - self._onnx_blend) * lexicon_raw
            )
        else:
            raw_score = lexicon_raw

        final_score = max(-1.0, min(1.0, raw_score))
        label       = self._label(final_score)

        logger.debug(
            "SentimentAnalyzer | engine=%s score=%.3f "
            "onnx=%.3f lex=%.3f label=%s",
            engine_used, final_score, onnx_raw, lexicon_raw, label,
        )

        return SentimentResult(
            score               = final_score,
            label               = label,
            matched_negative    = lex_result["matched_neg"],
            matched_positive    = lex_result["matched_pos"],
            matched_intensifiers= lex_result["matched_intens"],
            negation_flips      = lex_result["negation_flips"],
            emergency_override  = False,
            raw_score           = raw_score,
            engine_used         = engine_used,
            onnx_score          = onnx_raw,
            lexicon_score       = lexicon_raw,
        )

    def label(self, score: float) -> str:
        """Arabic label for a pre-computed score — useful for display."""
        return self._label(score)

    # ── Private helpers ──────────────────────────────────────────────────

    def _try_load_onnx(self, model_path: Path) -> bool:
        """
        Load ONNX INT8 model + tokenizer from model_path.
        Silent fail — returns False if unavailable so lexicon takes over.
        """
        if not _ORT_AVAILABLE:
            return False

        if not model_path.exists():
            logger.info(
                "ONNX model not found at %s — using lexicon-only mode.\n"
                "  Run train_sentiment.py to generate the model.",
                model_path,
            )
            return False

        try:
            # Load ONNX session (INT8 — CPU optimised)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self._onnx_threads   # FIX v2.1: configurable
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._ort_session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            # Load tokenizer from same directory
            try:
                from transformers import AutoTokenizer
                tokenizer_dir = model_path.parent
                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
                logger.info(
                    "ONNX model loaded | path=%s  tokenizer=%s",
                    model_path.name, tokenizer_dir.name,
                )
            except Exception as tok_err:
                logger.warning(
                    "ONNX model loaded but tokenizer failed: %s\n"
                    "  Falling back to lexicon-only mode.", tok_err
                )
                self._ort_session = None
                return False

            return True

        except Exception as exc:
            logger.warning("Failed to load ONNX model: %s — lexicon-only mode", exc)
            self._ort_session = None
            return False

    def _onnx_score(self, text: str) -> float:
        """
        Run ONNX inference and return a score in [−1.0, +1.0].

        The model outputs logits for [negative, positive].
        We apply softmax and map: positive_prob → [−1, +1]
          score = 2 × positive_prob − 1
          e.g. 90% positive → 0.80, 90% negative → −0.80
        """
        enc = self._tokenizer(
            text,
            truncation     = True,
            padding        = True,
            max_length     = 128,
            return_tensors = "np",
        )

        inputs = {
            "input_ids"     : enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        # Add token_type_ids only if the model expects it
        if "token_type_ids" in [i.name for i in self._ort_session.get_inputs()]:
            inputs["token_type_ids"] = enc.get(
                "token_type_ids", np.zeros_like(enc["input_ids"])
            ).astype(np.int64)

        logits = self._ort_session.run(None, inputs)[0][0]  # shape: (2,)

        # Softmax
        exp    = np.exp(logits - logits.max())
        probs  = exp / exp.sum()
        pos_prob = float(probs[1])

        # Map [0, 1] → [−1, +1]
        return round(2.0 * pos_prob - 1.0, 4)

    def _lexicon_score(self, text: str, tokens: list[str]) -> dict:
        """
        Pure lexicon scoring — uses _find_matches() for all occurrences.

        v2.1 changes
        ─────────────
        FIX 3 — All occurrences: "وجع في صدري وكمان وجع في ضهري" → 2× وجع.
        FIX 5 — Negation weight calibration:
          Heavy negative terms (weight < _HEAVY_NEG_THRESHOLD, e.g. "بيموت"):
            negated → contribution capped near 0 (not inflated positive)
            "الحمدلله المريض مش بيموت" = cautious relief, not joy
          Mild/moderate negative terms (weight ≥ _HEAVY_NEG_THRESHOLD):
            negated → mild positive (original behaviour)

        Returns
        -------
        dict with keys: raw_score, matched_neg, matched_pos,
                        matched_intens, negation_flips
        """
        accumulator    : float     = 0.0
        matched_neg    : list[str] = []
        matched_pos    : list[str] = []
        matched_intens : list[str] = []
        negation_flips : list[str] = []

        for term, weight in self._neg_lex.items():
            matches = self._find_matches(term, tokens)
            if not matches:
                continue
            for m in matches:
                if m["was_negated"]:
                    # FIX 5: heavy negative terms (emergencies) negated → near 0
                    # mild/moderate negatives → mild positive
                    if weight < self._HEAVY_NEG_THRESHOLD:
                        negation_contribution = weight * m["multiplier"] * -0.05
                    else:
                        negation_contribution = weight * m["multiplier"] * -0.5
                    accumulator -= negation_contribution
                    if term not in negation_flips:
                        negation_flips.append(term)
                else:
                    accumulator += weight * m["multiplier"]
                    if term not in matched_neg:
                        matched_neg.append(term)
                    if m["intensifier_used"] and m["intensifier_used"] not in matched_intens:
                        matched_intens.append(m["intensifier_used"])

        for term, weight in self._pos_lex.items():
            matches = self._find_matches(term, tokens)
            if not matches:
                continue
            for m in matches:
                if m["was_negated"]:
                    accumulator -= weight * m["multiplier"] * 0.5
                    if term not in negation_flips:
                        negation_flips.append(term)
                else:
                    accumulator += weight * m["multiplier"]
                    if term not in matched_pos:
                        matched_pos.append(term)
                    if m["intensifier_used"] and m["intensifier_used"] not in matched_intens:
                        matched_intens.append(m["intensifier_used"])

        return {
            "raw_score"     : max(-1.0, min(1.0, accumulator)),
            "matched_neg"   : matched_neg,
            "matched_pos"   : matched_pos,
            "matched_intens": matched_intens,
            "negation_flips": negation_flips,
        }

    # weight threshold below which negation means "neutral", not "positive"
    _HEAVY_NEG_THRESHOLD = -0.30

    def _find_matches(
        self,
        term  : str,
        tokens: list[str],
    ) -> list[dict]:
        """
        Find ALL occurrences of `term` as whole tokens and analyse each one.

        v2.1 changes
        ─────────────
        FIX 1 — Negation + Intensifier co-occur:
          "مش تعبان أوي" → negated=True AND multiplier=1.6
          Previously `break` after finding negation skipped right-window scan.
          Now BOTH windows are always scanned; negation and intensifier are
          independent flags, not mutually exclusive.

        FIX 3 — Multiple occurrences:
          "وجع في صدري وكمان وجع في ضهري" → two matches, each scored.
          Previously only the first occurrence was returned.

        Returns
        -------
        List of match dicts: [
          {
            "multiplier"      : float,
            "intensifier_used": str | None,
            "was_negated"     : bool,
          }, ...
        ]
        Empty list means term not found as whole tokens (no substring ghosts).
        """
        term_tokens = term.split()
        term_len    = len(term_tokens)
        matches     = []

        for i in range(len(tokens) - term_len + 1):
            if tokens[i : i + term_len] != term_tokens:
                continue

            multiplier       = 1.0
            intensifier_used: str | None = None
            was_negated      = False

            # ── Left window: negation check (closest token first) ─────────
            left_window = tokens[max(0, i - _WINDOW) : i]
            for w in reversed(left_window):
                if w in _NEGATION_MARKERS or (w.startswith("ما") and w.endswith("ش")):
                    was_negated = True
                    break
                # جداً/جدا can precede the term
                if w in _INTENSIFIERS and intensifier_used is None:
                    multiplier       = _INTENSIFIERS[w]
                    intensifier_used = w
                    break

            # ── Right window: intensifier check (always — FIX 1) ─────────
            # Scan right window regardless of negation so "مش تعبان أوي"
            # correctly picks up "أوي" even when was_negated=True
            right_window = tokens[i + term_len : min(len(tokens), i + term_len + _WINDOW)]
            for w in right_window:
                if w in _INTENSIFIERS:
                    # Right-window intensifier overrides left-window one
                    # (closer to the term = higher relevance)
                    multiplier       = _INTENSIFIERS[w]
                    intensifier_used = w
                    break

            matches.append({
                "multiplier"      : multiplier,
                "intensifier_used": intensifier_used,
                "was_negated"     : was_negated,
            })

        return matches

    # Keep old signature as a compatibility shim (used by tests that call it directly)
    def _context_multiplier(
        self,
        term  : str,
        tokens: list[str],
    ) -> tuple[bool, float, str | None, bool]:
        """
        Compatibility wrapper around _find_matches().
        Returns data for the FIRST occurrence only (legacy callers).
        New code should call _find_matches() directly.
        """
        matches = self._find_matches(term, tokens)
        if not matches:
            return False, 1.0, None, False
        m = matches[0]
        return True, m["multiplier"], m["intensifier_used"], m["was_negated"]

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
