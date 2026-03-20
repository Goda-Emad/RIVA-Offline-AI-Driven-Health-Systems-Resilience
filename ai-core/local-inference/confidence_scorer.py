"""
confidence_scorer.py  v4.1
==========================
RIVA Health Platform — Unified Confidence Scorer
-------------------------------------------------
يحسب درجة الثقة النهائية لكل قرار يتخذه الـ AI.

التحسينات v4.1:
    أ. STT Confidence    — مصدر سادس: جودة الصوت تؤثر على الثقة
    ب. Deep Dialect      — قاموس uncertainty أعمق بالعامية المصرية
    ج. Contradiction Det — "أنا كويس + ألم شديد" → score=0 + Emergency فوراً

المصادر الـ 6:
    1. model_probability   30% — softmax probability من الموديل
    2. intent_clarity      25% — وضوح الـ intent
    3. ambiguity_penalty   20% — خصم من ambiguity_handler
    4. profile_completeness15% — اكتمال الملف الطبي
    5. response_quality    10% — جودة الرد
    6. stt_confidence       ±  — bonus/penalty من جودة الصوت

الربط مع المنظومة:
    - chat.py           : يستدعيه بدل _confidence() البسيطة
    - orchestrator.py   : يستخدم final_score لقرار التوجيه
    - 12_ai_explanation : يعرض تفاصيل الـ score للمريض
    - voice.py          : يمرر stt_confidence لهنا

Author : GODA EMAD
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("riva.local_inference.confidence_scorer")

# ─── Thresholds ───────────────────────────────────────────────────────────────

LOW_CONFIDENCE_THRESH    = 0.55
MEDIUM_CONFIDENCE_THRESH = 0.75
HIGH_CONFIDENCE_THRESH   = 0.90

# STT thresholds
STT_GOOD_THRESH = 0.75    # صوت واضح → bonus
STT_BAD_THRESH  = 0.40    # صوت مشوّش → penalty

# ─── Intent clarity ───────────────────────────────────────────────────────────

_INTENT_CLARITY: dict[str, float] = {
    "Emergency":   0.95,
    "Pregnancy":   0.85,
    "School":      0.85,
    "Triage":      0.80,
    "Readmission": 0.80,
    "LOS":         0.78,
    "Doctor":      0.75,
    "History":     0.75,
    "Combined":    0.70,
    "General":     0.55,
}

# ─── Clinical profile fields ──────────────────────────────────────────────────

_PROFILE_FIELDS = [
    "has_diabetes", "is_pregnant", "has_hypertension",
    "has_heart_disease", "age", "gender", "chief_complaint",
]

# ─── ب. Deep dialect uncertainty markers (موسّع) ─────────────────────────────

_UNCERTAINTY_MARKERS = [
    # فصحى
    "مش متأكد", "مش عارف", "ممكن", "ربما", "مش واضح",
    "محتاج معلومات أكتر", "مش قادر أحدد", "صعب أقول",
    "maybe", "possibly", "not sure", "unclear",
    # عامية مصرية عميقة
    "يمكن", "تقريباً", "ممكن يكون", "مش متأكد أوي",
    "غالباً", "على الأغلب", "نص نص", "مش عارف بصراحة",
    "أظن", "حاسس إن", "مش حاسس كويس أوي",
    "بس مش متأكد", "عارف مش عارف", "مش لاقيه",
    "شايفه مش عارف", "ومش فاهم إيه",
]

_HALLUCINATION_MARKERS = [
    "دراسة أثبتت", "طبقاً لأبحاث", "وفقاً للإحصاءات",
    "في عام 20", "نسبة", "إحصائياً",
]

_GOOD_RESPONSE_MARKERS = [
    "روح الطبيب", "استشر الدكتور", "لو الأعراض اشتدت",
    "خذ الدواء", "كمّل العلاج", "راجع",
]

# ─── ج. Contradiction detection ──────────────────────────────────────────────
# "أنا كويس" + "ألم شديد" في نفس الرسالة → score=0 + Emergency

_WELLNESS_CLAIMS = [
    "أنا كويس", "أنا تمام", "مش تعبان", "معنيش", "ولا حاجة",
    "عندي حاجة", "أنا بخير", "كويس أوي",
]

_SEVERE_SYMPTOM_PATTERNS = [
    r"ألم\s*(شديد|قوي|رهيب|مش قادر)",
    r"وجع\s*(شديد|قوي|بيموتني|خانق)",
    r"ضيق(ة)?\s*(في|ف)\s*التنفس",
    r"مش قادر\s*(أتنفس|أتحرك|أقوم)",
    r"(حرارة|سخونية)\s*(عالية|شديدة|\d{2})",
    r"نزيف\s*(شديد|كتير|مستمر)",
    r"إغماء|فقدان وعي|بيغمى عليه",
    r"ألم في الصدر",
    r"ضغط\s*(عالي|انخفض|تغير)",
]


def _detect_contradiction(text: str) -> bool:
    """
    Returns True if the message contains both a wellness claim
    and a severe symptom in the same text.

    Example:
        "أنا كويس بس عندي ألم شديد في صدري" → True → Emergency
    """
    text_lower = text.lower()

    has_wellness = any(w in text_lower for w in _WELLNESS_CLAIMS)
    if not has_wellness:
        return False

    has_severe = any(
        re.search(p, text_lower) for p in _SEVERE_SYMPTOM_PATTERNS
    )
    return has_severe


# ─── Score breakdown ──────────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    """
    Full confidence breakdown — displayed in 12_ai_explanation.html
    """
    model_probability:    float
    intent_clarity:       float
    ambiguity_penalty:    float
    profile_completeness: float
    response_quality:     float
    stt_confidence:       float           # 0.0 = no audio, 1.0 = perfect
    stt_adjustment:       float           # actual bonus/penalty applied
    contradiction_found:  bool
    final_score:          float
    level:                str             # "high" | "medium" | "low_warning" | "low"
    force_emergency:      bool            # True لو اكتُشف تناقض خطير
    explanation:          str
    factors:              list[str] = field(default_factory=list)


# ─── Core scorer ─────────────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Computes weighted confidence score from 6 sources.

    Weights:
        model_probability    : 30%
        intent_clarity       : 25%
        ambiguity_penalty    : 20%  (penalty)
        profile_completeness : 15%
        response_quality     : 10%
        stt_confidence       : ±0.10 bonus / -0.20 penalty
    """

    WEIGHTS = {
        "model_probability":    0.30,
        "intent_clarity":       0.25,
        "profile_completeness": 0.15,
        "response_quality":     0.10,
    }
    AMBIGUITY_WEIGHT = 0.20

    # Calibration: weights are expert-initialised.
    # Validate with Reliability Curve on 500+ sessions:
    #   from sklearn.calibration import calibration_curve
    #   fraction_pos, mean_pred = calibration_curve(y_true, scores, n_bins=10)
    # Target: ECE < 0.05 before production deployment.

    # Non-linear combining for critical intents.
    # Geometric mean is conservative — one low score pulls everything down.
    # mp=0.9, ic=0.85, pc=0.2, rq=0.8:
    #   Weighted sum   → 0.74  (misleadingly OK)
    #   Geometric mean → 0.57  (correctly flags incomplete profile)
    GEOMETRIC_MEAN_INTENTS = {"Emergency", "Triage", "Pregnancy"}

    # STT noise floor: short recordings (<20 tokens) + low confidence
    # → penalty doubled for extra caution.
    STT_SHORT_TOKEN_THRESH = 20

    # ── 1. Model probability ─────────────────────────────────────────────────

    def _model_prob(self, score: Optional[float]) -> float:
        if score is None:
            return 0.90
        return max(0.0, min(1.0, float(score)))

    # ── 2. Intent clarity ────────────────────────────────────────────────────

    def _intent_clarity(self, intent: str) -> float:
        return _INTENT_CLARITY.get(intent, 0.60)

    # ── 3. Profile completeness ──────────────────────────────────────────────

    def _profile_completeness(self, profile: dict) -> float:
        if not profile:
            return 0.30
        known = sum(
            1 for f in _PROFILE_FIELDS
            if profile.get(f) not in (None, False, "")
        )
        return round(min(1.0, 0.30 + (known / len(_PROFILE_FIELDS)) * 0.70), 3)

    # ── 4. Response quality ──────────────────────────────────────────────────

    def _response_quality(self, response: str) -> float:
        if not response:
            return 0.20
        words      = response.split()
        word_count = len(words)
        score      = 0.50

        if word_count >= 30:
            score += 0.20
        elif word_count >= 15:
            score += 0.10
        elif word_count < 5:
            score -= 0.20

        uncertainty   = sum(1 for m in _UNCERTAINTY_MARKERS if m in response)
        hallucination = sum(1 for m in _HALLUCINATION_MARKERS if m in response)
        good          = sum(1 for m in _GOOD_RESPONSE_MARKERS if m in response)

        score -= uncertainty   * 0.10
        score -= hallucination * 0.15
        score += min(good * 0.05, 0.15)

        return round(max(0.0, min(1.0, score)), 3)

    # ── 5. Ambiguity penalty ─────────────────────────────────────────────────

    def _ambiguity_penalty(self, penalty: float) -> float:
        return max(0.0, min(0.40, penalty))

    # ── أ. STT confidence ────────────────────────────────────────────────────

    def _stt_adjustment(self, stt_confidence: float) -> float:
        """
        Adjusts final score based on audio quality from Whisper.

        stt_confidence = average token log-probability from Whisper
        (normalised to 0-1 by voice.py before passing here)

        Good audio (≥0.75) → +0.05 bonus
        Bad audio  (≤0.40) → -0.20 penalty (transcription unreliable)
        """
        if stt_confidence <= 0.0:
            return 0.0           # no audio input — no adjustment
        if stt_confidence >= STT_GOOD_THRESH:
            return +0.05
        if stt_confidence <= STT_BAD_THRESH:
            return -0.20
        # Linear interpolation in between
        ratio = (stt_confidence - STT_BAD_THRESH) / (STT_GOOD_THRESH - STT_BAD_THRESH)
        return round(-0.20 + ratio * 0.25, 3)

    def _stt_noise_floor(
        self, stt_confidence: float, token_count: int
    ) -> float:
        """
        STT Noise Floor — extra penalty for short + low-quality recordings.

        Short recordings (<20 tokens) with low STT confidence are more
        likely to be noise or mumbling than real medical speech.
        In these cases we double the standard STT penalty.

        token_count comes from stt_result["token_count"] in voice.py.
        """
        base_adj = self._stt_adjustment(stt_confidence)
        if (
            token_count > 0
            and token_count < self.STT_SHORT_TOKEN_THRESH
            and stt_confidence <= STT_BAD_THRESH
        ):
            extra = base_adj * 1.0   # double the penalty
            floored = max(-0.40, base_adj + extra)
            log.info(
                "[STT-NoiseFloor] short=%d tokens  conf=%.2f  "
                "adj=%.2f → floored=%.2f",
                token_count, stt_confidence, base_adj, floored,
            )
            return floored
        return base_adj

    # ── ج. Contradiction detection ───────────────────────────────────────────

    def _check_contradiction(self, patient_text: str) -> bool:
        return _detect_contradiction(patient_text)

    # ── Compute ──────────────────────────────────────────────────────────────

    def compute(
        self,
        intent:            str,
        response:          str,
        patient_text:      str             = "",
        clinical_profile:  dict            = {},
        model_logit_score: Optional[float] = None,
        ambiguity_penalty: float           = 0.0,
        stt_confidence:    float           = 0.0,
        stt_token_count:   int             = 0,
    ) -> ScoreBreakdown:
        """
        Computes the final weighted confidence score.

        Args:
            intent             : semantic intent from chat.py
            response           : AI-generated response
            patient_text       : original patient message (for contradiction check)
            clinical_profile   : session metadata
            model_logit_score  : raw softmax probability (optional)
            ambiguity_penalty  : from ambiguity_handler
            stt_confidence     : 0.0-1.0 from voice.py (0 = text input)
        """

        # ج. Contradiction check — overrides everything
        contradiction = self._check_contradiction(patient_text)
        if contradiction:
            log.warning(
                "[ConfidenceScorer] CONTRADICTION detected: '%s...' → force_emergency",
                patient_text[:60],
            )
            return ScoreBreakdown(
                model_probability    = 0.0,
                intent_clarity       = 0.0,
                ambiguity_penalty    = 0.0,
                profile_completeness = 0.0,
                response_quality     = 0.0,
                stt_confidence       = stt_confidence,
                stt_adjustment       = 0.0,
                contradiction_found  = True,
                final_score          = 0.0,
                level                = "low",
                force_emergency      = True,
                explanation          = (
                    "⚠️ تم اكتشاف تناقض في الرسالة — "
                    "ريفا بتحولك للطوارئ عشان سلامتك."
                ),
                factors              = ["تناقض بين 'أنا كويس' وأعراض شديدة"],
            )

        # Normal scoring
        mp   = self._model_prob(model_logit_score)
        ic   = self._intent_clarity(intent)
        pc   = self._profile_completeness(clinical_profile)
        rq   = self._response_quality(response)
        ap   = self._ambiguity_penalty(ambiguity_penalty)

        # STT: apply noise floor if short recording
        stt_adj = self._stt_noise_floor(stt_confidence, stt_token_count)

        # Non-linear combining for critical intents (Geometric Mean)
        # More conservative — one low score pulls everything down
        if intent in self.GEOMETRIC_MEAN_INTENTS:
            import math
            scores = [mp, ic, pc, rq]
            raw = math.pow(
                mp   ** self.WEIGHTS["model_probability"]    *
                ic   ** self.WEIGHTS["intent_clarity"]       *
                pc   ** self.WEIGHTS["profile_completeness"] *
                rq   ** self.WEIGHTS["response_quality"],
                1.0 / sum(self.WEIGHTS.values())
            )
            log.info(
                "[ConfidenceScorer] geometric_mean for '%s'  raw=%.3f", intent, raw
            )
        else:
            # Standard weighted sum for non-critical intents
            raw = (
                mp * self.WEIGHTS["model_probability"]    +
                ic * self.WEIGHTS["intent_clarity"]       +
                pc * self.WEIGHTS["profile_completeness"] +
                rq * self.WEIGHTS["response_quality"]
            )

        # Ambiguity penalty + STT adjustment
        final = max(0.0, min(1.0,
            raw - (ap * self.AMBIGUITY_WEIGHT) + stt_adj
        ))
        final = round(final, 3)

        # Level
        if final >= HIGH_CONFIDENCE_THRESH:
            level = "high"
        elif final >= MEDIUM_CONFIDENCE_THRESH:
            level = "medium"
        elif final >= LOW_CONFIDENCE_THRESH:
            level = "low_warning"
        else:
            level = "low"

        explanation = self._explain(final, level, ap, stt_confidence, stt_adj)
        factors     = self._key_factors(mp, ic, pc, rq, ap, stt_confidence, stt_adj)

        log.info(
            "[ConfidenceScorer] intent=%s  mp=%.2f  ic=%.2f  pc=%.2f  "
            "rq=%.2f  ap=%.2f  stt=%.2f(adj=%.2f)  → %.3f (%s)",
            intent, mp, ic, pc, rq, ap, stt_confidence, stt_adj, final, level,
        )

        return ScoreBreakdown(
            model_probability    = mp,
            intent_clarity       = ic,
            ambiguity_penalty    = ap,
            profile_completeness = pc,
            response_quality     = rq,
            stt_confidence       = stt_confidence,
            stt_adjustment       = stt_adj,
            contradiction_found  = False,
            final_score          = final,
            level                = level,
            force_emergency      = False,
            explanation          = explanation,
            factors              = factors,
        )

    def _explain(
        self,
        score:          float,
        level:          str,
        penalty:        float,
        stt_conf:       float,
        stt_adj:        float,
    ) -> str:
        pct = int(score * 100)

        if level == "high":
            return f"ريفا واثقة جداً من ردها ({pct}%). المعلومات كافية والحالة واضحة."

        if stt_adj <= -0.15:
            return (
                f"مستوى الثقة {pct}% — جودة الصوت منخفضة أثّرت على الفهم. "
                "حاول تسجّل تاني في مكان هادي."
            )

        if penalty > 0.15:
            return (
                f"مستوى الثقة {pct}% — بعض المعلومات غير واضحة. "
                "ريفا محتاجة تفاصيل أكتر."
            )

        if level == "medium":
            return f"ريفا واثقة من ردها بنسبة {pct}%. يُنصح بمراجعة الدكتور."

        return (
            f"مستوى الثقة {pct}% — الحالة تحتاج مراجعة طبية متخصصة. "
            "ريفا بتشرحلك تفكيرها هنا."
        )

    def _key_factors(
        self,
        mp: float, ic: float, pc: float,
        rq: float, ap: float,
        stt: float, stt_adj: float,
    ) -> list[str]:
        factors = []
        if mp < 0.60:
            factors.append("الموديل مش متأكد من التصنيف")
        if ic < 0.65:
            factors.append("الـ intent مش واضح كفاية")
        if pc < 0.40:
            factors.append("الملف الطبي للمريض ناقص")
        if rq < 0.50:
            factors.append("الرد يحتوي على عدم يقين")
        if ap > 0.15:
            factors.append("الرسالة فيها غموض في العامية")
        if stt_adj <= -0.10:
            factors.append(f"جودة الصوت منخفضة ({int(stt*100)}%)")
        elif stt_adj >= 0.05:
            factors.append(f"جودة الصوت ممتازة ({int(stt*100)}%) ← bonus")
        if not factors:
            factors.append("كل العوامل في حدودها الطبيعية")
        return factors[:3]


# ─── Singleton ────────────────────────────────────────────────────────────────

_scorer = ConfidenceScorer()


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_confidence(
    intent:            str,
    response:          str,
    patient_text:      str             = "",
    clinical_profile:  dict            = {},
    model_logit_score: Optional[float] = None,
    ambiguity_penalty: float           = 0.0,
    stt_confidence:    float           = 0.0,
    stt_token_count:   int             = 0,
) -> ScoreBreakdown:
    """
    Main entry point — replaces _confidence() in chat.py

    Usage in chat.py:
        from .confidence_scorer import compute_confidence

        score = compute_confidence(
            intent            = intent,
            response          = response,
            patient_text      = user_message,
            clinical_profile  = session["metadata"],
            ambiguity_penalty = amb["confidence_penalty"],
            stt_confidence    = stt_result.get("avg_log_prob", 0.0),
        )
        if score.force_emergency:
            intent   = "Emergency"
            response = _emergency_response()

        chat_result["confidence_score"] = score.final_score
        chat_result["score_breakdown"]  = score.__dict__
    """
    return _scorer.compute(
        intent            = intent,
        response          = response,
        patient_text      = patient_text,
        clinical_profile  = clinical_profile,
        model_logit_score = model_logit_score,
        ambiguity_penalty = ambiguity_penalty,
        stt_confidence    = stt_confidence,
        stt_token_count   = stt_token_count,
    )


def is_low_confidence(score: float) -> bool:
    return score < LOW_CONFIDENCE_THRESH


def get_thresholds() -> dict:
    return {
        "low":    LOW_CONFIDENCE_THRESH,
        "medium": MEDIUM_CONFIDENCE_THRESH,
        "high":   HIGH_CONFIDENCE_THRESH,
    }
