"""
doctor_feedback_handler.py
==========================
RIVA Health Platform — Doctor Feedback Handler
-----------------------------------------------
يجمع ويعالج feedback الدكاترة على ردود الـ AI
عشان يحسّن الموديلات بشكل مستمر (Continuous Learning Loop).

التحسينات المطبّقة:
    1. Vercel-safe storage  — يدعم Supabase/Vercel KV كبديل للـ JSONL
    2. Weighted sentiment   — وزن الـ feedback بناءً على تخصص الدكتور
    3. Counter-based signal — عداد في الذاكرة بدل تحميل كل الملف O(N)
    4. Input sanitization   — تنظيف النصوص من الحقن الخبيث
    5. PII protection       — patient_id كـ UUID فقط، بدون بيانات شخصية
    6. Critical alert       — إشعار فوري للفريق التقني عند signal حرج

Author : GODA EMAD
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.doctor_validation.feedback")

# ─── Storage backend detection ───────────────────────────────────────────────
#
# Vercel is stateless — local JSONL files are wiped on every deployment.
# We detect the environment and route to the right backend:
#   - SUPABASE_URL set  → Supabase (recommended for production)
#   - VERCEL_KV_URL set → Vercel KV
#   - Neither           → local JSONL (dev / offline mode)

_STORAGE_BACKEND = (
    "supabase"   if os.getenv("SUPABASE_URL")    else
    "vercel_kv"  if os.getenv("VERCEL_KV_URL")   else
    "local"
)

_BASE_DIR        = Path(__file__).parent / "logs"
_FEEDBACK_FILE   = _BASE_DIR / "doctor_feedback.jsonl"
_RETRAINING_FILE = _BASE_DIR / "retraining_signals.jsonl"

if _STORAGE_BACKEND == "local":
    _BASE_DIR.mkdir(parents=True, exist_ok=True)

log.info("[Feedback] Storage backend: %s", _STORAGE_BACKEND)

# ─── Thresholds & weights ────────────────────────────────────────────────────

NEGATIVE_THRESHOLD   = 5      # عدد الـ feedback السلبية قبل الـ signal
LOW_RATING_CUTOFF    = 2      # 1-2 نجوم = سلبي
CRITICAL_FLAG_ALERT  = True   # إشعار فوري عند flag خطير

# Weighted scoring per doctor specialty
# الاستشاري والأخصائي ليهم وزن أكبر من الممارس العام
SPECIALTY_WEIGHTS: dict[str, float] = {
    "consultant":        3.0,
    "specialist":        2.0,
    "general_practice":  1.0,
    "resident":          0.8,
    "intern":            0.5,
    "unknown":           1.0,
}

# Feedback type weights
TYPE_WEIGHTS: dict[str, float] = {
    "flagging":    3.0,   # أخطر أنواع الـ feedback
    "correction":  2.0,   # تصحيح صريح
    "rating":      1.0,   # تقييم عام
    "validation":  0.0,   # تأكيد = إيجابي → لا يُحسب في السلبيات
}

# In-memory counter fallback (used only in local/dev mode)
# In Vercel serverless, this resets on every cold start → use KV instead
_intent_counters_local: dict[str, float] = {}


def _counter_get(intent: str) -> float:
    """Reads intent counter from KV (serverless-safe) or local dict."""
    if _STORAGE_BACKEND == "vercel_kv":
        try:
            import httpx
            key = f"riva:counter:{intent}"
            r   = httpx.get(
                f"{os.getenv('VERCEL_KV_URL')}/get/{key}",
                headers={"Authorization": f"Bearer {os.getenv('VERCEL_KV_TOKEN','')}"},
                timeout=3.0,
            )
            val = r.json().get("result")
            return float(val) if val else 0.0
        except Exception as e:
            log.warning("[Counter] KV get failed: %s", e)
    return _intent_counters_local.get(intent, 0.0)


def _counter_set(intent: str, value: float) -> None:
    """Writes intent counter to KV or local dict."""
    if _STORAGE_BACKEND == "vercel_kv":
        try:
            import httpx
            key = f"riva:counter:{intent}"
            httpx.post(
                f"{os.getenv('VERCEL_KV_URL')}/set/{key}",
                headers={"Authorization": f"Bearer {os.getenv('VERCEL_KV_TOKEN','')}"},
                json={"value": str(value)},
                timeout=3.0,
            )
            return
        except Exception as e:
            log.warning("[Counter] KV set failed: %s", e)
    _intent_counters_local[intent] = value


def _counter_delete(intent: str) -> None:
    """Resets counter after signal is emitted."""
    if _STORAGE_BACKEND == "vercel_kv":
        try:
            import httpx
            httpx.post(
                f"{os.getenv('VERCEL_KV_URL')}/del/riva:counter:{intent}",
                headers={"Authorization": f"Bearer {os.getenv('VERCEL_KV_TOKEN','')}"},
                timeout=3.0,
            )
            return
        except Exception as e:
            log.warning("[Counter] KV delete failed: %s", e)
    _intent_counters_local.pop(intent, None)


# ─── Enums ───────────────────────────────────────────────────────────────────

class FeedbackType(str, Enum):
    RATING     = "rating"
    CORRECTION = "correction"
    VALIDATION = "validation"
    FLAGGING   = "flagging"


class FlagReason(str, Enum):
    DANGEROUS_ADVICE     = "dangerous_medical_advice"
    WRONG_DIAGNOSIS      = "wrong_diagnosis"
    MISSED_EMERGENCY     = "missed_emergency"
    DRUG_CONTRAINDICATED = "contraindicated_drug_suggestion"
    HALLUCINATION        = "ai_hallucination"
    BIAS                 = "demographic_bias"
    OTHER                = "other"


class RetrainingPriority(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


# ─── Input sanitization ───────────────────────────────────────────────────────

_DANGEROUS_PATTERNS = re.compile(
    r"(<script|javascript:|on\w+=|<iframe|<object|<embed|eval\(|exec\()",
    re.IGNORECASE,
)

def _sanitize(text: str, max_len: int = 2000) -> str:
    """
    Sanitizes free-text fields to prevent XSS / code injection
    when displayed in the Doctor Dashboard.

    Steps:
        1. HTML-escape special characters
        2. Strip known dangerous patterns
        3. Truncate to max_len
    """
    if not text:
        return ""
    clean = html.escape(text)
    clean = _DANGEROUS_PATTERNS.sub("[REMOVED]", clean)
    return clean[:max_len]


_PID_SALT = os.getenv("PATIENT_ID_SALT", "riva-default-salt-change-in-prod")

def _anonymize_patient_id(patient_id: str) -> str:
    """
    Salted SHA-256 hash of patient_id.
    Salt is loaded from PATIENT_ID_SALT env var (set in .env).
    Without the salt, brute-forcing known IDs is impossible.
    Ensures HIPAA/GDPR compliance — raw ID never stored in logs.
    """
    salted = f"{_PID_SALT}:{patient_id}"
    return hashlib.sha256(salted.encode()).hexdigest()[:16]


# ─── Data classes ─────────────────────────────────────────────────────────────

class FeedbackRecord:

    def __init__(
        self,
        doctor_id:          str,
        patient_id:         str,        # will be anonymised before storage
        session_id:         str,
        ai_intent:          str,
        ai_confidence:      float,
        ai_response:        str,
        feedback_type:      FeedbackType,
        doctor_specialty:   str                  = "unknown",
        rating:             Optional[int]        = None,
        corrected_response: Optional[str]        = None,
        flag_reason:        Optional[FlagReason] = None,
        flag_notes:         str                  = "",
        is_validated:       bool                 = False,
        feedback_id:        Optional[str]        = None,
        timestamp:          Optional[str]        = None,
        patient_id_hash:    Optional[str]        = None,
    ):
        self.feedback_id        = feedback_id    or str(uuid.uuid4())
        self.timestamp          = timestamp      or datetime.now(timezone.utc).isoformat()
        self.doctor_id          = doctor_id
        self.patient_id_hash    = patient_id_hash or _anonymize_patient_id(patient_id)
        self.session_id         = session_id
        self.ai_intent          = ai_intent
        self.ai_confidence      = round(ai_confidence, 3)
        self.ai_response        = _sanitize(ai_response)
        self.feedback_type      = feedback_type
        self.doctor_specialty   = doctor_specialty
        self.rating             = rating
        self.corrected_response = _sanitize(corrected_response or "")
        self.flag_reason        = flag_reason
        self.flag_notes         = _sanitize(flag_notes)
        self.is_validated       = is_validated

    def weighted_score(self) -> float:
        """
        Returns the negative impact score of this feedback.
        Incorporates doctor specialty + feedback type weights.
        A consultant flagging a response scores 3×3 = 9.0
        vs an intern rating 2 stars = 0.5×1.0 = 0.5.
        """
        if self.feedback_type == FeedbackType.VALIDATION:
            return 0.0
        if self.feedback_type == FeedbackType.RATING and self.rating:
            if self.rating > LOW_RATING_CUTOFF:
                return 0.0

        spec_w = SPECIALTY_WEIGHTS.get(self.doctor_specialty, 1.0)
        type_w = TYPE_WEIGHTS.get(self.feedback_type, 1.0)
        return spec_w * type_w

    def is_negative(self) -> bool:
        return self.weighted_score() > 0

    def to_dict(self) -> dict:
        return {
            "feedback_id":         self.feedback_id,
            "timestamp":           self.timestamp,
            "doctor_id":           self.doctor_id,
            "patient_id_hash":     self.patient_id_hash,  # PII-safe
            "session_id":          self.session_id,
            "ai_intent":           self.ai_intent,
            "ai_confidence":       self.ai_confidence,
            "ai_response":         self.ai_response,
            "feedback_type":       self.feedback_type,
            "doctor_specialty":    self.doctor_specialty,
            "rating":              self.rating,
            "corrected_response":  self.corrected_response,
            "flag_reason":         self.flag_reason,
            "flag_notes":          self.flag_notes,
            "is_validated":        self.is_validated,
            "weighted_score":      self.weighted_score(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackRecord":
        obj = cls.__new__(cls)
        obj.feedback_id        = d["feedback_id"]
        obj.timestamp          = d["timestamp"]
        obj.doctor_id          = d["doctor_id"]
        obj.patient_id_hash    = d.get("patient_id_hash", "anonymised")
        obj.session_id         = d["session_id"]
        obj.ai_intent          = d["ai_intent"]
        obj.ai_confidence      = d["ai_confidence"]
        obj.ai_response        = d["ai_response"]
        obj.feedback_type      = FeedbackType(d["feedback_type"])
        obj.doctor_specialty   = d.get("doctor_specialty", "unknown")
        obj.rating             = d.get("rating")
        obj.corrected_response = d.get("corrected_response", "")
        obj.flag_reason        = FlagReason(d["flag_reason"]) if d.get("flag_reason") else None
        obj.flag_notes         = d.get("flag_notes", "")
        obj.is_validated       = d.get("is_validated", False)
        return obj


class RetrainingSignal:

    def __init__(
        self,
        intent:           str,
        priority:         RetrainingPriority,
        weighted_score:   float,
        trigger_count:    int,
        avg_rating:       Optional[float],
        flag_reasons:     list[str],
        sample_responses: list[str],
        signal_id:        Optional[str] = None,
        timestamp:        Optional[str] = None,
    ):
        self.signal_id        = signal_id or str(uuid.uuid4())
        self.timestamp        = timestamp or datetime.now(timezone.utc).isoformat()
        self.intent           = intent
        self.priority         = priority
        self.weighted_score   = round(weighted_score, 2)
        self.trigger_count    = trigger_count
        self.avg_rating       = round(avg_rating, 2) if avg_rating else None
        self.flag_reasons     = flag_reasons
        self.sample_responses = sample_responses[:3]

    def to_dict(self) -> dict:
        return {
            "signal_id":        self.signal_id,
            "timestamp":        self.timestamp,
            "intent":           self.intent,
            "priority":         self.priority,
            "weighted_score":   self.weighted_score,
            "trigger_count":    self.trigger_count,
            "avg_rating":       self.avg_rating,
            "flag_reasons":     self.flag_reasons,
            "sample_responses": self.sample_responses,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RetrainingSignal":
        obj = cls.__new__(cls)
        for k, v in d.items():
            setattr(obj, k, v)
        obj.priority = RetrainingPriority(d["priority"])
        return obj


# ─── Storage backends ─────────────────────────────────────────────────────────

def _store(collection: str, data: dict) -> None:
    """Routes storage to the appropriate backend."""
    if _STORAGE_BACKEND == "supabase":
        _store_supabase(collection, data)
    elif _STORAGE_BACKEND == "vercel_kv":
        _store_vercel_kv(collection, data)
    else:
        _store_local(collection, data)


def _store_local(collection: str, data: dict) -> None:
    path = _FEEDBACK_FILE if collection == "feedback" else _RETRAINING_FILE
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _store_supabase(collection: str, data: dict) -> None:
    """
    Stores to Supabase (Postgres) via REST API.
    Table names: riva_feedback / riva_retraining_signals
    Set SUPABASE_URL and SUPABASE_KEY in environment.
    """
    try:
        import httpx
        table = "riva_feedback" if collection == "feedback" else "riva_retraining_signals"
        httpx.post(
            f"{os.getenv('SUPABASE_URL')}/rest/v1/{table}",
            headers={
                "apikey":        os.getenv("SUPABASE_KEY", ""),
                "Authorization": f"Bearer {os.getenv('SUPABASE_KEY', '')}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=data,
            timeout=5.0,
        )
    except Exception as e:
        log.error("[Feedback] Supabase store failed: %s — falling back to local", e)
        _store_local(collection, data)


def _store_vercel_kv(collection: str, data: dict) -> None:
    """
    Stores to Vercel KV (Redis-compatible) via REST API.
    Key pattern: riva:{collection}:{id}
    Set VERCEL_KV_URL and VERCEL_KV_TOKEN in environment.
    """
    try:
        import httpx
        key = f"riva:{collection}:{data.get('feedback_id') or data.get('signal_id')}"
        httpx.post(
            f"{os.getenv('VERCEL_KV_URL')}/set/{key}",
            headers={"Authorization": f"Bearer {os.getenv('VERCEL_KV_TOKEN', '')}"},
            json={"value": json.dumps(data, ensure_ascii=False)},
            timeout=5.0,
        )
    except Exception as e:
        log.error("[Feedback] Vercel KV store failed: %s — falling back to local", e)
        _store_local(collection, data)


def _load_local(collection: str, cls) -> list:
    path = _FEEDBACK_FILE if collection == "feedback" else _RETRAINING_FILE
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(cls.from_dict(json.loads(line)))
                except Exception as e:
                    log.warning("[Feedback] Skipping malformed line: %s", e)
    return records


# ─── Critical alert ───────────────────────────────────────────────────────────

def _alert_tech_team(signal: RetrainingSignal) -> None:
    """
    Sends immediate notification to the tech team when a CRITICAL
    retraining signal is emitted (e.g. doctor flags dangerous AI advice).

    Configured via environment variables:
        SLACK_WEBHOOK_URL  — Slack channel webhook
        ALERT_EMAIL        — fallback email (requires SMTP config)

    Medical errors cannot wait for the next training cycle.
    """
    message = (
        f"🚨 RIVA CRITICAL RETRAINING SIGNAL\n"
        f"Intent  : {signal.intent}\n"
        f"Score   : {signal.weighted_score}\n"
        f"Reasons : {', '.join(signal.flag_reasons)}\n"
        f"Signal  : {signal.signal_id}\n"
        f"Time    : {signal.timestamp}"
    )

    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_url:
        try:
            import httpx
            httpx.post(slack_url, json={"text": message}, timeout=5.0)
            log.info("[Alert] Slack notification sent for signal %s", signal.signal_id[:8])
            return
        except Exception as e:
            log.error("[Alert] Slack failed: %s", e)

    # Fallback — log prominently so monitoring tools pick it up
    log.critical("=" * 60)
    log.critical(message)
    log.critical("=" * 60)


# ─── Handler ─────────────────────────────────────────────────────────────────

class DoctorFeedbackHandler:

    def receive_feedback(self, record: FeedbackRecord) -> dict:
        # 1. Persist
        _store("feedback", record.to_dict())
        log.info(
            "[Feedback] doctor=%s  intent=%s  type=%s  weight=%.1f",
            record.doctor_id, record.ai_intent,
            record.feedback_type, record.weighted_score(),
        )

        signal = None

        # 2. Immediate critical signal
        if (
            record.feedback_type == FeedbackType.FLAGGING
            and CRITICAL_FLAG_ALERT
        ):
            signal = self._emit_signal(
                intent         = record.ai_intent,
                priority       = RetrainingPriority.CRITICAL,
                weighted_score = record.weighted_score(),
                trigger_count  = 1,
                avg_rating     = None,
                flag_reasons   = [str(record.flag_reason or FlagReason.OTHER)],
                samples        = [record.ai_response],
            )
            _alert_tech_team(signal)

        # 3. Counter-based signal (O(1) — no file scan)
        elif record.is_negative():
            signal = self._counter_check(record)

        return {
            "feedback_id":       record.feedback_id,
            "retraining_signal": signal.to_dict() if signal else None,
            "message": (
                "تم إرسال إشارة إعادة تدريب للفريق التقني فوراً"
                if signal else
                "شكراً على تقييمك — هيساعد ريفا تتحسن"
            ),
        }

    def _counter_check(self, record: FeedbackRecord) -> Optional[RetrainingSignal]:
        """
        Serverless-safe counter check using KV storage.
        Avoids O(N) file reads and survives cold starts on Vercel.
        Logs the final weighted_score inside the Signal for audit reference.
        """
        intent    = record.ai_intent
        current   = _counter_get(intent) + record.weighted_score()

        threshold = NEGATIVE_THRESHOLD * SPECIALTY_WEIGHTS.get(
            record.doctor_specialty, 1.0
        )

        if current >= threshold:
            final_score = current
            _counter_delete(intent)   # reset after signal
            log.info(
                "[Counter] Threshold reached for '%s'  score=%.1f  threshold=%.1f",
                intent, final_score, threshold,
            )
            priority = (
                RetrainingPriority.HIGH   if final_score >= threshold * 2 else
                RetrainingPriority.MEDIUM if final_score >= threshold      else
                RetrainingPriority.LOW
            )
            return self._emit_signal(
                intent         = intent,
                priority       = priority,
                weighted_score = final_score,   # مرجع لقوة النقد
                trigger_count  = int(final_score),
                avg_rating     = None,
                flag_reasons   = [],
                samples        = [record.ai_response],
            )

        _counter_set(intent, current)
        log.debug("[Counter] '%s' score=%.1f / %.1f", intent, current, threshold)
        return None

    def _emit_signal(
        self,
        intent:         str,
        priority:       RetrainingPriority,
        weighted_score: float,
        trigger_count:  int,
        avg_rating:     Optional[float],
        flag_reasons:   list[str],
        samples:        list[str],
    ) -> RetrainingSignal:
        signal = RetrainingSignal(
            intent           = intent,
            priority         = priority,
            weighted_score   = weighted_score,
            trigger_count    = trigger_count,
            avg_rating       = avg_rating,
            flag_reasons     = flag_reasons,
            sample_responses = samples,
        )
        _store("retraining", signal.to_dict())
        log.info(
            "[Retraining] signal=%s  intent=%s  priority=%s  score=%.1f",
            signal.signal_id[:8], intent, priority, weighted_score,
        )
        return signal

    def load_feedback(
        self,
        doctor_id:     Optional[str] = None,
        intent:        Optional[str] = None,
        negative_only: bool          = False,
    ) -> list[dict]:
        records = _load_local("feedback", FeedbackRecord)
        if doctor_id:
            records = [r for r in records if r.doctor_id == doctor_id]
        if intent:
            records = [r for r in records if r.ai_intent == intent]
        if negative_only:
            records = [r for r in records if r.is_negative()]
        return [r.to_dict() for r in records]

    def load_signals(self, priority: Optional[RetrainingPriority] = None) -> list[dict]:
        signals = _load_local("retraining", RetrainingSignal)
        if priority:
            signals = [s for s in signals if s.priority == priority]
        return [s.to_dict() for s in signals]

    def summary(self) -> dict:
        records = _load_local("feedback", FeedbackRecord)
        signals = _load_local("retraining", RetrainingSignal)
        if not records:
            return {"total_feedback": 0, "retraining_signals": 0,
                    "storage_backend": _STORAGE_BACKEND}

        ratings  = [r.rating for r in records if r.rating is not None]
        avg_rat  = round(sum(ratings) / len(ratings), 2) if ratings else None

        by_intent: dict[str, dict] = {}
        for r in records:
            if r.ai_intent not in by_intent:
                by_intent[r.ai_intent] = {"total": 0, "weighted_negative": 0.0}
            by_intent[r.ai_intent]["total"] += 1
            by_intent[r.ai_intent]["weighted_negative"] += r.weighted_score()

        return {
            "total_feedback":      len(records),
            "avg_rating":          avg_rat,
            "negative_weighted":   sum(r.weighted_score() for r in records),
            "flagged_count":       sum(1 for r in records
                                       if r.feedback_type == FeedbackType.FLAGGING),
            "validated_count":     sum(1 for r in records if r.is_validated),
            "by_intent":           by_intent,
            "retraining_signals":  len(signals),
            "critical_signals":    sum(1 for s in signals
                                       if s.priority == RetrainingPriority.CRITICAL),
            "storage_backend":     _STORAGE_BACKEND,
            "pending_counters":    dict(_intent_counters),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_handler = DoctorFeedbackHandler()


# ─── Public API ───────────────────────────────────────────────────────────────

def submit_rating(
    doctor_id:        str,
    patient_id:       str,
    session_id:       str,
    ai_intent:        str,
    ai_confidence:    float,
    ai_response:      str,
    rating:           int,
    doctor_specialty: str = "unknown",
) -> dict:
    assert 1 <= rating <= 5
    return _handler.receive_feedback(FeedbackRecord(
        doctor_id=doctor_id, patient_id=patient_id, session_id=session_id,
        ai_intent=ai_intent, ai_confidence=ai_confidence, ai_response=ai_response,
        feedback_type=FeedbackType.RATING, rating=rating,
        doctor_specialty=doctor_specialty,
    ))


def submit_correction(
    doctor_id:          str,
    patient_id:         str,
    session_id:         str,
    ai_intent:          str,
    ai_confidence:      float,
    ai_response:        str,
    corrected_response: str,
    doctor_specialty:   str = "unknown",
) -> dict:
    return _handler.receive_feedback(FeedbackRecord(
        doctor_id=doctor_id, patient_id=patient_id, session_id=session_id,
        ai_intent=ai_intent, ai_confidence=ai_confidence, ai_response=ai_response,
        feedback_type=FeedbackType.CORRECTION,
        corrected_response=corrected_response,
        doctor_specialty=doctor_specialty,
    ))


def submit_validation(
    doctor_id:        str,
    patient_id:       str,
    session_id:       str,
    ai_intent:        str,
    ai_confidence:    float,
    ai_response:      str,
    doctor_specialty: str = "unknown",
) -> dict:
    return _handler.receive_feedback(FeedbackRecord(
        doctor_id=doctor_id, patient_id=patient_id, session_id=session_id,
        ai_intent=ai_intent, ai_confidence=ai_confidence, ai_response=ai_response,
        feedback_type=FeedbackType.VALIDATION, is_validated=True,
        doctor_specialty=doctor_specialty,
    ))


def submit_flag(
    doctor_id:        str,
    patient_id:       str,
    session_id:       str,
    ai_intent:        str,
    ai_confidence:    float,
    ai_response:      str,
    flag_reason:      FlagReason,
    flag_notes:       str = "",
    doctor_specialty: str = "unknown",
) -> dict:
    return _handler.receive_feedback(FeedbackRecord(
        doctor_id=doctor_id, patient_id=patient_id, session_id=session_id,
        ai_intent=ai_intent, ai_confidence=ai_confidence, ai_response=ai_response,
        feedback_type=FeedbackType.FLAGGING,
        flag_reason=flag_reason, flag_notes=flag_notes,
        doctor_specialty=doctor_specialty,
    ))


def get_summary() -> dict:
    return _handler.summary()


def get_signals(priority: Optional[RetrainingPriority] = None) -> list[dict]:
    return _handler.load_signals(priority=priority)


def get_feedback(
    doctor_id:     Optional[str] = None,
    intent:        Optional[str] = None,
    negative_only: bool          = False,
) -> list[dict]:
    return _handler.load_feedback(
        doctor_id=doctor_id, intent=intent, negative_only=negative_only
    )
