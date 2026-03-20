"""
clinical_override_log.py
========================
RIVA Health Platform — Doctor Clinical Override Logger
------------------------------------------------------
يسجّل كل قرار طبي اتخذه الدكتور بشكل يخالف توصية الـ AI.

التحسينات المطبّقة:
    1. Chain of Trust     — كل سجل عنده record_hash مشتق من بياناته
                            يمنع التعديل اليدوي بعد الحفظ
    2. PII Salted Hashing — patient_id يتحوّل لـ salted SHA-256
                            مش ممكن يتعكس حتى لو الـ hash اتسرّب
    3. Cloud Storage      — يدعم Supabase / Vercel KV / local JSONL
                            الملفات المحلية بتتمسح على Vercel deployments
    4. O(1) Outcome Update— تحديث النتيجة عبر قاعدة البيانات مباشرة
                            من غير إعادة كتابة الملف كله

Author : GODA EMAD
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.doctor_validation.override_log")

# ─── Storage backend ─────────────────────────────────────────────────────────

_STORAGE_BACKEND = (
    "supabase"  if os.getenv("SUPABASE_URL")  else
    "vercel_kv" if os.getenv("VERCEL_KV_URL") else
    "local"
)

_LOG_DIR  = Path(__file__).parent / "logs"
_LOG_FILE = _LOG_DIR / "clinical_overrides.jsonl"

if _STORAGE_BACKEND == "local":
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

log.info("[Override] Storage backend: %s", _STORAGE_BACKEND)

# ─── PII Salted Hashing ───────────────────────────────────────────────────────

_PID_SALT = os.getenv("PATIENT_ID_SALT", "riva-default-salt-change-in-prod")


def _anonymize(patient_id: str) -> str:
    """
    Salted SHA-256 hash — without the salt, brute-forcing is infeasible.
    Raw patient_id is never persisted anywhere.
    """
    return hashlib.sha256(f"{_PID_SALT}:{patient_id}".encode()).hexdigest()[:16]


# ─── Chain of Trust ───────────────────────────────────────────────────────────

def _compute_hash(data: dict) -> str:
    """
    Computes a deterministic SHA-256 fingerprint of the record fields
    (excluding record_hash itself and outcome which can be updated).

    Any post-hoc manual edit to the stored JSON will invalidate the hash,
    making tampering immediately detectable during audit.
    """
    stable_fields = {
        k: v for k, v in data.items()
        if k not in ("record_hash", "outcome")
    }
    canonical = json.dumps(stable_fields, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def verify_record_integrity(record_dict: dict) -> bool:
    """
    Public utility — call this when loading records to verify
    they haven't been tampered with since creation.
    Returns True if the hash matches, False if tampered.
    """
    stored_hash   = record_dict.get("record_hash", "")
    expected_hash = _compute_hash(record_dict)
    return stored_hash == expected_hash


# ─── Enums ───────────────────────────────────────────────────────────────────

class OverrideReason(str, Enum):
    AI_MISSED_SYMPTOM      = "ai_missed_symptom"
    AI_OVERESTIMATED       = "ai_overestimated_severity"
    AI_UNDERESTIMATED      = "ai_underestimated_severity"
    PATIENT_CONTEXT        = "additional_patient_context"
    CLINICAL_EXAM_DIFFERS  = "clinical_exam_differs"
    LOCAL_PROTOCOL         = "local_protocol"
    PATIENT_PREFERENCE     = "patient_preference"
    COMORBIDITY            = "comorbidity_consideration"
    MEDICATION_INTERACTION = "medication_interaction"
    OTHER                  = "other"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MODERATE = "moderate"
    LOW      = "low"
    MONITOR  = "monitor"


class Outcome(str, Enum):
    IMPROVED = "improved"
    STABLE   = "stable"
    WORSENED = "worsened"
    REFERRED = "referred"
    PENDING  = "pending"


# ─── Record ──────────────────────────────────────────────────────────────────

class ClinicalOverrideRecord:

    def __init__(
        self,
        doctor_id:        str,
        patient_id:       str,
        session_id:       str,
        ai_intent:        str,
        ai_confidence:    float,
        ai_suggestion:    str,
        doctor_decision:  str,
        override_reason:  OverrideReason,
        severity:         Severity,
        reason_notes:     str     = "",
        outcome:          Outcome = Outcome.PENDING,
        override_id:      Optional[str] = None,
        timestamp:        Optional[str] = None,
        patient_id_hash:  Optional[str] = None,
        record_hash:      Optional[str] = None,
    ):
        self.override_id      = override_id     or str(uuid.uuid4())
        self.timestamp        = timestamp       or datetime.now(timezone.utc).isoformat()
        self.doctor_id        = doctor_id
        self.patient_id_hash  = patient_id_hash or _anonymize(patient_id)
        self.session_id       = session_id
        self.ai_intent        = ai_intent
        self.ai_confidence    = round(ai_confidence, 3)
        self.ai_suggestion    = ai_suggestion
        self.doctor_decision  = doctor_decision
        self.override_reason  = override_reason
        self.severity         = severity
        self.reason_notes     = reason_notes
        self.outcome          = outcome

        # Chain of Trust — computed after all fields are set
        base = self._base_dict()
        self.record_hash = record_hash or _compute_hash(base)

    def _base_dict(self) -> dict:
        """Fields used for hash computation (outcome excluded — it can change)."""
        return {
            "override_id":     self.override_id,
            "timestamp":       self.timestamp,
            "doctor_id":       self.doctor_id,
            "patient_id_hash": self.patient_id_hash,
            "session_id":      self.session_id,
            "ai_intent":       self.ai_intent,
            "ai_confidence":   self.ai_confidence,
            "ai_suggestion":   self.ai_suggestion,
            "doctor_decision": self.doctor_decision,
            "override_reason": self.override_reason,
            "severity":        self.severity,
            "reason_notes":    self.reason_notes,
        }

    def to_dict(self) -> dict:
        d = self._base_dict()
        d["outcome"]     = self.outcome
        d["record_hash"] = self.record_hash
        return d

    def is_intact(self) -> bool:
        """Returns True if the record hasn't been tampered with."""
        return self.record_hash == _compute_hash(self._base_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "ClinicalOverrideRecord":
        obj = cls.__new__(cls)
        obj.override_id     = d["override_id"]
        obj.timestamp       = d["timestamp"]
        obj.doctor_id       = d["doctor_id"]
        obj.patient_id_hash = d.get("patient_id_hash", "anonymised")
        obj.session_id      = d["session_id"]
        obj.ai_intent       = d["ai_intent"]
        obj.ai_confidence   = d["ai_confidence"]
        obj.ai_suggestion   = d["ai_suggestion"]
        obj.doctor_decision = d["doctor_decision"]
        obj.override_reason = OverrideReason(d["override_reason"])
        obj.severity        = Severity(d["severity"])
        obj.reason_notes    = d.get("reason_notes", "")
        obj.outcome         = Outcome(d.get("outcome", Outcome.PENDING))
        obj.record_hash     = d.get("record_hash", "")
        return obj

    def __repr__(self) -> str:
        intact = "✓" if self.is_intact() else "✗ TAMPERED"
        return (
            f"ClinicalOverride("
            f"id={self.override_id[:8]}… "
            f"doctor={self.doctor_id} "
            f"ai={self.ai_intent}({self.ai_confidence:.0%}) "
            f"→ {self.doctor_decision}  [{intact}])"
        )


# ─── Storage helpers ──────────────────────────────────────────────────────────

def _append_local(data: dict) -> None:
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _append_supabase(data: dict) -> None:
    try:
        import httpx
        httpx.post(
            f"{os.getenv('SUPABASE_URL')}/rest/v1/riva_clinical_overrides",
            headers={
                "apikey":        os.getenv("SUPABASE_KEY", ""),
                "Authorization": f"Bearer {os.getenv('SUPABASE_KEY','')}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=data,
            timeout=5.0,
        )
    except Exception as e:
        log.error("[Override] Supabase insert failed: %s — fallback local", e)
        _append_local(data)


def _append_vercel_kv(data: dict) -> None:
    try:
        import httpx
        key = f"riva:override:{data['override_id']}"
        httpx.post(
            f"{os.getenv('VERCEL_KV_URL')}/set/{key}",
            headers={"Authorization": f"Bearer {os.getenv('VERCEL_KV_TOKEN','')}"},
            json={"value": json.dumps(data, ensure_ascii=False)},
            timeout=5.0,
        )
    except Exception as e:
        log.error("[Override] Vercel KV set failed: %s — fallback local", e)
        _append_local(data)


def _store(data: dict) -> None:
    if _STORAGE_BACKEND == "supabase":
        _append_supabase(data)
    elif _STORAGE_BACKEND == "vercel_kv":
        _append_vercel_kv(data)
    else:
        _append_local(data)


# ─── O(1) Outcome Update ─────────────────────────────────────────────────────

def _update_outcome_supabase(override_id: str, outcome: Outcome) -> bool:
    """
    PATCH a single row in Supabase — O(1), no full file rewrite.
    """
    try:
        import httpx
        r = httpx.patch(
            f"{os.getenv('SUPABASE_URL')}/rest/v1/riva_clinical_overrides"
            f"?override_id=eq.{override_id}",
            headers={
                "apikey":        os.getenv("SUPABASE_KEY", ""),
                "Authorization": f"Bearer {os.getenv('SUPABASE_KEY','')}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json={"outcome": outcome},
            timeout=5.0,
        )
        return r.status_code in (200, 204)
    except Exception as e:
        log.error("[Override] Supabase update failed: %s", e)
        return False


def _update_outcome_vercel_kv(override_id: str, outcome: Outcome) -> bool:
    """
    Fetch → mutate → set back in Vercel KV — still O(1) per record.
    """
    try:
        import httpx
        key   = f"riva:override:{override_id}"
        base  = os.getenv("VERCEL_KV_URL")
        token = os.getenv("VERCEL_KV_TOKEN", "")
        hdrs  = {"Authorization": f"Bearer {token}"}

        r = httpx.get(f"{base}/get/{key}", headers=hdrs, timeout=3.0)
        if r.status_code != 200:
            return False

        data = json.loads(r.json().get("result", "{}"))
        data["outcome"] = outcome
        httpx.post(
            f"{base}/set/{key}",
            headers=hdrs,
            json={"value": json.dumps(data, ensure_ascii=False)},
            timeout=3.0,
        )
        return True
    except Exception as e:
        log.error("[Override] Vercel KV update failed: %s", e)
        return False


def _update_outcome_local(override_id: str, outcome: Outcome) -> bool:
    """
    Local fallback — rewrites the file (acceptable: local = dev only).
    """
    if not _LOG_FILE.exists():
        return False

    lines   = _LOG_FILE.read_text(encoding="utf-8").splitlines()
    found   = False
    updated = []

    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("override_id") == override_id:
                obj["outcome"] = outcome
                found = True
            updated.append(json.dumps(obj, ensure_ascii=False))
        except Exception:
            updated.append(line)

    if found:
        _LOG_FILE.write_text("\n".join(updated) + "\n", encoding="utf-8")

    return found


# ─── Logger ──────────────────────────────────────────────────────────────────

class ClinicalOverrideLogger:

    def log(self, record: ClinicalOverrideRecord) -> str:
        """Persists one override record. Returns override_id."""
        _store(record.to_dict())
        log.info(
            "[Override] id=%s  doctor=%s  patient_hash=%s  "
            "ai=%s(%.0f%%)  reason=%s  hash=%s",
            record.override_id[:8],
            record.doctor_id,
            record.patient_id_hash,
            record.ai_intent,
            record.ai_confidence * 100,
            record.override_reason,
            record.record_hash[:8],
        )
        return record.override_id

    def update_outcome(self, override_id: str, outcome: Outcome) -> bool:
        """
        O(1) outcome update — routes to the right backend.
        Does NOT rewrite the whole file.
        """
        if _STORAGE_BACKEND == "supabase":
            ok = _update_outcome_supabase(override_id, outcome)
        elif _STORAGE_BACKEND == "vercel_kv":
            ok = _update_outcome_vercel_kv(override_id, outcome)
        else:
            ok = _update_outcome_local(override_id, outcome)

        if ok:
            log.info("[Override] outcome updated: %s → %s", override_id[:8], outcome)
        else:
            log.warning("[Override] outcome update failed: %s", override_id[:8])
        return ok

    def load_all(self) -> list[ClinicalOverrideRecord]:
        """Loads from local JSONL (dev mode). In production query DB directly."""
        if not _LOG_FILE.exists():
            return []
        records = []
        tampered = 0
        with open(_LOG_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = ClinicalOverrideRecord.from_dict(json.loads(line))
                    if not r.is_intact():
                        tampered += 1
                        log.warning("[Override] TAMPERED record: %s", r.override_id[:8])
                    records.append(r)
                except Exception as e:
                    log.warning("[Override] Malformed line: %s", e)

        if tampered:
            log.error("[Override] %d tampered record(s) detected in audit log!", tampered)
        return records

    def load_by_patient(self, patient_id: str) -> list[ClinicalOverrideRecord]:
        pid_hash = _anonymize(patient_id)
        return [r for r in self.load_all() if r.patient_id_hash == pid_hash]

    def load_by_session(self, session_id: str) -> list[ClinicalOverrideRecord]:
        return [r for r in self.load_all() if r.session_id == session_id]

    def summary(self) -> dict:
        records = self.load_all()
        if not records:
            return {
                "total":           0,
                "storage_backend": _STORAGE_BACKEND,
                "chain_of_trust":  "enabled",
            }

        reason_counts: dict[str, int] = {}
        intent_counts: dict[str, int] = {}
        tampered_count = 0

        for r in records:
            reason_counts[r.override_reason] = reason_counts.get(r.override_reason, 0) + 1
            intent_counts[r.ai_intent]       = intent_counts.get(r.ai_intent, 0) + 1
            if not r.is_intact():
                tampered_count += 1

        confs = [r.ai_confidence for r in records]

        return {
            "total":              len(records),
            "avg_ai_confidence":  round(sum(confs) / len(confs), 3),
            "by_reason":          reason_counts,
            "by_ai_intent":       intent_counts,
            "pending_outcomes":   sum(1 for r in records if r.outcome == Outcome.PENDING),
            "tampered_records":   tampered_count,
            "storage_backend":    _STORAGE_BACKEND,
            "chain_of_trust":     "enabled",
            "log_file":           str(_LOG_FILE) if _STORAGE_BACKEND == "local" else "cloud",
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_logger = ClinicalOverrideLogger()


# ─── Public API ───────────────────────────────────────────────────────────────

def log_override(
    doctor_id:       str,
    patient_id:      str,
    session_id:      str,
    ai_intent:       str,
    ai_confidence:   float,
    ai_suggestion:   str,
    doctor_decision: str,
    override_reason: OverrideReason,
    severity:        Severity,
    reason_notes:    str    = "",
    outcome:         Outcome = Outcome.PENDING,
) -> str:
    """
    Logs one clinical override. Returns the override_id.

    Example:
        override_id = log_override(
            doctor_id       = "DR-001",
            patient_id      = "PT-4892",      # will be salted & hashed
            session_id      = "abc-123",
            ai_intent       = "Triage",
            ai_confidence   = 0.42,
            ai_suggestion   = "راجع طبيب...",
            doctor_decision = "حالة طوارئ — إدخال فوري",
            override_reason = OverrideReason.AI_UNDERESTIMATED,
            severity        = Severity.CRITICAL,
            reason_notes    = "ألم صدر ينتشر للذراع اليسرى",
        )
    """
    record = ClinicalOverrideRecord(
        doctor_id       = doctor_id,
        patient_id      = patient_id,
        session_id      = session_id,
        ai_intent       = ai_intent,
        ai_confidence   = ai_confidence,
        ai_suggestion   = ai_suggestion,
        doctor_decision = doctor_decision,
        override_reason = override_reason,
        severity        = severity,
        reason_notes    = reason_notes,
        outcome         = outcome,
    )
    return _logger.log(record)


def update_outcome(override_id: str, outcome: Outcome) -> bool:
    """O(1) outcome update — no full file rewrite."""
    return _logger.update_outcome(override_id, outcome)


def get_summary() -> dict:
    """Aggregate stats + chain-of-trust integrity report."""
    return _logger.summary()


def get_by_patient(patient_id: str) -> list[dict]:
    return [r.to_dict() for r in _logger.load_by_patient(patient_id)]


def get_by_session(session_id: str) -> list[dict]:
    return [r.to_dict() for r in _logger.load_by_session(session_id)]


def verify_audit_log() -> dict:
    """
    Runs integrity check on all local records.
    Returns a report showing how many records are intact vs tampered.
    Use this in the doctor dashboard or CI pipeline.
    """
    records  = _logger.load_all()
    intact   = sum(1 for r in records if r.is_intact())
    tampered = len(records) - intact
    return {
        "total":    len(records),
        "intact":   intact,
        "tampered": tampered,
        "status":   "clean" if tampered == 0 else "COMPROMISED",
    }
