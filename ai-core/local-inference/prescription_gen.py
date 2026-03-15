"""
RIVA - Prescription Generator v2.0
====================================
مولّد الروشتات الذكي مع:
✅ توقيع رقمي مُصلح
✅ Hash كامل للروشتة
✅ فحص تعارض الأدوية
✅ انتهاء صلاحية
✅ QR Payload محسّن
Author: GODA EMAD
"""
import json
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("RIVA.PrescriptionGen")

class PrescriptionGenerator:

    VERSION = "RIVA_RX_v1"

    # تعارضات أساسية — يتحدث من drug_conflicts.json
    RISKY_PAIRS = [
        ("warfarin",   "aspirin"),
        ("warfarin",   "ibuprofen"),
        ("warfarin",   "paracetamol"),
        ("metformin",  "alcohol"),
        ("ibuprofen",  "captopril"),
        ("digoxin",    "amiodarone"),
        ("simvastatin","amlodipine"),
    ]

    def __init__(self):
        logger.info("PrescriptionGenerator v2.0 initialized")

    # ── GENERATE ─────────────────────────────────
    def generate(
        self,
        patient_id:  str,
        doctor_id:   str,
        diagnosis:   str,
        medications: list,
        notes:       str = ""
    ) -> dict:
        rx_id     = str(uuid.uuid4())[:12].upper()
        timestamp = datetime.now().isoformat()
        issued_date = timestamp[:10]

        warnings = self.check_interactions(medications)

        rx = {
            "version":     self.VERSION,
            "rx_id":       rx_id,
            "patient_id":  patient_id,
            "doctor_id":   doctor_id,
            "diagnosis":   diagnosis,
            "medications": medications,
            "notes":       notes,
            "warnings":    warnings,
            "issued_at":   timestamp,
            "valid_until": self._valid_until(medications),
            "status":      "active",
            "signature":   self._sign(rx_id, patient_id, doctor_id, issued_date),
        }

        # Hash بعد بناء الروشتة كاملة
        rx["hash"]       = self._rx_hash(rx)
        rx["qr_payload"] = self.to_qr_payload(rx)

        logger.info(
            f"RX Generated | id={rx_id} | "
            f"patient={patient_id} | "
            f"meds={len(medications)} | "
            f"warnings={len(warnings)}"
        )
        return rx

    # ── SIGN ─────────────────────────────────────
    def _sign(
        self,
        rx_id:      str,
        patient_id: str,
        doctor_id:  str,
        issued_date:str
    ) -> str:
        data = f"{rx_id}{patient_id}{doctor_id}{issued_date}"
        return hashlib.sha256(data.encode()).hexdigest()[:16].upper()

    # ── HASH ─────────────────────────────────────
    def _rx_hash(self, rx: dict) -> str:
        # شيل hash و qr_payload من الحساب عشان مش موجودين وقت الحساب
        rx_copy = {k: v for k, v in rx.items()
                   if k not in ("hash", "qr_payload")}
        payload = json.dumps(rx_copy, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    # ── QR PAYLOAD ───────────────────────────────
    def to_qr_payload(self, rx: dict) -> str:
        payload = (
            f"RIVA-RX:{rx['rx_id']}|"
            f"P:{rx['patient_id']}|"
            f"D:{rx['doctor_id']}|"
            f"DX:{rx['diagnosis']}|"
            f"DATE:{rx['issued_at'][:10]}|"
            f"H:{rx.get('hash','')[:10]}"
        )
        return payload

    # ── VALID UNTIL ──────────────────────────────
    def _valid_until(self, medications: list) -> str:
        max_days = max(
            (m.get("days", 7) for m in medications),
            default=7
        )
        valid = datetime.now() + timedelta(days=max_days + 3)
        return valid.strftime("%Y-%m-%d")

    # ── CHECK INTERACTIONS ───────────────────────
    def check_interactions(self, medications: list) -> list:
        names    = {m["name"].lower().strip() for m in medications}
        warnings = []
        for a, b in self.RISKY_PAIRS:
            if a in names and b in names:
                warnings.append(f"تعارض دوائي: {a} + {b}")
                logger.warning(f"Drug interaction: {a} + {b}")
        return warnings

    # ── VERIFY ───────────────────────────────────
    def verify(self, rx: dict) -> bool:
        expected = self._sign(
            rx["rx_id"],
            rx["patient_id"],
            rx["doctor_id"],
            rx["issued_at"][:10]
        )
        sig_ok  = rx.get("signature", "") == expected
        hash_ok = rx.get("hash", "") == self._rx_hash(rx)
        return sig_ok and hash_ok

    # ── IS EXPIRED ───────────────────────────────
    def is_expired(self, rx: dict) -> bool:
        return datetime.now().date() > datetime.fromisoformat(
            rx["valid_until"]
        ).date()

    # ── STATUS ───────────────────────────────────
    def get_status(self, rx: dict) -> dict:
        return {
            "rx_id":      rx["rx_id"],
            "valid":      self.verify(rx),
            "expired":    self.is_expired(rx),
            "warnings":   rx.get("warnings", []),
            "status":     "expired" if self.is_expired(rx) else rx.get("status","active")
        }
