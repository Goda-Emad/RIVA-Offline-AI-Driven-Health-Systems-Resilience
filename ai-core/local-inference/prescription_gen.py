"""
RIVA - Prescription Generator v3.0
====================================
✅ منع KeyError في medications
✅ Constant-time comparison (hmac)
✅ تحميل التعارضات من JSON
✅ QR Payload محسّن مع version
✅ verify_qr للصيدليات
Author: GODA EMAD
"""
import json
import uuid
import hashlib
import hmac
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("RIVA.PrescriptionGen")

class PrescriptionGenerator:

    VERSION = "RIVA_RX_v1"

    DEFAULT_RISKY_PAIRS = [
        ["warfarin",    "aspirin"],
        ["warfarin",    "ibuprofen"],
        ["warfarin",    "paracetamol"],
        ["metformin",   "alcohol"],
        ["ibuprofen",   "captopril"],
        ["digoxin",     "amiodarone"],
        ["simvastatin", "amlodipine"],
    ]

    def __init__(
        self,
        conflicts_path: str = "business-intelligence/medical-content/drug_conflicts.json"
    ):
        self.RISKY_PAIRS = self.DEFAULT_RISKY_PAIRS.copy()
        self._load_interactions(conflicts_path)
        logger.info("PrescriptionGenerator v3.0 initialized")

    # ── LOAD INTERACTIONS ────────────────────────
    def _load_interactions(self, path: str) -> None:
        try:
            import os
            if not os.path.exists(path):
                logger.warning(f"conflicts not found: {path} — using defaults")
                return
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            conflicts = data.get("conflicts", [])
            self.RISKY_PAIRS = [
                [c["drug_a"], c["drug_b"]]
                for c in conflicts
                if c.get("severity") == "high"
            ]
            logger.info(f"Loaded {len(self.RISKY_PAIRS)} high-risk pairs")
        except Exception as e:
            logger.warning(f"Failed to load conflicts: {e} — using defaults")

    # ── GENERATE ─────────────────────────────────
    def generate(
        self,
        patient_id:  str,
        doctor_id:   str,
        diagnosis:   str,
        medications: list,
        notes:       str = ""
    ) -> dict:
        rx_id       = str(uuid.uuid4())[:12].upper()
        timestamp   = datetime.now().isoformat()
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
        rx_id:       str,
        patient_id:  str,
        doctor_id:   str,
        issued_date: str
    ) -> str:
        data = f"{rx_id}{patient_id}{doctor_id}{issued_date}"
        return hashlib.sha256(data.encode()).hexdigest()[:16].upper()

    # ── HASH ─────────────────────────────────────
    def _rx_hash(self, rx: dict) -> str:
        rx_copy = {k: v for k, v in rx.items()
                   if k not in ("hash", "qr_payload")}
        payload = json.dumps(rx_copy, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    # ── QR PAYLOAD ───────────────────────────────
    def to_qr_payload(self, rx: dict) -> str:
        return (
            f"RIVA:{self.VERSION}|"
            f"RX:{rx['rx_id']}|"
            f"P:{rx['patient_id']}|"
            f"D:{rx['doctor_id']}|"
            f"DATE:{rx['issued_at'][:10]}|"
            f"H:{rx.get('hash','')[:10]}"
        )

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
        # تحسين 1: منع KeyError لو name ناقص
        names = {
            m.get("name", "").lower().strip()
            for m in medications
            if m.get("name")
        }
        warnings = []
        for pair in self.RISKY_PAIRS:
            a, b = pair[0].lower(), pair[1].lower()
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
        # تحسين 2: constant-time comparison
        try:
            sig_ok  = hmac.compare_digest(rx.get("signature",""), expected)
            hash_ok = hmac.compare_digest(rx.get("hash",""), self._rx_hash(rx))
            return sig_ok and hash_ok
        except Exception:
            return False

    # ── VERIFY QR ────────────────────────────────
    def verify_qr(self, qr_payload: str) -> dict:
        """للصيدليات — يتحقق من QR Code"""
        try:
            parts = dict(
                p.split(":", 1) for p in qr_payload.split("|")
                if ":" in p
            )
            return {
                "rx_id":        parts.get("RX"),
                "patient_id":   parts.get("P"),
                "doctor_id":    parts.get("D"),
                "issued_date":  parts.get("DATE"),
                "hash":         parts.get("H"),
                "version":      parts.get("RIVA"),
                "valid_format": True
            }
        except Exception:
            return {"valid_format": False}

    # ── IS EXPIRED ───────────────────────────────
    def is_expired(self, rx: dict) -> bool:
        return datetime.now().date() > datetime.fromisoformat(
            rx["valid_until"]
        ).date()

    # ── GET STATUS ───────────────────────────────
    def get_status(self, rx: dict) -> dict:
        return {
            "rx_id":    rx["rx_id"],
            "valid":    self.verify(rx),
            "expired":  self.is_expired(rx),
            "warnings": rx.get("warnings", []),
            "status":   "expired" if self.is_expired(rx) else rx.get("status","active")
        }
