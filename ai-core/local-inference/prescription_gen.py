"""
prescription_gen.py
===================
RIVA Health Platform — Prescription Generator v4.0
----------------------------------------------------
يولّد روشتات طبية آمنة مع التحقق من التعارضات الدوائية.

التحسينات على v3.0:
    1. ربط مع drug_interaction.py  — smart_normalize + pregnancy check
    2. ربط مع data_compressor.py   — QR payload مضغوط < 800 bytes
    3. ربط مع clinical_profile     — تحذيرات إضافية للحوامل والسكريين
    4. PII protection              — patient_id مش بيتحفظ خام
    5. Salted signature            — PATIENT_ID_SALT من .env
    6. Arabic medication format    — اسم + جرعة + مدة بالعربي للأم
    7. Hallucination guard hook    — يتحقق من أي دواء يقترحه الـ chat

Author : GODA EMAD
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.local_inference.prescription_gen")

# ─── Paths ───────────────────────────────────────────────────────────────────

_BASE           = Path(__file__).parent.parent
_CONFLICTS_PATH = _BASE / "business-intelligence/medical-content/drug_conflicts.json"
_MED_RISK_MAP   = _BASE / "business-intelligence/medical-content/medication_risk_mapping.json"

# ─── Security ────────────────────────────────────────────────────────────────

_SALT = os.getenv("PATIENT_ID_SALT", "riva-default-salt-change-in-prod")


def _anonymize(patient_id: str) -> str:
    """Salted SHA-256 — matches clinical_override_log.py."""
    return hashlib.sha256(f"{_SALT}:{patient_id}".encode()).hexdigest()[:16]


# ─── Default interactions ─────────────────────────────────────────────────────

# ─── Interaction alternatives (مراجعة طبية — RIVA Team) ─────────────────────
# لكل تعارض خطير، بديل آمن موصى به طبياً
# يظهر في warnings بجانب التحذير ليساعد الدكتور مباشرة

_INTERACTION_ALTERNATIVES: dict[str, str] = {
    "warfarin+aspirin":       "استخدم clopidogrel بجرعة أقل بإشراف طبي",
    "warfarin+ibuprofen":     "استبدل ibuprofen بـ paracetamol بجرعة 500mg",
    "warfarin+paracetamol":   "قلل جرعة paracetamol لأقل من 2g/يوم وراقب INR",
    "ibuprofen+captopril":    "استبدل ibuprofen بـ paracetamol لمرضى الضغط",
    "digoxin+amiodarone":     "قلل جرعة digoxin 50% وراقب مستواه في الدم",
    "simvastatin+amlodipine": "حول simvastatin لـ rosuvastatin 10mg",
    "ciprofloxacin+warfarin": "استخدم amoxicillin بديلاً وراقب INR يومياً",
    "metformin+contrast":     "أوقف metformin 48 ساعة قبل الصبغة وبعدها",
    "insulin+alcohol":        "تجنب الكحول كلياً مع الإنسولين — هبوط سكر مفاجئ",
}


_DEFAULT_RISKY_PAIRS = [
    ["warfarin",    "aspirin"],
    ["warfarin",    "ibuprofen"],
    ["warfarin",    "paracetamol"],
    ["metformin",   "alcohol"],
    ["ibuprofen",   "captopril"],
    ["digoxin",     "amiodarone"],
    ["simvastatin", "amlodipine"],
    ["ciprofloxacin","warfarin"],
    ["metformin",   "contrast"],
    ["insulin",     "alcohol"],
]


# ─── Medication format helper ─────────────────────────────────────────────────

def _format_med_ar(med: dict) -> str:
    """
    Formats medication as Arabic string for the patient.
    Example: "باراسيتامول 500mg — حبة كل 8 ساعات لمدة 5 أيام"
    """
    name  = med.get("name", "—")
    dose  = med.get("dose", "")
    freq  = med.get("frequency", "")
    days  = med.get("days", "")
    route = med.get("route", "")

    parts = [name]
    if dose:
        parts.append(dose)
    if route:
        parts.append(f"({route})")
    if freq:
        parts.append(f"— {freq}")
    if days:
        parts.append(f"لمدة {days} يوم")
    return " ".join(parts)


# ─── Core generator ───────────────────────────────────────────────────────────

class PrescriptionGenerator:
    """
    Generates, signs, verifies, and compresses RIVA prescriptions.

    Integration points:
        - drug_interaction.py : smart normalization + pregnancy safety
        - data_compressor.py  : QR payload compression
        - clinical_profile    : extra warnings for pregnant/diabetic patients
        - chat.py guardrail   : verify any AI-suggested drug before prescribing
    """

    VERSION = "RIVA_RX_v4"

    def __init__(
        self,
        conflicts_path: Path = _CONFLICTS_PATH,
    ):
        self.risky_pairs = _DEFAULT_RISKY_PAIRS.copy()
        self._load_interactions(conflicts_path)
        log.info("[PrescriptionGen] v4.0 ready  pairs=%d", len(self.risky_pairs))

    # ── Load interactions ─────────────────────────────────────────────────────

    def _load_interactions(self, path: Path) -> None:
        if not path.exists():
            log.warning("[PrescriptionGen] conflicts file not found: %s", path)
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            conflicts = data.get("conflicts", [])
            self.risky_pairs = [
                [c["drug_a"], c["drug_b"]]
                for c in conflicts
                if c.get("severity") == "high"
            ]
            log.info("[PrescriptionGen] loaded %d high-risk pairs", len(self.risky_pairs))
        except Exception as e:
            log.warning("[PrescriptionGen] failed to load conflicts: %s", e)

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate(
        self,
        patient_id:       str,
        doctor_id:        str,
        diagnosis:        str,
        medications:      list[dict],
        notes:            str  = "",
        clinical_profile: dict = {},
        session_id:       Optional[str] = None,
    ) -> dict:
        """
        Generates a signed prescription with interaction checks.

        Args:
            patient_id       : raw patient ID (will be anonymised)
            doctor_id        : doctor identifier
            diagnosis        : diagnosis text
            medications      : list of {name, dose, frequency, days, route}
            notes            : additional notes
            clinical_profile : session metadata (is_pregnant, has_diabetes...)
            session_id       : for audit trail

        Returns:
            Full prescription dict with QR payload and warnings.
        """
        rx_id           = str(uuid.uuid4())[:12].upper()
        timestamp       = datetime.now(timezone.utc).isoformat()
        issued_date     = timestamp[:10]
        patient_id_hash = _anonymize(patient_id)

        # 1. Interaction check via drug_interaction.py (smart normalize)
        warnings        = self._check_interactions_smart(medications, clinical_profile)

        # 2. Arabic medication list for patient
        meds_ar = [_format_med_ar(m) for m in medications]

        rx = {
            "version":          self.VERSION,
            "rx_id":            rx_id,
            "patient_id_hash":  patient_id_hash,   # PII-safe
            "doctor_id":        doctor_id,
            "diagnosis":        diagnosis,
            "medications":      medications,
            "medications_ar":   meds_ar,
            "notes":            notes,
            "warnings":         warnings,
            "issued_at":        timestamp,
            "valid_until":      self._valid_until(medications),
            "status":           "active",
            "session_id":       session_id,
            "signature":        self._sign(rx_id, patient_id_hash, doctor_id, issued_date),
        }

        rx["hash"]       = self._rx_hash(rx)
        rx["qr_payload"] = self._build_qr_payload(rx)

        log.info(
            "[PrescriptionGen] RX=%s  patient=%s  meds=%d  warnings=%d",
            rx_id, patient_id_hash, len(medications), len(warnings),
        )

        if warnings:
            log.warning("[PrescriptionGen] WARNINGS: %s", warnings)

        # Store in audit trail for pattern analysis (17_sustainability.html)
        _store_audit(rx)

        return rx

    # ── Interaction check (smart) ─────────────────────────────────────────────

    def _check_interactions_smart(
        self,
        medications:      list[dict],
        clinical_profile: dict,
    ) -> list[str]:
        """
        Uses drug_interaction.py for smart normalization + pregnancy check.
        Falls back to local risky_pairs if drug_interaction unavailable.
        """
        warnings: list[str] = []
        med_names = [m.get("name", "") for m in medications if m.get("name")]

        # Try smart check via drug_interaction module
        try:
            from .drug_interaction import check_interaction, smart_normalize

            for i, drug in enumerate(med_names):
                others = med_names[:i] + med_names[i+1:]
                if not others:
                    continue
                result = check_interaction(
                    new_drug         = drug,
                    current_drugs    = others,
                    clinical_profile = clinical_profile,
                )
                for alert in result.get("alerts", []):
                    msg = (
                        f"تعارض دوائي: {alert['drug_a']} + {alert['drug_b']} "
                        f"| {alert['severity']} | {alert['effect_ar']}"
                    )
                    if msg not in warnings:
                        warnings.append(msg)

                # Pregnancy alert
                if result.get("pregnancy_alert"):
                    pa  = result["pregnancy_alert"]
                    msg = f"تحذير حمل: {drug} — {pa['effect_ar']}"
                    if msg not in warnings:
                        warnings.append(msg)

                # Food warnings
                for fw in result.get("food_warnings", []):
                    msg = f"تحذير غذائي: {drug} + {fw['food']} — {fw['effect_ar']}"
                    if msg not in warnings:
                        warnings.append(msg)

        except ImportError:
            # Fallback to local pairs
            warnings = self._check_interactions_local(medications)

        # Extra warnings from clinical profile
        warnings += self._clinical_warnings(med_names, clinical_profile)

        return warnings

    def _check_interactions_local(self, medications: list[dict]) -> list[str]:
        """Local fallback — uses simple string matching."""
        names = {
            m.get("name", "").lower().strip()
            for m in medications
            if m.get("name")
        }
        warnings = []
        for pair in self.risky_pairs:
            a, b = pair[0].lower(), pair[1].lower()
            if a in names and b in names:
                alt = _INTERACTION_ALTERNATIVES.get(f"{a}+{b}") or                       _INTERACTION_ALTERNATIVES.get(f"{b}+{a}", "")
                msg = f"تعارض دوائي: {a} + {b}"
                if alt:
                    msg += f" | البديل: {alt}"
                if msg not in warnings:
                    warnings.append(msg)
        return warnings

    def _clinical_warnings(
        self, med_names: list[str], profile: dict
    ) -> list[str]:
        """Extra warnings based on patient clinical profile."""
        warnings = []
        names_lower = {n.lower() for n in med_names}

        if profile.get("is_pregnant"):
            nsaids = {"ibuprofen", "diclofenac", "aspirin", "naproxen"}
            found  = nsaids & names_lower
            if found:
                warnings.append(
                    f"⚠️ تحذير للحامل: {', '.join(found)} — استشيري الطبيب قبل الاستخدام"
                )

        if profile.get("has_diabetes"):
            steroids = {"cortisone", "dexamethasone", "prednisolone"}
            found    = steroids & names_lower
            if found:
                warnings.append(
                    f"⚠️ تحذير لمريض السكر: {', '.join(found)} قد ترفع السكر — راقب المستوى"
                )

        if profile.get("has_kidney_disease"):
            nephrotoxic = {"ibuprofen", "naproxen", "gentamicin"}
            found       = nephrotoxic & names_lower
            if found:
                warnings.append(
                    f"⚠️ تحذير لمريض الكلى: {', '.join(found)} — قد تضر الكلى"
                )

        return warnings

    # ── QR payload (compressed) ───────────────────────────────────────────────

    def _build_qr_payload(self, rx: dict) -> str:
        """
        Builds a compressed QR payload using data_compressor.py.
        Falls back to plain text if compressor unavailable.

        Target: < 800 bytes so QR scans reliably in rural clinics.
        """
        qr_data = {
            "v":    self.VERSION,
            "rx":   rx["rx_id"],
            "p":    rx["patient_id_hash"],
            "d":    rx["doctor_id"],
            "date": rx["issued_at"][:10],
            "h":    rx.get("hash", "")[:12],
            "meds": [m.get("name","") for m in rx.get("medications",[])],
        }

        try:
            from .data_compressor import compress_for_qr
            result = compress_for_qr(qr_data, strict=False)
            import base64
            return base64.b64encode(result.data).decode()
        except Exception:
            # Plain text fallback
            return (
                f"RIVA:{self.VERSION}|"
                f"RX:{rx['rx_id']}|"
                f"P:{rx['patient_id_hash']}|"
                f"D:{rx['doctor_id']}|"
                f"DATE:{rx['issued_at'][:10]}|"
                f"H:{rx.get('hash','')[:10]}"
            )

    # ── Signature ─────────────────────────────────────────────────────────────

    def _sign(
        self,
        rx_id:           str,
        patient_id_hash: str,
        doctor_id:       str,
        issued_date:     str,
    ) -> str:
        """Salted signature — uses PATIENT_ID_SALT from .env."""
        data = f"{_SALT}:{rx_id}:{patient_id_hash}:{doctor_id}:{issued_date}"
        return hashlib.sha256(data.encode()).hexdigest()[:16].upper()

    def _rx_hash(self, rx: dict) -> str:
        """Chain-of-trust hash — excludes mutable fields."""
        stable = {k: v for k, v in rx.items() if k not in ("hash", "qr_payload")}
        payload = json.dumps(stable, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    # ── Verify ────────────────────────────────────────────────────────────────

    def verify(self, rx: dict) -> bool:
        """Verifies signature + hash integrity (constant-time comparison)."""
        try:
            expected_sig = self._sign(
                rx["rx_id"],
                rx["patient_id_hash"],
                rx["doctor_id"],
                rx["issued_at"][:10],
            )
            sig_ok  = hmac.compare_digest(rx.get("signature", ""), expected_sig)
            hash_ok = hmac.compare_digest(rx.get("hash", ""), self._rx_hash(rx))
            return sig_ok and hash_ok
        except Exception:
            return False

    def verify_qr(self, qr_payload: str) -> dict:
        """
        Verifies QR payload for pharmacies.
        Tries compressed format first, falls back to plain text.
        """
        # Try compressed
        try:
            import base64
            from .data_compressor import decompress_from_qr
            data = decompress_from_qr(base64.b64decode(qr_payload))
            return {
                "rx_id":       data.get("rx"),
                "patient_hash":data.get("p"),
                "doctor_id":   data.get("d"),
                "issued_date": data.get("date"),
                "hash":        data.get("h"),
                "version":     data.get("v"),
                "valid_format":True,
                "compressed":  True,
            }
        except Exception:
            pass

        # Plain text fallback
        try:
            parts = dict(
                p.split(":", 1) for p in qr_payload.split("|") if ":" in p
            )
            return {
                "rx_id":       parts.get("RX"),
                "patient_hash":parts.get("P"),
                "doctor_id":   parts.get("D"),
                "issued_date": parts.get("DATE"),
                "hash":        parts.get("H"),
                "version":     parts.get("RIVA"),
                "valid_format":True,
                "compressed":  False,
            }
        except Exception:
            return {"valid_format": False}

    # ── Hallucination guard hook ──────────────────────────────────────────────

    def verify_ai_suggestion(
        self,
        suggested_drug:   str,
        current_meds:     list[str],
        clinical_profile: dict = {},
    ) -> dict:
        """
        Called by chat.py hallucination guard before including any drug in response.

        If the AI suggests a drug that has dangerous interactions or is
        contraindicated for this patient, returns is_safe=False + warning.

        Usage in chat.py _apply_guardrail():
            from .prescription_gen import generator
            check = generator.verify_ai_suggestion(
                suggested_drug   = extracted_drug_name,
                current_meds     = session["metadata"]["current_medications"],
                clinical_profile = session["metadata"],
            )
            if not check["is_safe"]:
                response += f"\\n⚕️ {check['warning']}"
        """
        try:
            from .drug_interaction import check_interaction
            result = check_interaction(
                new_drug         = suggested_drug,
                current_drugs    = current_meds,
                clinical_profile = clinical_profile,
            )
            is_safe = result["is_safe"]
            warning = (
                f"تحذير: {suggested_drug} قد يتعارض مع أدوية المريض الحالية. "
                "لا يتم صرف هذا الدواء إلا بروشتة طبيب مختص بعد الكشف السريري."
                if not is_safe else ""
            )
            return {
                "is_safe":         is_safe,
                "warning":         warning,
                "confidence_impact":result.get("confidence_impact", 0.0),
                "alerts":          result.get("alerts", []),
            }
        except Exception as e:
            log.warning("[PrescriptionGen] AI suggestion check failed: %s", e)
            return {
                "is_safe":         True,
                "warning":         "",
                "confidence_impact":0.0,
                "alerts":          [],
            }

    # ── Status helpers ────────────────────────────────────────────────────────

    def _valid_until(self, medications: list[dict]) -> str:
        max_days = max((m.get("days", 7) for m in medications), default=7)
        valid    = datetime.now(timezone.utc) + timedelta(days=max_days + 3)
        return valid.strftime("%Y-%m-%d")

    def is_expired(self, rx: dict) -> bool:
        return datetime.now(timezone.utc).date() > datetime.fromisoformat(
            rx["valid_until"]
        ).date()

    def is_unfilled_alert(self, rx: dict) -> bool:
        """
        Returns True if the prescription was issued 48+ hours ago
        and status is still "active" (not dispensed).

        Triggers a patient reminder in 04_result.html and background-sync.js:
            if (rx.is_unfilled_alert) {
                sendReminder(rx.patient_id_hash, "روشتتك لسه ما اتصرفتش — روح الصيدلية!")
            }
        """
        if rx.get("status") in ("dispensed", "expired", "cancelled"):
            return False
        try:
            issued = datetime.fromisoformat(rx["issued_at"])
            hours_passed = (datetime.now(timezone.utc) - issued).total_seconds() / 3600
            return hours_passed >= 48
        except Exception:
            return False

    def mark_dispensed(self, rx: dict) -> dict:
        """Marks prescription as dispensed — called by pharmacy QR scan."""
        rx["status"]       = "dispensed"
        rx["dispensed_at"] = datetime.now(timezone.utc).isoformat()
        rx["hash"]         = self._rx_hash(rx)
        log.info("[PrescriptionGen] RX=%s marked dispensed", rx.get("rx_id"))
        return rx

    def get_status(self, rx: dict) -> dict:
        return {
            "rx_id":    rx["rx_id"],
            "valid":    self.verify(rx),
            "expired":  self.is_expired(rx),
            "warnings": rx.get("warnings", []),
            "status":   "expired" if self.is_expired(rx) else rx.get("status", "active"),
            "meds_ar":  rx.get("medications_ar", []),
        }


# ─── Audit trail ─────────────────────────────────────────────────────────────

def _store_audit(rx: dict) -> None:
    """
    Stores prescription hash in SQLite for Prescription Pattern Analysis.
    Used in 17_sustainability.html to show prescribing trends.

    Table: prescription_audit
        rx_id, patient_hash, doctor_id, diagnosis, med_count,
        warning_count, issued_at, hash, status
    """
    try:
        import sqlite3
        db_path = _BASE / "data/databases/prescriptions_audit.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prescription_audit (
                rx_id         TEXT PRIMARY KEY,
                patient_hash  TEXT,
                doctor_id     TEXT,
                diagnosis     TEXT,
                med_count     INTEGER,
                warning_count INTEGER,
                issued_at     TEXT,
                hash          TEXT,
                status        TEXT
            )
        """)
        cur.execute("""
            INSERT OR REPLACE INTO prescription_audit VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rx["rx_id"],
            rx["patient_id_hash"],
            rx["doctor_id"],
            rx["diagnosis"],
            len(rx.get("medications", [])),
            len(rx.get("warnings", [])),
            rx["issued_at"],
            rx["hash"],
            rx.get("status", "active"),
        ))
        con.commit()
        con.close()
        log.info("[PrescriptionGen] audit stored: %s", rx["rx_id"])
    except Exception as e:
        log.warning("[PrescriptionGen] audit store failed: %s", e)


def get_prescription_analytics() -> dict:
    """
    Returns prescribing pattern analytics for 17_sustainability.html.
    Shows most prescribed drugs, warning frequency, and doctor patterns.
    """
    try:
        import sqlite3
        db_path = _BASE / "data/databases/prescriptions_audit.db"
        if not db_path.exists():
            return {"total": 0, "message": "لا يوجد بيانات بعد"}

        con = sqlite3.connect(str(db_path))
        cur = con.cursor()

        total         = cur.execute("SELECT COUNT(*) FROM prescription_audit").fetchone()[0]
        with_warnings = cur.execute(
            "SELECT COUNT(*) FROM prescription_audit WHERE warning_count > 0"
        ).fetchone()[0]
        top_diagnoses = cur.execute(
            "SELECT diagnosis, COUNT(*) as n FROM prescription_audit "
            "GROUP BY diagnosis ORDER BY n DESC LIMIT 5"
        ).fetchall()
        dispensed = cur.execute(
            "SELECT COUNT(*) FROM prescription_audit WHERE status='dispensed'"
        ).fetchone()[0]

        con.close()

        return {
            "total_prescriptions":   total,
            "with_warnings":         with_warnings,
            "warning_rate_pct":      round(with_warnings / max(total, 1) * 100, 1),
            "dispensed":             dispensed,
            "dispensed_rate_pct":    round(dispensed / max(total, 1) * 100, 1),
            "top_diagnoses":         [{"diagnosis": r[0], "count": r[1]} for r in top_diagnoses],
        }
    except Exception as e:
        log.warning("[PrescriptionGen] analytics failed: %s", e)
        return {"error": str(e)}


# ─── Singleton ────────────────────────────────────────────────────────────────

generator = PrescriptionGenerator()


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_prescription(
    patient_id:       str,
    doctor_id:        str,
    diagnosis:        str,
    medications:      list[dict],
    notes:            str  = "",
    clinical_profile: dict = {},
    session_id:       Optional[str] = None,
) -> dict:
    """
    Main entry point.

    Usage in 09_doctor_dashboard.html → POST /prescriptions:
        result = generate_prescription(
            patient_id       = "PT-4892",
            doctor_id        = "DR-001",
            diagnosis        = "ارتفاع ضغط الدم",
            medications      = [
                {"name": "amlodipine", "dose": "5mg",
                 "frequency": "مرة يومياً", "days": 30},
            ],
            clinical_profile = session["metadata"],
            session_id       = session_id,
        )
        # QR جاهز في result["qr_payload"]
        # تحذيرات في result["warnings"]
        # روشتة بالعربي في result["medications_ar"]
    """
    return generator.generate(
        patient_id       = patient_id,
        doctor_id        = doctor_id,
        diagnosis        = diagnosis,
        medications      = medications,
        notes            = notes,
        clinical_profile = clinical_profile,
        session_id       = session_id,
    )


def verify_prescription(rx: dict) -> bool:
    return generator.verify(rx)


def verify_qr(qr_payload: str) -> dict:
    return generator.verify_qr(qr_payload)


def check_ai_drug(
    drug:             str,
    current_meds:     list[str],
    clinical_profile: dict = {},
) -> dict:
    """Hallucination guard — verifies any AI-suggested drug is safe."""
    return generator.verify_ai_suggestion(drug, current_meds, clinical_profile)


def is_unfilled_alert(rx: dict) -> bool:
    """Returns True if prescription unfilled after 48 hours."""
    return generator.is_unfilled_alert(rx)


def mark_dispensed(rx: dict) -> dict:
    """Marks prescription as dispensed by pharmacy."""
    dispensed = generator.mark_dispensed(rx)
    _store_audit(dispensed)   # update audit record
    return dispensed


def get_analytics() -> dict:
    """Prescription pattern analytics for 17_sustainability.html."""
    return get_prescription_analytics()
