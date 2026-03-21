"""
encryption_handler.py
=====================
RIVA Health Platform v4.1 — AES-256-GCM Encryption Handler
-----------------------------------------------------------
المسار: ai-core/security/encryption_handler.py

التحسينات على v4.0:
    1. ربط صريح بالـ 13 قاعدة بيانات الموجودة في data/databases/
    2. read_all_databases()  — يقرأ كل الـ 13 ملف دفعة واحدة
    3. write_all_databases() — يكتب الكل في عملية واحدة
    4. integrity_check()     — يتحقق من سلامة كل الملفات
    5. ربط مع clinical_override_log — يقرأ override records
    6. ربط مع prescription_gen audit — يقرأ audit trail
    7. database_stats()      — إحصائيات سريعة للـ dashboard

الخوارزمية: AES-256-GCM
    - Key    : 32 bytes (256-bit)
    - Nonce  : 12 bytes (96-bit) — عشوائي لكل عملية تشفير
    - Tag    : 16 bytes — مدمج في ciphertext تلقائياً
    - Format : nonce(12) + ciphertext+tag

Author: GODA EMAD · Harvard HSIL Hackathon 2026
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.security.encryption_handler")

# ─── Path resolution ─────────────────────────────────────────────────────────

_HERE     = Path(__file__).resolve().parent   # ai-core/security/
_AICORE   = _HERE.parent                      # ai-core/
_ROOT     = _AICORE.parent                    # project-root/
_DB_DIR   = _ROOT / "data" / "databases"
_KEY_FILE = _DB_DIR / "keys" / "riva_master.key"

# ─── الـ 13 قاعدة بيانات الموجودة في data/databases/ ─────────────────────────

ALL_DATABASES: dict[str, dict] = {
    "patients.encrypted": {
        "description": "بيانات المرضى الأساسية",
        "owner":       "db_loader.py",
        "critical":    True,
    },
    "pregnancy.encrypted": {
        "description": "بيانات الحوامل ومخاطر الحمل",
        "owner":       "pregnancy_risk.py",
        "critical":    True,
    },
    "medications.encrypted": {
        "description": "أدوية المرضى الحالية",
        "owner":       "drug_interaction.py",
        "critical":    True,
    },
    "lab_results.encrypted": {
        "description": "نتائج التحاليل المخبرية",
        "owner":       "triage_engine.py",
        "critical":    True,
    },
    "prescriptions.encrypted": {
        "description": "الروشتات الطبية",
        "owner":       "prescription_gen.py",
        "critical":    True,
    },
    "family_links.encrypted": {
        "description": "روابط الأسرة والمرافقين",
        "owner":       "db_loader.py",
        "critical":    False,
    },
    "predictions.encrypted": {
        "description": "توقعات الـ AI المحفوظة",
        "owner":       "unified_predictor.py",
        "critical":    True,
    },
    "readmission_risk.encrypted": {
        "description": "مخاطر إعادة الإدخال",
        "owner":       "readmission_predictor.py",
        "critical":    True,
    },
    "school.encrypted": {
        "description": "بيانات صحة الطلاب",
        "owner":       "school_health.py",
        "critical":    False,
    },
    "school_links.encrypted": {
        "description": "روابط الطلاب بالمدارس",
        "owner":       "school_health.py",
        "critical":    False,
    },
    "hospital.encrypted": {
        "description": "بيانات المستشفيات والعيادات",
        "owner":       "db_loader.py",
        "critical":    False,
    },
    "feedback.encrypted": {
        "description": "تقييمات الأطباء على الـ AI",
        "owner":       "doctor_feedback_handler.py",
        "critical":    False,
    },
    "sentiment_log.encrypted": {
        "description": "سجل تحليل مشاعر المرضى",
        "owner":       "chat.py",
        "critical":    False,
    },
}


def _resolve_key_path() -> Path:
    env_path = os.environ.get("RIVA_KEY_PATH")
    if env_path:
        return Path(env_path)
    return _KEY_FILE


# ─── Core handler ─────────────────────────────────────────────────────────────

class EncryptionHandler:
    """
    AES-256-GCM encryption/decryption for all 13 RIVA databases.

    Integrated with:
        - db_loader.py            : read_db / write_db
        - prescription_gen.py     : prescriptions.encrypted
        - doctor_feedback_handler : feedback.encrypted
        - readmission_predictor   : readmission_risk.encrypted
        - unified_predictor       : predictions.encrypted
        - school_health.py        : school.encrypted
        - 17_sustainability.html  : database_stats()
    """

    def __init__(self, key_path: Optional[Path] = None) -> None:
        self._key: Optional[bytes] = None
        self._load_key(key_path or _resolve_key_path())

    # ── Key loading ───────────────────────────────────────────────────────────

    def _load_key(self, key_path: Path) -> None:
        env_b64 = os.environ.get("RIVA_MASTER_KEY")
        if env_b64:
            try:
                self._key = base64.b64decode(env_b64)
                log.info("[EncryptionHandler] key loaded from RIVA_MASTER_KEY")
                return
            except Exception as exc:
                log.warning("[EncryptionHandler] env key decode failed: %s", exc)

        if key_path.exists():
            self._key = key_path.read_bytes()
            log.info("[EncryptionHandler] key loaded from %s", key_path)
            return

        log.error(
            "[EncryptionHandler] key not found. "
            "Run seed_all_databases.py on Colab to generate keys."
        )

    def is_ready(self) -> bool:
        return self._key is not None and len(self._key) == 32

    # ── Core encrypt / decrypt ────────────────────────────────────────────────

    def encrypt(self, raw: bytes) -> bytes:
        if not self.is_ready():
            raise RuntimeError("EncryptionHandler: key not loaded")
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = secrets.token_bytes(12)
        ct    = AESGCM(self._key).encrypt(nonce, raw, None)
        return nonce + ct

    def decrypt(self, data: bytes) -> bytes:
        if not self.is_ready():
            raise RuntimeError("EncryptionHandler: key not loaded")
        if len(data) < 28:
            raise ValueError(f"Data too short ({len(data)} bytes)")
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce, ct = data[:12], data[12:]
        return AESGCM(self._key).decrypt(nonce, ct, None)

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def encrypt_json(self, data: list | dict) -> bytes:
        return self.encrypt(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def decrypt_json(self, data: bytes) -> list | dict:
        return json.loads(self.decrypt(data).decode("utf-8"))

    # ── File helpers ──────────────────────────────────────────────────────────

    def encrypt_file(self, src: Path, dst: Optional[Path] = None) -> Path:
        dst = dst or src.with_suffix(".encrypted")
        dst.write_bytes(self.encrypt(src.read_bytes()))
        log.info("[EncryptionHandler] encrypted %s → %s", src.name, dst.name)
        return dst

    def decrypt_file(self, src: Path, dst: Optional[Path] = None) -> Path:
        dst = dst or src.with_suffix(".json")
        dst.write_bytes(self.decrypt(src.read_bytes()))
        log.info("[EncryptionHandler] decrypted %s → %s", src.name, dst.name)
        return dst

    # ── Single DB read/write ──────────────────────────────────────────────────

    def read_db(self, db_name: str) -> list | dict:
        """
        Read and decrypt one database.
        Returns [] gracefully if file not found (offline degradation).

        Usage:
            patients = handler.read_db("patients.encrypted")
        """
        path = _DB_DIR / db_name
        if not path.exists():
            log.info("[EncryptionHandler] %s not found — returning []", db_name)
            return []
        try:
            return self.decrypt_json(path.read_bytes())
        except Exception as exc:
            log.error("[EncryptionHandler] read_db(%s) failed: %s", db_name, exc)
            return []

    def write_db(self, db_name: str, data: list | dict) -> bool:
        """
        Encrypt and write one database.

        Usage:
            handler.write_db("predictions.encrypted", predictions_list)
        """
        path = _DB_DIR / db_name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.encrypt_json(data))
            count = len(data) if isinstance(data, list) else 1
            log.info("[EncryptionHandler] write_db(%s) — %d records", db_name, count)
            return True
        except Exception as exc:
            log.error("[EncryptionHandler] write_db(%s) failed: %s", db_name, exc)
            return False

    # ── Bulk read/write (all 13 databases) ────────────────────────────────────

    def read_all_databases(
        self,
        critical_only: bool = False,
    ) -> dict[str, list | dict]:
        """
        Reads all 13 encrypted databases at once.

        Args:
            critical_only: if True, only reads critical=True databases

        Returns:
            {
                "patients":         [...],
                "pregnancy":        [...],
                "medications":      [...],
                ...
            }

        Usage in db_loader.py:
            all_data = handler.read_all_databases()
            patients = all_data["patients"]
        """
        result = {}
        for db_name, meta in ALL_DATABASES.items():
            if critical_only and not meta["critical"]:
                continue
            key = db_name.replace(".encrypted", "")
            result[key] = self.read_db(db_name)

        log.info(
            "[EncryptionHandler] read_all_databases  loaded=%d  records=%d",
            len(result),
            sum(len(v) if isinstance(v, list) else 1 for v in result.values()),
        )
        return result

    def write_all_databases(self, data: dict[str, list | dict]) -> dict[str, bool]:
        """
        Writes multiple databases at once.

        Args:
            data: {"patients": [...], "medications": [...], ...}
                  keys can be with or without .encrypted extension

        Returns:
            {"patients": True, "medications": True, ...}
        """
        results = {}
        for key, records in data.items():
            db_name = key if key.endswith(".encrypted") else f"{key}.encrypted"
            results[key] = self.write_db(db_name, records)

        success = sum(1 for v in results.values() if v)
        log.info(
            "[EncryptionHandler] write_all_databases  %d/%d succeeded",
            success, len(results),
        )
        return results

    # ── Integrity check ───────────────────────────────────────────────────────

    def integrity_check(self) -> dict:
        """
        Verifies all 13 database files can be decrypted successfully.

        Returns:
            {
                "total":   13,
                "ok":      12,
                "failed":  1,
                "missing": 0,
                "details": {"patients.encrypted": "ok", ...}
                "timestamp": "..."
            }

        Usage in 17_sustainability.html → GET /security/integrity
        """
        details  = {}
        ok       = 0
        failed   = 0
        missing  = 0

        for db_name in ALL_DATABASES:
            path = _DB_DIR / db_name
            if not path.exists():
                details[db_name] = "missing"
                missing += 1
                continue
            try:
                data = self.decrypt_json(path.read_bytes())
                count = len(data) if isinstance(data, list) else 1
                details[db_name] = f"ok ({count} records)"
                ok += 1
            except Exception as e:
                details[db_name] = f"FAILED: {e}"
                failed += 1
                log.error("[EncryptionHandler] integrity FAILED: %s — %s", db_name, e)

        result = {
            "total":     len(ALL_DATABASES),
            "ok":        ok,
            "failed":    failed,
            "missing":   missing,
            "status":    "clean" if failed == 0 else "COMPROMISED",
            "details":   details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "[EncryptionHandler] integrity check: ok=%d  failed=%d  missing=%d",
            ok, failed, missing,
        )
        return result

    # ── Database stats ────────────────────────────────────────────────────────

    def database_stats(self) -> dict:
        """
        Returns record counts for all databases.
        Used in 17_sustainability.html and doctor dashboard.

        Returns:
            {
                "patients":       5,
                "pregnancy":      3,
                "medications":    6,
                ...
                "total_records":  47,
                "total_size_kb":  14.2,
            }
        """
        stats = {}
        total_records = 0
        total_size    = 0

        for db_name, meta in ALL_DATABASES.items():
            key  = db_name.replace(".encrypted", "")
            path = _DB_DIR / db_name

            if not path.exists():
                stats[key] = {"records": 0, "size_bytes": 0, "status": "missing"}
                continue

            size = path.stat().st_size
            total_size += size

            try:
                data    = self.decrypt_json(path.read_bytes())
                count   = len(data) if isinstance(data, list) else 1
                total_records += count
                stats[key] = {
                    "records":     count,
                    "size_bytes":  size,
                    "description": meta["description"],
                    "status":      "ok",
                }
            except Exception:
                stats[key] = {
                    "records":    0,
                    "size_bytes": size,
                    "status":     "decrypt_failed",
                }

        return {
            "databases":      stats,
            "total_records":  total_records,
            "total_size_kb":  round(total_size / 1024, 2),
            "total_files":    len(ALL_DATABASES),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }

    # ── Append to DB (used by feedback + predictions) ─────────────────────────

    def append_to_db(self, db_name: str, record: dict) -> bool:
        """
        Appends one record to an existing database.
        Read → append → write back.

        Usage in doctor_feedback_handler.py (Vercel-safe fallback):
            handler.append_to_db("feedback.encrypted", feedback_record)
        """
        existing = self.read_db(db_name)
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(record)
        return self.write_db(db_name, existing)

    @property
    def status(self) -> dict:
        return {
            "ready":          self.is_ready(),
            "algorithm":      "AES-256-GCM",
            "key_size_bits":  len(self._key) * 8 if self._key else 0,
            "key_source":     "env_var" if os.environ.get("RIVA_MASTER_KEY") else "file",
            "db_dir":         str(_DB_DIR),
            "databases_count":len(ALL_DATABASES),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_handler: Optional[EncryptionHandler] = None


def get_encryption_handler() -> EncryptionHandler:
    """
    Singleton — used by db_loader.py and all RIVA modules.

    Usage:
        from ai_core.security.encryption_handler import get_encryption_handler

        handler  = get_encryption_handler()
        patients = handler.read_db("patients.encrypted")
        all_data = handler.read_all_databases()
        stats    = handler.database_stats()
    """
    global _handler
    if _handler is None:
        _handler = EncryptionHandler()
    return _handler
