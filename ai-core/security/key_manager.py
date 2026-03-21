"""
key_manager.py
==============
RIVA Health Platform — Cryptographic Key Manager
-------------------------------------------------
المسار: ai-core/security/key_manager.py

المهمة:
    - إدارة دورة حياة المفاتيح الكاملة (generate → store → rotate → revoke)
    - يُنسّق بين encryption_handler.py و digital_signature.py
    - يضمن إن riva_master.key محفوظ بأمان ومش بيتسرّب

المفاتيح اللي بيديرها:
    1. riva_master.key     — AES-256-GCM لتشفير قواعد البيانات الـ 13
    2. {doctor_id}.privkey — Ed25519 للتوقيع الرقمي
    3. {doctor_id}.pubkey  — Ed25519 للتحقق (مشترك)
    4. RIVA_LOG_SALT       — HMAC salt لحماية PII في الـ logs

Key rotation policy:
    - riva_master.key: كل 90 يوم
    - doctor keys    : كل 365 يوم أو عند انتهاء عقد الدكتور

الربط مع المنظومة:
    - encryption_handler.py  : يستخدم riva_master.key
    - digital_signature.py   : يستخدم doctor keypairs
    - access_control.py      : ADMIN فقط يدير المفاتيح
    - 17_sustainability.html : يعرض key status

Author: GODA EMAD · Harvard HSIL Hackathon 2026
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.security.key_manager")

# ─── Paths ───────────────────────────────────────────────────────────────────

_HERE     = Path(__file__).resolve().parent
_AICORE   = _HERE.parent
_ROOT     = _AICORE.parent
_KEYS_DIR = _ROOT / "data" / "databases" / "keys"
_META_FILE= _KEYS_DIR / "key_metadata.json"

# ─── Rotation policy ─────────────────────────────────────────────────────────

MASTER_KEY_ROTATION_DAYS  = 90
DOCTOR_KEY_ROTATION_DAYS  = 365
KEY_WARNING_DAYS          = 14   # warn X days before expiry


# ─── Key metadata ─────────────────────────────────────────────────────────────

@dataclass
class KeyMetadata:
    key_id:      str
    key_type:    str          # "master" | "doctor_private" | "doctor_public" | "salt"
    owner_id:    str          # doctor_id or "system"
    created_at:  str
    expires_at:  str
    rotated_at:  Optional[str] = None
    revoked:     bool          = False
    fingerprint: str           = ""    # SHA-256 of public key (first 16 chars)

    def is_expired(self) -> bool:
        if self.revoked:
            return True
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return datetime.now(timezone.utc) > exp
        except Exception:
            return False

    def days_until_expiry(self) -> int:
        try:
            exp = datetime.fromisoformat(self.expires_at)
            delta = exp - datetime.now(timezone.utc)
            return max(0, delta.days)
        except Exception:
            return 0

    def needs_rotation_warning(self) -> bool:
        return (
            not self.revoked and
            0 < self.days_until_expiry() <= KEY_WARNING_DAYS
        )

    def to_dict(self) -> dict:
        return {
            "key_id":     self.key_id,
            "key_type":   self.key_type,
            "owner_id":   self.owner_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "rotated_at": self.rotated_at,
            "revoked":    self.revoked,
            "fingerprint":self.fingerprint,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeyMetadata":
        return cls(
            key_id      = d["key_id"],
            key_type    = d["key_type"],
            owner_id    = d["owner_id"],
            created_at  = d["created_at"],
            expires_at  = d["expires_at"],
            rotated_at  = d.get("rotated_at"),
            revoked     = d.get("revoked", False),
            fingerprint = d.get("fingerprint", ""),
        )


# ─── Core Key Manager ─────────────────────────────────────────────────────────

class KeyManager:
    """
    Central key lifecycle manager for RIVA.

    Responsibilities:
        1. Generate & store riva_master.key (AES-256)
        2. Generate & store doctor Ed25519 keypairs
        3. Track expiry & rotation schedule
        4. Provide key status for 17_sustainability.html
        5. Revoke compromised keys

    Security principles:
        - Private keys never leave _KEYS_DIR unencrypted
        - All metadata stored in key_metadata.json (no sensitive material)
        - Key fingerprints use first 16 chars of SHA-256(public_key)
        - Rotation creates new key BEFORE revoking old (zero-downtime)
    """

    def __init__(self) -> None:
        _KEYS_DIR.mkdir(parents=True, exist_ok=True)
        self._metadata: dict[str, KeyMetadata] = {}
        self._load_metadata()
        log.info("[KeyManager] ready  keys=%d  dir=%s", len(self._metadata), _KEYS_DIR)

    # ── Metadata persistence ──────────────────────────────────────────────────

    def _load_metadata(self) -> None:
        if not _META_FILE.exists():
            return
        try:
            data = json.loads(_META_FILE.read_text(encoding="utf-8"))
            for entry in data:
                m = KeyMetadata.from_dict(entry)
                self._metadata[m.key_id] = m
            log.info("[KeyManager] loaded %d key records", len(self._metadata))
        except Exception as e:
            log.warning("[KeyManager] metadata load failed: %s", e)

    def _save_metadata(self) -> None:
        try:
            data = [m.to_dict() for m in self._metadata.values()]
            _META_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            log.error("[KeyManager] metadata save failed: %s", e)

    def _register(self, meta: KeyMetadata) -> None:
        self._metadata[meta.key_id] = meta
        self._save_metadata()

    # ── 1. Master key ─────────────────────────────────────────────────────────

    def generate_master_key(self, force: bool = False) -> Path:
        """
        Generates riva_master.key (32 random bytes = AES-256).

        Args:
            force: overwrite existing key (use with caution — re-encrypts all DBs)

        Returns:
            Path to the key file

        Usage (run once on Colab during project setup):
            km = KeyManager()
            key_path = km.generate_master_key()
            # Download key_path to your local machine
            # Set RIVA_MASTER_KEY=base64(key) in .env
        """
        key_path = _KEYS_DIR / "riva_master.key"

        if key_path.exists() and not force:
            log.info("[KeyManager] master key already exists — use force=True to rotate")
            return key_path

        key_bytes   = secrets.token_bytes(32)
        key_b64     = base64.b64encode(key_bytes).decode()
        fingerprint = hashlib.sha256(key_bytes).hexdigest()[:16]

        key_path.write_bytes(key_bytes)

        now     = datetime.now(timezone.utc)
        expires = now + timedelta(days=MASTER_KEY_ROTATION_DAYS)

        meta = KeyMetadata(
            key_id      = "riva_master_key",
            key_type    = "master",
            owner_id    = "system",
            created_at  = now.isoformat(),
            expires_at  = expires.isoformat(),
            fingerprint = fingerprint,
        )
        self._register(meta)

        log.info(
            "[KeyManager] master key generated  fingerprint=%s  expires=%s",
            fingerprint, expires.date(),
        )

        # Also export as base64 env var hint
        env_hint_path = _KEYS_DIR / "RIVA_MASTER_KEY.b64.txt"
        env_hint_path.write_text(
            f"# Add this to your .env file:\n"
            f"RIVA_MASTER_KEY={key_b64}\n"
            f"# Fingerprint: {fingerprint}\n"
            f"# Expires: {expires.date()}\n",
            encoding="utf-8",
        )
        log.info("[KeyManager] env hint written to %s", env_hint_path.name)

        return key_path

    def rotate_master_key(self) -> tuple[Path, str]:
        """
        Rotates the master key:
            1. Generates new key
            2. Re-encrypts all 13 databases with new key
            3. Marks old key as rotated

        Returns:
            (new_key_path, new_fingerprint)

        ⚠️ Warning: data loss risk if interrupted mid-rotation.
        Run on Colab with full Drive access.
        """
        log.info("[KeyManager] starting master key rotation")

        # Load all databases with old key
        try:
            from .encryption_handler import get_encryption_handler
            old_handler = get_encryption_handler()
            all_data    = old_handler.read_all_databases()
        except Exception as e:
            raise RuntimeError(f"Cannot load existing data before rotation: {e}")

        # Generate new key
        new_path = self.generate_master_key(force=True)

        # Reload handler with new key
        import importlib
        try:
            import ai_core.security.encryption_handler as _mod
            _mod._handler = None   # reset singleton
            new_handler   = get_encryption_handler()
        except Exception:
            from .encryption_handler import EncryptionHandler
            new_handler = EncryptionHandler(new_path)

        # Re-encrypt all databases
        results = new_handler.write_all_databases(all_data)
        ok      = sum(1 for v in results.values() if v)

        # Update rotation timestamp
        if "riva_master_key" in self._metadata:
            self._metadata["riva_master_key"].rotated_at = (
                datetime.now(timezone.utc).isoformat()
            )
            self._save_metadata()

        log.info(
            "[KeyManager] rotation complete  re-encrypted=%d/%d",
            ok, len(results),
        )

        fingerprint = self._metadata.get("riva_master_key", KeyMetadata(
            "","","","","")).fingerprint

        return new_path, fingerprint

    # ── 2. Doctor keypairs ────────────────────────────────────────────────────

    def generate_doctor_keys(self, doctor_id: str) -> dict:
        """
        Generates Ed25519 keypair for a doctor via digital_signature.py.

        Returns:
            {
                "doctor_id":   str,
                "public_key":  str (base64),
                "fingerprint": str,
                "expires":     str,
            }
        """
        from .digital_signature import get_signer
        signer = get_signer()
        kp     = signer.generate_keypair(doctor_id)

        fingerprint = hashlib.sha256(kp.public_key_bytes).hexdigest()[:16]
        now         = datetime.now(timezone.utc)
        expires     = now + timedelta(days=DOCTOR_KEY_ROTATION_DAYS)

        meta = KeyMetadata(
            key_id      = f"doctor_{doctor_id}_private",
            key_type    = "doctor_private",
            owner_id    = doctor_id,
            created_at  = now.isoformat(),
            expires_at  = expires.isoformat(),
            fingerprint = fingerprint,
        )
        self._register(meta)

        pub_meta = KeyMetadata(
            key_id      = f"doctor_{doctor_id}_public",
            key_type    = "doctor_public",
            owner_id    = doctor_id,
            created_at  = now.isoformat(),
            expires_at  = expires.isoformat(),
            fingerprint = fingerprint,
        )
        self._register(pub_meta)

        log.info(
            "[KeyManager] doctor keys generated  id=%s  fp=%s  expires=%s",
            doctor_id, fingerprint, expires.date(),
        )

        return {
            "doctor_id":   doctor_id,
            "public_key":  kp.public_key_b64,
            "fingerprint": fingerprint,
            "expires":     expires.isoformat(),
        }

    # ── 3. Revocation ─────────────────────────────────────────────────────────

    def revoke_key(self, key_id: str, reason: str = "") -> bool:
        """
        Marks a key as revoked.
        Revoked keys fail all signature/encryption checks.

        Usage (when doctor leaves clinic):
            km.revoke_key("doctor_DR-001_private", "contract_ended")
        """
        if key_id not in self._metadata:
            log.warning("[KeyManager] revoke: key not found: %s", key_id)
            return False

        self._metadata[key_id].revoked = True
        self._save_metadata()
        log.warning(
            "[KeyManager] KEY REVOKED: %s  reason=%s",
            key_id, reason or "unspecified",
        )
        return True

    # ── 4. Status & health ────────────────────────────────────────────────────

    def key_status(self) -> dict:
        """
        Full key status report for 17_sustainability.html.

        Returns:
            {
                "master_key":     {valid, fingerprint, expires_in_days},
                "doctor_keys":    [{doctor_id, valid, expires_in_days}],
                "warnings":       ["DR-001 key expires in 5 days"],
                "revoked_count":  int,
                "overall_health": "green" | "yellow" | "red",
            }
        """
        warnings      = []
        revoked_count = 0
        doctor_keys   = []

        # Master key
        master_meta  = self._metadata.get("riva_master_key")
        master_status = {
            "valid":          master_meta and not master_meta.is_expired() if master_meta else False,
            "fingerprint":    master_meta.fingerprint if master_meta else "not_generated",
            "expires_in_days":master_meta.days_until_expiry() if master_meta else 0,
            "file_exists":    (_KEYS_DIR / "riva_master.key").exists(),
        }

        if master_meta:
            if master_meta.is_expired():
                warnings.append("⚠️ riva_master.key منتهي الصلاحية — rotate فوراً")
            elif master_meta.needs_rotation_warning():
                warnings.append(
                    f"⚠️ riva_master.key ينتهي في {master_meta.days_until_expiry()} يوم"
                )

        # Doctor keys
        seen_doctors = set()
        for meta in self._metadata.values():
            if meta.key_type == "doctor_private":
                seen_doctors.add(meta.owner_id)
                entry = {
                    "doctor_id":      meta.owner_id,
                    "valid":          not meta.is_expired(),
                    "revoked":        meta.revoked,
                    "fingerprint":    meta.fingerprint,
                    "expires_in_days":meta.days_until_expiry(),
                }
                doctor_keys.append(entry)

                if meta.revoked:
                    revoked_count += 1
                elif meta.is_expired():
                    warnings.append(f"⚠️ مفتاح {meta.owner_id} منتهي")
                elif meta.needs_rotation_warning():
                    warnings.append(
                        f"ℹ️ مفتاح {meta.owner_id} ينتهي في {meta.days_until_expiry()} يوم"
                    )

        # Overall health
        if any("منتهي" in w for w in warnings):
            health = "red"
        elif warnings:
            health = "yellow"
        else:
            health = "green"

        return {
            "master_key":     master_status,
            "doctor_keys":    doctor_keys,
            "warnings":       warnings,
            "revoked_count":  revoked_count,
            "total_keys":     len(self._metadata),
            "overall_health": health,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }

    def check_rotation_needed(self) -> list[str]:
        """
        Returns list of key_ids that need rotation.
        Called by a scheduled task or /health endpoint.
        """
        return [
            m.key_id
            for m in self._metadata.values()
            if m.needs_rotation_warning() or m.is_expired()
        ]

    @property
    def status(self) -> dict:
        return {
            "keys_dir":        str(_KEYS_DIR),
            "total_keys":      len(self._metadata),
            "master_key_file": (_KEYS_DIR / "riva_master.key").exists(),
            "rotation_needed": self.check_rotation_needed(),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_km: Optional[KeyManager] = None


def get_key_manager() -> KeyManager:
    """
    Singleton — used during setup and by /health endpoint.

    Usage (initial setup on Colab):
        from ai_core.security.key_manager import get_key_manager

        km = get_key_manager()

        # 1. Generate master key (run once)
        km.generate_master_key()

        # 2. Generate doctor keys
        km.generate_doctor_keys("DR-001")
        km.generate_doctor_keys("DR-002")

        # 3. Check status
        print(km.key_status())

    Usage in 17_sustainability.html → GET /security/key-status:
        status = km.key_status()
        # {"overall_health": "green", "warnings": [], ...}
    """
    global _km
    if _km is None:
        _km = KeyManager()
    return _km


# ─── Public API ───────────────────────────────────────────────────────────────

def setup_riva_keys(doctor_ids: list[str]) -> dict:
    """
    One-shot setup — generates all keys needed for RIVA.
    Run on Colab during initial project configuration.

    Usage:
        from ai_core.security.key_manager import setup_riva_keys

        result = setup_riva_keys(["DR-001", "DR-002", "NR-001"])
        print(result["master_key_path"])
        # Download riva_master.key to local machine
        # Set RIVA_MASTER_KEY in .env
    """
    km = get_key_manager()

    # Master key
    master_path = km.generate_master_key()

    # Doctor keys
    doctor_results = {}
    for doc_id in doctor_ids:
        try:
            doctor_results[doc_id] = km.generate_doctor_keys(doc_id)
        except Exception as e:
            log.error("[KeyManager] doctor key generation failed: %s — %s", doc_id, e)
            doctor_results[doc_id] = {"error": str(e)}

    status = km.key_status()

    log.info(
        "[KeyManager] setup complete  master=%s  doctors=%d  health=%s",
        master_path.name, len(doctor_ids), status["overall_health"],
    )

    return {
        "master_key_path": str(master_path),
        "master_key_b64":  base64.b64encode(master_path.read_bytes()).decode()
                           if master_path.exists() else "",
        "doctor_keys":     doctor_results,
        "status":          status,
        "next_step":       (
            "1. احفظ riva_master.key على جهازك\n"
            "2. أضف RIVA_MASTER_KEY للـ .env\n"
            "3. ارفع الملفات .pubkey على GitHub\n"
            "4. احتفظ بالـ .privkey على جهازك فقط"
        ),
    }


def get_key_status() -> dict:
    """Key status for /security/key-status endpoint."""
    return get_key_manager().key_status()
