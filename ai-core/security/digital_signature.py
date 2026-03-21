"""
digital_signature.py
====================
RIVA Health Platform — Digital Signature System
------------------------------------------------
المسار: ai-core/security/digital_signature.py

المهمة:
    - توقيع الروشتات والقرارات الطبية رقمياً
    - التحقق من صحة التوقيع (للصيدليات والمستشفيات)
    - ربط كل توقيع بـ doctor_id + timestamp + record_hash
    - Chain of Trust: الروشتة موقّعة → التوقيع موثّق → الموثّق مشفّر

الخوارزمية: Ed25519
    - مفتاح خاص (private key): 32 bytes — عند الدكتور فقط
    - مفتاح عام (public key) : 32 bytes — يُشارك مع الصيدليات
    - التوقيع               : 64 bytes
    - السرعة                : < 1ms للتوقيع و< 1ms للتحقق

لماذا Ed25519 وليس RSA؟
    - أسرع بكثير (مهم للبيئة الريفية offline)
    - مفاتيح أصغر (مناسب للـ QR codes)
    - مقاوم لهجمات التوقيع المزيّف (EUF-CMA secure)

الربط مع المنظومة:
    - prescription_gen.py    : يوقّع كل روشتة قبل QR
    - prescription_signer.py : wrapper مبسّط للدكتور
    - encryption_handler.py  : يحفظ المفاتيح مشفّرة
    - access_control.py      : DOCTOR فقط يوقّع
    - 04_result.html         : يعرض "✓ موقّعة رقمياً"

Author: GODA EMAD · Harvard HSIL Hackathon 2026
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.security.digital_signature")

# ─── Paths ───────────────────────────────────────────────────────────────────

_HERE       = Path(__file__).resolve().parent   # ai-core/security/
_AICORE     = _HERE.parent
_ROOT       = _AICORE.parent
_KEYS_DIR   = _ROOT / "data" / "databases" / "keys"
_SIGLOG_DIR = _ROOT / "data" / "databases"

# ─── Signature record ─────────────────────────────────────────────────────────

@dataclass
class SignatureRecord:
    """
    Complete signature package attached to any signed document.
    Stored alongside the document for verification.
    """
    signature_b64:  str     # Ed25519 signature — base64
    public_key_b64: str     # signer's public key — base64
    doctor_id:      str     # who signed
    record_hash:    str     # SHA-256 of the signed content
    signed_at:      str     # ISO 8601 timestamp
    algorithm:      str = "Ed25519"
    version:        str = "RIVA-SIG-v1"

    def to_dict(self) -> dict:
        return {
            "signature":   self.signature_b64,
            "public_key":  self.public_key_b64,
            "doctor_id":   self.doctor_id,
            "record_hash": self.record_hash,
            "signed_at":   self.signed_at,
            "algorithm":   self.algorithm,
            "version":     self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SignatureRecord":
        return cls(
            signature_b64  = d["signature"],
            public_key_b64 = d["public_key"],
            doctor_id      = d["doctor_id"],
            record_hash    = d["record_hash"],
            signed_at      = d["signed_at"],
            algorithm      = d.get("algorithm", "Ed25519"),
            version        = d.get("version", "RIVA-SIG-v1"),
        )


# ─── Key pair ─────────────────────────────────────────────────────────────────

@dataclass
class KeyPair:
    private_key_bytes: bytes   # 32 bytes — never leave the device
    public_key_bytes:  bytes   # 32 bytes — safe to share

    @property
    def public_key_b64(self) -> str:
        return base64.b64encode(self.public_key_bytes).decode()

    @property
    def private_key_b64(self) -> str:
        return base64.b64encode(self.private_key_bytes).decode()


# ─── Core signer ─────────────────────────────────────────────────────────────

class DigitalSigner:
    """
    Ed25519 digital signature system for RIVA medical documents.

    Key storage strategy:
        - Private key: encrypted with AES-256-GCM via encryption_handler
          stored at data/databases/keys/{doctor_id}.privkey
        - Public key:  stored plaintext at data/databases/keys/{doctor_id}.pubkey
          shared with pharmacies for verification

    Offline-first:
        - Keys generated once on setup (Colab/device)
        - All signing/verification works without internet
        - Public keys cached in memory after first load
    """

    def __init__(self) -> None:
        self._keypairs: dict[str, KeyPair] = {}
        self._pubkeys:  dict[str, bytes]   = {}
        _KEYS_DIR.mkdir(parents=True, exist_ok=True)
        log.info("[DigitalSigner] ready  keys_dir=%s", _KEYS_DIR)

    # ── Key generation ────────────────────────────────────────────────────────

    def generate_keypair(self, doctor_id: str) -> KeyPair:
        """
        Generates an Ed25519 key pair for a doctor.
        Private key is encrypted and stored.
        Public key is stored plaintext for pharmacy verification.

        Usage (run once during doctor onboarding):
            signer = DigitalSigner()
            kp = signer.generate_keypair("DR-001")
            print(f"Public key: {kp.public_key_b64}")
        """
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.generate()
        public_key  = private_key.public_key()

        priv_bytes  = private_key.private_bytes_raw()
        pub_bytes   = public_key.public_bytes_raw()
        kp          = KeyPair(priv_bytes, pub_bytes)

        # Save public key (plaintext — safe to share)
        pub_path = _KEYS_DIR / f"{doctor_id}.pubkey"
        pub_path.write_bytes(pub_bytes)

        # Save private key (encrypted)
        self._store_private_key(doctor_id, priv_bytes)

        self._keypairs[doctor_id] = kp
        self._pubkeys[doctor_id]  = pub_bytes

        log.info(
            "[DigitalSigner] keypair generated for %s  pub=%s…",
            doctor_id, kp.public_key_b64[:16],
        )
        return kp

    # ── Key loading ───────────────────────────────────────────────────────────

    def _store_private_key(self, doctor_id: str, priv_bytes: bytes) -> None:
        """Encrypts and stores private key via encryption_handler."""
        try:
            from .encryption_handler import get_encryption_handler
            handler   = get_encryption_handler()
            encrypted = handler.encrypt(priv_bytes)
            priv_path = _KEYS_DIR / f"{doctor_id}.privkey"
            priv_path.write_bytes(encrypted)
            log.info("[DigitalSigner] private key stored (encrypted): %s", doctor_id)
        except Exception as e:
            # Fallback: store raw (dev mode only)
            log.warning("[DigitalSigner] encryption unavailable, storing raw key: %s", e)
            priv_path = _KEYS_DIR / f"{doctor_id}.privkey.raw"
            priv_path.write_bytes(priv_bytes)

    def _load_private_key(self, doctor_id: str) -> Optional[bytes]:
        """Loads and decrypts private key."""
        # Try encrypted
        priv_path = _KEYS_DIR / f"{doctor_id}.privkey"
        if priv_path.exists():
            try:
                from .encryption_handler import get_encryption_handler
                handler = get_encryption_handler()
                return handler.decrypt(priv_path.read_bytes())
            except Exception as e:
                log.warning("[DigitalSigner] decrypt privkey failed: %s", e)

        # Try raw (dev mode)
        raw_path = _KEYS_DIR / f"{doctor_id}.privkey.raw"
        if raw_path.exists():
            log.warning("[DigitalSigner] loading raw (unencrypted) private key — dev only")
            return raw_path.read_bytes()

        log.error("[DigitalSigner] private key not found for %s", doctor_id)
        return None

    def _load_public_key(self, doctor_id: str) -> Optional[bytes]:
        """Loads public key from file or memory cache."""
        if doctor_id in self._pubkeys:
            return self._pubkeys[doctor_id]
        pub_path = _KEYS_DIR / f"{doctor_id}.pubkey"
        if pub_path.exists():
            pub_bytes = pub_path.read_bytes()
            self._pubkeys[doctor_id] = pub_bytes
            return pub_bytes
        log.error("[DigitalSigner] public key not found for %s", doctor_id)
        return None

    def _get_keypair(self, doctor_id: str) -> Optional[KeyPair]:
        """Returns cached or loads keypair for doctor."""
        if doctor_id in self._keypairs:
            return self._keypairs[doctor_id]
        priv = self._load_private_key(doctor_id)
        pub  = self._load_public_key(doctor_id)
        if priv is None or pub is None:
            return None
        kp = KeyPair(priv, pub)
        self._keypairs[doctor_id] = kp
        return kp

    # ── Sign ──────────────────────────────────────────────────────────────────

    def sign(
        self,
        content:   dict | str | bytes,
        doctor_id: str,
    ) -> SignatureRecord:
        """
        Signs a medical document with the doctor's Ed25519 private key.

        Args:
            content   : the document to sign (dict/str/bytes)
            doctor_id : the signing doctor's ID

        Returns:
            SignatureRecord — attach this to the document before QR generation

        Usage in prescription_gen.py:
            sig = signer.sign(rx_dict, doctor_id="DR-001")
            rx_dict["digital_signature"] = sig.to_dict()

        Raises:
            KeyError if doctor's private key not found
            RuntimeError if cryptography library unavailable
        """
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        kp = self._get_keypair(doctor_id)
        if kp is None:
            raise KeyError(
                f"[DigitalSigner] no keypair for {doctor_id} — "
                "run generate_keypair() first"
            )

        # Canonicalise content → bytes
        raw = self._to_bytes(content)

        # SHA-256 hash of the content (for Chain of Trust)
        record_hash = hashlib.sha256(raw).hexdigest()

        # Ed25519 sign
        private_key = Ed25519PrivateKey.from_private_bytes(kp.private_key_bytes)
        sig_bytes   = private_key.sign(raw)

        record = SignatureRecord(
            signature_b64  = base64.b64encode(sig_bytes).decode(),
            public_key_b64 = kp.public_key_b64,
            doctor_id      = doctor_id,
            record_hash    = record_hash,
            signed_at      = datetime.now(timezone.utc).isoformat(),
        )

        log.info(
            "[DigitalSigner] signed  doctor=%s  hash=%s…  sig=%s…",
            doctor_id, record_hash[:8], record.signature_b64[:12],
        )
        return record

    # ── Verify ────────────────────────────────────────────────────────────────

    def verify(
        self,
        content:   dict | str | bytes,
        signature: SignatureRecord | dict,
    ) -> dict:
        """
        Verifies a digital signature.

        Args:
            content   : the original document
            signature : SignatureRecord or dict

        Returns:
            {
                valid        : bool,
                doctor_id    : str,
                signed_at    : str,
                hash_match   : bool,
                error        : str | None,
            }

        Usage in pharmacy QR scanner:
            result = signer.verify(rx_dict, rx_dict["digital_signature"])
            if result["valid"]:
                show_green_checkmark()
        """
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature

        if isinstance(signature, dict):
            try:
                signature = SignatureRecord.from_dict(signature)
            except Exception as e:
                return {"valid": False, "error": f"Invalid signature format: {e}"}

        raw         = self._to_bytes(content)
        record_hash = hashlib.sha256(raw).hexdigest()
        hash_match  = (record_hash == signature.record_hash)

        try:
            pub_bytes  = base64.b64decode(signature.public_key_b64)
            sig_bytes  = base64.b64decode(signature.signature_b64)
            public_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
            public_key.verify(sig_bytes, raw)
            valid = True
            error = None
        except InvalidSignature:
            valid = False
            error = "التوقيع الرقمي غير صحيح — الوثيقة قد تكون معدّلة"
            log.warning(
                "[DigitalSigner] INVALID signature  doctor=%s  hash=%s",
                signature.doctor_id, signature.record_hash[:8],
            )
        except Exception as e:
            valid = False
            error = str(e)
            log.error("[DigitalSigner] verify error: %s", e)

        return {
            "valid":      valid and hash_match,
            "sig_valid":  valid,
            "hash_match": hash_match,
            "doctor_id":  signature.doctor_id,
            "signed_at":  signature.signed_at,
            "algorithm":  signature.algorithm,
            "error":      error,
        }

    # ── Batch verify ──────────────────────────────────────────────────────────

    def verify_batch(self, documents: list[dict]) -> list[dict]:
        """
        Verifies multiple documents.
        Used by pharmacy systems to batch-check prescriptions.

        Returns list of verify() results in same order.
        """
        results = []
        for doc in documents:
            sig = doc.get("digital_signature")
            if sig is None:
                results.append({"valid": False, "error": "no_signature"})
            else:
                content = {k: v for k, v in doc.items() if k != "digital_signature"}
                results.append(self.verify(content, sig))
        return results

    # ── Audit log ─────────────────────────────────────────────────────────────

    def log_signature(
        self,
        record:     SignatureRecord,
        doc_type:   str,
        patient_id: str = "",
    ) -> None:
        """
        Appends signature event to signature_audit.jsonl.
        Used for forensic audit trail — who signed what when.
        """
        audit_path = _SIGLOG_DIR / "signature_audit.jsonl"
        entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "doctor_id":  record.doctor_id,
            "doc_type":   doc_type,
            "record_hash":record.record_hash,
            "signed_at":  record.signed_at,
            "patient_hash": hashlib.sha256(patient_id.encode()).hexdigest()[:8]
                            if patient_id else "",
        }
        try:
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("[DigitalSigner] audit log failed: %s", e)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_bytes(content: dict | str | bytes) -> bytes:
        if isinstance(content, bytes):
            return content
        if isinstance(content, str):
            return content.encode("utf-8")
        if isinstance(content, dict):
            # Canonical JSON — sorted keys, no whitespace
            return json.dumps(
                content, sort_keys=True,
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
        raise TypeError(f"Unsupported type: {type(content)}")

    def has_keypair(self, doctor_id: str) -> bool:
        """Returns True if doctor has a key pair."""
        priv = _KEYS_DIR / f"{doctor_id}.privkey"
        raw  = _KEYS_DIR / f"{doctor_id}.privkey.raw"
        pub  = _KEYS_DIR / f"{doctor_id}.pubkey"
        return (priv.exists() or raw.exists()) and pub.exists()

    @property
    def status(self) -> dict:
        pub_keys = list(_KEYS_DIR.glob("*.pubkey"))
        return {
            "algorithm":       "Ed25519",
            "keys_dir":        str(_KEYS_DIR),
            "doctors_with_keys": [p.stem for p in pub_keys],
            "loaded_keypairs": list(self._keypairs.keys()),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_signer: Optional[DigitalSigner] = None


def get_signer() -> DigitalSigner:
    """
    Singleton — used by prescription_gen.py and prescription_signer.py.

    Usage in prescription_gen.py:
        from ai_core.security.digital_signature import get_signer

        signer = get_signer()
        sig    = signer.sign(rx_dict, doctor_id=doctor_id)
        rx_dict["digital_signature"] = sig.to_dict()
        signer.log_signature(sig, "prescription", patient_id)

    Usage in pharmacy (04_result.html → backend):
        result = signer.verify(rx_dict, rx_dict["digital_signature"])
        # {"valid": True, "doctor_id": "DR-001", "signed_at": "..."}
    """
    global _signer
    if _signer is None:
        _signer = DigitalSigner()
    return _signer


# ─── Public API ───────────────────────────────────────────────────────────────

def sign_document(
    content:   dict | str | bytes,
    doctor_id: str,
    doc_type:  str = "prescription",
    patient_id:str = "",
) -> dict:
    """
    Signs a document and returns the full document with signature attached.

    Usage:
        signed_rx = sign_document(rx_dict, "DR-001", "prescription", patient_id)
        # signed_rx["digital_signature"] contains the signature
    """
    signer = get_signer()
    sig    = signer.sign(content, doctor_id)
    signer.log_signature(sig, doc_type, patient_id)

    if isinstance(content, dict):
        return {**content, "digital_signature": sig.to_dict()}
    return {"content": content, "digital_signature": sig.to_dict()}


def verify_document(document: dict) -> dict:
    """
    Verifies a signed document.
    Extracts signature automatically from document["digital_signature"].
    """
    signer = get_signer()
    sig    = document.get("digital_signature")
    if not sig:
        return {"valid": False, "error": "الوثيقة غير موقّعة"}
    content = {k: v for k, v in document.items() if k != "digital_signature"}
    return signer.verify(content, sig)
