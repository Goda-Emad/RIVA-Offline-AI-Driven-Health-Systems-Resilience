"""
encryption_handler.py
=====================
RIVA Health Platform v4.0 — AES-256-GCM Encryption Handler
-----------------------------------------------------------
المسار: ai-core/security/encryption_handler.py

المهمة:
    - تشفير وفك تشفير كل ملفات .encrypted في data/databases/
    - يستخدم riva_master.key المحفوظ على جهازك / Drive
    - يُستدعى من db_loader.py و seed_all_databases.py

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
from pathlib import Path
from typing import Optional

log = logging.getLogger("riva.security.encryption_handler")

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

_HERE       = Path(__file__).resolve().parent   # ai-core/security/
_AICORE     = _HERE.parent                      # ai-core/
_ROOT       = _AICORE.parent                    # project-root/
_DB_DIR     = _ROOT / "data" / "databases"
_KEY_FILE   = _DB_DIR / "keys" / "riva_master.key"


def _resolve_key_path() -> Path:
    """
    Priority:
      1. RIVA_KEY_PATH env var     (server deployment)
      2. RIVA_MASTER_KEY env var   (base64 — Colab / CI)
      3. data/databases/keys/riva_master.key  (local)
    """
    env_path = os.environ.get("RIVA_KEY_PATH")
    if env_path:
        return Path(env_path)
    return _KEY_FILE


# ─────────────────────────────────────────────────────────────────────────────
# EncryptionHandler
# ─────────────────────────────────────────────────────────────────────────────

class EncryptionHandler:
    """
    AES-256-GCM encryption/decryption for all RIVA databases.

    Usage:
        handler = EncryptionHandler()

        # Encrypt
        raw       = json.dumps(data).encode()
        encrypted = handler.encrypt(raw)

        # Decrypt
        raw_back  = handler.decrypt(encrypted)
        data_back = json.loads(raw_back)
    """

    def __init__(self, key_path: Optional[Path] = None) -> None:
        self._key: Optional[bytes] = None
        self._load_key(key_path or _resolve_key_path())

    # ── Key loading ───────────────────────────────────────────────────────────

    def _load_key(self, key_path: Path) -> None:
        """
        Load AES-256 key.
        Priority:
          1. RIVA_MASTER_KEY env var (base64) — Colab / CI
          2. key file on disk
        """
        # Option 1: env var (base64-encoded)
        env_b64 = os.environ.get("RIVA_MASTER_KEY")
        if env_b64:
            try:
                self._key = base64.b64decode(env_b64)
                log.info("[EncryptionHandler] key loaded from RIVA_MASTER_KEY env var")
                return
            except Exception as exc:
                log.warning("[EncryptionHandler] RIVA_MASTER_KEY decode failed: %s", exc)

        # Option 2: key file
        if key_path.exists():
            self._key = key_path.read_bytes()
            log.info("[EncryptionHandler] key loaded from %s", key_path)
            return

        log.error(
            "[EncryptionHandler] key not found at %s and RIVA_MASTER_KEY not set. "
            "Run seed_all_databases.py on Colab to generate keys.",
            key_path,
        )

    def is_ready(self) -> bool:
        return self._key is not None and len(self._key) == 32

    # ── Core: encrypt / decrypt ───────────────────────────────────────────────

    def encrypt(self, raw: bytes) -> bytes:
        """
        Encrypt raw bytes → nonce(12) + ciphertext+tag.

        AES-256-GCM guarantees:
            - Confidentiality: nobody reads without the key
            - Integrity: any tampering with the file → decrypt fails
        """
        if not self.is_ready():
            raise RuntimeError("EncryptionHandler: key not loaded")

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = secrets.token_bytes(12)
        ct    = AESGCM(self._key).encrypt(nonce, raw, None)
        return nonce + ct

    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt nonce(12) + ciphertext+tag → raw bytes.
        Raises ValueError if data is tampered or key is wrong.
        """
        if not self.is_ready():
            raise RuntimeError("EncryptionHandler: key not loaded")

        if len(data) < 28:   # 12 nonce + 16 tag minimum
            raise ValueError(f"Data too short to be encrypted ({len(data)} bytes)")

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce, ct = data[:12], data[12:]
        return AESGCM(self._key).decrypt(nonce, ct, None)

    # ── JSON helpers (used by db_loader.py) ───────────────────────────────────

    def encrypt_json(self, data: list | dict) -> bytes:
        """Serialize to JSON → encrypt."""
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        return self.encrypt(raw)

    def decrypt_json(self, data: bytes) -> list | dict:
        """Decrypt → parse JSON."""
        raw = self.decrypt(data)
        return json.loads(raw.decode("utf-8"))

    # ── File helpers ──────────────────────────────────────────────────────────

    def encrypt_file(self, src: Path, dst: Optional[Path] = None) -> Path:
        """
        Encrypt a JSON file → .encrypted file.

        Args:
            src: path to plain JSON file
            dst: output path (default: src with .encrypted extension)
        """
        if dst is None:
            dst = src.with_suffix(".encrypted")
        raw = src.read_bytes()
        dst.write_bytes(self.encrypt(raw))
        log.info("[EncryptionHandler] encrypted %s → %s", src.name, dst.name)
        return dst

    def decrypt_file(self, src: Path, dst: Optional[Path] = None) -> Path:
        """
        Decrypt .encrypted file → JSON file.

        Args:
            src: path to .encrypted file
            dst: output path (default: src with .json extension)
        """
        if dst is None:
            dst = src.with_suffix(".json")
        raw = src.read_bytes()
        dst.write_bytes(self.decrypt(raw))
        log.info("[EncryptionHandler] decrypted %s → %s", src.name, dst.name)
        return dst

    def read_db(self, db_name: str) -> list | dict:
        """
        Read and decrypt a database file from data/databases/.

        Usage in db_loader.py:
            handler = EncryptionHandler()
            patients = handler.read_db("patients.encrypted")

        Returns [] on file-not-found (graceful offline degradation).
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
        Encrypt and write a database file to data/databases/.

        Usage in db_loader.py:
            handler.write_db("predictions.encrypted", predictions_list)
        """
        path = _DB_DIR / db_name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.encrypt_json(data))
            log.info("[EncryptionHandler] write_db(%s) — %d records", db_name, len(data) if isinstance(data, list) else 1)
            return True
        except Exception as exc:
            log.error("[EncryptionHandler] write_db(%s) failed: %s", db_name, exc)
            return False

    @property
    def status(self) -> dict:
        return {
            "ready":       self.is_ready(),
            "algorithm":   "AES-256-GCM",
            "key_size":    len(self._key) * 8 if self._key else 0,
            "key_source":  "env_var" if os.environ.get("RIVA_MASTER_KEY") else "file",
            "db_dir":      str(_DB_DIR),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_handler: Optional[EncryptionHandler] = None


def get_encryption_handler() -> EncryptionHandler:
    """
    Singleton — used by db_loader.py.

    Usage:
        from ai_core.security.encryption_handler import get_encryption_handler
        handler  = get_encryption_handler()
        patients = handler.read_db("patients.encrypted")
    """
    global _handler
    if _handler is None:
        _handler = EncryptionHandler()
    return _handler


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, tempfile
    logging.basicConfig(level=logging.INFO)

    print("=" * 55)
    print("RIVA EncryptionHandler — self-test")
    print("=" * 55)

    # ── Generate test key ─────────────────────────────────
    test_key = secrets.token_bytes(32)
    os.environ["RIVA_MASTER_KEY"] = base64.b64encode(test_key).decode()

    handler = EncryptionHandler()
    assert handler.is_ready()
    print(f"\n✅ Key loaded — {handler.status['key_size']} bits")

    # ── Encrypt / Decrypt bytes ───────────────────────────
    raw       = "RIVA test data - بيانات اختبار".encode("utf-8")
    encrypted = handler.encrypt(raw)
    decrypted = handler.decrypt(encrypted)
    assert decrypted == raw
    assert len(encrypted) == 12 + len(raw) + 16   # nonce + data + tag
    print(f"✅ encrypt/decrypt bytes — {len(raw)}B → {len(encrypted)}B → {len(decrypted)}B")

    # ── Encrypt / Decrypt JSON ────────────────────────────
    data      = [{"patient_id": "P001", "risk": 0.72, "label": "خطر متوسط"}]
    enc_json  = handler.encrypt_json(data)
    dec_json  = handler.decrypt_json(enc_json)
    assert dec_json == data
    print(f"✅ encrypt_json/decrypt_json — {len(data)} records")

    # ── Tamper detection ──────────────────────────────────
    tampered = bytearray(encrypted)
    tampered[20] ^= 0xFF
    try:
        handler.decrypt(bytes(tampered))
        print("❌ tamper detection FAILED")
    except Exception:
        print("✅ tamper detection — modified ciphertext rejected")

    # ── Wrong key ─────────────────────────────────────────
    wrong_key = secrets.token_bytes(32)
    os.environ["RIVA_MASTER_KEY"] = base64.b64encode(wrong_key).decode()
    handler2 = EncryptionHandler()
    try:
        handler2.decrypt(encrypted)
        print("❌ wrong key not detected")
    except Exception:
        print("✅ wrong key rejected correctly")

    # ── read_db graceful on missing file ──────────────────
    os.environ["RIVA_MASTER_KEY"] = base64.b64encode(test_key).decode()
    h3   = EncryptionHandler()
    data = h3.read_db("nonexistent.encrypted")
    assert data == []
    print("✅ read_db missing file → [] (graceful)")

    # ── write_db + read_db roundtrip ──────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        # Patch _DB_DIR temporarily
        import ai_core.security.encryption_handler as _mod
        orig = _mod._DB_DIR
        _mod._DB_DIR = Path(tmp)
        sample = [{"subject_id": "T001", "age": 45}]
        h3.write_db("test.encrypted", sample)
        back = h3.read_db("test.encrypted")
        assert back == sample
        _mod._DB_DIR = orig
    print("✅ write_db + read_db roundtrip OK")

    print(f"\n✅ EncryptionHandler self-test complete")
    print(f"   Status: {handler.status}")
    sys.exit(0)
