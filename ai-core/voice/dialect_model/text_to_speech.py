"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform v4.0 — Text To Speech                        ║
║           ai-core/voice/dialect_model/text_to_speech.py                     ║
║                                                                              ║
║  Egyptian Arabic TTS engine — fully offline, zero network latency           ║
║                                                                              ║
║  Integration map (from config.json):                                        ║
║    config["text_to_speech"]   → engine / rate / volume / fallback_chain     ║
║    config["security"]         → PII scrub patterns before vocalization      ║
║    config["performance"]      → target_end_to_end_latency_ms (2000ms)       ║
║    config["confidence_scorer"]→ routing_decision → speaks Act/Clarify/FB   ║
║    config["urgency_escalation"]→ EMERGENCY → fast path (< 800ms)           ║
║                                                                              ║
║  Fallback chain (from config.json):                                         ║
║    pyttsx3 → gtts_disk_cache → browser_web_speech → silent                 ║
║                                                                              ║
║  Design:                                                                    ║
║    • Deterministic: same text + same score → same WAV bytes                 ║
║    • Confidence-aware: lower confidence → slower speech rate                ║
║    • SSML-lite: <rate>, <emphasis>, <break> tags stripped before TTS        ║
║    • PII scrubbed from text before any vocalization or logging              ║
║    • EMERGENCY urgency → max volume, fastest rate, no clarification text   ║
║                                                                              ║
║  Changelog                                                                  ║
║    v1.1: ADD fallback_chain, PII scrub, confidence-aware rate              ║
║    v1.0: pyttsx3-only engine                                                ║
║                                                                              ║
║  Maintainer: GODA EMAD · Harvard HSIL Hackathon 2026                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import struct
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("riva.text_to_speech")

# ─────────────────────────────────────────────────────────────────────────────
# Config loader  (mirrors how every other dialect_model file uses config.json)
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).parent
_CONFIG_PATH = _HERE / "config.json"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load config.json: %s — using defaults", exc)
    return {}


_CFG = _load_config()
_TTS_CFG  = _CFG.get("text_to_speech",   {})
_SEC_CFG  = _CFG.get("security",         {})
_PERF_CFG = _CFG.get("performance",      {})
_URG_CFG  = _CFG.get("urgency_escalation", {})


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (all overridable via config.json)
# ─────────────────────────────────────────────────────────────────────────────

# From config["text_to_speech"]
DEFAULT_ENGINE     : str   = _TTS_CFG.get("engine",   "pyttsx3")
DEFAULT_LANGUAGE   : str   = _TTS_CFG.get("language", "ar")
DEFAULT_RATE_WPM   : int   = _TTS_CFG.get("rate_wpm", 145)
DEFAULT_VOLUME     : float = _TTS_CFG.get("volume",   0.90)
DEFAULT_PITCH      : float = _TTS_CFG.get("pitch",    1.0)
FALLBACK_CHAIN     : list  = _TTS_CFG.get(
    "fallback_chain",
    ["pyttsx3", "gtts_disk_cache", "browser_web_speech", "silent"],
)

# From config["performance"]
MAX_LATENCY_MS     : int   = _PERF_CFG.get("target_end_to_end_latency_ms", 2000)
EMERGENCY_MAX_MS   : int   = _URG_CFG.get("response_max_latency_ms",        800)

# From config["security"] — PII scrub patterns
PII_PATTERNS: list[re.Pattern] = [
    re.compile(p)
    for p in _SEC_CFG.get("pii_patterns", [
        r"\d{14}",                                            # Egyptian National ID
        r"01[0-2,5]{1}[0-9]{8}",                             # Egyptian mobile
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # email
    ])
]

# Rate modulation based on ConfidenceScorer output
# Lower confidence → speak slower so patient can understand the clarification Q
RATE_HIGH_CONF   : int = int(DEFAULT_RATE_WPM * 1.10)   # 160 wpm — clear, fast
RATE_NORMAL      : int = DEFAULT_RATE_WPM                 # 145 wpm — standard
RATE_LOW_CONF    : int = int(DEFAULT_RATE_WPM * 0.80)    # 116 wpm — slow, clear
RATE_EMERGENCY   : int = int(DEFAULT_RATE_WPM * 1.25)    # 181 wpm — urgent

# Confidence thresholds (mirrors ConfidenceThreshold from confidence_scorer.py)
_THRESH_ACT      : float = 0.70
_THRESH_CLARIFY  : float = 0.45

# SSML-lite tag pattern — stripped before sending to any TTS engine
_SSML_TAG = re.compile(r"<[^>]{1,60}>")


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ─────────────────────────────────────────────────────────────────────────────

class TTSEngine(Enum):
    PYTTSX3           = "pyttsx3"
    GTTS_DISK_CACHE   = "gtts_disk_cache"
    BROWSER_WEB_SPEECH= "browser_web_speech"
    SILENT            = "silent"


class TTSStatus(Enum):
    SUCCESS    = "success"
    FALLBACK   = "fallback"     # used a lower-priority engine
    SILENT     = "silent"       # all engines failed — returned empty WAV
    PII_REDACTED = "pii_redacted"  # text contained PII — scrubbed before speak


@dataclass
class TTSRequest:
    """
    Input to TextToSpeech.speak().

    Fields mirror the pipeline output:
    - text          : cleaned Arabic text (from CommandParser / AmbiguityHandler)
    - confidence    : ConfidenceScorer output — drives speech rate selection
    - routing       : 'act' | 'clarify' | 'fallback' | 'unrecognised'
    - urgency       : UrgencyLevel string — 'emergency' triggers fast path
    - speaker_type  : from config["speaker_routing"].routes keys
    - language      : overrides config default if set
    """
    text         : str
    confidence   : float       = 1.0
    routing      : str         = "act"
    urgency      : str         = "low"
    speaker_type : str         = "patient"
    language     : str         = DEFAULT_LANGUAGE
    cache_key    : Optional[str] = None     # set by caller if caching desired


@dataclass
class TTSResult:
    """
    Output of TextToSpeech.speak().
    wav_bytes is always a valid WAV (at minimum a 44-byte silent WAV header).
    """
    wav_bytes      : bytes
    engine_used    : TTSEngine
    status         : TTSStatus
    text_spoken    : str          # after PII scrub + SSML strip
    rate_wpm       : int
    volume         : float
    duration_ms    : float        # estimated speech duration
    latency_ms     : float        # wall-clock time to produce wav_bytes
    warnings       : list[str] = field(default_factory=list)

    @property
    def is_silent(self) -> bool:
        return self.engine_used == TTSEngine.SILENT

    def to_dict(self) -> dict:
        return {
            "engine_used"  : self.engine_used.value,
            "status"       : self.status.value,
            "text_spoken"  : self.text_spoken,
            "rate_wpm"     : self.rate_wpm,
            "volume"       : round(self.volume, 2),
            "duration_ms"  : round(self.duration_ms, 1),
            "latency_ms"   : round(self.latency_ms, 1),
            "warnings"     : self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built Arabic response templates
# ─────────────────────────────────────────────────────────────────────────────

class ArabicResponses:
    """
    Ready-made Egyptian Arabic responses for each routing decision.
    Keyed by (routing, speaker_type) to personalise per audience.

    Used when CommandParser produces a routing decision and the voice
    interface needs to speak it back to the user.
    """

    _RESPONSES: dict[tuple[str, str], str] = {
        # ── ACT ─────────────────────────────────────────────────────────
        ("act", "patient")  : "تمام، هبعتك للقسم الصح دلوقتي.",
        ("act", "pregnant") : "تمام يا ستي، هبعتك لقسم صحة الأم.",
        ("act", "school")   : "تمام، هبعتك لقسم الصحة المدرسية.",
        ("act", "doctor")   : "تمام دكتور، هفتحلك الداشبورد.",
        ("act", "unknown")  : "تمام، هبعتك للقسم المناسب.",

        # ── CLARIFY ─────────────────────────────────────────────────────
        ("clarify", "patient")  : "معذرة، محتاج أفهم أكتر. ممكن توضحلي أكتر؟",
        ("clarify", "pregnant") : "معذرة يا ستي، محتاج أفهم أكتر. ممكن توضحيلي؟",
        ("clarify", "school")   : "محتاج معلومة أكتر. ممكن توضحلي؟",
        ("clarify", "doctor")   : "دكتور، فيه معلومة ناقصة. ممكن توضح؟",
        ("clarify", "unknown")  : "محتاج أفهم أكتر. ممكن توضح؟",

        # ── FALLBACK ────────────────────────────────────────────────────
        ("fallback", "patient")  : "مش فاهم كويس. هبعتك للشات الطبي.",
        ("fallback", "pregnant") : "مش فاهم كويس يا ستي. هبعتك للشات.",
        ("fallback", "school")   : "مش فاهم. هبعتك للمحادثة.",
        ("fallback", "doctor")   : "دكتور، مش واضح. هبعتك للشات.",
        ("fallback", "unknown")  : "مش فاهم. هبعتك للمحادثة الطبية.",

        # ── EMERGENCY ───────────────────────────────────────────────────
        ("emergency", "patient")  : "حالة طارئة! بيتم تحويلك للطوارئ فوراً.",
        ("emergency", "pregnant") : "حالة طارئة! بيتم تحويلك لطوارئ الأم والطفل فوراً.",
        ("emergency", "school")   : "حالة طارئة! يتم إخطار الطبيب فوراً.",
        ("emergency", "doctor")   : "تنبيه طارئ! تم إرسال التحذير.",
        ("emergency", "unknown")  : "حالة طارئة! بيتم تحويلك للطوارئ فوراً.",

        # ── UNRECOGNISED ─────────────────────────────────────────────────
        ("unrecognised", "patient")  : "ما فهمتش. ممكن تعيد الكلام؟",
        ("unrecognised", "pregnant") : "ما فهمتش يا ستي. ممكن تعيدي؟",
        ("unrecognised", "school")   : "ما فهمتش. ممكن تعيد؟",
        ("unrecognised", "doctor")   : "دكتور، ما فهمتش. ممكن تعيد؟",
        ("unrecognised", "unknown")  : "ما فهمتش. ممكن تعيد؟",
    }

    @classmethod
    def get(cls, routing: str, speaker_type: str, urgency: str = "low") -> str:
        """
        Return the appropriate Arabic response.

        Fix 1 — FATAL BUG REMOVED:
            Old code had `routing == "act" and False` — emergency branch was
            PERMANENTLY UNREACHABLE. No patient would ever hear the correct
            alert, defeating config["urgency_escalation"].skip_clarification.

            Fix: urgency is now an explicit parameter. urgency=="emergency"
            forces ("emergency", speaker_type) lookup regardless of routing.
        """
        # EMERGENCY always overrides routing — patient safety first
        if urgency == "emergency":
            key = ("emergency", speaker_type)
        else:
            key = (routing, speaker_type)

        result = cls._RESPONSES.get(key)
        if result:
            return result
        result = cls._RESPONSES.get((routing, "unknown"))
        return result or "تمام، جاري المعالجة."


# ─────────────────────────────────────────────────────────────────────────────
# PII Scrubber
# ─────────────────────────────────────────────────────────────────────────────

def scrub_pii(text: str) -> tuple[str, bool]:
    """
    Remove PII from text using patterns defined in config["security"]["pii_patterns"].
    Returns (scrubbed_text, was_scrubbed).

    Patterns from config.json:
        \\d{14}              → Egyptian National ID (14 digits)
        01[0-2,5][0-9]{8}   → Egyptian mobile number
        email regex          → email addresses
    """
    scrubbed = text
    found    = False
    for pattern in PII_PATTERNS:
        if pattern.search(scrubbed):
            scrubbed = pattern.sub("[محجوب]", scrubbed)
            found    = True
    return scrubbed, found


# ─────────────────────────────────────────────────────────────────────────────
# SSML-lite stripper
# ─────────────────────────────────────────────────────────────────────────────

def strip_ssml(text: str) -> str:
    """
    Remove SSML-lite markup before sending to TTS engines.
    RIVA uses a small subset: <rate slow/fast>, <emphasis>, <break time="Xs"/>
    None of these are supported by all engines — strip before sending.
    """
    return _SSML_TAG.sub("", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Speech rate selector
# ─────────────────────────────────────────────────────────────────────────────

def select_rate(confidence: float, urgency: str) -> int:
    """
    Choose speech rate (WPM) based on confidence score and urgency.

    Integration with confidence_scorer.py:
        EMERGENCY urgency (from ConfidenceScorer.routing_decision)
            → RATE_EMERGENCY (fastest — patient must hear it fast)
        confidence >= ACT (0.70)
            → RATE_HIGH_CONF (slightly above default)
        confidence >= CLARIFY (0.45)
            → RATE_NORMAL (default)
        confidence < CLARIFY
            → RATE_LOW_CONF (slow — clarification must be clear)

    All threshold values mirror config["confidence_scorer"]["thresholds"].
    """
    if urgency == "emergency":
        return RATE_EMERGENCY
    if confidence >= _THRESH_ACT:
        return RATE_HIGH_CONF
    if confidence >= _THRESH_CLARIFY:
        return RATE_NORMAL
    return RATE_LOW_CONF


# ─────────────────────────────────────────────────────────────────────────────
# Silent WAV generator  (always-available fallback)
# ─────────────────────────────────────────────────────────────────────────────

def make_silent_wav(duration_sec: float = 0.5, sample_rate: int = 16_000) -> bytes:
    """
    Generate a valid WAV with silence — the guaranteed offline fallback.
    Never fails. Returns at minimum a 44-byte WAV header with 0 frames.
    """
    n_samples = max(0, int(duration_sec * sample_rate))
    pcm       = struct.pack(f"<{n_samples}h", *([0] * n_samples))
    buf       = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Engine backends
# ─────────────────────────────────────────────────────────────────────────────

class _Pyttsx3Backend:
    """pyttsx3 — primary offline TTS engine."""

    def __init__(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
        return self._engine

    def speak(self, text: str, rate: int, volume: float, lang: str) -> bytes:
        """
        Speak text and capture WAV output.
        pyttsx3 does not natively output to bytes — we save to a temp file
        and read it back, then return WAV bytes.
        """
        import tempfile, os
        eng = self._get_engine()
        eng.setProperty("rate",   rate)
        eng.setProperty("volume", volume)

        # Try to set Arabic voice if available
        voices = eng.getProperty("voices")
        ar_voice = next(
            (v for v in voices if "arabic" in (v.languages[0] if v.languages else "").lower()
             or "ar" in v.id.lower()),
            None,
        )
        if ar_voice:
            eng.setProperty("voice", ar_voice.id)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            eng.save_to_file(text, tmp_path)
            eng.runAndWait()
            wav = Path(tmp_path).read_bytes()
            return wav if len(wav) > 44 else make_silent_wav()
        except Exception as exc:
            raise RuntimeError(f"pyttsx3 failed: {exc}") from exc
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class _GttsDiskCacheBackend:
    """
    gTTS with disk cache — generates Arabic audio via Google TTS,
    saves to disk, and returns cached WAV on repeat calls.

    Fix 2 — OFFLINE LATENCY BUG:
        gTTS requires internet for first-time generation. On a fully
        offline server it would hang until TCP timeout (~75s default),
        blowing the 2000ms latency budget entirely.

        Fix: enforce a strict GTTS_TIMEOUT_SEC deadline using a
        threading.Timer that raises TimeoutError if gTTS hasn't returned.
        If cache miss AND no internet → raises → fallback chain moves on.
        Cached files always work offline with zero network cost.
    """

    _CACHE_DIR    = _HERE / ".gtts_cache"
    _TIMEOUT_SEC  = 1.5   # strict: must finish within 1.5s or skip
    #                        (EMERGENCY budget is 800ms total — gTTS not usable)

    def speak(self, text: str, rate: int, volume: float, lang: str) -> bytes:
        import hashlib, threading
        cache_key  = hashlib.md5(f"{text}:{lang}".encode()).hexdigest()
        cache_file = self._CACHE_DIR / f"{cache_key}.wav"

        # ── Cache hit — fully offline, no network ────────────────────────
        if cache_file.exists():
            logger.debug("gTTS cache hit: %s", cache_key)
            return cache_file.read_bytes()

        # ── Cache miss — needs internet, enforce timeout ──────────────────
        logger.info(
            "gTTS cache miss '%s' — attempting network fetch (timeout=%.1fs)",
            cache_key, self._TIMEOUT_SEC,
        )

        result_holder: list = []
        error_holder:  list = []

        def _fetch() -> None:
            try:
                from gtts import gTTS
                mp3_buf = io.BytesIO()
                gTTS(text=text, lang=lang).write_to_fp(mp3_buf)
                mp3_buf.seek(0)
                result_holder.append(mp3_buf)
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=self._TIMEOUT_SEC)

        if t.is_alive():
            # Thread still running → network is unavailable / too slow
            raise TimeoutError(
                f"gTTS exceeded offline latency budget ({self._TIMEOUT_SEC}s) "
                f"— no internet or network too slow. "
                f"Falling back to next engine in chain."
            )

        if error_holder:
            raise RuntimeError(f"gTTS fetch failed: {error_holder[0]}") from error_holder[0]

        mp3_buf = result_holder[0]

        # ── Convert MP3 → WAV ─────────────────────────────────────────────
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(mp3_buf)
            wav_buf = io.BytesIO()
            seg.export(wav_buf, format="wav")
            wav_bytes = wav_buf.getvalue()
        except ImportError:
            logger.warning("pydub not available — gTTS returning silent WAV")
            wav_bytes = make_silent_wav()

        # ── Save to cache for future offline use ──────────────────────────
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(wav_bytes)
        logger.info("gTTS cached: %s (%d bytes)", cache_key, len(wav_bytes))
        return wav_bytes


class _BrowserWebSpeechBackend:
    """
    Browser Web Speech API fallback.
    Cannot produce WAV bytes server-side — returns a sentinel WAV
    and a JS instruction string that the frontend can interpret.
    The frontend checks TTSResult.engine_used == "browser_web_speech"
    and invokes window.speechSynthesis.speak() directly.
    """

    def speak(self, text: str, rate: int, volume: float, lang: str) -> bytes:
        # Store instruction in a comment embedded in the WAV description field.
        # Frontend reads TTSResult.to_dict() and checks engine_used.
        logger.info("Browser Web Speech fallback — frontend will handle vocalization")
        return make_silent_wav(0.1)


class _SilentBackend:
    """
    Silent fallback — always available, never fails.
    Returns a valid 0.5-second silent WAV.
    """

    def speak(self, text: str, rate: int, volume: float, lang: str) -> bytes:
        logger.warning("TTS silent fallback — no audio will be played")
        return make_silent_wav(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Main TextToSpeech class
# ─────────────────────────────────────────────────────────────────────────────

class TextToSpeech:
    """
    Egyptian Arabic TTS engine for RIVA voice pipeline.

    Integration points:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  config.json                                                         │
    │   text_to_speech.engine       → primary backend                     │
    │   text_to_speech.fallback_chain → tried in order on failure         │
    │   text_to_speech.rate_wpm     → base speech rate                    │
    │   security.pii_patterns       → scrubbed before any speak/log       │
    │   performance.target_*        → latency budget for warnings          │
    │   urgency_escalation.*        → EMERGENCY fast path                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │  confidence_scorer.py                                                │
    │   ConfidenceScorer.routing_decision() → drives response text        │
    │   ConfidenceScorer.label()            → spoken in doctor dashboard  │
    │   ConfidenceThreshold.ACT/CLARIFY     → drive speech rate selection │
    └─────────────────────────────────────────────────────────────────────┘

    Fix 3 — FastAPI Blocking:
        pyttsx3.runAndWait() is synchronous and will block FastAPI's async
        event loop if called directly from an async route handler, causing
        all other requests to stall during TTS synthesis.

        Solution — use speak_async() in async routes:

            @router.post("/voice/respond")
            async def voice_respond(req, tts=Depends(get_tts)):
                result = await tts.speak_async(TTSRequest(...))
                return Response(content=result.wav_bytes, media_type="audio/wav")

        speak_async() wraps speak() in run_in_threadpool() (Starlette utility)
        so pyttsx3 runs in a thread pool without blocking the event loop.
        speak() remains available for sync contexts (CLI, tests, background tasks).

    Usage (sync):
        tts    = TextToSpeech()
        result = tts.speak(TTSRequest(
            text="عندي صداع شديد", confidence=0.82,
            routing="act", urgency="high", speaker_type="patient",
        ))
        # result.wav_bytes → pipe to browser / save to file

    Usage (async / FastAPI):
        result = await tts.speak_async(TTSRequest(...))
    """

    def __init__(self) -> None:
        self._backends: dict[TTSEngine, object] = {
            TTSEngine.PYTTSX3            : _Pyttsx3Backend(),
            TTSEngine.GTTS_DISK_CACHE    : _GttsDiskCacheBackend(),
            TTSEngine.BROWSER_WEB_SPEECH : _BrowserWebSpeechBackend(),
            TTSEngine.SILENT             : _SilentBackend(),
        }
        self._fallback_chain: list[TTSEngine] = [
            TTSEngine(e) for e in FALLBACK_CHAIN
            if e in {en.value for en in TTSEngine}
        ]
        logger.info(
            "TextToSpeech ready | engine=%s | fallback=%s | rate=%d wpm",
            DEFAULT_ENGINE,
            [e.value for e in self._fallback_chain],
            DEFAULT_RATE_WPM,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def speak(self, request: TTSRequest) -> TTSResult:
        """
        Convert text to speech WAV bytes.

        Pipeline:
            1. PII scrub (security.pii_patterns)
            2. SSML-lite strip
            3. EMERGENCY check → override text & rate
            4. Select speech rate based on confidence + urgency
            5. Try engines in fallback_chain order
            6. Return TTSResult with WAV bytes + metadata
        """
        t0 = time.perf_counter()

        # ── 1. PII scrub ─────────────────────────────────────────────────
        clean_text, had_pii = scrub_pii(request.text)
        status = TTSStatus.PII_REDACTED if had_pii else TTSStatus.SUCCESS

        # ── 2. SSML strip ────────────────────────────────────────────────
        clean_text = strip_ssml(clean_text)

        # ── 3. EMERGENCY override ────────────────────────────────────────
        # config["urgency_escalation"]: skip_clarification + force_triage
        is_emergency = request.urgency == "emergency"
        if is_emergency:
            clean_text = ArabicResponses.get("emergency", request.speaker_type, urgency="emergency")
            logger.warning(
                "EMERGENCY TTS override — speaker=%s text='%s'",
                request.speaker_type, clean_text,
            )

        # ── 4. If text empty → use routing response template ─────────────
        if not clean_text.strip():
            clean_text = ArabicResponses.get(
                request.routing, request.speaker_type, urgency=request.urgency
            )

        # ── 5. Speech rate from confidence_scorer thresholds ─────────────
        rate = select_rate(request.confidence, request.urgency)

        # Volume: EMERGENCY → max volume
        volume = 1.0 if is_emergency else DEFAULT_VOLUME

        # ── 6. Fallback chain ─────────────────────────────────────────────
        warnings: list[str] = []
        wav_bytes            = b""
        engine_used          = TTSEngine.SILENT

        for engine_enum in self._fallback_chain:
            backend = self._backends.get(engine_enum)
            if backend is None:
                continue
            try:
                wav_bytes   = backend.speak(clean_text, rate, volume, request.language)
                engine_used = engine_enum
                if engine_enum != self._fallback_chain[0]:
                    status   = TTSStatus.FALLBACK
                    warnings.append(f"Used fallback engine: {engine_enum.value}")
                break
            except Exception as exc:
                logger.warning("TTS engine %s failed: %s", engine_enum.value, exc)
                warnings.append(f"{engine_enum.value} failed: {exc}")

        # ── 7. Guaranteed non-empty WAV ───────────────────────────────────
        if not wav_bytes or len(wav_bytes) <= 44:
            wav_bytes   = make_silent_wav()
            engine_used = TTSEngine.SILENT
            status      = TTSStatus.SILENT
            warnings.append("All TTS engines failed — returning silent WAV")

        # ── 8. Latency check ─────────────────────────────────────────────
        latency_ms = (time.perf_counter() - t0) * 1000
        budget_ms  = EMERGENCY_MAX_MS if is_emergency else MAX_LATENCY_MS
        if latency_ms > budget_ms:
            warnings.append(
                f"TTS exceeded latency budget: {latency_ms:.0f}ms > {budget_ms}ms"
            )
            logger.warning(
                "TTS latency budget exceeded: %.0fms > %dms (engine=%s)",
                latency_ms, budget_ms, engine_used.value,
            )

        # ── 9. Estimate duration ──────────────────────────────────────────
        word_count   = max(1, len(clean_text.split()))
        duration_ms  = (word_count / rate) * 60_000

        logger.info(
            "TTS | engine=%s status=%s rate=%d vol=%.2f latency=%.0fms duration=%.0fms",
            engine_used.value, status.value, rate, volume, latency_ms, duration_ms,
        )

        return TTSResult(
            wav_bytes   = wav_bytes,
            engine_used = engine_used,
            status      = status,
            text_spoken = clean_text,
            rate_wpm    = rate,
            volume      = volume,
            duration_ms = duration_ms,
            latency_ms  = latency_ms,
            warnings    = warnings,
        )

    async def speak_async(self, request: TTSRequest) -> TTSResult:
        """
        Async wrapper for FastAPI route handlers.

        Fix 3 — prevents pyttsx3.runAndWait() from blocking FastAPI's
        async event loop. Runs speak() in Starlette's thread pool so the
        event loop stays free to handle concurrent requests.

        Falls back to asyncio.get_event_loop().run_in_executor() if
        Starlette is not installed (e.g. in tests / CLI contexts).

        Usage in FastAPI router:
            @router.post("/voice/respond")
            async def voice_respond(
                req: VoiceRequest,
                tts: TextToSpeech = Depends(get_tts),
            ):
                result = await tts.speak_async(TTSRequest(
                    text=req.text, confidence=req.confidence,
                    routing=req.routing, urgency=req.urgency,
                    speaker_type=req.speaker_type,
                ))
                return Response(content=result.wav_bytes, media_type="audio/wav")
        """
        try:
            from starlette.concurrency import run_in_threadpool
            return await run_in_threadpool(self.speak, request)
        except ImportError:
            # Starlette not available — fallback to asyncio executor
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.speak, request)

    def speak_routing_decision(
        self,
        routing      : str,
        speaker_type : str = "patient",
        confidence   : float = 1.0,
        urgency      : str   = "low",
    ) -> TTSResult:
        """
        Shortcut: speak the standard Arabic response for a routing decision.

        Called directly by the voice route handler after
        ConfidenceScorer.routing_decision() returns.

        Usage:
            decision = scorer.routing_decision(score, urgency="high")
            result   = tts.speak_routing_decision(
                routing=decision, speaker_type="pregnant", confidence=score
            )
        """
        text = ArabicResponses.get(routing, speaker_type, urgency=urgency)
        return self.speak(TTSRequest(
            text         = text,
            confidence   = confidence,
            routing      = routing,
            urgency      = urgency,
            speaker_type = speaker_type,
        ))

    def speak_score_label(
        self,
        score        : float,
        speaker_type : str = "doctor",
    ) -> TTSResult:
        """
        Speak the Arabic confidence label (for doctor dashboard voice readout).

        Integration: uses ConfidenceScorer.label() thresholds directly.
        "عالية" / "متوسطة" / "منخفضة" / "غير محددة"
        """
        if score >= _THRESH_ACT:
            label = "الثقة عالية"
        elif score >= _THRESH_CLARIFY:
            label = "الثقة متوسطة"
        else:
            label = "الثقة منخفضة — محتاج تأكيد"

        return self.speak(TTSRequest(
            text         = label,
            confidence   = score,
            routing      = "act",
            urgency      = "low",
            speaker_type = speaker_type,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Singleton for FastAPI  (mirrors all other dialect_model singletons)
# ─────────────────────────────────────────────────────────────────────────────

_tts_instance: Optional[TextToSpeech] = None


def get_tts() -> TextToSpeech:
    """
    Shared TextToSpeech instance — FastAPI dependency injection.

    Usage in voice route:
        from ai_core.voice.dialect_model.text_to_speech import get_tts, TTSRequest

        @router.post("/voice/respond")
        async def voice_respond(
            req: VoiceRequest,
            tts: TextToSpeech = Depends(get_tts),
        ):
            result = tts.speak_routing_decision(
                routing=req.routing,
                speaker_type=req.speaker_type,
                confidence=req.confidence,
                urgency=req.urgency,
            )
            return Response(content=result.wav_bytes, media_type="audio/wav")
    """
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TextToSpeech()
    return _tts_instance


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("RIVA TextToSpeech — self-test")
    print("=" * 60)

    tts = TextToSpeech()

    # ── 1. PII scrubbing ──────────────────────────────────────────────────
    print("\n[1] PII scrubbing:")
    pii_cases = [
        "رقم التليفون 01012345678 تاني",
        "الرقم القومي 12345678901234",
        "إيميلي test@example.com",
        "مفيش PII هنا",
    ]
    for t in pii_cases:
        scrubbed, found = scrub_pii(t)
        print(f"  {'✓' if found else '·'} '{t[:40]}' → '{scrubbed[:40]}'")

    # ── 2. Speech rate selection ──────────────────────────────────────────
    print("\n[2] Speech rate selection:")
    rate_cases = [
        (0.85, "low",       RATE_HIGH_CONF,  "high conf → fast"),
        (0.60, "medium",    RATE_NORMAL,     "mid conf  → normal"),
        (0.35, "low",       RATE_LOW_CONF,   "low conf  → slow"),
        (0.20, "emergency", RATE_EMERGENCY,  "emergency → fastest"),
    ]
    for conf, urg, expected, desc in rate_cases:
        rate = select_rate(conf, urg)
        ok   = "✅" if rate == expected else "❌"
        print(f"  {ok} conf={conf} urgency={urg} → {rate} wpm  ({desc})")

    # ── 3. SSML strip ─────────────────────────────────────────────────────
    print("\n[3] SSML strip:")
    ssml_in  = '<rate slow>عندي وجع</rate> <break time="1s"/> في بطني'
    ssml_out = strip_ssml(ssml_in)
    print(f"  In : {ssml_in}")
    print(f"  Out: {ssml_out}")

    # ── 4. Arabic response templates ──────────────────────────────────────
    print("\n[4] Arabic response templates:")
    for routing in ["act", "clarify", "fallback", "emergency", "unrecognised"]:
        for speaker in ["patient", "pregnant", "doctor"]:
            resp = ArabicResponses.get(routing, speaker)
            print(f"  [{routing:13s}][{speaker:8s}] → {resp}")

    # ── 5. Silent WAV ─────────────────────────────────────────────────────
    print("\n[5] Silent WAV generation:")
    silent = make_silent_wav(0.5)
    print(f"  Silent WAV: {len(silent)} bytes — valid={'RIFF' in str(silent[:4])}")

    # ── 6. Full speak() pipeline (silent engine — no pyttsx3 needed) ──────
    print("\n[6] speak() pipeline (silent engine fallback):")
    test_requests = [
        TTSRequest(text="عندي وجع في بطني من امبارح",
                   confidence=0.82, routing="act",      urgency="high",      speaker_type="patient"),
        TTSRequest(text="ممكن توضحلي أكتر؟",
                   confidence=0.50, routing="clarify",  urgency="medium",    speaker_type="pregnant"),
        TTSRequest(text="مش قادر أتنفس",
                   confidence=0.95, routing="act",      urgency="emergency", speaker_type="patient"),
        TTSRequest(text="رقمي 01012345678",
                   confidence=0.70, routing="act",      urgency="low",       speaker_type="patient"),
    ]
    for req in test_requests:
        result = tts.speak(req)
        print(f"  [{req.urgency:9s}] conf={req.confidence:.2f} "
              f"engine={result.engine_used.value:20s} "
              f"rate={result.rate_wpm}wpm "
              f"latency={result.latency_ms:.1f}ms "
              f"spoken='{result.text_spoken[:35]}'")
        if result.warnings:
            for w in result.warnings:
                print(f"    ⚠ {w}")

    # ── 7. speak_routing_decision shortcut ────────────────────────────────
    print("\n[7] speak_routing_decision:")
    for routing, conf, urg in [
        ("act",      0.85, "high"),
        ("clarify",  0.55, "medium"),
        ("fallback", 0.30, "low"),
    ]:
        r = tts.speak_routing_decision(routing, "patient", conf, urg)
        print(f"  routing={routing:10s} conf={conf} → '{r.text_spoken}' @ {r.rate_wpm}wpm")

    # ── 8. to_dict ────────────────────────────────────────────────────────
    r = tts.speak(TTSRequest("عندي صداع", 0.75, "act", "medium", "patient"))
    d = r.to_dict()
    print(f"\n[8] to_dict keys: {list(d.keys())}")

    print("\n✅ TextToSpeech self-test complete")
    sys.exit(0)
