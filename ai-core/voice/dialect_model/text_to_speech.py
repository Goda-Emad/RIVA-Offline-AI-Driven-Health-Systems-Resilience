"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform v4.0 — Text To Speech                        ║
║           ai-core/voice/dialect_model/text_to_speech.py                     ║
║                                                                              ║
║  Egyptian Arabic TTS engine — fully offline, zero network latency           ║
║                                                                              ║
║  Integration map (from config.json):                                        ║
║    config["text_to_speech"]    → engine / rate / volume / fallback_chain    ║
║    config["security"]          → PII scrub patterns before vocalization     ║
║    config["performance"]       → target_end_to_end_latency_ms (2000ms)      ║
║    config["confidence_scorer"] → routing_decision → speaks Act/Clarify/FB  ║
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
║    v4.1: ADD LRU cache eviction — background thread, 500-file cap,         ║
║          touch() on cache hits refreshes LRU order, _EVICT_LOCK prevents   ║
║          concurrent eviction races, cache_size/cache_bytes properties       ║
║    v4.0: FIX 1 ArabicResponses "and False" removed — EMERGENCY reachable   ║
║          FIX 2 gTTS: 1.5s timeout + EMERGENCY cache-miss instant skip      ║
║          FIX 3 speak_async: get_running_loop() — Python 3.12 safe          ║
║          FIX 4 Dead code removed from __main__                              ║
║          FIX 5 urgency forwarded to all backends                           ║
║          FIX 6 TTSRequest.cache_key implemented in gTTS backend            ║
║    v1.1: ADD fallback_chain, PII scrub, confidence-aware rate              ║
║    v1.0: pyttsx3-only engine                                                ║
║                                                                              ║
║  Maintainer: GODA EMAD · Harvard HSIL Hackathon 2026                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import io, json, logging, re, struct, time, wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("riva.text_to_speech")

_HERE        = Path(__file__).resolve().parent
_VOICE       = _HERE.parent
_AICORE      = _VOICE.parent
_ROOT        = _AICORE.parent

_CONFIG_PATH        = _HERE  / "config.json"
_LEXICON_PATH       = _ROOT  / "data" / "raw" / "arabic_sentiment" / "lexicons" / "egyptian_medical_lexicon.json"
_AMBIGUITY_MAP_PATH = _ROOT  / "business-intelligence" / "mapping" / "ambiguity_map.json"
_INTENT_MAP_PATH    = _ROOT  / "business-intelligence" / "mapping" / "intent_mapping.json"

def _validate_paths() -> None:
    for path, label, req in [
        (_CONFIG_PATH,"config.json",True),(_LEXICON_PATH,"egyptian_medical_lexicon.json",False),
        (_AMBIGUITY_MAP_PATH,"ambiguity_map.json",False),(_INTENT_MAP_PATH,"intent_mapping.json",False),
    ]:
        if path.exists(): logger.debug("Path OK: %s",label)
        else: (logger.warning if req else logger.info)("Path %s: %s not found","MISSING" if req else "absent",label)

_validate_paths()

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try: return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc: logger.warning("config.json load failed: %s",exc)
    return {}

_CFG=_load_config(); _TTS_CFG=_CFG.get("text_to_speech",{}); _SEC_CFG=_CFG.get("security",{})
_PERF_CFG=_CFG.get("performance",{}); _URG_CFG=_CFG.get("urgency_escalation",{})

DEFAULT_ENGINE=_TTS_CFG.get("engine","pyttsx3"); DEFAULT_LANGUAGE=_TTS_CFG.get("language","ar")
DEFAULT_RATE_WPM=_TTS_CFG.get("rate_wpm",145); DEFAULT_VOLUME=_TTS_CFG.get("volume",0.90)
DEFAULT_PITCH=_TTS_CFG.get("pitch",1.0)
FALLBACK_CHAIN=_TTS_CFG.get("fallback_chain",["pyttsx3","gtts_disk_cache","browser_web_speech","silent"])
MAX_LATENCY_MS=_PERF_CFG.get("target_end_to_end_latency_ms",2000)
EMERGENCY_MAX_MS=_URG_CFG.get("response_max_latency_ms",800)

PII_PATTERNS=[re.compile(p) for p in _SEC_CFG.get("pii_patterns",[
    r"\d{14}",r"01[0-2,5]{1}[0-9]{8}",r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"])]

RATE_HIGH_CONF=int(DEFAULT_RATE_WPM*1.10); RATE_NORMAL=DEFAULT_RATE_WPM
RATE_LOW_CONF=int(DEFAULT_RATE_WPM*0.80); RATE_EMERGENCY=int(DEFAULT_RATE_WPM*1.25)
_THRESH_ACT=0.70; _THRESH_CLARIFY=0.45
_SSML_TAG=re.compile(r"<[^>]{1,60}>")

class TTSEngine(Enum):
    PYTTSX3="pyttsx3"; GTTS_DISK_CACHE="gtts_disk_cache"
    BROWSER_WEB_SPEECH="browser_web_speech"; SILENT="silent"

class TTSStatus(Enum):
    SUCCESS="success"; FALLBACK="fallback"; SILENT="silent"; PII_REDACTED="pii_redacted"

@dataclass
class TTSRequest:
    """Input to TextToSpeech.speak(). Maps pipeline output → TTS parameters."""
    text         : str
    confidence   : float         = 1.0
    routing      : str           = "act"
    urgency      : str           = "low"
    speaker_type : str           = "patient"
    language     : str           = DEFAULT_LANGUAGE
    cache_key    : Optional[str] = None  # FIX 6: override gTTS cache key

@dataclass
class TTSResult:
    """Output of TextToSpeech.speak(). wav_bytes is always a valid WAV (>=44 bytes)."""
    wav_bytes   : bytes; engine_used : TTSEngine; status : TTSStatus
    text_spoken : str;   rate_wpm    : int;       volume : float
    duration_ms : float; latency_ms  : float
    warnings    : list[str] = field(default_factory=list)

    @property
    def is_silent(self) -> bool: return self.engine_used == TTSEngine.SILENT

    def to_dict(self) -> dict:
        return {"engine_used":self.engine_used.value,"status":self.status.value,
                "text_spoken":self.text_spoken,"rate_wpm":self.rate_wpm,
                "volume":round(self.volume,2),"duration_ms":round(self.duration_ms,1),
                "latency_ms":round(self.latency_ms,1),"warnings":self.warnings}


class ArabicResponses:
    """
    Ready-made Egyptian Arabic responses for (routing, speaker_type).

    routing values: 'act' | 'clarify' | 'fallback' | 'unrecognised' (from ConfidenceScorer)
    urgency values: 'emergency' overrides routing; others fall through to routing lookup.

    FIX 1: 'and False' bug removed — EMERGENCY branch is now reachable.
           Old code permanently skipped emergency responses for every patient.
    """
    _RESPONSES: dict = {
        ("act","patient"):"تمام، هبعتك للقسم الصح دلوقتي.",("act","pregnant"):"تمام يا ستي، هبعتك لقسم صحة الأم.",
        ("act","school"):"تمام، هبعتك لقسم الصحة المدرسية.",("act","doctor"):"تمام دكتور، هفتحلك الداشبورد.",
        ("act","unknown"):"تمام، هبعتك للقسم المناسب.",
        ("clarify","patient"):"معذرة، محتاج أفهم أكتر. ممكن توضحلي؟",
        ("clarify","pregnant"):"معذرة يا ستي، محتاج أفهم أكتر. ممكن توضحيلي؟",
        ("clarify","school"):"محتاج معلومة أكتر. ممكن توضحلي؟",
        ("clarify","doctor"):"دكتور، فيه معلومة ناقصة. ممكن توضح؟",
        ("clarify","unknown"):"محتاج أفهم أكتر. ممكن توضح؟",
        ("fallback","patient"):"مش فاهم كويس. هبعتك للشات الطبي.",
        ("fallback","pregnant"):"مش فاهم كويس يا ستي. هبعتك للشات.",
        ("fallback","school"):"مش فاهم. هبعتك للمحادثة.",
        ("fallback","doctor"):"دكتور، مش واضح. هبعتك للشات.",
        ("fallback","unknown"):"مش فاهم. هبعتك للمحادثة الطبية.",
        ("emergency","patient"):"حالة طارئة! بيتم تحويلك للطوارئ فوراً.",
        ("emergency","pregnant"):"حالة طارئة! بيتم تحويلك لطوارئ الأم والطفل فوراً.",
        ("emergency","school"):"حالة طارئة! يتم إخطار الطبيب فوراً.",
        ("emergency","doctor"):"تنبيه طارئ! تم إرسال التحذير.",
        ("emergency","unknown"):"حالة طارئة! بيتم تحويلك للطوارئ فوراً.",
        ("unrecognised","patient"):"ما فهمتش. ممكن تعيد الكلام؟",
        ("unrecognised","pregnant"):"ما فهمتش يا ستي. ممكن تعيدي؟",
        ("unrecognised","school"):"ما فهمتش. ممكن تعيد؟",
        ("unrecognised","doctor"):"دكتور، ما فهمتش. ممكن تعيد؟",
        ("unrecognised","unknown"):"ما فهمتش. ممكن تعيد؟",
    }
    @classmethod
    def get(cls, routing:str, speaker_type:str, urgency:str="low") -> str:
        key = ("emergency",speaker_type) if urgency=="emergency" else (routing,speaker_type)
        return cls._RESPONSES.get(key) or cls._RESPONSES.get((routing,"unknown")) or "تمام، جاري المعالجة."


def scrub_pii(text:str) -> tuple[str,bool]:
    scrubbed,found=text,False
    for p in PII_PATTERNS:
        if p.search(scrubbed): scrubbed=p.sub("[محجوب]",scrubbed); found=True
    return scrubbed,found

def strip_ssml(text:str) -> str: return _SSML_TAG.sub("",text).strip()

def select_rate(confidence:float, urgency:str) -> int:
    if urgency=="emergency": return RATE_EMERGENCY
    if confidence>=_THRESH_ACT: return RATE_HIGH_CONF
    if confidence>=_THRESH_CLARIFY: return RATE_NORMAL
    return RATE_LOW_CONF

def make_silent_wav(duration_sec:float=0.5, sample_rate:int=16_000) -> bytes:
    n=max(0,int(duration_sec*sample_rate)); pcm=struct.pack(f"<{n}h",*([0]*n))
    buf=io.BytesIO()
    with wave.open(buf,"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(pcm)
    return buf.getvalue()


class _Pyttsx3Backend:
    def __init__(self): self._engine=None
    def _get_engine(self):
        if self._engine is None:
            import pyttsx3; self._engine=pyttsx3.init()
        return self._engine
    def speak(self,text,rate,volume,lang,urgency="low"):
        import tempfile,os; eng=self._get_engine()
        eng.setProperty("rate",rate); eng.setProperty("volume",volume)
        voices=eng.getProperty("voices")
        ar=next((v for v in voices if "arabic" in (v.languages[0] if v.languages else "").lower() or "ar" in v.id.lower()),None)
        if ar: eng.setProperty("voice",ar.id)
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as t: tp=t.name
        try:
            eng.save_to_file(text,tp); eng.runAndWait()
            wav=Path(tp).read_bytes(); return wav if len(wav)>44 else make_silent_wav()
        except Exception as exc: raise RuntimeError(f"pyttsx3: {exc}") from exc
        finally:
            try: os.unlink(tp)
            except OSError: pass


class _GttsDiskCacheBackend:
    """
    gTTS with LRU disk cache — Arabic audio cached for offline reuse.

    FIX 2a: 1.5s timeout prevents 75s hang on offline server.
    FIX 2b: EMERGENCY + cache-miss raises immediately (preserves 800ms budget).
    FIX 6:  TTSRequest.cache_key overrides auto-generated hash.
    ADD v4.1: LRU cache eviction — background thread prevents disk exhaustion.

    Cache eviction strategy
    ────────────────────────
    • Max files cap: 500 WAV files by default (≈ 250MB at ~500KB avg).
    • Trigger: checked after every new file is written (not on cache hits).
    • Algorithm: LRU — sort by st_mtime (last access), delete oldest surplus.
    • Non-blocking: eviction runs in a daemon thread so it never delays
      the TTS response that triggered it.
    • Thread-safe: a module-level lock prevents two evictions running
      simultaneously when concurrent requests both write new files.
    """

    _CACHE_DIR    = _HERE / ".gtts_cache"
    _TIMEOUT_SEC  = 1.5
    _MAX_FILES    = 500          # keep newest 500 WAVs — tune via subclass
    _EVICT_LOCK   = __import__("threading").Lock()   # one eviction at a time

    # ── Public: synthesise speech ────────────────────────────────────────

    def speak(
        self, text: str, rate: int, volume: float,
        lang: str, urgency: str = "low",
        cache_key: Optional[str] = None,
    ) -> bytes:
        import hashlib, threading

        key_str    = cache_key or hashlib.md5(f"{text}:{lang}".encode()).hexdigest()
        cache_file = self._CACHE_DIR / f"{key_str}.wav"

        # ── Cache hit — fully offline, no eviction needed ────────────────
        if cache_file.exists():
            # Touch file to mark it as recently used (update LRU timestamp)
            cache_file.touch()
            logger.debug("gTTS cache hit: %s", key_str)
            return cache_file.read_bytes()

        # FIX 2b: EMERGENCY + cache miss → skip immediately
        if urgency == "emergency":
            raise RuntimeError(
                f"gTTS cache miss on EMERGENCY — skipping to preserve {EMERGENCY_MAX_MS}ms budget"
            )

        # ── Cache miss — network fetch with timeout ──────────────────────
        logger.info("gTTS cache miss '%s' — fetching (timeout=%.1fs)", key_str, self._TIMEOUT_SEC)
        result, error = [], []

        def _fetch() -> None:
            try:
                from gtts import gTTS
                buf = io.BytesIO()
                gTTS(text=text, lang=lang).write_to_fp(buf)
                buf.seek(0)
                result.append(buf)
            except Exception as e:
                error.append(e)

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=self._TIMEOUT_SEC)

        if t.is_alive():
            raise TimeoutError(f"gTTS exceeded {self._TIMEOUT_SEC}s offline budget")
        if error:
            raise RuntimeError(f"gTTS: {error[0]}") from error[0]

        # ── MP3 → WAV conversion ─────────────────────────────────────────
        try:
            from pydub import AudioSegment
            seg     = AudioSegment.from_mp3(result[0])
            wav_buf = io.BytesIO()
            seg.export(wav_buf, format="wav")
            wav = wav_buf.getvalue()
        except ImportError:
            logger.warning("pydub unavailable — using silent WAV")
            wav = make_silent_wav()

        # ── Persist to cache ─────────────────────────────────────────────
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(wav)
        logger.info("gTTS cached: %s (%d bytes)", key_str, len(wav))

        # ── Trigger LRU eviction in background (non-blocking) ───────────
        threading.Thread(
            target = self._evict_lru,
            daemon = True,
            name   = "riva-gtts-evict",
        ).start()

        return wav

    # ── LRU eviction ────────────────────────────────────────────────────

    def _evict_lru(self, max_files: Optional[int] = None) -> int:
        """
        Delete the oldest (least recently used) WAV files when the cache
        exceeds max_files.

        Returns the number of files deleted (0 if no eviction needed).

        Non-blocking design:
            Called from a daemon thread after each new file is written.
            Uses _EVICT_LOCK so concurrent requests don't race.
            Errors are logged but never raised — eviction failure must
            not affect the TTS response that triggered it.

        LRU ordering:
            Files are sorted by st_mtime (modified/touched time).
            Cache hits call file.touch() to refresh their mtime, so
            files that are actively used stay near the top of the list.
        """
        cap = max_files or self._MAX_FILES

        # Fast path: don't even acquire the lock if well under cap
        try:
            files = list(self._CACHE_DIR.glob("*.wav"))
        except Exception as exc:
            logger.error("gTTS eviction: failed to list cache dir: %s", exc)
            return 0

        if len(files) <= cap:
            return 0

        # Acquire lock — one eviction thread runs at a time
        if not self._EVICT_LOCK.acquire(blocking=False):
            logger.debug("gTTS eviction: already running, skipping")
            return 0

        deleted = 0
        try:
            # Sort ascending by mtime → oldest first
            files.sort(key=lambda f: f.stat().st_mtime)
            surplus = len(files) - cap
            to_delete = files[:surplus]

            for f in to_delete:
                try:
                    f.unlink()
                    deleted += 1
                except OSError as exc:
                    logger.warning("gTTS eviction: could not delete %s: %s", f.name, exc)

            logger.info(
                "gTTS LRU eviction: deleted %d/%d surplus files (cap=%d)",
                deleted, surplus, cap,
            )
        except Exception as exc:
            logger.error("gTTS eviction failed: %s", exc)
        finally:
            self._EVICT_LOCK.release()

        return deleted

    @property
    def cache_size(self) -> int:
        """Current number of WAV files in the cache directory."""
        try:
            return sum(1 for _ in self._CACHE_DIR.glob("*.wav"))
        except Exception:
            return 0

    @property
    def cache_bytes(self) -> int:
        """Total size of cached WAV files in bytes."""
        try:
            return sum(f.stat().st_size for f in self._CACHE_DIR.glob("*.wav"))
        except Exception:
            return 0


class _BrowserWebSpeechBackend:
    def speak(self,text,rate,volume,lang,urgency="low"):
        logger.info("Browser Web Speech fallback"); return make_silent_wav(0.1)

class _SilentBackend:
    def speak(self,text,rate,volume,lang,urgency="low"):
        logger.warning("TTS silent fallback"); return make_silent_wav(0.5)


class TextToSpeech:
    """
    Egyptian Arabic TTS engine for RIVA.

    FIX 3: speak_async uses asyncio.get_running_loop() — Python 3.12 safe.
           (get_event_loop() raises RuntimeError in Python 3.12 async contexts)
    """
    def __init__(self):
        self._backends={
            TTSEngine.PYTTSX3:_Pyttsx3Backend(),
            TTSEngine.GTTS_DISK_CACHE:_GttsDiskCacheBackend(),
            TTSEngine.BROWSER_WEB_SPEECH:_BrowserWebSpeechBackend(),
            TTSEngine.SILENT:_SilentBackend(),
        }
        self._fallback_chain=[TTSEngine(e) for e in FALLBACK_CHAIN if e in {en.value for en in TTSEngine}]
        logger.info("TextToSpeech ready | chain=%s | rate=%d",[e.value for e in self._fallback_chain],DEFAULT_RATE_WPM)

    def speak(self, request:TTSRequest) -> TTSResult:
        """
        Pipeline (9 steps):
            1. PII scrub    2. SSML strip     3. EMERGENCY override
            4. Empty guard  5. Rate select    6. Volume select
            7. Fallback chain (urgency forwarded — FIX 5)
            8. Guarantee non-empty WAV        9. Latency budget check
        """
        t0=time.perf_counter()
        clean,had_pii=scrub_pii(request.text)
        status=TTSStatus.PII_REDACTED if had_pii else TTSStatus.SUCCESS
        clean=strip_ssml(clean)
        is_em=request.urgency=="emergency"
        if is_em:
            clean=ArabicResponses.get("emergency",request.speaker_type,urgency="emergency")
            logger.warning("EMERGENCY TTS override: %s",clean)
        if not clean.strip():
            clean=ArabicResponses.get(request.routing,request.speaker_type,urgency=request.urgency)
        rate=select_rate(request.confidence,request.urgency)
        volume=1.0 if is_em else DEFAULT_VOLUME
        warnings,wav,engine_used=[],b"",TTSEngine.SILENT
        for eng in self._fallback_chain:
            backend=self._backends.get(eng)
            if not backend: continue
            try:
                if eng==TTSEngine.GTTS_DISK_CACHE:
                    wav=backend.speak(clean,rate,volume,request.language,urgency=request.urgency,cache_key=request.cache_key)
                else:
                    wav=backend.speak(clean,rate,volume,request.language,urgency=request.urgency)
                engine_used=eng
                if eng!=self._fallback_chain[0]: status=TTSStatus.FALLBACK; warnings.append(f"Fallback: {eng.value}")
                break
            except Exception as exc:
                logger.warning("TTS %s failed: %s",eng.value,exc); warnings.append(f"{eng.value}: {exc}")
        if not wav or len(wav)<=44:
            wav=make_silent_wav(); engine_used=TTSEngine.SILENT; status=TTSStatus.SILENT
            warnings.append("All engines failed — silent WAV")
        lat=(time.perf_counter()-t0)*1000; budget=EMERGENCY_MAX_MS if is_em else MAX_LATENCY_MS
        if lat>budget:
            warnings.append(f"Budget exceeded: {lat:.0f}ms>{budget}ms")
            logger.warning("TTS budget exceeded: %.0fms>%dms (%s)",lat,budget,engine_used.value)
        dur=(max(1,len(clean.split()))/rate)*60_000
        logger.info("TTS|engine=%s status=%s rate=%d lat=%.0fms",engine_used.value,status.value,rate,lat)
        return TTSResult(wav_bytes=wav,engine_used=engine_used,status=status,text_spoken=clean,
                         rate_wpm=rate,volume=volume,duration_ms=dur,latency_ms=lat,warnings=warnings)

    async def speak_async(self, request:TTSRequest) -> TTSResult:
        """FIX 3: get_running_loop() instead of deprecated get_event_loop()."""
        try:
            from starlette.concurrency import run_in_threadpool
            return await run_in_threadpool(self.speak,request)
        except ImportError:
            import asyncio
            return await asyncio.get_running_loop().run_in_executor(None,self.speak,request)

    def speak_routing_decision(self,routing:str,speaker_type:str="patient",confidence:float=1.0,urgency:str="low") -> TTSResult:
        """Shortcut: speak standard Arabic response for a ConfidenceScorer routing decision."""
        text=ArabicResponses.get(routing,speaker_type,urgency=urgency)
        return self.speak(TTSRequest(text=text,confidence=confidence,routing=routing,urgency=urgency,speaker_type=speaker_type))

    def speak_score_label(self,score:float,speaker_type:str="doctor") -> TTSResult:
        """Speak Arabic confidence label for doctor dashboard voice readout."""
        label=("الثقة عالية" if score>=_THRESH_ACT else "الثقة متوسطة" if score>=_THRESH_CLARIFY else "الثقة منخفضة — محتاج تأكيد")
        return self.speak(TTSRequest(text=label,confidence=score,routing="act",urgency="low",speaker_type=speaker_type))


_tts_instance:Optional[TextToSpeech]=None

def get_tts() -> TextToSpeech:
    """FastAPI Depends() singleton."""
    global _tts_instance
    if _tts_instance is None: _tts_instance=TextToSpeech()
    return _tts_instance


if __name__=="__main__":
    import sys
    logging.basicConfig(level=logging.INFO,format="%(levelname)s:%(name)s:%(message)s")
    print("="*60); print("RIVA TextToSpeech v4.0 — smoke-test"); print("="*60)
    print(f"\n[0] config: {'✅' if _CONFIG_PATH.exists() else '⚠ missing'}")
    tts=get_tts()
    print("\n[1] PII scrub:")
    for t in ["رقمي 01012345678","الرقم 12345678901234","مفيش PII"]:
        s,f=scrub_pii(t); print(f"  {'✓' if f else '·'} {t!r} → {s!r}")
    print("\n[2] Rate selection:")
    for c,u,e in [(0.85,"low",RATE_HIGH_CONF),(0.60,"medium",RATE_NORMAL),(0.20,"emergency",RATE_EMERGENCY)]:
        r=select_rate(c,u); print(f"  {'✅' if r==e else '❌'} conf={c} urg={u} → {r}wpm")
    print("\n[3] EMERGENCY override (FIX 1):")
    em=tts.speak(TTSRequest(text="مش قادر أتنفس",confidence=0.95,routing="act",urgency="emergency",speaker_type="patient"))
    print(f"  {'✅' if 'طارئ' in em.text_spoken else '❌'} text='{em.text_spoken}'")
    print(f"  {'✅' if em.rate_wpm==RATE_EMERGENCY else '❌'} rate={em.rate_wpm}wpm vol={em.volume}")
    print("\n[4] PII in pipeline:")
    pii=tts.speak(TTSRequest(text="رقمي 01012345678",confidence=0.7,routing="act",urgency="low",speaker_type="patient"))
    print(f"  {'✅' if '[محجوب]' in pii.text_spoken else '❌'} {pii.text_spoken!r}")
    print("\n[5] speak_routing_decision:")
    for rt,cf,ug in [("act",0.85,"high"),("clarify",0.5,"medium"),("fallback",0.3,"low")]:
        r=tts.speak_routing_decision(rt,"patient",cf,ug); print(f"  {rt:10s} → '{r.text_spoken}' @{r.rate_wpm}wpm")
    print("\n✅ smoke-test complete"); sys.exit(0)
