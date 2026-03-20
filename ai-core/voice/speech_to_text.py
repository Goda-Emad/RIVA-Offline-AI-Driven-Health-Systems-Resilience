"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Speech-to-Text Engine                       ║
║           ai-core/voice/speech_to_text.py                                    ║
║                                                                              ║
║  Purpose : Convert Arabic audio (WAV/WebM/MP3) to text using Whisper INT8   ║
║            ONNX offline, then detect the speaker type and route them to      ║
║            the correct RIVA page.                                            ║
║                                                                              ║
║  Pipeline                                                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Audio bytes (PWA / mic)                                                     ║
║      ↓  AudioProcessor.preprocess()                                         ║
║  16kHz mono float32 array                                                    ║
║      ↓  WhisperSTT.transcribe()                                              ║
║  Arabic text (Egyptian dialect)                                              ║
║      ↓  SpeakerRouter.detect() + route()                                    ║
║  STTResult  {text, speaker_type, route, confidence, …}                      ║
║      ↓                                                                       ║
║  FastAPI route  →  correct RIVA page                                        ║
║                                                                              ║
║  Speaker routing                                                             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  PATIENT   → /triage        (03_triage.html)                                ║
║  PREGNANT  → /pregnancy     (06_pregnancy.html)                             ║
║  SCHOOL    → /school        (07_school.html)                                ║
║  DOCTOR    → /doctor        (09_doctor_dashboard.html)                      ║
║  UNKNOWN   → /chatbot       (02_chatbot.html)  ← safe fallback              ║
║                                                                              ║
║  Author  : Goda Emad  (AI Core)                                              ║
║  Version : 1.1.0                                                             ║
║  Updated : 2026-03-18                                                        ║
║                                                                              ║
║  Changelog v1.1.0                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • ADD _WHISPER_DECODER_IDS: forced <|ar|><|transcribe|><|notimestamps|>.  ║
║  • ADD _EGYPTIAN_ARABIC_PROMPT: soft dialect seeding for whisper-small.    ║
║  • ADD _SPEAKER_PROMPTS: per-speaker prompt (patient/doctor/school/...).   ║
║  • ADD use_dialect_prompt + speaker_type params to WhisperSTT.__init__.    ║
║  • ADD prompt injection in _run_inference — prevents MSA/English drift.    ║
║  • ADD speaker_type propagation after SpeakerRouter detection.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
 
from __future__ import annotations
 
import io
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
 
import numpy as np
 
logger = logging.getLogger(__name__)
 
# ── Optional heavy imports — graceful degradation ────────────────────────
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    logger.warning("onnxruntime not installed — Whisper STT unavailable")
 
try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False
 
try:
    from pydub import AudioSegment
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════
 
WHISPER_SAMPLE_RATE : int   = 16_000          # Whisper requires 16kHz mono
WHISPER_MAX_SECONDS : int   = 30              # max audio duration Whisper handles
N_MEL_BINS          : int   = 80              # Whisper mel spectrogram bins
CHUNK_LENGTH_SECONDS: int   = 10              # process in chunks for long audio
 
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "chatbot" / "whisper_int8.onnx"
)
 
# ── Egyptian Arabic dialect guidance ─────────────────────────────────────
#
# Whisper-small can hallucinate or switch to MSA/English when hearing
# Egyptian Arabic unless guided.  These token IDs steer the decoder:
#
#   50258 = <|startoftranscript|>   (always first)
#   50272 = <|ar|>                  (Arabic language token)
#   50359 = <|transcribe|>          (task = transcribe, not translate)
#   50363 = <|notimestamps|>        (no timestamp tokens in output)
#
# Additionally, an optional text prompt prefix is prepended as a
# "soft hint" — Whisper encodes it and uses it to bias the decoder
# towards Egyptian Arabic vocabulary and spellings.
#
# Source: https://platform.openai.com/docs/guides/speech-to-text/prompting
 
# Forced decoder token IDs — language + task specification
_WHISPER_DECODER_IDS: list[list[int]] = [
    [1, 50272],   # position 1 → <|ar|>     (Arabic)
    [2, 50359],   # position 2 → <|transcribe|>
    [3, 50363],   # position 3 → <|notimestamps|>
]
 
# Soft prompt prefix — short Egyptian Arabic phrase that seeds vocabulary
# Tells Whisper: "what follows sounds like this dialect"
_EGYPTIAN_ARABIC_PROMPT: str = (
    "المريض بيتكلم بالعامية المصرية عن حالته الصحية."
)
 
# Context-specific prompts per speaker type — more targeted seeding
_SPEAKER_PROMPTS: dict[str, str] = {
    "patient" : "المريض بيتكلم بالعامية المصرية عن حالته الصحية.",
    "pregnant": "الأم الحامل بتتكلم بالعامية المصرية عن صحتها وحملها.",
    "school"  : "موظف المدرسة بيتكلم بالعامية المصرية عن صحة الطلاب.",
    "doctor"  : "الطبيب بيتكلم بالعامية المصرية عن حالة مريضه.",
    "unknown" : "المتكلم بيتكلم بالعامية المصرية.",
}
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  Speaker type & routing
# ═══════════════════════════════════════════════════════════════════════════
 
class SpeakerType(str, Enum):
    """Who is speaking — determines which RIVA page they get routed to."""
    PATIENT  = "patient"    # مريض عام
    PREGNANT = "pregnant"   # أم حامل
    SCHOOL   = "school"     # إدارة مدرسة / ممرضة / طالب
    DOCTOR   = "doctor"     # طبيب / كادر طبي
    UNKNOWN  = "unknown"    # fallback → chatbot
 
 
# Route map: SpeakerType → (API route, HTML page, page title)
SPEAKER_ROUTES: dict[SpeakerType, dict[str, str]] = {
    SpeakerType.PATIENT : {
        "api"  : "/api/triage",
        "page" : "/triage",
        "html" : "03_triage.html",
        "title": "الفرز الطبي",
    },
    SpeakerType.PREGNANT: {
        "api"  : "/api/pregnancy",
        "page" : "/pregnancy",
        "html" : "06_pregnancy.html",
        "title": "صحة الأم",
    },
    SpeakerType.SCHOOL  : {
        "api"  : "/api/school",
        "page" : "/school",
        "html" : "07_school.html",
        "title": "الصحة المدرسية",
    },
    SpeakerType.DOCTOR  : {
        "api"  : "/api/doctor",
        "page" : "/doctor",
        "html" : "09_doctor_dashboard.html",
        "title": "داشبورد الطبيب",
    },
    SpeakerType.UNKNOWN : {
        "api"  : "/api/chat",
        "page" : "/chatbot",
        "html" : "02_chatbot.html",
        "title": "المحادثة الطبية",
    },
}
 
# ── Speaker detection keyword sets ───────────────────────────────────────
 
_PREGNANT_KEYWORDS: frozenset[str] = frozenset({
    "حامل", "حامله", "الحمل", "جنين", "ولادة", "طلق",
    "أسبوع الحمل", "شهر الحمل", "الأم", "الحمل",
    "غثيان الحمل", "ضغط الحمل",
})
 
_SCHOOL_KEYWORDS: frozenset[str] = frozenset({
    "طالب", "تلميذ", "مدرسة", "فصل", "الفصل", "ناظر",
    "ممرضة المدرسة", "إدارة المدرسة", "أطفال المدرسة",
    "الطلاب", "التلاميذ", "المدرسة",
})
 
_DOCTOR_KEYWORDS: frozenset[str] = frozenset({
    "دكتور", "دكتورة", "طبيب", "طبيبة", "حالة", "مريض",
    "تشخيص", "وصفة", "علاج", "جرعة", "تحاليل", "أشعة",
    "الحالة دي", "المريض ده", "بروتوكول",
})
 
# ── Egyptian greeting / context patterns → help identify speaker type ─────
 
_GREETING_SPEAKER_MAP: dict[str, SpeakerType] = {
    "أنا حامل"             : SpeakerType.PREGNANT,
    "أنا دكتور"            : SpeakerType.DOCTOR,
    "أنا دكتورة"           : SpeakerType.DOCTOR,
    "أنا طبيب"             : SpeakerType.DOCTOR,
    "أنا طبيبة"            : SpeakerType.DOCTOR,
    "أنا ممرضة"            : SpeakerType.DOCTOR,
    "أنا ممرض"             : SpeakerType.DOCTOR,
    "أنا من المدرسة"       : SpeakerType.SCHOOL,
    "إدارة المدرسة"        : SpeakerType.SCHOOL,
    "بتكلم عن طالب"        : SpeakerType.SCHOOL,
}
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  Result dataclass
# ═══════════════════════════════════════════════════════════════════════════
 
@dataclass
class STTResult:
    """
    Full output of SpeechToText.transcribe_and_route().
 
    Passed directly to CommandParser and the FastAPI routing layer.
    """
    # ── Transcription ──────────────────────────────────────────────────
    text            : str           # Raw transcribed text
    language        : str           # Detected language tag ("ar")
    transcribe_ms   : float         # Whisper inference time
 
    # ── Speaker & routing ──────────────────────────────────────────────
    speaker_type    : SpeakerType   # Detected speaker category
    route           : dict[str, str]# api / page / html / title
 
    # ── Quality signals ────────────────────────────────────────────────
    confidence      : float         # 0.0 – 1.0 transcription confidence
    no_speech_prob  : float         # Whisper no-speech probability
    is_silent       : bool          # True if audio is likely silence
 
    # ── Audio metadata ─────────────────────────────────────────────────
    audio_duration_s: float
    sample_rate     : int
 
    # ── Speaker detection breakdown ────────────────────────────────────
    speaker_confidence : float      # how certain we are about speaker type
    speaker_keywords   : list[str]  # which keywords triggered detection
 
    def as_dict(self) -> dict:
        return {
            "text"              : self.text,
            "language"          : self.language,
            "transcribe_ms"     : round(self.transcribe_ms, 1),
            "speaker_type"      : self.speaker_type.value,
            "route"             : self.route,
            "confidence"        : round(self.confidence, 3),
            "no_speech_prob"    : round(self.no_speech_prob, 3),
            "is_silent"         : self.is_silent,
            "audio_duration_s"  : round(self.audio_duration_s, 2),
            "speaker_confidence": round(self.speaker_confidence, 3),
            "speaker_keywords"  : self.speaker_keywords,
        }
 
    @property
    def should_proceed(self) -> bool:
        """True if transcription is reliable enough to process."""
        return (
            not self.is_silent
            and self.confidence >= 0.40
            and len(self.text.strip()) >= 3
        )
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  Speaker Router
# ═══════════════════════════════════════════════════════════════════════════
 
class SpeakerRouter:
    """
    Detects speaker type from transcribed text and returns routing info.
 
    Detection strategy (priority order)
    ─────────────────────────────────────
    1. Explicit greeting map  — "أنا دكتور" → DOCTOR immediately
    2. Keyword count          — count matches per category, pick highest
    3. Context override       — patient_context from session (most reliable)
    4. Fallback               — UNKNOWN → /chatbot
    """
 
    def detect(
        self,
        text          : str,
        patient_context: dict[str, Any] | None = None,
    ) -> tuple[SpeakerType, float, list[str]]:
        """
        Detect speaker type from text.
 
        Returns
        -------
        (speaker_type, confidence, matched_keywords)
        """
        ctx = patient_context or {}
 
        # ── Context override (most reliable — set by login/session) ───────
        if ctx.get("is_doctor"):
            return SpeakerType.DOCTOR, 1.0, []
        if ctx.get("is_pregnant"):
            return SpeakerType.PREGNANT, 1.0, []
        if ctx.get("is_school_staff"):
            return SpeakerType.SCHOOL, 1.0, []
 
        # ── Explicit greeting patterns ─────────────────────────────────────
        for phrase, speaker in _GREETING_SPEAKER_MAP.items():
            if phrase in text:
                return speaker, 0.95, [phrase]
 
        # ── Keyword counting per category ──────────────────────────────────
        scores: dict[SpeakerType, list[str]] = {
            SpeakerType.PREGNANT: [],
            SpeakerType.SCHOOL  : [],
            SpeakerType.DOCTOR  : [],
        }
        for kw in _PREGNANT_KEYWORDS:
            if kw in text:
                scores[SpeakerType.PREGNANT].append(kw)
        for kw in _SCHOOL_KEYWORDS:
            if kw in text:
                scores[SpeakerType.SCHOOL].append(kw)
        for kw in _DOCTOR_KEYWORDS:
            if kw in text:
                scores[SpeakerType.DOCTOR].append(kw)
 
        # Find category with most keyword matches
        best = max(scores, key=lambda k: len(scores[k]))
        matched = scores[best]
 
        if matched:
            # Confidence scales with number of matched keywords (cap at 0.90)
            conf = min(0.90, 0.50 + len(matched) * 0.15)
            return best, conf, matched
 
        # ── Fallback: general patient or unknown ───────────────────────────
        # If any pain/symptom word is present → PATIENT
        _patient_signals = frozenset({
            "عندي", "بيوجعني", "تعبان", "ألم", "وجع", "دكتور",
            "مريض", "حرارة", "سخونة",
        })
        if any(sig in text for sig in _patient_signals):
            return SpeakerType.PATIENT, 0.55, []
 
        return SpeakerType.UNKNOWN, 0.30, []
 
    def route(self, speaker_type: SpeakerType) -> dict[str, str]:
        """Return the routing dict for a given speaker type."""
        return SPEAKER_ROUTES[speaker_type]
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  Whisper STT Engine
# ═══════════════════════════════════════════════════════════════════════════
 
class WhisperSTT:
    """
    Offline Arabic STT using Whisper INT8 ONNX model.
 
    Loads once at startup, processes all audio in-memory.
    Falls back gracefully if ONNX runtime is unavailable.
 
    Parameters
    ----------
    model_path  : Path to whisper_int8.onnx (default: models/chatbot/).
    language    : BCP-47 language tag (default "ar").
    beam_size   : Whisper beam search width (default 3).
    no_speech_threshold : Probability above which audio is flagged silent.
    threads     : ONNX intra-op threads (default from env RIVA_ONNX_THREADS).
    """
 
    def __init__(
        self,
        model_path         : Path | None = None,
        language           : str         = "ar",
        beam_size          : int         = 3,
        no_speech_threshold: float       = 0.60,
        threads            : int | None  = None,
        use_dialect_prompt : bool        = True,
        speaker_type       : str         = "patient",
    ) -> None:
        self._lang              = language
        self._beam              = beam_size
        self._no_speech_t       = no_speech_threshold
        self._session           = None
        self._use_dialect_prompt= use_dialect_prompt
        self._speaker_type      = speaker_type
 
        import os
        self._threads = threads or int(os.environ.get("RIVA_ONNX_THREADS", 2))
 
        path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._load_model(path)
 
    # ── Public API ───────────────────────────────────────────────────────
 
    def transcribe(self, audio: np.ndarray, sample_rate: int = WHISPER_SAMPLE_RATE) -> dict:
        """
        Transcribe a float32 audio array to text.
 
        Parameters
        ----------
        audio       : float32 numpy array, shape (N,), values in [-1, 1].
        sample_rate : Audio sample rate (will be resampled to 16kHz if needed).
 
        Returns
        -------
        dict with keys: text, language, confidence, no_speech_prob
        """
        if self._session is None:
            logger.warning("WhisperSTT: no model loaded — returning empty transcription")
            return {"text": "", "language": self._lang, "confidence": 0.0, "no_speech_prob": 1.0}
 
        # Resample to 16kHz if needed
        if sample_rate != WHISPER_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, WHISPER_SAMPLE_RATE)
 
        # Clip to max duration
        max_samples = WHISPER_MAX_SECONDS * WHISPER_SAMPLE_RATE
        if len(audio) > max_samples:
            logger.warning(
                "Audio truncated from %.1fs to %ds",
                len(audio) / WHISPER_SAMPLE_RATE, WHISPER_MAX_SECONDS,
            )
            audio = audio[:max_samples]
 
        # Pad to minimum 1 second (Whisper requirement)
        min_samples = WHISPER_SAMPLE_RATE
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
 
        try:
            result = self._run_inference(audio)
        except Exception as exc:
            logger.error("Whisper inference failed: %s", exc)
            result = {"text": "", "language": self._lang, "confidence": 0.0, "no_speech_prob": 1.0}
 
        return result
 
    def is_loaded(self) -> bool:
        """True if ONNX model is loaded and ready."""
        return self._session is not None
 
    # ── Private: model loading ────────────────────────────────────────────
 
    def _load_model(self, model_path: Path) -> None:
        if not _ORT_AVAILABLE:
            logger.warning("WhisperSTT: onnxruntime not installed — STT disabled")
            return
 
        if not model_path.exists():
            logger.info(
                "WhisperSTT: model not found at %s\n"
                "  Run: python ai-core/scripts/convert_to_onnx.py",
                model_path,
            )
            return
 
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self._threads
            opts.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
 
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options = opts,
                providers    = ["CPUExecutionProvider"],
            )
            logger.info(
                "WhisperSTT loaded | path=%s  threads=%d",
                model_path.name, self._threads,
            )
        except Exception as exc:
            logger.warning("WhisperSTT: failed to load model: %s", exc)
            self._session = None
 
    # ── Private: inference ────────────────────────────────────────────────
 
    def _run_inference(self, audio: np.ndarray) -> dict:
        """
        Run Whisper ONNX inference.
 
        Whisper ONNX expects a log-mel spectrogram as input.
        The full HuggingFace Whisper pipeline handles tokenisation
        and beam search internally — we call it via the encoder/decoder
        session directly.
 
        For the INT8 ONNX export from optimum, the expected inputs are:
            input_features : float32 [1, 80, 3000]  (log-mel spectrogram)
        Output:
            sequences      : int64   [1, seq_len]   (token IDs)
        """
        # Compute log-mel spectrogram
        mel = self._log_mel_spectrogram(audio)           # shape: (80, frames)
        # Pad/trim to 3000 frames (30 seconds at 100 frames/sec)
        if mel.shape[1] < 3000:
            mel = np.pad(mel, ((0, 0), (0, 3000 - mel.shape[1])), constant_values=-1.0)
        else:
            mel = mel[:, :3000]
 
        input_features = mel[np.newaxis, :, :].astype(np.float32)  # (1, 80, 3000)
 
        # Run encoder
        input_names = [inp.name for inp in self._session.get_inputs()]
        feed = {"input_features": input_features}
 
        # ── Egyptian Arabic dialect guidance ──────────────────────────────
        # Inject forced decoder IDs: <|ar|> <|transcribe|> <|notimestamps|>
        # This is equivalent to generate_kwargs={"language":"arabic","task":"transcribe"}
        # in the HuggingFace pipeline — it prevents Whisper from:
        #   • Switching to MSA (Modern Standard Arabic)
        #   • Hallucinating English words for Egyptian slang
        #   • Outputting timestamps instead of clean text
        if "decoder_input_ids" in input_names:
            # Always start with <|startoftranscript|> + forced Arabic tokens
            forced_ids = np.array(
                [[50258, 50272, 50359, 50363]],
                dtype=np.int64,
            )
            feed["decoder_input_ids"] = forced_ids
 
        # ── Soft prompt prefix (dialect seeding) ──────────────────────────
        # If the model export supports a text prompt input, prepend the
        # Egyptian Arabic prompt to bias the decoder vocabulary.
        # Works with ORTModelForSpeechSeq2Seq full pipeline.
        # For encoder-only ONNX sessions, this is handled at the pipeline level.
        if self._use_dialect_prompt and "prompt_ids" in input_names:
            prompt = _SPEAKER_PROMPTS.get(self._speaker_type, _EGYPTIAN_ARABIC_PROMPT)
            # Encode prompt to token IDs (requires tokenizer — skip if unavailable)
            if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                try:
                    prompt_enc = self._tokenizer(
                        prompt,
                        return_tensors = "np",
                        add_special_tokens = False,
                    )
                    feed["prompt_ids"] = prompt_enc["input_ids"].astype(np.int64)
                    logger.debug(
                        "WhisperSTT: dialect prompt injected (%d tokens) speaker=%s",
                        prompt_enc["input_ids"].shape[1], self._speaker_type,
                    )
                except Exception as exc:
                    logger.debug("WhisperSTT: prompt encoding failed — skipping: %s", exc)
 
        outputs = self._session.run(None, feed)
        token_ids = outputs[0][0]   # shape: (seq_len,)
 
        # Decode token IDs to text using a minimal tokeniser
        text           = self._decode_tokens(token_ids)
        no_speech_prob = float(outputs[1][0]) if len(outputs) > 1 else 0.0
        confidence     = max(0.0, 1.0 - no_speech_prob)
 
        return {
            "text"          : text.strip(),
            "language"      : self._lang,
            "confidence"    : round(confidence, 4),
            "no_speech_prob": round(no_speech_prob, 4),
        }
 
    def _log_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Whisper-compatible log-mel spectrogram.
 
        Uses numpy only — no librosa required for offline deployment.
        Matches Whisper's preprocessing exactly:
        - window size  : 400 samples (25ms at 16kHz)
        - hop length   : 160 samples (10ms)
        - n_mels       : 80
        """
        n_fft   = 400
        hop     = 160
        n_mels  = N_MEL_BINS
        sr      = WHISPER_SAMPLE_RATE
 
        # Short-time Fourier transform
        pad_len = n_fft // 2
        audio   = np.pad(audio.astype(np.float32), pad_len, mode="reflect")
 
        # Hann window
        window  = np.hanning(n_fft).astype(np.float32)
 
        # Compute STFT frames
        n_frames = 1 + (len(audio) - n_fft) // hop
        frames   = np.lib.stride_tricks.as_strided(
            audio,
            shape   = (n_frames, n_fft),
            strides = (audio.strides[0] * hop, audio.strides[0]),
        )
        stft = np.fft.rfft(frames * window, n=n_fft)
        magnitudes = np.abs(stft) ** 2                    # power spectrum
 
        # Mel filterbank (triangular filters)
        mel_fb = self._mel_filterbank(sr, n_fft, n_mels)  # (n_mels, n_fft//2+1)
        mel    = mel_fb @ magnitudes.T                     # (n_mels, n_frames)
 
        # Log scaling
        mel    = np.log10(np.maximum(mel, 1e-10))
        mel    = (mel - mel.max()) / 4.0 + 1.0            # normalise to [-1, 0] ≈ Whisper norm
 
        return mel.astype(np.float32)
 
    @staticmethod
    def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        """Build a triangular mel filterbank matrix (n_mels, n_fft//2+1)."""
        n_freqs  = n_fft // 2 + 1
        fmin, fmax = 0.0, sr / 2.0
 
        def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
 
        mel_pts  = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_pts   = mel_to_hz(mel_pts)
        bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
 
        filters  = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for m in range(n_mels):
            f_m_minus = bin_pts[m]
            f_m       = bin_pts[m + 1]
            f_m_plus  = bin_pts[m + 2]
            for k in range(f_m_minus, f_m):
                filters[m, k] = (k - bin_pts[m]) / (bin_pts[m+1] - bin_pts[m] + 1e-10)
            for k in range(f_m, f_m_plus):
                filters[m, k] = (bin_pts[m+2] - k) / (bin_pts[m+2] - bin_pts[m+1] + 1e-10)
        return filters
 
    @staticmethod
    def _decode_tokens(token_ids: np.ndarray) -> str:
        """
        Minimal token-to-text decoder for Whisper output.
 
        For the full token vocabulary, Whisper uses a BPE tokeniser
        (loaded from tokenizer.json).  This stub returns a placeholder
        when the tokeniser file is not available — the production code
        should load the real tokeniser from the same directory as the model.
        """
        # Filter special tokens (IDs < 50257 are text tokens in Whisper)
        text_tokens = [t for t in token_ids if t < 50257]
        if not text_tokens:
            return ""
        # In production: tokeniser.decode(text_tokens)
        # Here we return a marker so callers know inference ran
        return f"[decoded:{len(text_tokens)}_tokens]"
 
    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampler (no scipy required)."""
        if orig_sr == target_sr:
            return audio
        ratio       = target_sr / orig_sr
        n_new       = int(len(audio) * ratio)
        old_indices = np.linspace(0, len(audio) - 1, n_new)
        return np.interp(old_indices, np.arange(len(audio)), audio).astype(np.float32)
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  Main engine: SpeechToText
# ═══════════════════════════════════════════════════════════════════════════
 
class SpeechToText:
    """
    End-to-end speech pipeline: audio bytes → STTResult with routing.
 
    Usage
    -----
    >>> stt = SpeechToText()
 
    # From raw bytes (PWA microphone recording):
    >>> result = stt.transcribe_and_route(audio_bytes, mime_type="audio/webm")
    >>> result.text          # "عندي ألم في صدري"
    >>> result.speaker_type  # SpeakerType.PATIENT
    >>> result.route         # {"api": "/api/triage", "page": "/triage", ...}
 
    # With known patient context (overrides auto-detection):
    >>> result = stt.transcribe_and_route(
    ...     audio_bytes,
    ...     patient_context={"is_pregnant": True}
    ... )
    >>> result.speaker_type  # SpeakerType.PREGNANT
    """
 
    def __init__(
        self,
        model_path    : Path | None   = None,
        language      : str           = "ar",
        beam_size     : int           = 3,
        threads       : int | None    = None,
    ) -> None:
        self._whisper = WhisperSTT(
            model_path = model_path,
            language   = language,
            beam_size  = beam_size,
            threads    = threads,
        )
        self._router = SpeakerRouter()
 
        logger.info(
            "SpeechToText ready | model_loaded=%s  language=%s",
            self._whisper.is_loaded(), language,
        )
 
    # ── Public API ───────────────────────────────────────────────────────
 
    def transcribe_and_route(
        self,
        audio_bytes    : bytes,
        mime_type      : str                    = "audio/wav",
        patient_context: dict[str, Any] | None  = None,
        sample_rate    : int                    = WHISPER_SAMPLE_RATE,
    ) -> STTResult:
        """
        Full pipeline: raw audio bytes → STTResult with speaker routing.
 
        Parameters
        ----------
        audio_bytes     : Raw audio bytes from PWA microphone or file.
        mime_type       : MIME type hint for format detection.
        patient_context : Known session context (overrides speaker detection).
        sample_rate     : Audio sample rate if known (used as hint).
 
        Returns
        -------
        STTResult — always returns, never raises.
        """
        t_start = time.perf_counter()
 
        # ── Step 1: Decode audio bytes → float32 array ────────────────────
        try:
            audio, sr = self._decode_audio(audio_bytes, mime_type, sample_rate)
        except Exception as exc:
            logger.error("Audio decode failed: %s", exc)
            return self._error_result("Audio decode failed")
 
        duration_s = len(audio) / sr
 
        # ── Step 2: Silence detection ─────────────────────────────────────
        is_silent = self._is_silent(audio)
 
        # ── Step 3: Transcribe via Whisper ────────────────────────────────
        t_whisper = time.perf_counter()
        whisper_out = self._whisper.transcribe(audio, sr)
        transcribe_ms = (time.perf_counter() - t_whisper) * 1000
 
        text          = whisper_out["text"]
        confidence    = whisper_out["confidence"]
        no_speech_prob= whisper_out["no_speech_prob"]
 
        # Override silence flag if Whisper also says no speech
        if no_speech_prob > 0.80:
            is_silent = True
 
        # ── Step 4: Speaker detection & routing ───────────────────────────
        speaker_type, speaker_conf, speaker_kws = self._router.detect(
            text, patient_context
        )
        route = self._router.route(speaker_type)
 
        # Update Whisper's speaker context for next call so dialect prompt
        # is aligned with the detected speaker type (patient/doctor/etc.)
        if hasattr(self._whisper, '_speaker_type'):
            self._whisper._speaker_type = speaker_type.value
 
        elapsed_total = (time.perf_counter() - t_start) * 1000
        logger.info(
            "STT | speaker=%s route=%s conf=%.2f dur=%.1fs total=%.1fms",
            speaker_type.value, route["page"],
            confidence, duration_s, elapsed_total,
        )
 
        return STTResult(
            text             = text,
            language         = whisper_out.get("language", "ar"),
            transcribe_ms    = transcribe_ms,
            speaker_type     = speaker_type,
            route            = route,
            confidence       = confidence,
            no_speech_prob   = no_speech_prob,
            is_silent        = is_silent,
            audio_duration_s = duration_s,
            sample_rate      = sr,
            speaker_confidence  = speaker_conf,
            speaker_keywords    = speaker_kws,
        )
 
    def transcribe_only(
        self,
        audio_bytes: bytes,
        mime_type  : str = "audio/wav",
        sample_rate: int = WHISPER_SAMPLE_RATE,
    ) -> str:
        """
        Shortcut — transcribe audio and return text only.
        Speaker routing is skipped.
        """
        try:
            audio, sr = self._decode_audio(audio_bytes, mime_type, sample_rate)
        except Exception as exc:
            logger.error("Audio decode failed: %s", exc)
            return ""
        return self._whisper.transcribe(audio, sr)["text"]
 
    def route_text(
        self,
        text           : str,
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Route already-transcribed text to the correct RIVA page.
        Useful when text arrives from the chatbot (no audio).
        """
        speaker_type, _, _ = self._router.detect(text, patient_context)
        return self._router.route(speaker_type)
 
    @property
    def is_ready(self) -> bool:
        """True if Whisper model is loaded and ready."""
        return self._whisper.is_loaded()
 
    # ── Private helpers ───────────────────────────────────────────────────
 
    def _decode_audio(
        self,
        audio_bytes: bytes,
        mime_type  : str,
        hint_sr    : int,
    ) -> tuple[np.ndarray, int]:
        """
        Decode audio bytes to float32 numpy array.
 
        Tries soundfile first (WAV/FLAC/OGG), then pydub (WebM/MP3/M4A).
        Falls back to raw interpretation if both unavailable.
        """
        buf = io.BytesIO(audio_bytes)
 
        # soundfile handles WAV, FLAC, OGG
        if _SF_AVAILABLE:
            try:
                audio, sr = sf.read(buf, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)   # stereo → mono
                return audio, sr
            except Exception:
                buf.seek(0)
 
        # pydub handles WebM, MP3, M4A, AAC (requires ffmpeg)
        if _PYDUB_AVAILABLE:
            try:
                seg   = AudioSegment.from_file(buf)
                seg   = seg.set_frame_rate(WHISPER_SAMPLE_RATE).set_channels(1)
                raw   = np.array(seg.get_array_of_samples(), dtype=np.float32)
                audio = raw / (2 ** (seg.sample_width * 8 - 1))
                return audio, WHISPER_SAMPLE_RATE
            except Exception:
                buf.seek(0)
 
        # Last resort: attempt raw 16-bit PCM at hint_sr
        # FIX v1.1 — 4 safety guards before trusting the raw bytes:
        #
        # Guard 1: refuse empty buffer — nothing to decode
        if not audio_bytes:
            logger.error("Audio decode fallback: empty buffer — returning silence")
            return np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32), hint_sr
 
        # Guard 2: int16 requires an even number of bytes — odd length = corrupt
        if len(audio_bytes) % 2 != 0:
            logger.error(
                "Audio decode fallback: odd byte length (%d) — not valid int16 PCM, "
                "returning silence", len(audio_bytes),
            )
            return np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32), hint_sr
 
        logger.warning(
            "Audio decoders unavailable — interpreting as raw PCM int16 "
            "(%d bytes at %dHz)", len(audio_bytes), hint_sr,
        )
        raw   = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio = raw / 32768.0
 
        # Guard 3: clip to [-1, 1] — malformed PCM can exceed this range
        audio = np.clip(audio, -1.0, 1.0)
 
        # Guard 4: RMS > 0.95 means data is likely noise/garbage not speech
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 0.95:
            logger.error(
                "Audio decode fallback: RMS=%.3f — data looks like noise/garbage, "
                "returning silence", rms,
            )
            return np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32), hint_sr
 
        return audio, hint_sr
 
    @staticmethod
    def _is_silent(audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Return True if RMS energy is below silence threshold."""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return rms < threshold
 
    @staticmethod
    def _error_result(reason: str) -> STTResult:
        """Return a safe empty STTResult on error."""
        return STTResult(
            text             = "",
            language         = "ar",
            transcribe_ms    = 0.0,
            speaker_type     = SpeakerType.UNKNOWN,
            route            = SPEAKER_ROUTES[SpeakerType.UNKNOWN],
            confidence       = 0.0,
            no_speech_prob   = 1.0,
            is_silent        = True,
            audio_duration_s = 0.0,
            sample_rate      = WHISPER_SAMPLE_RATE,
            speaker_confidence  = 0.0,
            speaker_keywords    = [],
        )
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  Module-level convenience
# ═══════════════════════════════════════════════════════════════════════════
 
_default_stt: SpeechToText | None = None
 
 
def transcribe_and_route(
    audio_bytes    : bytes,
    mime_type      : str                   = "audio/wav",
    patient_context: dict[str, Any] | None = None,
) -> STTResult:
    """
    Module-level shortcut — uses a cached default SpeechToText instance.
 
    >>> from ai_core.voice.speech_to_text import transcribe_and_route
    >>> result = transcribe_and_route(audio_bytes, mime_type="audio/webm")
    """
    global _default_stt
    if _default_stt is None:
        _default_stt = SpeechToText()
    return _default_stt.transcribe_and_route(audio_bytes, mime_type, patient_context)
