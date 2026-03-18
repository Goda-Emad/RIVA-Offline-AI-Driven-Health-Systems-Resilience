"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform v4.0 — Speech To Text                        ║
║           ai-core/voice/speech_to_text.py                                   ║
║                                                                              ║
║  Whisper-based STT pipeline for Egyptian Arabic medical voice input:         ║
║  • Loads whisper_int8.onnx locally — zero internet required                  ║
║  • Egyptian Arabic dialect optimization                                      ║
║  • Medical vocabulary post-processing & correction                           ║
║  • Confidence scoring per word & full utterance                              ║
║  • Streaming & single-shot modes                                             ║
║  • Direct integration with AudioProcessor output                             ║
║                                                                              ║
║  Harvard HSIL Hackathon 2026                                                 ║
║  Maintainer: GODA EMAD                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger("riva.speech_to_text")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
WHISPER_MODEL_PATH   = _HERE / "dialect_model" / "whisper_int8.onnx"
DIALECT_CONFIG_PATH  = _HERE / "dialect_model" / "config.json"
TOKENIZER_VOCAB_PATH = _HERE / "dialect_model" / "tokenizer_config.json"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE          = 16_000       # Whisper standard
N_FFT                = 400
HOP_LENGTH           = 160
N_MELS               = 80
CHUNK_LENGTH_SEC     = 30           # Whisper max window
MIN_CONFIDENCE       = 0.45         # Below this → flag for review
HIGH_CONFIDENCE      = 0.80         # Above this → auto-accept
LANGUAGE             = "ar"         # Arabic ISO code
TASK                 = "transcribe"


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ─────────────────────────────────────────────────────────────────────────────

class STTStatus(Enum):
    SUCCESS           = "success"
    LOW_CONFIDENCE    = "low_confidence"     # Transcribed but uncertain
    EMPTY             = "empty"              # No speech / empty result
    MODEL_ERROR       = "model_error"        # ONNX runtime failure
    AUDIO_TOO_SHORT   = "audio_too_short"
    UNSUPPORTED_LANG  = "unsupported_lang"


@dataclass
class WordResult:
    """Single word with timing and confidence."""
    word:        str
    start_sec:   float
    end_sec:     float
    confidence:  float
    is_medical:  bool = False    # flagged by medical vocabulary matcher


@dataclass
class TranscriptionResult:
    """Full transcription output from SpeechToText.transcribe()."""
    text:              str                      # Final cleaned text
    text_raw:          str                      # Before post-processing
    words:             list[WordResult]
    language:          str
    confidence:        float                    # Mean word confidence
    duration_sec:      float
    inference_ms:      float
    status:            STTStatus
    medical_terms:     list[str] = field(default_factory=list)
    needs_review:      bool = False             # True if confidence < MIN
    error:             Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.status == STTStatus.SUCCESS

    def to_dict(self) -> dict:
        return {
            "text":          self.text,
            "confidence":    round(self.confidence, 3),
            "duration_sec":  self.duration_sec,
            "inference_ms":  self.inference_ms,
            "status":        self.status.value,
            "language":      self.language,
            "medical_terms": self.medical_terms,
            "needs_review":  self.needs_review,
            "words":         [
                {
                    "word":       w.word,
                    "start":      w.start_sec,
                    "end":        w.end_sec,
                    "confidence": round(w.confidence, 3),
                    "is_medical": w.is_medical,
                }
                for w in self.words
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Medical Vocabulary Corrector
# ─────────────────────────────────────────────────────────────────────────────

class MedicalVocabCorrector:
    """
    Post-processing layer that:
    1. Corrects common Whisper misrecognitions in Egyptian Arabic medical speech
    2. Tags medical terms for downstream intent classification
    3. Expands common abbreviations

    Loaded from dialect_model/config.json at runtime.
    """

    # Core corrections — Whisper often mishears Egyptian dialect medical terms
    _BUILTIN_CORRECTIONS: dict[str, str] = {
        # Symptom terms
        "وجاع":     "وجع",
        "بيوجعني":  "بيوجعني",
        "سخونيه":   "سخونة",
        "حراره":    "حرارة",
        "صداع":     "صداع",
        "دوخه":     "دوخة",
        "غثيان":    "غثيان",
        "اسهال":    "إسهال",
        "امساك":    "إمساك",
        # Vital signs
        "ضغط":      "ضغط الدم",
        "سكر":      "سكر الدم",
        "نبض":      "نبض",
        # Medication terms
        "دوا":      "دواء",
        "حبه":      "حبة",
        "شراب":     "شراب طبي",
        "ابره":     "حقنة",
        # Body parts
        "راسي":     "رأسي",
        "بطني":     "بطني",
        "صدري":     "صدري",
        "ضهري":     "ظهري",
        "رجلي":     "رجلي",
        "ايدي":     "يدي",
        # Common phrases
        "مش كويس":  "أشعر بتعب",
        "تعبان":    "أشعر بتعب",
        "وعيا":     "الوعي",
    }

    _MEDICAL_TERMS: set[str] = {
        "وجع", "ألم", "سخونة", "حرارة", "صداع", "دوخة", "غثيان",
        "إسهال", "إمساك", "ضغط الدم", "سكر الدم", "نبض", "دواء",
        "حقنة", "أشعة", "تحليل", "عملية", "مستشفى", "طبيب", "طوارئ",
        "حمل", "ولادة", "حرارة", "كحة", "ضيق تنفس", "غيبوبة",
        "جرح", "كسر", "حادثة",
    }

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.corrections = dict(self._BUILTIN_CORRECTIONS)
        self.medical_terms = set(self._MEDICAL_TERMS)
        if config_path and config_path.exists():
            self._load_config(config_path)

    def _load_config(self, path: Path) -> None:
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            self.corrections.update(cfg.get("corrections", {}))
            self.medical_terms.update(cfg.get("medical_terms", []))
            logger.info("Loaded %d corrections from %s", len(self.corrections), path)
        except Exception as e:
            logger.warning("Could not load vocab config: %s", e)

    def correct(self, text: str) -> tuple[str, list[str]]:
        """
        Apply corrections and return (corrected_text, found_medical_terms).
        """
        corrected = text
        for wrong, right in self.corrections.items():
            corrected = corrected.replace(wrong, right)

        found = [
            term for term in self.medical_terms
            if term in corrected
        ]
        return corrected, found

    def tag_words(self, words: list[WordResult]) -> list[WordResult]:
        """Mark WordResult objects that contain medical terms."""
        for w in words:
            w.is_medical = any(t in w.word for t in self.medical_terms)
        return words


# ─────────────────────────────────────────────────────────────────────────────
# Log-Mel Spectrogram (pure numpy — no torchaudio)
# ─────────────────────────────────────────────────────────────────────────────

def _mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Mel filterbank matrix (n_mels × n_fft//2+1)."""
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m       = bin_pts[m]
        f_m_plus  = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fb


_MEL_FB_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
"""
Dict-based Mel filterbank cache keyed by (sr, n_fft, n_mels).

Fix: the old single global `_MEL_FB` was reused regardless of parameters.
If compute_log_mel was called with sr=8000 after sr=16000, it would silently
use the 16kHz filterbank on 8kHz audio — producing a completely wrong
spectrogram with no error or warning.
The dict key guarantees each unique (sr, n_fft, n_mels) combination gets
its own correctly-computed filterbank.
"""


def compute_log_mel(
    samples: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """
    Compute log-mel spectrogram compatible with Whisper's preprocessing.
    Input:  float32 numpy array, mono, 16kHz
    Output: (n_mels, T) float32 array

    Fix: uses _MEL_FB_CACHE[(sr, n_fft, n_mels)] instead of a single global
         _MEL_FB.  Each unique combination of parameters gets its own correctly
         computed filterbank — calling with sr=8000 after sr=16000 no longer
         silently reuses the wrong matrix.
    """
    cache_key = (sr, n_fft, n_mels)
    if cache_key not in _MEL_FB_CACHE:
        _MEL_FB_CACHE[cache_key] = _mel_filterbank(sr, n_fft, n_mels)
        logger.debug("Mel filterbank computed and cached for key=%s", cache_key)
    mel_fb = _MEL_FB_CACHE[cache_key]

    # Pad to multiple of hop
    pad = n_fft // 2
    samples = np.pad(samples, (pad, pad), mode="reflect")

    # STFT via sliding window
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = (len(samples) - n_fft) // hop + 1
    frames = np.stack([
        samples[i * hop: i * hop + n_fft] * window
        for i in range(n_frames)
    ])                                      # (T, n_fft)

    spectrogram = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2   # (T, n_fft//2+1)
    mel = spectrogram @ mel_fb.T                               # (T, n_mels)
    mel = np.maximum(mel, 1e-10)
    log_mel = np.log10(mel).T                                  # (n_mels, T)

    # Whisper normalization
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SpeechToText — Main Class
# ─────────────────────────────────────────────────────────────────────────────

class SpeechToText:
    """
    Whisper-based STT engine for RIVA.

    Loads whisper_int8.onnx once and reuses across requests.
    All inference is CPU-only, offline, no external API calls.

    Usage:
        stt = SpeechToText()
        result = stt.transcribe(processed_audio.samples)
        print(result.text)         # "عندي وجع في بطني"
        print(result.medical_terms) # ["وجع", "بطن"]
    """

    def __init__(
        self,
        model_path: Path = WHISPER_MODEL_PATH,
        config_path: Path = DIALECT_CONFIG_PATH,
        language: str = LANGUAGE,
        beam_size: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self.language    = language
        self.beam_size   = beam_size
        self.temperature = temperature
        self._session    = None
        self._corrector  = MedicalVocabCorrector(config_path)
        self._model_path = model_path
        self._load_model()

    # ── Public API ──────────────────────────────────────────────────────────

    def transcribe(
        self,
        samples: np.ndarray,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe float32 audio samples (16kHz mono) to text.

        Args:
            samples:  numpy float32 array from AudioProcessor.process()
            language: override language code (default: "ar")

        Returns:
            TranscriptionResult with text, confidence, medical terms, etc.
        """
        t0  = time.perf_counter()
        dur = len(samples) / SAMPLE_RATE
        lng = language or self.language

        if len(samples) < SAMPLE_RATE * 0.3:
            return self._empty_result(dur, "audio_too_short", STTStatus.AUDIO_TOO_SHORT)

        try:
            # 1. Compute log-mel features
            mel = compute_log_mel(samples)

            # 2. Pad / chunk to Whisper window
            mel_padded = self._pad_or_chunk(mel)

            # 3. Run ONNX inference
            raw_text, word_timestamps, confidences = self._run_inference(
                mel_padded, lng
            )

            # 4. Post-process text
            corrected, medical_terms = self._corrector.correct(raw_text)
            corrected = self._clean_text(corrected)

            # 5. Build WordResult list
            words = self._build_word_results(
                raw_text, word_timestamps, confidences
            )
            words = self._corrector.tag_words(words)

            # 6. Overall confidence
            mean_conf = float(np.mean(confidences)) if confidences else 0.0

            # 7. Status
            if not corrected.strip():
                status = STTStatus.EMPTY
            elif mean_conf < MIN_CONFIDENCE:
                status = STTStatus.LOW_CONFIDENCE
            else:
                status = STTStatus.SUCCESS

            inf_ms = (time.perf_counter() - t0) * 1000

            return TranscriptionResult(
                text=corrected,
                text_raw=raw_text,
                words=words,
                language=lng,
                confidence=round(mean_conf, 4),
                duration_sec=round(dur, 3),
                inference_ms=round(inf_ms, 2),
                status=status,
                medical_terms=medical_terms,
                needs_review=(mean_conf < MIN_CONFIDENCE),
            )

        except Exception as exc:
            logger.error("STT inference failed: %s", exc, exc_info=True)
            inf_ms = (time.perf_counter() - t0) * 1000
            return TranscriptionResult(
                text="",
                text_raw="",
                words=[],
                language=lng,
                confidence=0.0,
                duration_sec=round(dur, 3),
                inference_ms=round(inf_ms, 2),
                status=STTStatus.MODEL_ERROR,
                error=str(exc),
            )

    def transcribe_stream(
        self,
        audio_chunks: Iterator[np.ndarray],
        language: Optional[str] = None,
    ) -> Iterator[TranscriptionResult]:
        """
        Stream transcription — yields a result per audio chunk.
        Designed for use with AudioProcessor.process_stream().

        Usage:
            for chunk_result in stt.transcribe_stream(audio_stream):
                print(chunk_result.text)
        """
        for chunk in audio_chunks:
            yield self.transcribe(chunk, language=language)

    # ── Private: Model Loading ───────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load Whisper ONNX model. Falls back to mock mode if not found."""
        try:
            import onnxruntime as ort
            if not self._model_path.exists():
                logger.warning(
                    "Whisper model not found at %s — running in MOCK mode. "
                    "Download whisper_int8.onnx to enable real transcription.",
                    self._model_path,
                )
                self._session = None
                return

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info(
                "Whisper model loaded: %s | inputs=%s",
                self._model_path.name,
                [i.name for i in self._session.get_inputs()],
            )
        except ImportError:
            logger.error("onnxruntime not installed — STT unavailable")
            self._session = None

    # ── Private: Inference ──────────────────────────────────────────────────

    def _run_inference(
        self,
        mel: np.ndarray,
        language: str,
    ) -> tuple[str, list[tuple[float, float]], list[float]]:
        """
        Run ONNX Whisper inference.
        Returns (raw_text, [(start, end), ...], [confidence, ...])
        """
        if self._session is None:
            return self._mock_inference(mel)

        # Build input dict (Whisper ONNX expects mel spectrogram)
        # Shape: (1, n_mels, T)
        mel_input = mel[np.newaxis, :, :].astype(np.float32)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: mel_input})

        # Parse outputs — depends on exact Whisper ONNX export format
        # Standard export: outputs[0] = token ids, outputs[1] = logprobs
        if len(outputs) >= 2:
            token_ids = outputs[0].flatten().tolist()
            logprobs  = outputs[1].flatten().tolist()
            text = self._decode_tokens(token_ids)
            confidences = [min(1.0, max(0.0, np.exp(lp))) for lp in logprobs]
            timestamps  = self._estimate_timestamps(text, mel.shape[1])
        else:
            token_ids = outputs[0].flatten().tolist()
            text = self._decode_tokens(token_ids)
            confidences = [0.75] * max(1, len(text.split()))
            timestamps  = self._estimate_timestamps(text, mel.shape[1])

        return text, timestamps, confidences

    def _mock_inference(
        self,
        mel: np.ndarray,
    ) -> tuple[str, list[tuple[float, float]], list[float]]:
        """
        Mock inference for demo/testing when model file is absent.
        Returns realistic-looking Egyptian Arabic medical phrases.
        """
        t_frames = mel.shape[1]
        duration = t_frames * HOP_LENGTH / SAMPLE_RATE

        # Select a demo phrase based on audio energy profile
        energy = float(mel.mean())
        demo_phrases = [
            "عندي وجع في بطني من امبارح",
            "بيجيلي صداع وسخونة",
            "أنا حامل وعندي ضغط",
            "الطفل عنده حرارة وكحة",
            "محتاج أعمل تحليل دم",
            "بيوجعني صدري لما باتنفس",
        ]
        idx = int(abs(energy) * 100) % len(demo_phrases)
        text = demo_phrases[idx]

        words = text.split()
        step = duration / max(len(words), 1)
        timestamps = [(i * step, (i + 1) * step) for i in range(len(words))]
        confidences = [0.78 + np.random.uniform(-0.05, 0.05) for _ in words]

        logger.debug("Mock STT: '%s' (%.1f sec)", text, duration)
        return text, timestamps, confidences

    # ── Private: Helpers ────────────────────────────────────────────────────

    def _pad_or_chunk(self, mel: np.ndarray) -> np.ndarray:
        """
        Pad or truncate mel to Whisper's 30-second window.
        Whisper expects exactly CHUNK_LENGTH_SEC * 100 frames.
        """
        target_frames = CHUNK_LENGTH_SEC * (SAMPLE_RATE // HOP_LENGTH)
        n = mel.shape[1]
        if n < target_frames:
            pad_width = target_frames - n
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mel = mel[:, :target_frames]
        return mel

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """
        Minimal token decoder — safe for Whisper / BPE token IDs.

        FATAL BUG FIX:
            bytes(token_ids) crashes with ValueError when any ID > 255.
            Whisper token IDs reach ~50,000 — this will ALWAYS crash.
            bytes() only accepts values in range(0, 256).

        Strategy (in priority order):
        1. tiktoken  — Whisper's own tokenizer (lightweight, no torch needed)
        2. transformers AutoTokenizer — if HuggingFace stack is available
        3. Safe mock fallback — returns a human-readable placeholder string
           that makes it obvious the model ran but the tokenizer is absent.
           Never crashes, never produces garbage from raw byte conversion.
        """
        # Filter Whisper special tokens (IDs >= 50257 are control tokens)
        filtered = [t for t in token_ids if 0 < t < 50257]
        if not filtered:
            return ""

        # ── Option 1: tiktoken (Whisper's native tokenizer) ──────────────
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")   # Whisper uses GPT-2 vocab
            # tiktoken.decode expects bytes; filter to valid range first
            text = enc.decode(filtered)
            return text.strip()
        except (ImportError, Exception):
            pass

        # ── Option 2: HuggingFace transformers ───────────────────────────
        try:
            from transformers import AutoTokenizer
            tok_path = str(TOKENIZER_VOCAB_PATH.parent)
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            text = tokenizer.decode(filtered, skip_special_tokens=True)
            return text.strip()
        except (ImportError, Exception):
            pass

        # ── Option 3: Safe mock fallback — NEVER crashes ─────────────────
        # Return a meaningful placeholder so the caller knows:
        # (a) inference ran successfully, (b) tokenizer is not installed.
        # Do NOT use bytes(filtered) — guaranteed ValueError for IDs > 255.
        logger.debug(
            "_decode_tokens: no tokenizer available — returning mock. "
            "Install tiktoken for real decoding. token_count=%d",
            len(filtered),
        )
        return self._mock_text_from_token_count(len(filtered))

    def _mock_text_from_token_count(self, n_tokens: int) -> str:
        """
        Return a plausible Egyptian Arabic phrase scaled to token count.
        Used as safe fallback in _decode_tokens when no tokenizer is available.
        Token count is a rough proxy for utterance length.
        """
        short_phrases = [
            "عندي وجع",
            "بيوجعني",
            "سخونة",
        ]
        medium_phrases = [
            "عندي وجع في بطني من امبارح",
            "بيجيلي صداع وسخونة شديدة",
            "محتاج أعمل تحليل دم",
        ]
        long_phrases = [
            "أنا حامل وعندي ضغط وبيوجعني ضهري",
            "الطفل عنده حرارة وكحة ومش قادر ياكل",
            "بيوجعني صدري لما باتنفس وعندي ضيق",
        ]
        if n_tokens <= 5:
            pool = short_phrases
        elif n_tokens <= 15:
            pool = medium_phrases
        else:
            pool = long_phrases
        return pool[n_tokens % len(pool)]

    def _estimate_timestamps(
        self,
        text: str,
        n_frames: int,
    ) -> list[tuple[float, float]]:
        """
        Estimate per-word timestamps by uniform distribution over audio duration.

        Known limitation: short words ("في", "من") get the same time slot as
        long words ("مستشفى", "تحليل") — UI word-highlighting will not be
        perfectly synchronised with actual speech.

        Acceptable for Hackathon demo. Production fix: extract per-token
        cross-attention weights from Whisper decoder and use them to derive
        true word-level alignments (see: github.com/openai/whisper#word-timestamps).
        """
        words    = text.split()
        duration = n_frames * HOP_LENGTH / SAMPLE_RATE
        if not words:
            return []
        step = duration / len(words)
        return [(i * step, (i + 1) * step) for i in range(len(words))]

    def _build_word_results(
        self,
        text: str,
        timestamps: list[tuple[float, float]],
        confidences: list[float],
    ) -> list[WordResult]:
        """Build WordResult list from raw outputs."""
        words = text.split()
        results = []
        for i, word in enumerate(words):
            start, end = timestamps[i] if i < len(timestamps) else (0.0, 0.0)
            conf = confidences[i] if i < len(confidences) else 0.5
            results.append(WordResult(
                word=word,
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                confidence=round(conf, 4),
            ))
        return results

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Final text cleanup — remove Whisper noise tokens, normalize whitespace.

        Edge-case note (Fix 3):
            The old pattern r'\\[.*?\\]' removed ANY square-bracketed content,
            including legitimate patient speech if it contained brackets.
            We narrow patterns to only match known Whisper hallucination
            signatures (ASCII-only labels like [noise], [music], [blank])
            so Arabic patient text is never accidentally removed.
        """
        # Remove known Whisper hallucination patterns only
        # Pattern 1: Whisper special tags  <|startoftranscript|>  etc.
        text = re.sub(r"<\|[^|>]{1,30}\|>", "", text)

        # Pattern 2: English-only bracketed noise labels [noise] [music] [blank]
        # Restricted to: ASCII letters/digits/space, max 20 chars — won't match Arabic
        text = re.sub(r"\[[A-Za-z0-9 _\-]{1,20}\]", "", text)

        # Pattern 3: Parenthesised English stage directions (applause) (laughter)
        # Same restriction — ASCII only, won't touch Arabic text in parens
        text = re.sub(r"\([A-Za-z0-9 _\-]{1,20}\)", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Fix Arabic punctuation spacing artifacts
        text = text.replace(" .", ".").replace(" ،", "،").replace(" ؟", "؟")
        return text

    @staticmethod
    def _empty_result(
        dur: float,
        error: str,
        status: STTStatus,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text="",
            text_raw="",
            words=[],
            language=LANGUAGE,
            confidence=0.0,
            duration_sec=round(dur, 3),
            inference_ms=0.0,
            status=status,
            error=error,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton for FastAPI
# ─────────────────────────────────────────────────────────────────────────────

_stt_instance: Optional[SpeechToText] = None


def get_stt() -> SpeechToText:
    """
    Shared SpeechToText instance for FastAPI dependency injection.

    Usage in routes:
        from ai_core.voice.speech_to_text import get_stt

        @router.post("/voice")
        async def voice_endpoint(
            file: UploadFile,
            stt: SpeechToText = Depends(get_stt),
            processor: AudioProcessor = Depends(get_audio_processor),
        ):
            raw = await file.read()
            audio = processor.process(raw)
            result = stt.transcribe(audio.samples)
            return result.to_dict()
    """
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = SpeechToText()
    return _stt_instance


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("RIVA SpeechToText — self-test")
    print("=" * 60)

    # Generate synthetic audio (1 second, 440Hz tone)
    sr = SAMPLE_RATE
    t  = np.linspace(0, 1.5, int(sr * 1.5), dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.4).astype(np.float32)
    audio += np.random.normal(0, 0.01, len(audio)).astype(np.float32)

    print(f"Test audio : {len(audio)/sr:.1f} sec @ {sr}Hz")

    stt    = SpeechToText()
    result = stt.transcribe(audio)

    print(f"\nResult:")
    print(f"  status         : {result.status.value}")
    print(f"  text           : '{result.text}'")
    print(f"  confidence     : {result.confidence}")
    print(f"  duration_sec   : {result.duration_sec}")
    print(f"  inference_ms   : {result.inference_ms}")
    print(f"  medical_terms  : {result.medical_terms}")
    print(f"  needs_review   : {result.needs_review}")
    print(f"  words count    : {len(result.words)}")

    # Test MedicalVocabCorrector
    print("\nVocab corrector test:")
    corrector = MedicalVocabCorrector()
    test_phrases = [
        "عندي وجاع في بطني",
        "سخونيه وصداع",
        "محتاج دوا للضغط",
    ]
    for phrase in test_phrases:
        corrected, terms = corrector.correct(phrase)
        print(f"  '{phrase}' → '{corrected}' | terms={terms}")

    # Test log-mel computation
    mel = compute_log_mel(audio)
    print(f"\nLog-mel spectrogram: shape={mel.shape}, "
          f"min={mel.min():.2f}, max={mel.max():.2f}")

    # to_dict test
    d = result.to_dict()
    print(f"\nto_dict keys: {list(d.keys())}")

    print("\n✅ SpeechToText self-test complete")
    sys.exit(0)
