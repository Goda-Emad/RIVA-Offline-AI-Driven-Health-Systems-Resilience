"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Audio Processor                             ║
║           ai-core/voice/dialect_model/audio_processor.py                     ║
║                                                                              ║
║  Purpose : Preprocess raw audio from PWA microphone before Whisper STT.     ║
║            Runs fully offline — no external API calls.                       ║
║                                                                              ║
║  Pipeline position                                                           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  PWA mic (WebM/WAV/MP3 bytes)                                                ║
║      ↓  AudioProcessor.preprocess()                                         ║
║  16kHz mono float32 array  (clean, VAD-trimmed)                             ║
║      ↓  WhisperSTT.transcribe()                                              ║
║  Arabic text                                                                 ║
║                                                                              ║
║  Processing steps                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. Decode bytes → float32 array (WAV / WebM / MP3 / raw PCM)               ║
║  2. Stereo → mono (average channels)                                         ║
║  3. Resample → 16kHz  (Whisper requirement)                                  ║
║  4. Normalize amplitude → peak −3 dBFS                                       ║
║  5. Voice Activity Detection — trim leading/trailing silence                 ║
║  6. Spectral noise gate  — attenuate non-speech frequency bands              ║
║  7. Chunk long recordings (>30s) into overlapping segments                  ║
║                                                                              ║
║  Design                                                                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Pure numpy — no librosa, no scipy required for core pipeline.            ║
║  • Graceful fallback at every step — never raises, always returns audio.    ║
║  • Returns ProcessedAudio dataclass with full quality metadata.             ║
║                                                                              ║
║  Author  : Goda Emad  (AI Core)                                              ║
║  Version : 1.1.0                                                             ║
║  Updated : 2026-03-18                                                        ║
║                                                                              ║
║  Changelog v1.1.0                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • FIX 1 VAD: added ZCR — catches whispered consonants (س، ش، ف).          ║
║  • FIX 2 Async: preprocess_async() + ThreadPoolExecutor(max_workers=4).    ║
║  • FIX 3 Noise gate: n_fft 512→1024 for sharper frequency resolution.      ║
║  • FIX 4 Memory: _make_chunks uses array views — no redundant float32 copy.║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional decoders — graceful degradation ─────────────────────────────
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

TARGET_SR        : int   = 16_000      # Whisper requirement
MAX_DURATION_S   : int   = 30          # max chunk length in seconds
OVERLAP_S        : float = 0.5         # overlap between chunks (seconds)
SILENCE_THRESHOLD: float = 0.01        # RMS below this = silence
VAD_FRAME_MS     : int   = 20          # VAD analysis frame length (ms)
VAD_PAD_MS       : int   = 200         # silence padding kept around speech (ms)
PEAK_DBFS        : float = -3.0        # target normalisation level
NOISE_GATE_HZ    : int   = 80          # high-pass cutoff (remove hum <80Hz)


# ═══════════════════════════════════════════════════════════════════════════
#  Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessedAudio:
    """
    Output of AudioProcessor.preprocess().

    Passed to WhisperSTT.transcribe() — the `chunks` list is the primary
    output; process each chunk independently when the audio is long.
    """
    # ── Primary output ────────────────────────────────────────────────
    chunks          : list[np.ndarray]  # list of float32 arrays at 16kHz
    sample_rate     : int               # always TARGET_SR (16000)

    # ── Quality metadata ──────────────────────────────────────────────
    original_duration_s : float
    processed_duration_s: float
    is_silent           : bool          # True if all audio below threshold
    was_clipped         : bool          # True if audio was clipped to max duration
    speech_ratio        : float         # fraction of audio containing speech (0–1)
    peak_dbfs           : float         # peak amplitude after normalisation (dBFS)
    noise_floor_rms     : float         # estimated background noise level

    # ── Processing flags ──────────────────────────────────────────────
    stereo_to_mono : bool
    resampled      : bool
    vad_trimmed    : bool
    noise_gated    : bool
    processing_ms  : float

    def as_dict(self) -> dict:
        return {
            "n_chunks"            : len(self.chunks),
            "sample_rate"         : self.sample_rate,
            "original_duration_s" : round(self.original_duration_s, 2),
            "processed_duration_s": round(self.processed_duration_s, 2),
            "is_silent"           : self.is_silent,
            "was_clipped"         : self.was_clipped,
            "speech_ratio"        : round(self.speech_ratio, 3),
            "peak_dbfs"           : round(self.peak_dbfs, 1),
            "noise_floor_rms"     : round(self.noise_floor_rms, 4),
            "stereo_to_mono"      : self.stereo_to_mono,
            "resampled"           : self.resampled,
            "vad_trimmed"         : self.vad_trimmed,
            "noise_gated"         : self.noise_gated,
            "processing_ms"       : round(self.processing_ms, 1),
        }

    @property
    def total_samples(self) -> int:
        return sum(len(c) for c in self.chunks)

    @property
    def should_transcribe(self) -> bool:
        """True if audio is worth sending to Whisper."""
        return not self.is_silent and len(self.chunks) > 0 and self.speech_ratio >= 0.05


# ═══════════════════════════════════════════════════════════════════════════
#  AudioProcessor
# ═══════════════════════════════════════════════════════════════════════════

class AudioProcessor:
    """
    Stateless audio preprocessor — instantiate once, call preprocess() many times.

    All steps are configurable via constructor parameters.
    Every step degrades gracefully — if it fails, the audio passes through unchanged.

    Parameters
    ----------
    target_sr          : Output sample rate (default 16000 for Whisper).
    silence_threshold  : RMS below which a frame is considered silent (default 0.01).
    vad_frame_ms       : VAD analysis frame length in milliseconds (default 20ms).
    vad_pad_ms         : Silence padding around speech segments (default 200ms).
    peak_dbfs          : Target peak level after normalisation (default −3 dBFS).
    max_duration_s     : Max chunk length in seconds — longer audio is chunked (default 30).
    overlap_s          : Overlap between chunks to avoid cutting words (default 0.5s).
    enable_noise_gate  : Apply spectral noise gate (default True).
    noise_gate_hz      : High-pass cutoff for noise gate in Hz (default 80).

    Examples
    --------
    >>> ap = AudioProcessor()
    >>> result = ap.preprocess(audio_bytes, mime_type="audio/webm", sample_rate=48000)
    >>> result.should_transcribe     # True if speech detected
    >>> for chunk in result.chunks:
    ...     text = whisper.transcribe(chunk)
    """

    def __init__(
        self,
        target_sr        : int   = TARGET_SR,
        silence_threshold: float = SILENCE_THRESHOLD,
        vad_frame_ms     : int   = VAD_FRAME_MS,
        vad_pad_ms       : int   = VAD_PAD_MS,
        peak_dbfs        : float = PEAK_DBFS,
        max_duration_s   : int   = MAX_DURATION_S,
        overlap_s        : float = OVERLAP_S,
        enable_noise_gate: bool  = True,
        noise_gate_hz    : int   = NOISE_GATE_HZ,
    ) -> None:
        self._sr          = target_sr
        self._sil_thr     = silence_threshold
        self._vad_frame   = vad_frame_ms
        self._vad_pad     = vad_pad_ms
        self._peak_dbfs   = peak_dbfs
        self._max_dur     = max_duration_s
        self._overlap     = overlap_s
        self._noise_gate  = enable_noise_gate
        self._gate_hz     = noise_gate_hz

        logger.info(
            "AudioProcessor ready | sr=%d vad_frame=%dms peak=%.1fdBFS noise_gate=%s",
            target_sr, vad_frame_ms, peak_dbfs, enable_noise_gate,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def preprocess(
        self,
        audio_bytes: bytes,
        mime_type  : str = "audio/wav",
        sample_rate: int = TARGET_SR,
    ) -> ProcessedAudio:
        """
        Full preprocessing pipeline: raw bytes → clean 16kHz float32 chunks.

        Parameters
        ----------
        audio_bytes : Raw audio bytes (WAV, WebM, MP3, or raw PCM int16).
        mime_type   : MIME type hint for format detection.
        sample_rate : Source sample rate (used when decoding raw PCM).

        Returns
        -------
        ProcessedAudio — always returns, never raises.
        """
        t_start = time.perf_counter()

        # ── Step 1: Decode bytes → float32 array ──────────────────────────
        try:
            audio, orig_sr = self._decode(audio_bytes, mime_type, sample_rate)
        except Exception as exc:
            logger.error("Audio decode failed: %s — returning empty result", exc)
            return self._empty_result(0.0, time.perf_counter() - t_start)

        if len(audio) == 0:
            return self._empty_result(0.0, time.perf_counter() - t_start)

        original_duration = len(audio) / orig_sr
        stereo_flag       = False
        resample_flag     = False

        # ── Step 2: Stereo → mono ─────────────────────────────────────────
        if audio.ndim > 1:
            audio       = audio.mean(axis=1)
            stereo_flag = True

        audio = audio.astype(np.float32)

        # ── Step 3: Resample → target_sr ─────────────────────────────────
        if orig_sr != self._sr:
            audio         = self._resample(audio, orig_sr, self._sr)
            resample_flag = True

        # ── Step 4: Estimate noise floor (before normalisation) ───────────
        noise_floor = self._estimate_noise_floor(audio)

        # ── Step 5: Normalize amplitude ───────────────────────────────────
        audio, peak_db = self._normalize(audio)

        # ── Step 6: Spectral noise gate (high-pass) ───────────────────────
        gated_flag = False
        if self._noise_gate:
            try:
                audio      = self._spectral_noise_gate(audio)
                gated_flag = True
            except Exception as exc:
                logger.debug("Noise gate failed — skipping: %s", exc)

        # ── Step 7: Voice Activity Detection — trim silence ───────────────
        audio_vad, speech_ratio = self._apply_vad(audio)
        vad_flag = len(audio_vad) < len(audio)
        audio    = audio_vad

        is_silent = speech_ratio < 0.05 or self._rms(audio) < self._sil_thr

        # ── Step 8: Clip & chunk long audio ───────────────────────────────
        was_clipped = False
        max_samples = self._max_dur * self._sr
        if len(audio) > max_samples:
            was_clipped = True
        chunks = self._make_chunks(audio)

        processed_duration = sum(len(c) for c in chunks) / self._sr
        processing_ms      = (time.perf_counter() - t_start) * 1000

        logger.info(
            "AudioProcessor | orig=%.1fs proc=%.1fs chunks=%d "
            "speech=%.0f%% silent=%s peak=%.1fdBFS %.1fms",
            original_duration, processed_duration, len(chunks),
            speech_ratio * 100, is_silent, peak_db, processing_ms,
        )

        return ProcessedAudio(
            chunks               = chunks,
            sample_rate          = self._sr,
            original_duration_s  = original_duration,
            processed_duration_s = processed_duration,
            is_silent            = is_silent,
            was_clipped          = was_clipped,
            speech_ratio         = speech_ratio,
            peak_dbfs            = peak_db,
            noise_floor_rms      = noise_floor,
            stereo_to_mono       = stereo_flag,
            resampled            = resample_flag,
            vad_trimmed          = vad_flag,
            noise_gated          = gated_flag,
            processing_ms        = processing_ms,
        )

    def preprocess_array(
        self,
        audio : np.ndarray,
        src_sr: int = TARGET_SR,
    ) -> ProcessedAudio:
        """
        Preprocess a numpy array directly (skips byte decoding).
        Useful when audio is already decoded by another module.
        """
        encoded = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return self.preprocess(encoded, mime_type="audio/pcm", sample_rate=src_sr)

    # ── Private: decoding ─────────────────────────────────────────────────

    def _decode(
        self,
        audio_bytes: bytes,
        mime_type  : str,
        hint_sr    : int,
    ) -> tuple[np.ndarray, int]:
        """
        Decode raw bytes to float32 numpy array + sample rate.

        Decode priority:
        1. soundfile — WAV, FLAC, OGG (fast, no ffmpeg needed)
        2. pydub     — WebM, MP3, M4A, AAC (needs ffmpeg)
        3. raw PCM   — last resort, with safety guards
        """
        if not audio_bytes:
            raise ValueError("Empty audio bytes")

        buf = io.BytesIO(audio_bytes)

        # soundfile path
        if _SF_AVAILABLE:
            try:
                audio, sr = sf.read(buf, dtype="float32", always_2d=False)
                return audio, sr
            except Exception:
                buf.seek(0)

        # pydub path
        if _PYDUB_AVAILABLE:
            try:
                seg   = AudioSegment.from_file(buf)
                seg   = seg.set_channels(1).set_frame_rate(self._sr)
                raw   = np.array(seg.get_array_of_samples(), dtype=np.float32)
                audio = raw / (2 ** (seg.sample_width * 8 - 1))
                return audio, self._sr
            except Exception:
                buf.seek(0)

        # Raw PCM int16 fallback with safety guards
        return self._decode_raw_pcm(audio_bytes, hint_sr)

    def _decode_raw_pcm(
        self,
        audio_bytes: bytes,
        hint_sr    : int,
    ) -> tuple[np.ndarray, int]:
        """Decode raw int16 PCM with 4-layer safety guard."""

        # Guard 1: minimum viable length (at least 0.1s of audio)
        min_bytes = int(hint_sr * 0.1) * 2
        if len(audio_bytes) < min_bytes:
            logger.error("Raw PCM: too short (%d bytes) — silence", len(audio_bytes))
            return np.zeros(self._sr, dtype=np.float32), hint_sr

        # Guard 2: int16 requires even byte count
        if len(audio_bytes) % 2 != 0:
            logger.error("Raw PCM: odd byte count (%d) — silence", len(audio_bytes))
            return np.zeros(self._sr, dtype=np.float32), hint_sr

        logger.warning("Raw PCM fallback (%d bytes at %dHz)", len(audio_bytes), hint_sr)
        raw   = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio = np.clip(raw / 32768.0, -1.0, 1.0)

        # Guard 3: RMS sanity check — true noise has RMS > 0.9
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 0.90:
            logger.error("Raw PCM: RMS=%.3f — looks like noise, returning silence", rms)
            return np.zeros(self._sr, dtype=np.float32), hint_sr

        return audio, hint_sr

    # ── Private: processing steps ─────────────────────────────────────────

    def _resample(
        self,
        audio    : np.ndarray,
        orig_sr  : int,
        target_sr: int,
    ) -> np.ndarray:
        """
        Linear interpolation resampler — no scipy required.

        Accurate enough for speech at 8kHz–48kHz → 16kHz.
        For extreme ratios (>4×) consider a polyphase filter.
        """
        if orig_sr == target_sr:
            return audio
        n_new       = max(1, int(len(audio) * target_sr / orig_sr))
        old_indices = np.linspace(0, len(audio) - 1, n_new)
        return np.interp(old_indices, np.arange(len(audio)), audio).astype(np.float32)

    def _normalize(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Normalize peak amplitude to self._peak_dbfs.

        Returns (normalized_audio, achieved_peak_dbfs).
        Skips normalization if audio is silent to avoid amplifying noise.
        """
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-6:
            return audio, -96.0   # effectively silent

        target_peak = 10 ** (self._peak_dbfs / 20.0)
        audio       = audio * (target_peak / peak)
        audio       = np.clip(audio, -1.0, 1.0)
        achieved_db = 20.0 * np.log10(float(np.max(np.abs(audio))) + 1e-10)
        return audio, achieved_db

    def _spectral_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """
        Lightweight spectral noise gate using FFT high-pass filtering.

        Attenuates frequencies below self._gate_hz (default 80Hz) which
        typically contain hum, air conditioning, and low-frequency noise
        rather than speech.

        Implemented as a real-FFT brick-wall high-pass — fast and
        dependency-free.
        """
        n_fft    = min(1024, len(audio))   # FIX v1.1: 31Hz → 15Hz bin resolution
        if n_fft < 4:
            return audio

        # Compute FFT, zero out below cutoff, invert
        hop      = n_fft // 2
        cutoff_bin = int(self._gate_hz * n_fft / self._sr)

        output   = np.zeros_like(audio)
        window   = np.hanning(n_fft).astype(np.float32)
        norm     = np.zeros_like(audio)

        for start in range(0, len(audio) - n_fft + 1, hop):
            frame  = audio[start : start + n_fft] * window
            spec   = np.fft.rfft(frame)
            # Zero out DC and low-frequency noise bins
            spec[:cutoff_bin] = 0.0
            filtered = np.fft.irfft(spec).real.astype(np.float32)
            output[start : start + n_fft] += filtered * window
            norm[start : start + n_fft]   += window ** 2

        # Overlap-add normalisation — avoid division by zero at edges
        mask         = norm > 1e-8
        output[mask] /= norm[mask]
        output[~mask] = 0.0

        return output.astype(np.float32)

    def _apply_vad(
        self,
        audio: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Voice Activity Detection — trim leading/trailing silence and compute
        the fraction of audio containing speech.

        Algorithm
        ─────────
        1. Split audio into fixed-length frames (default 20ms)
        2. Compute RMS energy per frame
        3. Compute Zero-Crossing Rate (ZCR) per frame   ← FIX v1.1
        4. Mark frames as speech if RMS > threshold OR ZCR > 0.10
           ZCR catches whispered consonants (س، ش، ف) that have low
           RMS but high zero-crossing rate — pure noise has low ZCR.
        5. Add padding (200ms) around speech frames
        6. Trim audio to [first_speech, last_speech] with padding

        Returns
        -------
        (trimmed_audio, speech_ratio)
        """
        frame_len   = int(self._sr * self._vad_frame / 1000)
        pad_frames  = int(self._vad_pad / self._vad_frame)

        if frame_len == 0 or len(audio) < frame_len:
            return audio, 1.0   # too short to analyse

        # Compute RMS + ZCR per frame
        n_frames = len(audio) // frame_len
        frames   = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms      = np.sqrt(np.mean(frames ** 2, axis=1))

        # FIX v1.1: Zero-Crossing Rate — count sign changes per frame
        # ZCR > 0.10 = whispered speech (س، ش، ف) or fricatives
        # ZCR < 0.05 + low RMS = background noise (AC, fan, hum)
        zcr       = np.mean(np.abs(np.diff(np.sign(frames))), axis=1) / 2
        is_speech = (rms > self._sil_thr) | (zcr > 0.10)

        speech_ratio = float(is_speech.mean())

        if not is_speech.any():
            return np.zeros(frame_len, dtype=np.float32), 0.0  # all silence

        # Find first and last speech frame with padding
        speech_idx   = np.where(is_speech)[0]
        first_frame  = max(0,          speech_idx[0]  - pad_frames)
        last_frame   = min(n_frames-1, speech_idx[-1] + pad_frames)

        start_sample = first_frame * frame_len
        end_sample   = (last_frame + 1) * frame_len
        trimmed      = audio[start_sample : end_sample]

        return trimmed.astype(np.float32), speech_ratio

    def _make_chunks(self, audio: np.ndarray) -> list[np.ndarray]:
        """
        Split audio into non-silent chunks ≤ max_duration_s.

        Long recordings are split with overlap_s overlap to avoid
        cutting words at boundaries.  Short recordings are returned
        as a single chunk.

        Each chunk is padded to at least 1 second (Whisper minimum).
        """
        max_samples = self._max_dur * self._sr
        min_samples = self._sr      # Whisper requires at least 1 second
        overlap_samples = int(self._overlap * self._sr)

        if len(audio) == 0:
            return []

        # Pad very short audio to 1 second minimum
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        if len(audio) <= max_samples:
            # FIX v1.1: audio is already float32 from decode step — no copy needed
            return [audio]

        # Split into overlapping chunks
        # FIX v1.1: slices are views into the original array — zero memory copy.
        # np.pad() only allocates when the last chunk is short (rare case).
        chunks = []
        start  = 0
        while start < len(audio):
            end   = min(start + max_samples, len(audio))
            chunk = audio[start : end]          # view, not copy
            # Pad last chunk to 1 second if too short (allocates only here)
            if len(chunk) < min_samples:
                chunk = np.pad(chunk, (0, min_samples - len(chunk)))
            chunks.append(chunk)
            if end >= len(audio):
                break
            start += max_samples - overlap_samples

        return chunks

    # ── Private: helpers ──────────────────────────────────────────────────

    @staticmethod
    def _rms(audio: np.ndarray) -> float:
        """Root mean square energy."""
        return float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

    def _estimate_noise_floor(self, audio: np.ndarray) -> float:
        """
        Estimate background noise level from the quietest 10% of frames.

        Uses the 10th percentile RMS across all frames as the noise floor
        estimate — robust to occasional speech bursts.
        """
        frame_len = int(self._sr * self._vad_frame / 1000)
        if frame_len == 0 or len(audio) < frame_len:
            return 0.0

        n_frames = len(audio) // frame_len
        frames   = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms_vals = np.sqrt(np.mean(frames ** 2, axis=1))
        return float(np.percentile(rms_vals, 10))

    @staticmethod
    def _empty_result(original_duration: float, elapsed_s: float) -> ProcessedAudio:
        """Return a safe empty ProcessedAudio on decode failure."""
        return ProcessedAudio(
            chunks               = [],
            sample_rate          = TARGET_SR,
            original_duration_s  = original_duration,
            processed_duration_s = 0.0,
            is_silent            = True,
            was_clipped          = False,
            speech_ratio         = 0.0,
            peak_dbfs            = -96.0,
            noise_floor_rms      = 0.0,
            stereo_to_mono       = False,
            resampled            = False,
            vad_trimmed          = False,
            noise_gated          = False,
            processing_ms        = elapsed_s * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level convenience
# ═══════════════════════════════════════════════════════════════════════════

_default_processor: AudioProcessor | None = None

# FIX v1.1: dedicated thread pool for CPU-bound audio processing
# Keeps FastAPI event loop unblocked when multiple patients send audio
# simultaneously — each preprocess() call runs in its own thread.
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
_audio_pool = _ThreadPoolExecutor(
    max_workers = 4,
    thread_name_prefix = "riva_audio",
)


def preprocess(
    audio_bytes: bytes,
    mime_type  : str = "audio/wav",
    sample_rate: int = TARGET_SR,
) -> ProcessedAudio:
    """
    Module-level shortcut — uses a cached default AudioProcessor instance.

    >>> from ai_core.voice.dialect_model.audio_processor import preprocess
    >>> result = preprocess(audio_bytes, mime_type="audio/webm")
    >>> for chunk in result.chunks:
    ...     text = whisper.transcribe(chunk)
    """
    global _default_processor
    if _default_processor is None:
        _default_processor = AudioProcessor()
    return _default_processor.preprocess(audio_bytes, mime_type, sample_rate)


async def preprocess_async(
    audio_bytes: bytes,
    mime_type  : str = "audio/wav",
    sample_rate: int = TARGET_SR,
) -> ProcessedAudio:
    """
    Non-blocking async wrapper for use inside FastAPI / asyncio handlers.

    FIX v1.1 — CPU-bound operations (FFT, resampling, VAD) run in a
    dedicated ThreadPoolExecutor so they never block the FastAPI event loop.
    Without this, 5 concurrent patients would be processed one-by-one;
    with this they run in parallel across 4 threads.

    Usage (inside a FastAPI route)
    ───────────────────────────────
    >>> @app.post("/api/voice")
    ... async def voice_endpoint(audio: UploadFile):
    ...     result = await preprocess_async(await audio.read(), "audio/webm")
    ...     for chunk in result.chunks:
    ...         text = await stt.transcribe_async(chunk)

    Notes
    ─────
    • Thread-safe — AudioProcessor is stateless.
    • max_workers=4 covers typical clinic loads; tune via RIVA_AUDIO_WORKERS.
    """
    import asyncio
    import os

    global _audio_pool
    # Allow overriding pool size via environment variable
    workers = int(os.environ.get("RIVA_AUDIO_WORKERS", 4))
    if _audio_pool._max_workers != workers:
        _audio_pool = _ThreadPoolExecutor(
            max_workers        = workers,
            thread_name_prefix = "riva_audio",
        )

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _audio_pool,
        preprocess,
        audio_bytes,
        mime_type,
        sample_rate,
    )
