"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform v4.0 — Audio Processor                       ║
║           ai-core/voice/audio_processor.py                                  ║
║                                                                              ║
║  Handles all audio input processing for RIVA's voice interface:             ║
║  • Raw audio capture & validation                                            ║
║  • Noise reduction & normalization                                           ║
║  • Chunking & VAD (Voice Activity Detection)                                 ║
║  • Format conversion → Whisper-compatible float32 mono 16kHz                ║
║  • Fully offline — zero internet dependency                                  ║
║                                                                              ║
║  Harvard HSIL Hackathon 2026                                                 ║
║  Maintainer: GODA EMAD                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import io
import logging
import math
import struct
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("riva.audio_processor")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SAMPLE_RATE: int = 16_000          # Whisper expects 16kHz
TARGET_CHANNELS: int = 1                  # Mono
TARGET_BIT_DEPTH: int = 16               # 16-bit PCM
MAX_AUDIO_DURATION_SEC: float = 30.0     # Max recording length
MIN_AUDIO_DURATION_SEC: float = 0.3      # Min valid speech length
CHUNK_DURATION_SEC: float = 0.5          # Streaming chunk size
VAD_ENERGY_THRESHOLD: float = 0.01       # Voice activity threshold (RMS)
SILENCE_TIMEOUT_SEC: float = 1.5         # Stop after N seconds of silence
NOISE_REDUCTION_ALPHA: float = 0.95      # Spectral subtraction smoothing


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ─────────────────────────────────────────────────────────────────────────────

class AudioFormat(Enum):
    WAV   = "wav"
    WEBM  = "webm"
    OGG   = "ogg"
    MP3   = "mp3"
    RAW   = "raw"       # raw PCM bytes from browser MediaRecorder
    FLOAT32 = "float32" # already-decoded numpy array


class AudioQuality(Enum):
    EXCELLENT = "excellent"   # SNR > 20dB, duration OK
    GOOD      = "good"        # SNR 10-20dB
    POOR      = "poor"        # SNR < 10dB, may affect accuracy
    SILENT    = "silent"      # No speech detected
    TOO_SHORT = "too_short"   # < MIN_AUDIO_DURATION_SEC
    TOO_LONG  = "too_long"    # > MAX_AUDIO_DURATION_SEC


@dataclass
class AudioMetadata:
    """Metadata extracted from audio before transcription."""
    duration_sec: float
    sample_rate: int
    channels: int
    bit_depth: int
    rms_energy: float
    snr_db: float
    quality: AudioQuality
    has_speech: bool
    speech_ratio: float          # fraction of audio containing speech
    format_detected: AudioFormat
    processing_time_ms: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class ProcessedAudio:
    """
    Output of AudioProcessor.process().
    Always float32 numpy array, mono, 16kHz — ready for Whisper.
    """
    samples: np.ndarray          # shape: (N,) float32, range [-1, 1]
    sample_rate: int             # always TARGET_SAMPLE_RATE
    metadata: AudioMetadata
    success: bool
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Core Processor
# ─────────────────────────────────────────────────────────────────────────────

class AudioProcessor:
    """
    Full audio processing pipeline for RIVA voice interface.

    Usage:
        processor = AudioProcessor()
        result = processor.process(raw_bytes, fmt=AudioFormat.WAV)
        if result.success:
            transcription = whisper_model.transcribe(result.samples)

    All processing is CPU-only, offline, no external API calls.
    Designed to run on low-end hardware (Raspberry Pi, basic Android).
    """

    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        vad_threshold: float = VAD_ENERGY_THRESHOLD,
        noise_reduction: bool = True,
        normalize: bool = True,
        trim_silence: bool = True,
    ) -> None:
        self.target_sr = target_sr
        self.vad_threshold = vad_threshold
        self.noise_reduction = noise_reduction
        self.normalize = normalize
        self.trim_silence = trim_silence
        self._noise_profile: Optional[np.ndarray] = None
        logger.info(
            "AudioProcessor initialized | sr=%d | vad_thr=%.3f | "
            "noise_reduction=%s | normalize=%s",
            target_sr, vad_threshold, noise_reduction, normalize,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def process(
        self,
        audio_input: bytes | np.ndarray,
        fmt: AudioFormat = AudioFormat.WAV,
    ) -> ProcessedAudio:
        """
        Main entry point. Accepts raw bytes or numpy array.
        Returns ProcessedAudio ready for Whisper transcription.
        """
        t_start = time.perf_counter()
        try:
            # 1. Decode to float32
            samples, sr = self._decode(audio_input, fmt)

            # 2. Convert to mono
            if samples.ndim == 2:
                samples = samples.mean(axis=1)

            # 3. Resample to 16kHz if needed
            if sr != self.target_sr:
                samples = self._resample(samples, sr, self.target_sr)

            # 4. Noise reduction
            if self.noise_reduction:
                samples = self._reduce_noise(samples)

            # 5. Normalize amplitude
            if self.normalize:
                samples = self._normalize(samples)

            # 6. Trim leading/trailing silence
            if self.trim_silence:
                samples = self._trim_silence(samples)

            # 7. Build metadata & quality assessment
            proc_ms = (time.perf_counter() - t_start) * 1000
            metadata = self._assess_quality(samples, fmt, proc_ms)

            # 8. Final validation
            if not metadata.has_speech:
                return ProcessedAudio(
                    samples=samples,
                    sample_rate=self.target_sr,
                    metadata=metadata,
                    success=False,
                    error="no_speech_detected",
                )

            if metadata.quality == AudioQuality.TOO_SHORT:
                return ProcessedAudio(
                    samples=samples,
                    sample_rate=self.target_sr,
                    metadata=metadata,
                    success=False,
                    error="audio_too_short",
                )

            return ProcessedAudio(
                samples=samples,
                sample_rate=self.target_sr,
                metadata=metadata,
                success=True,
            )

        except Exception as exc:
            logger.error("AudioProcessor.process failed: %s", exc, exc_info=True)
            proc_ms = (time.perf_counter() - t_start) * 1000
            dummy_meta = AudioMetadata(
                duration_sec=0.0, sample_rate=0, channels=0, bit_depth=0,
                rms_energy=0.0, snr_db=0.0, quality=AudioQuality.POOR,
                has_speech=False, speech_ratio=0.0,
                format_detected=fmt, processing_time_ms=proc_ms,
                warnings=[str(exc)],
            )
            return ProcessedAudio(
                samples=np.zeros(1, dtype=np.float32),
                sample_rate=self.target_sr,
                metadata=dummy_meta,
                success=False,
                error=str(exc),
            )

    def calibrate_noise(self, noise_bytes: bytes, fmt: AudioFormat = AudioFormat.WAV) -> None:
        """
        Calibrate noise profile from a short silence recording (0.5–2 sec).
        Call before processing patient speech for better noise reduction.
        """
        samples, sr = self._decode(noise_bytes, fmt)
        if sr != self.target_sr:
            samples = self._resample(samples, sr, self.target_sr)
        self._noise_profile = self._compute_noise_profile(samples)
        logger.info("Noise profile calibrated from %.2f sec of silence", len(samples)/self.target_sr)

    def process_stream(
        self,
        chunk_iterator,
        fmt: AudioFormat = AudioFormat.RAW,
    ):
        """
        Generator for streaming audio (e.g. browser MediaRecorder chunks).
        Yields ProcessedAudio when a complete utterance is detected via VAD.

        Usage:
            for result in processor.process_stream(ws_chunks):
                transcript = whisper.transcribe(result.samples)
        """
        buffer = np.array([], dtype=np.float32)
        silence_frames = 0
        chunk_samples = int(self.target_sr * CHUNK_DURATION_SEC)
        silence_limit = int(SILENCE_TIMEOUT_SEC / CHUNK_DURATION_SEC)

        for chunk in chunk_iterator:
            samples, sr = self._decode(chunk, fmt)
            if sr != self.target_sr:
                samples = self._resample(samples, sr, self.target_sr)
            samples = self._normalize(samples)
            buffer = np.concatenate([buffer, samples])

            rms = self._rms(samples[-chunk_samples:])
            if rms < self.vad_threshold:
                silence_frames += 1
            else:
                silence_frames = 0

            if silence_frames >= silence_limit and len(buffer) > chunk_samples:
                yield self.process(buffer, fmt=AudioFormat.FLOAT32)
                buffer = np.array([], dtype=np.float32)
                silence_frames = 0

        if len(buffer) > int(self.target_sr * MIN_AUDIO_DURATION_SEC):
            yield self.process(buffer, fmt=AudioFormat.FLOAT32)

    # ── Private: Decode ─────────────────────────────────────────────────────

    def _decode(
        self,
        audio_input: bytes | np.ndarray,
        fmt: AudioFormat,
    ) -> tuple[np.ndarray, int]:
        """Decode any supported format to float32 numpy array + sample rate."""

        if fmt == AudioFormat.FLOAT32 or isinstance(audio_input, np.ndarray):
            arr = np.asarray(audio_input, dtype=np.float32)
            return arr, self.target_sr

        if fmt == AudioFormat.WAV:
            return self._decode_wav(audio_input)

        if fmt == AudioFormat.RAW:
            # Assume 16-bit signed PCM at target_sr
            pcm = np.frombuffer(audio_input, dtype=np.int16).astype(np.float32)
            return pcm / 32768.0, self.target_sr

        if fmt in (AudioFormat.WEBM, AudioFormat.OGG, AudioFormat.MP3):
            return self._decode_compressed(audio_input, fmt)

        raise ValueError(f"Unsupported audio format: {fmt}")

    def _decode_wav(self, data: bytes) -> tuple[np.ndarray, int]:
        """Pure-Python WAV decoder — no external dependencies."""
        with wave.open(io.BytesIO(data)) as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            n_frames   = wf.getnframes()
            raw        = wf.readframes(n_frames)

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth, np.int16)
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)

        # Normalize to [-1, 1]
        max_val = float(2 ** (8 * sampwidth - 1))
        samples /= max_val

        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)

        return samples, framerate

    def _decode_compressed(
        self, data: bytes, fmt: AudioFormat
    ) -> tuple[np.ndarray, int]:
        """
        Decode compressed formats (WebM/OGG/MP3).
        Uses soundfile or pydub if available; falls back to WAV conversion hint.
        """
        try:
            import soundfile as sf
            samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            return samples, sr
        except ImportError:
            pass

        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(io.BytesIO(data), format=fmt.value)
            seg = seg.set_channels(1).set_frame_rate(self.target_sr).set_sample_width(2)
            samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
            return samples / 32768.0, self.target_sr
        except ImportError:
            pass

        raise RuntimeError(
            f"Cannot decode {fmt.value}: install soundfile or pydub. "
            "On low-resource devices, use WAV or RAW format directly."
        )

    # ── Private: DSP Pipeline ───────────────────────────────────────────────

    def _resample(
        self, samples: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """
        Linear interpolation resampling — no scipy needed.
        Good enough for speech; Whisper is robust to minor SR variations.
        """
        if orig_sr == target_sr:
            return samples

        ratio = target_sr / orig_sr
        n_out = int(len(samples) * ratio)
        x_orig = np.arange(len(samples))
        x_new  = np.linspace(0, len(samples) - 1, n_out)
        resampled = np.interp(x_new, x_orig, samples).astype(np.float32)
        logger.debug("Resampled %dHz → %dHz (%d → %d samples)", orig_sr, target_sr, len(samples), n_out)
        return resampled

    def _reduce_noise(self, samples: np.ndarray) -> np.ndarray:
        """
        Spectral subtraction noise reduction.
        Estimates noise floor from first 0.2s if no calibration profile.
        """
        n_fft   = 512
        hop     = n_fft // 2
        n_frames = (len(samples) - n_fft) // hop

        if n_frames < 1:
            return samples

        # Estimate noise from first 0.2s if no profile
        if self._noise_profile is None:
            noise_len = min(int(self.target_sr * 0.2), len(samples) // 4)
            noise_profile = self._compute_noise_profile(samples[:noise_len])
        else:
            noise_profile = self._noise_profile

        # Frame-by-frame spectral subtraction
        output = np.zeros_like(samples)
        window = np.hanning(n_fft).astype(np.float32)

        for i in range(n_frames):
            start = i * hop
            frame = samples[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            mag = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Subtract noise floor
            mag_clean = np.maximum(mag - NOISE_REDUCTION_ALPHA * noise_profile, 0.0)
            spectrum_clean = mag_clean * np.exp(1j * phase)
            frame_clean = np.fft.irfft(spectrum_clean).real * window

            output[start:start + n_fft] += frame_clean

        return output.astype(np.float32)

    def _compute_noise_profile(self, silence: np.ndarray) -> np.ndarray:
        n_fft = 512
        if len(silence) < n_fft:
            return np.ones(n_fft // 2 + 1, dtype=np.float32) * 1e-6
        spectrum = np.fft.rfft(silence[:n_fft])
        return np.abs(spectrum).astype(np.float32)

    def _normalize(self, samples: np.ndarray) -> np.ndarray:
        """Peak normalization to [-1, 1]. Handles silent audio safely."""
        peak = np.abs(samples).max()
        if peak < 1e-7:
            return samples
        return (samples / peak).astype(np.float32)

    def _trim_silence(
        self,
        samples: np.ndarray,
        frame_ms: int = 20,
        padding_ms: int = 100,
    ) -> np.ndarray:
        """
        Trim leading/trailing silence using energy-based VAD.
        Keeps `padding_ms` of silence at each edge for natural speech.
        """
        frame_len = int(self.target_sr * frame_ms / 1000)
        padding   = int(self.target_sr * padding_ms / 1000)

        energies = [
            self._rms(samples[i:i + frame_len])
            for i in range(0, len(samples) - frame_len, frame_len)
        ]
        if not energies:
            return samples

        threshold = self.vad_threshold
        speech_frames = [e > threshold for e in energies]

        try:
            first = next(i for i, s in enumerate(speech_frames) if s)
            last  = len(speech_frames) - 1 - next(
                i for i, s in enumerate(reversed(speech_frames)) if s
            )
        except StopIteration:
            return samples  # no speech found — return as-is

        start = max(0, first * frame_len - padding)
        end   = min(len(samples), (last + 1) * frame_len + padding)
        return samples[start:end]

    # ── Private: Quality Assessment ─────────────────────────────────────────

    def _assess_quality(
        self,
        samples: np.ndarray,
        fmt: AudioFormat,
        proc_ms: float,
    ) -> AudioMetadata:
        """Compute metadata and quality label for the processed audio."""
        duration = len(samples) / self.target_sr
        rms      = float(self._rms(samples))
        snr_db   = self._estimate_snr(samples)

        # Speech ratio via frame-level VAD
        frame_len = int(self.target_sr * 0.02)
        frames = [
            self._rms(samples[i:i + frame_len]) > self.vad_threshold
            for i in range(0, len(samples) - frame_len, frame_len)
        ]
        speech_ratio = sum(frames) / max(len(frames), 1)
        has_speech   = speech_ratio > 0.05

        # Quality label
        warnings: list[str] = []
        if duration < MIN_AUDIO_DURATION_SEC:
            quality = AudioQuality.TOO_SHORT
            warnings.append(f"Duration {duration:.2f}s < minimum {MIN_AUDIO_DURATION_SEC}s")
        elif duration > MAX_AUDIO_DURATION_SEC:
            quality = AudioQuality.TOO_LONG
            warnings.append(f"Duration {duration:.2f}s > maximum {MAX_AUDIO_DURATION_SEC}s")
        elif not has_speech:
            quality = AudioQuality.SILENT
            warnings.append("No speech detected — check microphone or ambient noise")
        elif snr_db >= 20:
            quality = AudioQuality.EXCELLENT
        elif snr_db >= 10:
            quality = AudioQuality.GOOD
        else:
            quality = AudioQuality.POOR
            warnings.append(f"Low SNR ({snr_db:.1f}dB) — transcription accuracy may be reduced")

        return AudioMetadata(
            duration_sec=round(duration, 3),
            sample_rate=self.target_sr,
            channels=TARGET_CHANNELS,
            bit_depth=TARGET_BIT_DEPTH,
            rms_energy=round(rms, 6),
            snr_db=round(snr_db, 2),
            quality=quality,
            has_speech=has_speech,
            speech_ratio=round(speech_ratio, 3),
            format_detected=fmt,
            processing_time_ms=round(proc_ms, 2),
            warnings=warnings,
        )

    # ── Private: Utilities ──────────────────────────────────────────────────

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        """Root-Mean-Square energy."""
        if len(samples) == 0:
            return 0.0
        return float(math.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    @staticmethod
    def _estimate_snr(samples: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio (dB).
        Uses the lowest 10% energy frames as noise estimate.
        """
        if len(samples) < 320:
            return 0.0

        frame_len = 160
        energies  = sorted([
            np.mean(samples[i:i + frame_len] ** 2)
            for i in range(0, len(samples) - frame_len, frame_len)
        ])
        noise_floor = max(np.mean(energies[:max(1, len(energies)//10)]), 1e-12)
        signal_pwr  = max(np.mean(energies[len(energies)//2:]), 1e-12)

        snr = 10 * math.log10(signal_pwr / noise_floor)
        return float(min(max(snr, 0.0), 60.0))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_wav_file(path: str | Path) -> bytes:
    """Load a WAV file from disk as raw bytes."""
    return Path(path).read_bytes()


def samples_to_wav(samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """
    Convert float32 numpy array back to WAV bytes (for saving/sending).
    Useful for logging or sending processed audio to external STT APIs.
    """
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def detect_format(raw_bytes: bytes) -> AudioFormat:
    """
    Detect audio format from magic bytes (file signature).
    Used when format is unknown (e.g. browser upload).
    """
    if raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WAVE':
        return AudioFormat.WAV
    if raw_bytes[:4] == b'OggS':
        return AudioFormat.OGG
    if raw_bytes[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2', b'ID3'):
        return AudioFormat.MP3
    if raw_bytes[:4] == b'\x1aE\xdf\xa3':
        return AudioFormat.WEBM
    return AudioFormat.RAW


# ─────────────────────────────────────────────────────────────────────────────
# Singleton for FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

_processor_instance: Optional[AudioProcessor] = None


def get_audio_processor() -> AudioProcessor:
    """
    Returns a shared AudioProcessor instance (FastAPI dependency injection).

    Usage in routes:
        from ai_core.voice.audio_processor import get_audio_processor

        @router.post("/voice")
        async def voice_endpoint(
            file: UploadFile,
            processor: AudioProcessor = Depends(get_audio_processor),
        ):
            raw = await file.read()
            fmt = detect_format(raw)
            result = processor.process(raw, fmt)
            ...
    """
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = AudioProcessor(
            target_sr=TARGET_SAMPLE_RATE,
            vad_threshold=VAD_ENERGY_THRESHOLD,
            noise_reduction=True,
            normalize=True,
            trim_silence=True,
        )
    return _processor_instance


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run directly: python audio_processor.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("RIVA AudioProcessor — self-test")
    print("=" * 60)

    processor = AudioProcessor()

    # Generate a synthetic 440Hz sine wave (1 second) as test audio
    sr = TARGET_SAMPLE_RATE
    t  = np.linspace(0, 1.0, sr, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # Add slight noise
    tone += np.random.normal(0, 0.02, size=len(tone)).astype(np.float32)

    # Save as WAV bytes
    wav_bytes = samples_to_wav(tone, sr)
    print(f"Test signal : 440Hz sine, 1.0 sec, {len(wav_bytes)} bytes WAV")

    # Process
    result = processor.process(wav_bytes, fmt=AudioFormat.WAV)

    print(f"\nResult:")
    print(f"  success          : {result.success}")
    print(f"  samples shape    : {result.samples.shape}")
    print(f"  sample_rate      : {result.sample_rate}")
    print(f"  duration_sec     : {result.metadata.duration_sec}")
    print(f"  quality          : {result.metadata.quality.value}")
    print(f"  snr_db           : {result.metadata.snr_db}")
    print(f"  has_speech       : {result.metadata.has_speech}")
    print(f"  speech_ratio     : {result.metadata.speech_ratio}")
    print(f"  processing_ms    : {result.metadata.processing_time_ms}")
    print(f"  warnings         : {result.metadata.warnings}")
    print(f"  format_detected  : {result.metadata.format_detected.value}")

    # Test format detection
    fmt = detect_format(wav_bytes)
    print(f"\nFormat detection : {fmt.value}  (expected: wav)")

    print("\n✅ AudioProcessor self-test complete")
    sys.exit(0 if result.success or result.error == "no_speech_detected" else 1)
