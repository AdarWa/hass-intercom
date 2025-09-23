from __future__ import annotations

import asyncio
import base64
import math
import wave
from array import array
from collections import deque
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import iirnotch, butter, lfilter

try:  # pragma: no cover - optional dependency
    import webrtc_audio_processing as webrtc_ap  # type: ignore
except ImportError:  # pragma: no cover - gracefully degrade when missing
    webrtc_ap = None


@dataclass(slots=True)
class WebRTCProcessorConfig:
    enable_aec: bool = True
    enable_agc: bool = False
    enable_noise_suppression: bool = True
    enable_high_pass_filter: bool = True
    noise_suppression_level: str = "high"
    agc_mode: str = "adaptive_digital"
    agc_target_level_dbfs: int = 3
    agc_compression_gain_db: int = 9


class WebRTCAudioProcessor:
    """Wrapper around the optional WebRTC audio processing module."""

    def __init__(self, audio_format: AudioFormat, config: Optional[WebRTCProcessorConfig] = None) -> None:
        if webrtc_ap is None:
            raise RuntimeError("webrtc_audio_processing package is not available")
        if audio_format.channels != 1:
            raise ValueError("WebRTC audio processing currently supports mono audio only")
        self.format = audio_format
        self.config = config or WebRTCProcessorConfig()

        kwargs = {
            "enable_agc": self.config.enable_agc,
            "enable_ns": self.config.enable_noise_suppression,
            "enable_aec": self.config.enable_aec,
            "enable_hp": self.config.enable_high_pass_filter,
        }

        if self.config.enable_noise_suppression:
            ns_level = self._resolve_enum(
                "NoiseSuppressionLevel",
                self.config.noise_suppression_level,
                default="HIGH",
            )
            if ns_level is not None:
                kwargs["noise_suppression_level"] = ns_level

        if self.config.enable_agc:
            agc_mode = self._resolve_enum("AgcMode", self.config.agc_mode, default="ADAPTIVE_DIGITAL")
            if agc_mode is not None:
                kwargs["agc_mode"] = agc_mode
            kwargs["agc_target_level_dbfs"] = self.config.agc_target_level_dbfs
            kwargs["agc_compression_gain_db"] = self.config.agc_compression_gain_db

        self._processor = webrtc_ap.AudioProcessing(**kwargs)

    def process_capture_frame(self, frame: bytes) -> bytes:
        if not frame:
            return frame
        samples = np.frombuffer(frame, dtype=np.int16)
        samples = np.ascontiguousarray(samples)
        processed = self._processor.process_stream(samples, self.format.sample_rate, self.format.channels)
        processed_array = np.asarray(processed, dtype=np.int16)
        return processed_array.tobytes()

    def process_render_frame(self, frame: bytes) -> None:
        if not frame:
            return
        samples = np.frombuffer(frame, dtype=np.int16)
        samples = np.ascontiguousarray(samples)
        self._processor.process_reverse_stream(samples, self.format.sample_rate, self.format.channels)

    def reset(self) -> None:
        if hasattr(self._processor, "reset"):
            self._processor.reset()
        elif hasattr(self._processor, "initialize"):
            self._processor.initialize()

    @staticmethod
    def _resolve_enum(enum_name: str, value: str, *, default: str) -> Optional[object]:
        if webrtc_ap is None:
            return None
        enum_type = getattr(webrtc_ap, enum_name, None)
        if enum_type is None:
            return None
        try:
            return getattr(enum_type, value.upper())
        except AttributeError:
            try:
                return getattr(enum_type, default)
            except AttributeError:
                return None


class EchoCanceller:
    """Simple normalized LMS echo canceller for mono PCM streams."""

    def __init__(
        self,
        audio_format: AudioFormat,
        filter_length: int = 1024,
        adaptation_rate: float = 0.05,
        leakage: float = 0.999,
        min_power: float = 1e-3,
    ) -> None:
        if audio_format.channels != 1:
            raise ValueError("echo cancellation currently supports mono audio only")
        if filter_length <= 0:
            raise ValueError("filter_length must be positive")
        if adaptation_rate <= 0:
            raise ValueError("adaptation_rate must be positive")
        if min_power <= 0:
            raise ValueError("min_power must be positive")

        self.format = audio_format
        self.filter_length = int(filter_length)
        self.adaptation_rate = float(adaptation_rate)
        self.leakage = float(min(max(leakage, 0.0), 1.0))
        self.min_power = float(min_power)

        self._weights = np.zeros(self.filter_length, dtype=np.float32)
        self._far_history = np.zeros(self.filter_length, dtype=np.float32)
        self._max_frame_samples = max(1, self.format.samples_per_frame)
        self._history_limit = self.filter_length + self._max_frame_samples - 1

    def reset(self) -> None:
        self._weights.fill(0.0)
        self._far_history = np.zeros(self.filter_length, dtype=np.float32)
        self._max_frame_samples = max(1, self.format.samples_per_frame)
        self._history_limit = self.filter_length + self._max_frame_samples - 1

    def update_far(self, frame: bytes) -> None:
        if not frame:
            return
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        frame_len = samples.size
        if frame_len == 0:
            return
        self._ensure_history_limit(frame_len)
        self._far_history = np.concatenate((self._far_history, samples))
        if self._far_history.size > self._history_limit:
            self._far_history = self._far_history[-self._history_limit :]

    def cancel(self, near_frame: bytes) -> bytes:
        if not near_frame:
            return near_frame

        near = np.frombuffer(near_frame, dtype=np.int16).astype(np.float32)
        frame_len = near.size
        if frame_len == 0:
            return near_frame

        self._ensure_history_limit(frame_len)

        history = self._far_history
        required = self.filter_length + frame_len - 1
        if history.size < required:
            pad = np.zeros(required - history.size, dtype=np.float32)
            history = np.concatenate((pad, history))
        windows = np.lib.stride_tricks.sliding_window_view(history, self.filter_length, axis=0)
        if windows.shape[0] < frame_len:
            pad_rows = frame_len - windows.shape[0]
            pad_windows = np.zeros((pad_rows, self.filter_length), dtype=np.float32)
            windows = np.concatenate((pad_windows, windows), axis=0)
        window_block = windows[-frame_len:]

        weights = self._weights
        errors = np.empty_like(near)
        for i in range(frame_len):
            reference = window_block[i]
            estimated = float(np.dot(weights, reference))
            error = near[i] - estimated
            power = float(np.dot(reference, reference)) + self.min_power
            gain = (self.adaptation_rate / power) * error
            weights = self.leakage * weights + gain * reference
            errors[i] = error

        self._weights = weights

        cancelled = np.clip(errors, -32768, 32767).astype(np.int16)
        return cancelled.tobytes()

    def _ensure_history_limit(self, frame_samples: int) -> None:
        if frame_samples <= 0:
            return
        if frame_samples > self._max_frame_samples:
            self._max_frame_samples = frame_samples
        limit = self.filter_length + self._max_frame_samples - 1
        if limit != self._history_limit:
            self._history_limit = limit
        if self._far_history.size > self._history_limit:
            self._far_history = self._far_history[-self._history_limit :]

import simpleaudio as sa

BYTES_PER_SAMPLE = 2

def encode_audio_payload(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_audio_payload(payload: str) -> bytes:
    return base64.b64decode(payload.encode("ascii"), validate=True)


@dataclass(slots=True)
class AudioFormat:
    encoding: str = "pcm_s16le"
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 40

    def __post_init__(self) -> None:
        if self.encoding != "pcm_s16le":
            raise ValueError("only pcm_s16le is supported")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("invalid audio format")
        if self.frame_ms <= 0:
            raise ValueError("frame duration must be positive")

    @property
    def samples_per_frame(self) -> int:
        return (self.sample_rate * self.frame_ms) // 1000

    @property
    def frame_bytes(self) -> int:
        return self.samples_per_frame * self.channels * BYTES_PER_SAMPLE

class AudioFilter:
    def __init__(self, audio_format: AudioFormat):
        self.format = audio_format

    def process(self, frame: bytes) -> bytes:
        return frame


class NotchFilter(AudioFilter):
    def __init__(self, audio_format: AudioFormat, freq=50, q=30.0):
        super().__init__(audio_format)
        self.b, self.a = iirnotch(freq, q, audio_format.sample_rate)
        # keep filter state between frames
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)

    def process(self, frame: bytes) -> bytes:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        filtered, self.zi = lfilter(self.b, self.a, samples, zi=self.zi)
        return np.clip(filtered, -32768, 32767).astype(np.int16).tobytes()


class HighPassFilter(AudioFilter):
    def __init__(self, audio_format: AudioFormat, cutoff=100, order=4):
        super().__init__(audio_format)
        nyquist = 0.5 * audio_format.sample_rate
        norm_cutoff = cutoff / nyquist
        self.b, self.a = butter(order, norm_cutoff, btype="high", analog=False)
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)

    def process(self, frame: bytes) -> bytes:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        filtered, self.zi = lfilter(self.b, self.a, samples, zi=self.zi)
        return np.clip(filtered, -32768, 32767).astype(np.int16).tobytes()

class AudioSource:
    def __init__(self, audio_format: AudioFormat) -> None:
        self.format = audio_format

    async def start(self) -> None:  # pragma: no cover - subclasses may override
        return None

    async def stop(self) -> None:  # pragma: no cover
        return None

    async def read_frame(self) -> bytes:
        raise NotImplementedError


class SilenceAudioSource(AudioSource):
    async def read_frame(self) -> bytes:
        return bytes(self.format.frame_bytes)


class ToneAudioSource(AudioSource):
    def __init__(
        self,
        audio_format: AudioFormat,
        frequency: float = 440.0,
        amplitude: float = 0.2,
    ) -> None:
        super().__init__(audio_format)
        self.frequency = frequency
        self.amplitude = max(0.0, min(amplitude, 1.0))
        self._phase = 0.0
        if self.format.channels != 1:
            raise ValueError("tone source only supports mono")

    async def read_frame(self) -> bytes:
        step = 2.0 * math.pi * self.frequency / self.format.sample_rate
        samples = array("h")
        for _ in range(self.format.samples_per_frame):
            value = math.sin(self._phase) * self.amplitude
            self._phase += step
            if self._phase > 2.0 * math.pi:
                self._phase -= 2.0 * math.pi
            samples.append(int(max(-1.0, min(1.0, value)) * 32767))
        return samples.tobytes()


class MicrophoneAudioSource(AudioSource):
    def __init__(self, audio_format: AudioFormat, device: Optional[str] = None) -> None:
        super().__init__(audio_format)
        self.device = device
        self._stream = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)

    async def start(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sounddevice is required for microphone capture") from exc

        if self._stream is not None:
            return

        self._loop = asyncio.get_running_loop()
        frame_bytes = self.format.frame_bytes
        samples_per_frame = self.format.samples_per_frame
        queue = self._queue

        def callback(indata, frames, _time_info, status) -> None:
            if status:  # pragma: no cover - logging only
                logging.warning("Microphone stream status: %s", status)
            data = bytes(indata)
            if len(data) < frame_bytes:
                data = data + bytes(frame_bytes - len(data))
            elif len(data) > frame_bytes:
                data = data[:frame_bytes]
            if self._loop is None:
                return
            loop = self._loop

            def deliver() -> None:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:  # pragma: no cover - defensive
                        pass
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:  # pragma: no cover - defensive
                    pass

            loop.call_soon_threadsafe(deliver)

        def open_stream():
            return sd.RawInputStream(
                samplerate=self.format.sample_rate,
                channels=self.format.channels,
                dtype="int16",
                blocksize=samples_per_frame,
                device=self.device,
                callback=callback,
            )

        self._stream = await asyncio.to_thread(open_stream)
        await asyncio.to_thread(self._stream.start)

    async def stop(self) -> None:
        if self._stream is not None:
            await asyncio.to_thread(self._stream.stop)
            await asyncio.to_thread(self._stream.close)
            self._stream = None
        self._loop = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - defensive
                break

    async def read_frame(self) -> bytes:
        return await self._queue.get()


class FileAudioSource(AudioSource):
    def __init__(self, audio_format: AudioFormat, path: str | Path, loop: bool = True) -> None:
        super().__init__(audio_format)
        self.path = Path(path)
        self.loop = loop
        self._wave: Optional[wave.Wave_read] = None

    async def start(self) -> None:
        wave_file = wave.open(str(self.path), "rb")
        if wave_file.getsampwidth() != BYTES_PER_SAMPLE:
            wave_file.close()
            raise ValueError("source file must be 16-bit audio")
        if wave_file.getframerate() != self.format.sample_rate:
            wave_file.close()
            raise ValueError("source sample rate mismatch")
        if wave_file.getnchannels() != self.format.channels:
            wave_file.close()
            raise ValueError("source channel count mismatch")
        self._wave = wave_file

    async def stop(self) -> None:
        if self._wave is not None:
            self._wave.close()
            self._wave = None

    async def read_frame(self) -> bytes:
        if self._wave is None:
            raise RuntimeError("audio source not started")
        data = self._wave.readframes(self.format.samples_per_frame)
        if len(data) < self.format.frame_bytes:
            if not data and self.loop:
                self._wave.rewind()
                data = self._wave.readframes(self.format.samples_per_frame)
            if len(data) < self.format.frame_bytes:
                data = data + bytes(self.format.frame_bytes - len(data))
        return data


class AudioSink:
    def __init__(self, audio_format: AudioFormat) -> None:
        self.format = audio_format

    async def start(self) -> None:  # pragma: no cover
        return None

    async def stop(self) -> None:  # pragma: no cover
        return None

    async def play(self, frame: bytes) -> None:
        raise NotImplementedError


class NullAudioSink(AudioSink):
    async def play(self, frame: bytes) -> None:  # pragma: no cover - intentional no-op
        return None


class SimpleAudioSink(AudioSink):
    def __init__(self, audio_format: AudioFormat, device: Optional[str] = None) -> None:
        super().__init__(audio_format)
        self.device = device or None
        self._play_objects: deque[sa.PlayObject] = deque()
        self._sd_stream: Optional[object] = None

    async def start(self) -> None:
        if self.device is None or self._sd_stream is not None:
            return
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sounddevice is required to select an output device") from exc

        def open_stream():
            return sd.RawOutputStream(
                samplerate=self.format.sample_rate,
                channels=self.format.channels,
                dtype="int16",
                blocksize=self.format.samples_per_frame,
                device=self.device,
            )

        stream = await asyncio.to_thread(open_stream)
        await asyncio.to_thread(stream.start)
        self._sd_stream = stream

    async def play(self, frame: bytes) -> None:
        if self._sd_stream is not None:
            await asyncio.to_thread(self._sd_stream.write, frame)
            return

        play_obj = await asyncio.to_thread(
            sa.play_buffer,
            frame,
            self.format.channels,
            BYTES_PER_SAMPLE,
            self.format.sample_rate,
        )
        self._play_objects.append(play_obj)
        while self._play_objects and not self._play_objects[0].is_playing():
            self._play_objects.popleft()

    async def stop(self) -> None:
        if self._sd_stream is not None:
            stream = self._sd_stream
            self._sd_stream = None
            await asyncio.to_thread(stream.stop)
            await asyncio.to_thread(stream.close)

        while self._play_objects:
            play_obj = self._play_objects.popleft()
            play_obj.stop()


class WaveFileSink(AudioSink):
    def __init__(self, audio_format: AudioFormat, path: str | Path) -> None:
        super().__init__(audio_format)
        self.path = Path(path)
        self._wave: Optional[wave.Wave_write] = None

    async def start(self) -> None:
        wave_file = wave.open(str(self.path), "wb")
        wave_file.setnchannels(self.format.channels)
        wave_file.setsampwidth(BYTES_PER_SAMPLE)
        wave_file.setframerate(self.format.sample_rate)
        self._wave = wave_file

    async def stop(self) -> None:
        if self._wave is not None:
            await asyncio.to_thread(self._wave.close)
            self._wave = None

    async def play(self, frame: bytes) -> None:
        if self._wave is None:
            raise RuntimeError("audio sink not started")
        await asyncio.to_thread(self._wave.writeframes, frame)
