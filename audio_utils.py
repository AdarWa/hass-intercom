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
    def __init__(self, audio_format: AudioFormat) -> None:
        super().__init__(audio_format)
        self._play_objects: deque[sa.PlayObject] = deque()

    async def play(self, frame: bytes) -> None:
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
