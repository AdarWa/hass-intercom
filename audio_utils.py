from __future__ import annotations

import asyncio
import base64
from collections import deque
from dataclasses import dataclass
from typing import Optional

import simpleaudio as sa

try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency check
    sd = None
    _sounddevice_import_error = exc
else:
    _sounddevice_import_error = None


BYTES_PER_SAMPLE = 2


def encode_audio_payload(data: bytes) -> str:
    """Encode raw PCM bytes as an ASCII-safe payload."""
    return base64.b64encode(data).decode("ascii")


def decode_audio_payload(payload: str) -> bytes:
    """Decode an ASCII payload back into raw PCM bytes."""
    return base64.b64decode(payload.encode("ascii"), validate=True)


@dataclass(slots=True)
class AudioFormat:
    encoding: str = "pcm_s16le"
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 40

    def __post_init__(self) -> None:
        if self.encoding != "pcm_s16le":
            raise ValueError("only pcm_s16le audio is supported")
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

    async def start(self) -> None:  # pragma: no cover - default no-op
        return None

    async def stop(self) -> None:  # pragma: no cover - default no-op
        return None

    async def read_frame(self) -> bytes:
        raise NotImplementedError


class MicrophoneAudioSource(AudioSource):
    """Capture PCM frames from the system microphone using sounddevice."""

    def __init__(self, audio_format: AudioFormat) -> None:
        super().__init__(audio_format)
        self._stream: Optional[object] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)

    async def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is required for microphone capture") from _sounddevice_import_error
        if self._stream is not None:
            return

        self._loop = asyncio.get_running_loop()
        frame_bytes = self.format.frame_bytes
        samples_per_frame = self.format.samples_per_frame
        queue = self._queue

        def callback(indata, _frames, _time_info, status) -> None:
            if status:  # pragma: no cover - logging only
                import logging

                logging.warning("Microphone stream status: %s", status)
            data = bytes(indata)
            if len(data) < frame_bytes:
                data = data + bytes(frame_bytes - len(data))
            elif len(data) > frame_bytes:
                data = data[:frame_bytes]
            loop = self._loop
            if loop is None:
                return

            def deliver() -> None:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:  # pragma: no cover - defensive
                        return
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:  # pragma: no cover - defensive
                    return

            loop.call_soon_threadsafe(deliver)

        def open_stream():
            return sd.RawInputStream(
                samplerate=self.format.sample_rate,
                channels=self.format.channels,
                dtype="int16",
                blocksize=samples_per_frame,
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
        while not self._queue.empty():  # clear any buffered audio
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - defensive
                break

    async def read_frame(self) -> bytes:
        return await self._queue.get()


class AudioSink:
    def __init__(self, audio_format: AudioFormat) -> None:
        self.format = audio_format

    async def start(self) -> None:  # pragma: no cover - default no-op
        return None

    async def stop(self) -> None:  # pragma: no cover - default no-op
        return None

    async def play(self, frame: bytes) -> None:
        raise NotImplementedError


class SimpleAudioSink(AudioSink):
    """Playback PCM frames using simpleaudio."""

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
