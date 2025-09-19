"""Lean intercom client providing mandatory bi-directional PCM audio."""
from __future__ import annotations

import argparse
import asyncio
import asyncio.subprocess
import base64
import binascii
import contextlib
import json
import logging
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional

ENCODING = "utf-8"
REGISTER_TIMEOUT_SECONDS = 10
DEFAULT_ENCODING = "pcm_s16le"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1

CommandHandler = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]
AudioStreamFactory = Callable[[str, Dict[str, Any]], Awaitable["AudioStreamConfig"]]
AudioFrameHandler = Callable[[str, Dict[str, Any]], Awaitable[None]]


class RegistrationError(RuntimeError):
    """Raised when the client fails to register with the host."""


@dataclass
class IntercomConfig:
    host: str
    port: int
    client_id: str
    token: Optional[str] = None


@dataclass
class AudioStreamConfig:
    stream_id: str
    frame_source: AsyncIterator[bytes]
    encoding: str = DEFAULT_ENCODING
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS


_SIMPLEAUDIO_AVAILABLE = False
try:  # pragma: no cover - environment dependent
    import simpleaudio  # type: ignore  # noqa: F401

    _SIMPLEAUDIO_AVAILABLE = True
except Exception:  # pragma: no cover - best effort detection
    _SIMPLEAUDIO_AVAILABLE = False


class AudioPlayback:
    """Pipe PCM frames to `aplay` for local monitoring."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        if not shutil.which("aplay"):
            raise RuntimeError("aplay executable not found")
        self.sample_rate = sample_rate
        self.channels = channels
        self._proc: Optional[subprocess.Popen] = None

    async def start(self) -> None:
        if self._proc is not None:
            return
        cmd = [
            "aplay",
            "-q",
            "-c",
            str(self.channels),
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-t",
            "raw",
            "-",
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    async def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        self._proc.terminate()
        self._proc = None

    async def enqueue(self, data: bytes) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except BrokenPipeError:
            logging.warning("aplay pipe closed; disabling playback")
            await self.stop()


class IntercomClient:
    def __init__(
        self,
        config: IntercomConfig,
        handler: CommandHandler,
        *,
        audio_stream_factory: Optional[AudioStreamFactory] = None,
        audio_frame_handler: Optional[AudioFrameHandler] = None,
        audio_input_command: Optional[str] = None,
        enable_playback: bool = False,
    ) -> None:
        self._config = config
        self._handler = handler
        self._audio_stream_factory = audio_stream_factory
        self._audio_frame_handler = audio_frame_handler
        if not audio_input_command:
            raise ValueError("audio_input_command is required for audio capture")
        self._audio_input_command = audio_input_command
        self._playback_enabled = enable_playback

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._stopped = asyncio.Event()
        self._reader_task: Optional[asyncio.Task[None]] = None

        self._audio_streams: Dict[str, asyncio.Task[None]] = {}
        self._audio_stream_metadata: Dict[str, Dict[str, Any]] = {}
        self._audio_sequences: Dict[str, int] = {}
        self._playback: Optional[AudioPlayback] = None
        self._playback_stream_id: Optional[str] = None

    async def connect(self) -> None:
        logging.info("Connecting to host %s:%s", self._config.host, self._config.port)
        reader, writer = await asyncio.open_connection(self._config.host, self._config.port)
        self._reader = reader
        self._writer = writer
        self._stopped = asyncio.Event()

        register_message: Dict[str, Any] = {
            "type": "register",
            "role": "intercom",
            "client_id": self._config.client_id,
        }
        if self._config.token is not None:
            register_message["token"] = self._config.token

        await self._send_json(register_message)

        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=REGISTER_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise RegistrationError("Timed out waiting for registered message") from exc

        if not raw:
            raise RegistrationError("Server closed connection during registration")

        response = self._decode_message(raw)
        if response.get("type") != "registered" or response.get("status") != "ok":
            raise RegistrationError(f"Unexpected registration response: {response}")

        logging.info("Successfully registered as '%s'", self._config.client_id)

    def start_background(self) -> None:
        if self._reader_task is not None and not self._reader_task.done():
            return
        if self._reader is None:
            raise RuntimeError("connect() must be called before start_background()")
        self._reader_task = asyncio.create_task(self._reader_loop(), name="intercom-reader")

    async def run(self) -> None:
        if self._reader is None or self._writer is None:
            raise RuntimeError("connect() must be called before run()")

        self.start_background()
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            logging.info("Shutdown signal received")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                # Signals are not supported on all platforms (e.g. Windows event loop).
                pass

        await stop_event.wait()
        await self.close()

    async def _reader_loop(self) -> None:
        assert self._reader is not None
        reader = self._reader
        while not self._stopped.is_set():
            raw = await reader.readline()
            if not raw:
                logging.info("Server closed connection")
                self._stopped.set()
                break

            try:
                message = self._decode_message(raw)
            except json.JSONDecodeError as exc:
                logging.warning("Invalid JSON from host: %s", exc)
                continue

            message_type = message.get("type")
            if message_type == "command":
                await self._handle_command(message)
            elif message_type == "error":
                logging.error("Host reported error: %s", message)
            elif message_type == "event":
                logging.debug("Received event from host (ignored): %s", message)
            elif message_type == "command_ack":
                logging.debug("Host sent command_ack to intercom (ignored): %s", message)
            elif message_type == "audio_frame":
                await self._handle_audio_frame_message(message)
            else:
                logging.warning("Unknown message from host: %s", message)

    async def _handle_command(self, message: Dict[str, Any]) -> None:
        command = message.get("command")
        payload = message.get("payload", {})
        command_id = message.get("command_id")
        origin_id = message.get("origin_id")

        if not isinstance(command, str) or not command:
            logging.warning("Ignoring command without name: %s", message)
            return
        if not isinstance(payload, dict):
            logging.warning("Ignoring command with invalid payload: %s", message)
            payload = {}

        logging.info("Received command %s (origin: %s, id: %s)", command, origin_id, command_id)

        try:
            if command == "start_audio":
                response_payload = await self._process_start_audio(payload)
            elif command == "stop_audio":
                response_payload = await self._process_stop_audio(payload)
            else:
                response_payload = await self._handler(command, payload)
                if not isinstance(response_payload, dict):
                    raise TypeError("Command handler must return a dict payload")
            status = "ok"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.exception("Command handler raised for %s", command)
            status = "error"
            response_payload = {"reason": "handler_exception", "details": str(exc)}

        await self._send_json(
            {
                "type": "response",
                "command_id": command_id or str(uuid.uuid4()),
                "status": status,
                "payload": response_payload,
            }
        )
        logging.info(
            "Sent response for %s (status=%s, command_id=%s)", command, status, command_id
        )

    async def _process_start_audio(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        factory = self._audio_stream_factory or self._default_audio_factory
        config = await factory("start_audio", payload)
        await self.start_audio_stream(config)
        return {
            "stream_id": config.stream_id,
            "encoding": config.encoding,
            "sample_rate": config.sample_rate,
            "channels": config.channels,
        }

    async def _process_stop_audio(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stream_id = payload.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stop_audio requires a stream_id")
        metadata = await self.stop_audio_stream(stream_id)
        response: Dict[str, Any] = {"stream_id": stream_id}
        if metadata:
            response.update(metadata)
        return response

    async def _handle_audio_frame_message(self, message: Dict[str, Any]) -> None:
        stream_id = message.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            logging.warning("Dropped audio frame without stream_id: %s", message)
            return

        handler = self._audio_frame_handler
        if handler is not None:
            await handler(stream_id, message)
            return

        logging.debug(
            "Received talkback frame for %s (sequence=%s, bytes=%s)",
            stream_id,
            message.get("sequence"),
            len(message.get("data", "")),
        )

        if not self._playback_enabled:
            return

        if self._playback is None:
            sample_rate = int(message.get("sample_rate", DEFAULT_SAMPLE_RATE))
            channels = int(message.get("channels", DEFAULT_CHANNELS))
            try:
                self._playback = AudioPlayback(sample_rate, channels)
                await self._playback.start()
                self._playback_stream_id = stream_id
            except Exception as exc:
                logging.warning("Playback disabled: %s", exc)
                self._playback_enabled = False
                return

        if self._playback_stream_id != stream_id:
            # Switch to the new stream
            await self._stop_playback()
            try:
                sample_rate = int(message.get("sample_rate", DEFAULT_SAMPLE_RATE))
                channels = int(message.get("channels", DEFAULT_CHANNELS))
                self._playback = AudioPlayback(sample_rate, channels)
                await self._playback.start()
                self._playback_stream_id = stream_id
            except Exception as exc:
                logging.warning("Playback disabled: %s", exc)
                self._playback_enabled = False
                return

        try:
            pcm = base64.b64decode(message.get("data", ""), validate=True)
        except binascii.Error as exc:
            logging.warning("Failed to decode talkback frame for %s: %s", stream_id, exc)
            return

        await self._playback.enqueue(pcm)

    async def send_audio_frame(
        self,
        stream_id: str,
        data: bytes,
        *,
        encoding: str = DEFAULT_ENCODING,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        sequence: Optional[int] = None,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("Client is not connected")
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Audio data must be bytes")
        if sequence is not None and not isinstance(sequence, int):
            raise TypeError("sequence must be an int")

        if sequence is None:
            sequence = self._audio_sequences.get(stream_id, 0)
        payload = {
            "type": "audio_frame",
            "stream_id": stream_id,
            "encoding": encoding,
            "sample_rate": sample_rate,
            "channels": channels,
            "data": base64.b64encode(bytes(data)).decode("ascii"),
        }
        if sequence is not None:
            payload["sequence"] = sequence

        await self._send_json(payload)
        self._audio_sequences[stream_id] = sequence + 1
        logging.debug(
            "Sent audio frame for %s (sequence=%s, size=%s)",
            stream_id,
            sequence,
            len(payload["data"]),
        )

    async def send_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self._writer is None:
            raise RuntimeError("Client is not connected")
        if not isinstance(event, str) or not event:
            raise ValueError("event must be a non-empty string")

        message = {
            "type": "event",
            "event": event,
            "payload": payload or {},
        }
        await self._send_json(message)
        logging.info("Sent event %s", event)

    async def start_audio_stream(self, config: AudioStreamConfig) -> None:
        if config.stream_id in self._audio_streams:
            raise ValueError(f"Stream {config.stream_id} already active")

        metadata = {
            "encoding": config.encoding,
            "sample_rate": config.sample_rate,
            "channels": config.channels,
        }
        self._audio_stream_metadata[config.stream_id] = metadata
        self._audio_sequences[config.stream_id] = 0

        task = asyncio.create_task(
            self._audio_sender_loop(config.stream_id, config),
            name=f"audio-stream-{config.stream_id}",
        )
        self._audio_streams[config.stream_id] = task

    async def stop_audio_stream(self, stream_id: str) -> Optional[Dict[str, Any]]:
        metadata = self._audio_stream_metadata.get(stream_id)
        task = self._audio_streams.pop(stream_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._audio_stream_metadata.pop(stream_id, None)
        self._audio_sequences.pop(stream_id, None)
        return metadata

    async def _audio_sender_loop(self, stream_id: str, config: AudioStreamConfig) -> None:
        frame_source = config.frame_source
        sequence = 0
        try:
            async for frame in frame_source:
                if frame is None:
                    continue
                if not isinstance(frame, (bytes, bytearray)):
                    raise TypeError("Audio frame must be bytes")
                await self.send_audio_frame(
                    stream_id,
                    bytes(frame),
                    encoding=config.encoding,
                    sample_rate=config.sample_rate,
                    channels=config.channels,
                    sequence=sequence,
                )
                sequence += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Audio stream %s encountered an error", stream_id)
        finally:
            if hasattr(frame_source, "aclose"):
                with contextlib.suppress(Exception):
                    await frame_source.aclose()  # type: ignore[attr-defined]
            self._audio_streams.pop(stream_id, None)
            self._audio_stream_metadata.pop(stream_id, None)
            self._audio_sequences.pop(stream_id, None)

    async def close(self) -> None:
        self._stopped.set()

        for stream_id in list(self._audio_streams.keys()):
            await self.stop_audio_stream(stream_id)

        await self._stop_playback()

        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None

        if self._writer and not self._writer.is_closing():
            try:
                await self._send_json({"type": "close"})
            except Exception:
                pass
            self._writer.close()
        self._reader = None
        self._writer = None

    async def _send_json(self, message: Dict[str, Any]) -> None:
        writer = self._writer
        if writer is None:
            raise RuntimeError("Connection not established")
        data = json.dumps(message, separators=(",", ":")) + "\n"
        writer.write(data.encode(ENCODING))
        await writer.drain()

    @staticmethod
    def _decode_message(raw: bytes) -> Dict[str, Any]:
        text = raw.decode(ENCODING).strip()
        if not text:
            raise json.JSONDecodeError("empty message", text, 0)
        message = json.loads(text)
        if not isinstance(message, dict):
            raise json.JSONDecodeError("message not object", text, 0)
        return message

    async def _default_audio_factory(self, command: str, payload: Dict[str, Any]) -> AudioStreamConfig:
        stream_id = payload.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            stream_id = str(uuid.uuid4())
        sample_rate = int(payload.get("sample_rate", DEFAULT_SAMPLE_RATE))
        channels = int(payload.get("channels", DEFAULT_CHANNELS))
        frame_duration_ms = float(payload.get("frame_duration_ms", 20.0))

        frame_source = self._process_audio_source(
            self._audio_input_command,
            sample_rate=sample_rate,
            channels=channels,
            frame_duration_ms=frame_duration_ms,
        )

        return AudioStreamConfig(
            stream_id=stream_id,
            frame_source=frame_source,
            encoding=DEFAULT_ENCODING,
            sample_rate=sample_rate,
            channels=channels,
        )

    def _process_audio_source(
        self,
        command: str,
        *,
        sample_rate: int,
        channels: int,
        frame_duration_ms: float,
    ) -> AsyncIterator[bytes]:
        bytes_per_frame = max(
            1,
            int(sample_rate * frame_duration_ms / 1000.0) * channels * 2,
        )

        async def _generator() -> AsyncIterator[bytes]:
            logging.info("Starting audio input command: %s", command)
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            try:
                assert proc.stdout is not None
                while not self._stopped.is_set():
                    try:
                        chunk = await proc.stdout.readexactly(bytes_per_frame)
                    except asyncio.IncompleteReadError as exc:
                        if exc.partial:
                            yield exc.partial
                        break
                    if not chunk:
                        break
                    yield chunk
            finally:
                if proc.returncode is None:
                    proc.terminate()
                    with contextlib.suppress(ProcessLookupError):
                        await proc.wait()
                else:
                    await proc.wait()

        return _generator()

    async def _stop_playback(self) -> None:
        if self._playback is not None:
            await self._playback.stop()
            self._playback = None
            self._playback_stream_id = None


async def default_handler(command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if command == "ping":
        return {"pong": True}
    return {"handled": False}


async def run(args: argparse.Namespace) -> None:
    config = IntercomConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        token=args.token,
    )

    client = IntercomClient(
        config=config,
        handler=default_handler,
        audio_stream_factory=None,
        audio_frame_handler=None,
        audio_input_command=args.audio_input_command,
        enable_playback=args.playback,
    )

    await client.connect()
    await client.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intercom audio bridge client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--client-id", default="intercom", help="Client identifier")
    parser.add_argument("--token", help="Optional auth token")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--audio-input-command",
        required=True,
        help="Shell command producing raw PCM S16LE audio",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Play incoming audio using aplay",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        asyncio.run(run(args))
    except RegistrationError as exc:
        logging.error("Registration failed: %s", exc)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
