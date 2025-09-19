"""Minimal Home Assistant client for bi-directional PCM audio."""
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
import sys
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

ENCODING = "utf-8"
REGISTER_TIMEOUT_SECONDS = 10


class RegistrationError(RuntimeError):
    """Raised when registration with the host fails."""


@dataclass
class HomeAssistantConfig:
    host: str
    port: int
    client_id: str
    token: Optional[str] = None


class AudioPlayback:
    """Pipe PCM frames to `aplay`."""

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


async def pcm_capture(
    command: str,
    *,
    sample_rate: int,
    channels: int,
    frame_duration_ms: float,
) -> AsyncIterator[bytes]:
    if not command:
        raise RuntimeError("talkback command must be provided")

    bytes_per_frame = max(1, int(sample_rate * frame_duration_ms / 1000.0) * channels * 2)
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    try:
        assert proc.stdout is not None
        while True:
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


class MockHomeAssistantClient:
    def __init__(
        self,
        config: HomeAssistantConfig,
        *,
        playback: bool = False,
        talkback_command: Optional[str] = None,
        talkback_frame_ms: float = 20.0,
    ) -> None:
        self._config = config
        self._playback_enabled = playback
        self._talkback_command = talkback_command
        self._talkback_frame_ms = talkback_frame_ms

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._pending: Dict[str, asyncio.Future[Dict[str, Any]]] = {}
        self._stopped = asyncio.Event()

        self._playback: Optional[AudioPlayback] = None
        self._playback_stream_id: Optional[str] = None

        self._talkback_task: Optional[asyncio.Task[None]] = None
        self._talkback_stream_id: Optional[str] = None

    async def connect(self) -> None:
        reader, writer = await asyncio.open_connection(self._config.host, self._config.port)
        self._reader = reader
        self._writer = writer

        register = {
            "type": "register",
            "role": "home_assistant",
            "client_id": self._config.client_id,
        }
        if self._config.token:
            register["token"] = self._config.token

        await self._send_json(register)

        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=REGISTER_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise RegistrationError("Timed out waiting for registered message") from exc

        if not raw:
            raise RegistrationError("Server closed connection during registration")

        response = self._decode_message(raw)
        if response.get("type") != "registered" or response.get("status") != "ok":
            raise RegistrationError(f"Unexpected registration response: {response}")

        logging.info("Registered with host as '%s'", self._config.client_id)
        self._reader_task = asyncio.create_task(self._reader_loop(), name="ha-reader")

    async def send_command(
        self,
        command: str,
        payload: Optional[Dict[str, Any]] = None,
        command_id: Optional[str] = None,
    ) -> str:
        if self._writer is None:
            raise RuntimeError("Client is not connected")

        payload = payload or {}
        command_id = command_id or str(uuid.uuid4())
        message = {
            "type": "command",
            "command": command,
            "payload": payload,
            "command_id": command_id,
        }
        future: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[command_id] = future
        await self._send_json(message)
        logging.info("Sent command %s (id=%s)", command, command_id)
        return command_id

    async def wait_for_response(self, command_id: str, timeout: float) -> Dict[str, Any]:
        future = self._pending.get(command_id)
        if future is None:
            raise RuntimeError(f"Unknown command id {command_id}")
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(command_id, None)

    async def listen(self, duration: Optional[float] = None) -> None:
        if duration is None:
            await self._stopped.wait()
        else:
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=duration)
            except asyncio.TimeoutError:
                pass

    async def close(self) -> None:
        self._stopped.set()
        await self.stop_talkback()
        await self.stop_playback()
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._writer and not self._writer.is_closing():
            try:
                await self._send_json({"type": "close"})
            except Exception:
                pass
            self._writer.close()
        self._reader = None
        self._writer = None

    async def start_playback(self, *, stream_id: str, sample_rate: int, channels: int) -> None:
        if not self._playback_enabled:
            return
        if self._playback is None:
            try:
                self._playback = AudioPlayback(sample_rate, channels)
                await self._playback.start()
            except Exception as exc:
                logging.warning("Playback disabled: %s", exc)
                self._playback_enabled = False
                return
        self._playback_stream_id = stream_id

    async def stop_playback(self, stream_id: Optional[str] = None) -> None:
        if stream_id is not None and stream_id != self._playback_stream_id:
            return
        if self._playback is not None:
            await self._playback.stop()
        self._playback = None
        self._playback_stream_id = None

    async def start_talkback_stream(
        self,
        stream_id: str,
        *,
        sample_rate: int,
        channels: int,
    ) -> None:
        if not self._talkback_command or self._talkback_task is not None:
            return

        async def _loop() -> None:
            try:
                async for chunk in pcm_capture(
                    self._talkback_command,
                    sample_rate=sample_rate,
                    channels=channels,
                    frame_duration_ms=self._talkback_frame_ms,
                ):
                    await self.send_audio_frame(
                        stream_id,
                        chunk,
                        sample_rate=sample_rate,
                        channels=channels,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Talkback loop failed")
            finally:
                self._talkback_stream_id = None

        self._talkback_stream_id = stream_id
        self._talkback_task = asyncio.create_task(_loop(), name=f"ha-talkback-{stream_id}")

    async def stop_talkback(self) -> None:
        if self._talkback_task is not None:
            self._talkback_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._talkback_task
            self._talkback_task = None
        self._talkback_stream_id = None

    async def stop_talkback_stream(self, stream_id: str) -> None:
        if self._talkback_stream_id != stream_id:
            return
        await self.stop_talkback()

    async def send_audio_frame(
        self,
        stream_id: str,
        data: bytes,
        *,
        sample_rate: int,
        channels: int,
        sequence: Optional[int] = None,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("Client is not connected")
        payload = {
            "type": "audio_frame",
            "stream_id": stream_id,
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
            "channels": channels,
            "data": base64.b64encode(data).decode("ascii"),
        }
        if sequence is not None:
            payload["sequence"] = sequence
        await self._send_json(payload)

    async def _reader_loop(self) -> None:
        assert self._reader is not None
        reader = self._reader
        while not self._stopped.is_set():
            raw = await reader.readline()
            if not raw:
                logging.info("Host closed the connection")
                self._stopped.set()
                break
            try:
                message = self._decode_message(raw)
            except json.JSONDecodeError as exc:
                logging.warning("Discarding invalid JSON from host: %s", exc)
                continue

            mtype = message.get("type")
            if mtype == "command_ack":
                logging.debug("Command ack: %s", message)
            elif mtype == "response":
                await self._handle_response(message)
            elif mtype == "event":
                logging.info("Event: %s", message)
            elif mtype == "audio_frame":
                await self._handle_audio_frame(message)
            elif mtype == "error":
                logging.error("Error from host: %s", message)
                details = message.get("details", {})
                stream_id = details.get("stream_id") if isinstance(details, dict) else None
                if stream_id:
                    await self.stop_talkback_stream(stream_id)
                    await self.stop_playback(stream_id)
            else:
                logging.warning("Unhandled message: %s", message)

    async def _handle_response(self, message: Dict[str, Any]) -> None:
        command_id = message.get("command_id")
        if not isinstance(command_id, str):
            return
        future = self._pending.get(command_id)
        if future and not future.done():
            future.set_result(message)

        payload = message.get("payload")
        if not isinstance(payload, dict):
            return

        if payload.get("stream_id") and payload.get("status", "ok") == "ok":
            await self._maybe_start_streams(payload)

    async def _maybe_start_streams(self, payload: Dict[str, Any]) -> None:
        stream_id = payload.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            return
        sample_rate = int(payload.get("sample_rate", 16000))
        channels = int(payload.get("channels", 1))

        await self.start_playback(stream_id=stream_id, sample_rate=sample_rate, channels=channels)
        await self.start_talkback_stream(stream_id, sample_rate=sample_rate, channels=channels)

    async def _handle_audio_frame(self, message: Dict[str, Any]) -> None:
        stream_id = message.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            logging.warning("Ignoring audio frame without stream_id: %s", message)
            return

        sample_rate = int(message.get("sample_rate", 16000))
        channels = int(message.get("channels", 1))
        if self._playback_stream_id is None:
            await self.start_playback(stream_id=stream_id, sample_rate=sample_rate, channels=channels)

        if self._playback and self._playback_stream_id == stream_id:
            try:
                pcm = base64.b64decode(message.get("data", ""), validate=True)
            except binascii.Error as exc:
                logging.warning("Failed to decode audio frame for %s: %s", stream_id, exc)
                return
            await self._playback.enqueue(pcm)

    async def _send_json(self, message: Dict[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("Connection not established")
        data = json.dumps(message, separators=(",", ":")) + "\n"
        self._writer.write(data.encode(ENCODING))
        await self._writer.drain()

    @staticmethod
    def _decode_message(raw: bytes) -> Dict[str, Any]:
        text = raw.decode(ENCODING).strip()
        if not text:
            raise json.JSONDecodeError("empty message", text, 0)
        message = json.loads(text)
        if not isinstance(message, dict):
            raise json.JSONDecodeError("message not object", text, 0)
        return message


async def run_client(args: argparse.Namespace) -> None:
    config = HomeAssistantConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        token=args.token,
    )
    client = MockHomeAssistantClient(
        config,
        playback=args.playback,
        talkback_command=args.talkback_input_command,
        talkback_frame_ms=args.talkback_frame_ms,
    )
    await client.connect()

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logging.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    command_id: Optional[str] = None
    if args.command:
        payload: Dict[str, Any] = {}
        if args.payload:
            try:
                payload = json.loads(args.payload)
            except json.JSONDecodeError as exc:
                logging.error("Failed to parse payload JSON: %s", exc)
                payload = {}
        command_id = await client.send_command(args.command, payload)

    async def _waiters() -> None:
        if command_id and args.wait_for_response:
            try:
                response = await client.wait_for_response(command_id, timeout=args.response_timeout)
                logging.info("Received response: %s", response)
                payload = response.get("payload", {})
                if isinstance(payload, dict):
                    await client._maybe_start_streams(payload)
            except asyncio.TimeoutError:
                logging.error("Timed out waiting for response %s", command_id)
            except Exception as exc:
                logging.error("Error while waiting for response: %s", exc)
        if args.listen_duration is not None:
            await client.listen(duration=args.listen_duration)
        else:
            await stop_event.wait()

    try:
        await _waiters()
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Home Assistant audio test client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--client-id", default="ha-client", help="Client identifier")
    parser.add_argument("--token", help="Optional auth token")
    parser.add_argument("--command", help="Command name to send once connected")
    parser.add_argument("--payload", help="JSON payload for the command")
    parser.add_argument("--wait-for-response", action="store_true", help="Block until the command response arrives")
    parser.add_argument("--response-timeout", type=float, default=10.0, help="Seconds to wait for the command response")
    parser.add_argument("--listen-duration", type=float, help="Keep connection open for this many seconds")
    parser.add_argument("--playback", action="store_true", help="Play incoming audio via aplay")
    parser.add_argument("--talkback-input-command", help="Shell command producing raw PCM talkback audio")
    parser.add_argument("--talkback-frame-ms", type=float, default=20.0, help="Talkback frame duration in milliseconds")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        asyncio.run(run_client(args))
    except RegistrationError as exc:
        logging.error("Registration failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
