from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from audio_utils import (
    AudioFormat,
    AudioSink,
    AudioSource,
    FileAudioSource,
    HighPassFilter,
    MicrophoneAudioSource,
    NotchFilter,
    NullAudioSink,
    SilenceAudioSource,
    SimpleAudioSink,
    ToneAudioSource,
    WaveFileSink,
    decode_audio_payload,
    encode_audio_payload,
)
from protocol_client import ConnectionClosed, JsonConnection, open_connection, register_client

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_FRAME_MS = 40


@dataclass(slots=True)
class PendingCommand:
    command: str
    payload: Dict


@dataclass(slots=True)
class HomeAudioStream:
    stream_id: str
    format: AudioFormat
    source: AudioSource
    sink: AudioSink
    connection: JsonConnection
    sequence: int = 0
    task: Optional[asyncio.Task[None]] = field(default=None, repr=False)


class HomeAssistantClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        client_id: str,
        preferred_format: AudioFormat,
        source_factory: Callable[[AudioFormat], AudioSource],
        sink_factory: Callable[[AudioFormat], AudioSink],
        auto_start: bool,
        requested_stream_id: Optional[str],
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.preferred_format = preferred_format
        self._source_factory = source_factory
        self._sink_factory = sink_factory
        self._auto_start = auto_start
        self._requested_stream_id = requested_stream_id
        self._connection: Optional[JsonConnection] = None
        self._pending: Dict[str, PendingCommand] = {}
        self._streams: Dict[str, HomeAudioStream] = {}

    async def run(self) -> None:
        try:
            connection = await open_connection(self.host, self.port)
            await register_client(connection, role="home_assistant", client_id=self.client_id)
            self._connection = connection
            logging.info("Registered Home Assistant client '%s'", self.client_id)
            if self._auto_start:
                await self.request_start_audio(stream_id=self._requested_stream_id)
            await self._listen_loop(connection)
        except ConnectionClosed as exc:
            logging.error("Connection closed: %s", exc)
        finally:
            await self._shutdown()

    async def request_start_audio(self, *, stream_id: Optional[str] = None) -> Optional[str]:
        connection = self._connection
        if connection is None:
            logging.error("Cannot start audio without an active connection")
            return None
        payload: Dict = {
            "encoding": self.preferred_format.encoding,
            "sample_rate": self.preferred_format.sample_rate,
            "channels": self.preferred_format.channels,
        }
        if stream_id:
            payload["stream_id"] = stream_id
        command_id = str(uuid.uuid4())
        await self._send_command(command_id, "start_audio", payload)
        logging.info("Sent start_audio command %s", command_id)
        return command_id

    async def request_stop_audio(self, stream_id: str) -> Optional[str]:
        connection = self._connection
        if connection is None:
            return None
        command_id = str(uuid.uuid4())
        await self._send_command(command_id, "stop_audio", {"stream_id": stream_id})
        logging.info("Sent stop_audio command %s for stream %s", command_id, stream_id)
        return command_id

    async def _send_command(self, command_id: str, command: str, payload: Dict) -> None:
        connection = self._connection
        if connection is None:
            raise ConnectionClosed("connection not established")
        await connection.send(
            {
                "type": "command",
                "command": command,
                "command_id": command_id,
                "payload": payload,
            }
        )
        self._pending[command_id] = PendingCommand(command=command, payload=payload)

    async def _listen_loop(self, connection: JsonConnection) -> None:
        while True:
            try:
                message = await connection.receive()
            except ConnectionClosed:
                break
            message_type = message.get("type")
            if message_type == "command_ack":
                await self._handle_command_ack(message)
            elif message_type == "response":
                await self._handle_response(message)
            elif message_type == "event":
                await self._handle_event(message)
            elif message_type == "audio_frame":
                await self._handle_audio_frame(message)
            elif message_type == "error":
                await self._handle_error(message)
            elif message_type == "close":
                logging.info("Received close control message from host")
                break
            else:
                logging.debug("Unhandled message type '%s'", message_type)

    async def _handle_command_ack(self, message: Dict) -> None:
        command_id = message.get("command_id")
        if isinstance(command_id, str):
            logging.debug("Command %s acknowledged", command_id)

    async def _handle_response(self, message: Dict) -> None:
        command_id = message.get("command_id")
        if not isinstance(command_id, str):
            return
        pending = self._pending.pop(command_id, None)
        if pending is None:
            logging.debug("Ignoring response for unknown command %s", command_id)
            return
        status = message.get("status")
        payload = message.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        if pending.command == "start_audio":
            if status == "ok":
                await self._start_stream_from_response(payload)
            else:
                logging.error("start_audio rejected: %s", payload)
        elif pending.command == "stop_audio":
            stream_id = payload.get("stream_id")
            if status == "ok" and isinstance(stream_id, str):
                await self._terminate_stream(stream_id)
                logging.info("Stream %s stopped", stream_id)
            else:
                logging.error("stop_audio failed: %s", payload)

    async def _start_stream_from_response(self, payload: Dict) -> None:
        stream_id = payload.get("stream_id")
        encoding = payload.get("encoding")
        sample_rate = payload.get("sample_rate")
        channels = payload.get("channels")
        if not isinstance(stream_id, str):
            logging.error("Missing stream_id in start_audio response")
            return
        if encoding != "pcm_s16le":
            logging.error("Unsupported encoding %s", encoding)
            return
        if not isinstance(sample_rate, int) or not isinstance(channels, int):
            logging.error("Invalid audio parameters in response: %s", payload)
            return
        if stream_id in self._streams:
            logging.warning("Audio stream %s already active", stream_id)
            return

        audio_format = AudioFormat(
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
            frame_ms=self.preferred_format.frame_ms,
        )
        source = self._source_factory(audio_format)
        sink = self._sink_factory(audio_format)
        try:
            await source.start()
            await sink.start()
        except Exception as exc:
            logging.error("Failed to start local audio IO: %s", exc)
            await source.stop()
            await sink.stop()
            await self.request_stop_audio(stream_id)
            return

        connection = self._connection
        if connection is None:
            logging.error("Connection lost before stream start")
            await source.stop()
            await sink.stop()
            return

        stream = HomeAudioStream(
            stream_id=stream_id,
            format=audio_format,
            source=source,
            sink=sink,
            connection=connection,
        )
        task = asyncio.create_task(self._source_loop(stream))
        stream.task = task
        self._streams[stream_id] = stream
        logging.info("Audio stream %s established", stream_id)

    async def _source_loop(self, stream: HomeAudioStream) -> None:
        frame_interval = stream.format.frame_ms / 1000.0
        notch = NotchFilter(stream.format, freq=50)        # for hum
        highpass = HighPassFilter(stream.format, cutoff=100)  # for clunks/pops
        try:
            while True:
                frame = await stream.source.read_frame()
                frame = notch.process(frame)
                frame = highpass.process(frame)
                await stream.connection.send(
                    {
                        "type": "audio_frame",
                        "stream_id": stream.stream_id,
                        "sequence": stream.sequence,
                        "encoding": stream.format.encoding,
                        "sample_rate": stream.format.sample_rate,
                        "channels": stream.format.channels,
                        "direction": "client_to_intercom",
                        "data": encode_audio_payload(frame),
                    }
                )
                stream.sequence += 1
                await asyncio.sleep(frame_interval)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            logging.info("Connection closed while streaming audio for %s", stream.stream_id)
        except Exception as exc:
            logging.exception("Failed to stream audio for %s", stream.stream_id)
            await self._send_stream_error(stream.stream_id, str(exc))
        finally:
            await stream.source.stop()
            await stream.sink.stop()
            if self._streams.get(stream.stream_id) is stream:
                self._streams.pop(stream.stream_id, None)
            logging.info("Stopped audio stream %s", stream.stream_id)

    async def _handle_audio_frame(self, message: Dict) -> None:
        stream_id = message.get("stream_id")
        if not isinstance(stream_id, str):
            return
        stream = self._streams.get(stream_id)
        if stream is None:
            logging.debug("Dropping audio frame for inactive stream %s", stream_id)
            return
        direction = message.get("direction")
        if direction not in {"intercom_to_client", "client_to_intercom"}:
            return
        if direction != "intercom_to_client":
            return
        data_field = message.get("data")
        if not isinstance(data_field, str):
            return
        try:
            frame = decode_audio_payload(data_field)
        except Exception as exc:
            logging.warning("Failed to decode audio payload: %s", exc)
            return
        await stream.sink.play(frame)

    async def _handle_event(self, message: Dict) -> None:
        logging.info("Received event: %s", message.get("event"))

    async def _handle_error(self, message: Dict) -> None:
        details = message.get("details", {})
        if not isinstance(details, dict):
            details = {}
        stream_id = details.get("stream_id")
        if isinstance(stream_id, str):
            logging.warning("Host reported stream error for %s: %s", stream_id, message.get("reason"))
            await self._terminate_stream(stream_id)

    async def _send_stream_error(self, stream_id: str, details: str) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            await connection.send(
                {
                    "type": "error",
                    "reason": "audio_stream_failed",
                    "details": {"stream_id": stream_id, "details": details},
                }
            )
        except ConnectionClosed:
            logging.debug("Unable to report stream error for %s", stream_id)

    async def _terminate_stream(self, stream_id: str) -> None:
        stream = self._streams.get(stream_id)
        if stream is None:
            return
        task = stream.task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        else:
            await stream.source.stop()
            await stream.sink.stop()
        self._streams.pop(stream_id, None)

    async def _shutdown(self) -> None:
        tasks = [self._terminate_stream(stream_id) for stream_id in list(self._streams.keys())]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._connection is not None:
            self._connection.close()


def _build_source_factory(args: argparse.Namespace) -> Callable[[AudioFormat], AudioSource]:
    if args.source_file:
        path = args.source_file
        return lambda fmt: FileAudioSource(fmt, path, loop=True)
    if args.mic:
        device = args.mic_device
        return lambda fmt: MicrophoneAudioSource(fmt, device=device)
    if args.tone:
        return lambda fmt: ToneAudioSource(fmt, frequency=args.tone_frequency, amplitude=args.tone_amplitude)
    return lambda fmt: SilenceAudioSource(fmt)


def _build_sink_factory(args: argparse.Namespace) -> Callable[[AudioFormat], AudioSink]:
    if args.sink_file:
        path = args.sink_file
        return lambda fmt: WaveFileSink(fmt, path)
    if args.mute:
        return lambda fmt: NullAudioSink(fmt)
    device = args.speaker_device or None
    return lambda fmt: SimpleAudioSink(fmt, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Home Assistant client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--client-id", default="ha-client-1", help="Client identifier")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Preferred sample rate")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="Preferred channel count")
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS, help="Preferred frame duration in ms")
    parser.add_argument("--source-file", help="Path to WAV file used for outbound audio")
    parser.add_argument("--mic", action="store_true", help="Capture outbound audio from the default microphone")
    parser.add_argument("--mic-device", help="Optional sounddevice input identifier")
    parser.add_argument("--sink-file", help="Write inbound audio to a WAV file")
    parser.add_argument("--speaker-device", help="Optional sounddevice output identifier")
    parser.add_argument("--tone", action="store_true", help="Use tone generator as outbound audio source")
    parser.add_argument("--tone-frequency", type=float, default=440.0, help="Tone generator frequency")
    parser.add_argument("--tone-amplitude", type=float, default=0.2, help="Tone amplitude (0-1)")
    parser.add_argument("--mute", action="store_true", help="Disable playback of inbound audio")
    parser.add_argument("--no-auto-start", action="store_true", help="Skip automatic start_audio command")
    parser.add_argument("--stream-id", help="Explicit stream ID to request")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    preferred_format = AudioFormat(
        sample_rate=args.sample_rate,
        channels=args.channels,
        frame_ms=args.frame_ms,
    )
    client = HomeAssistantClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        preferred_format=preferred_format,
        source_factory=_build_source_factory(args),
        sink_factory=_build_sink_factory(args),
        auto_start=not args.no_auto_start,
        requested_stream_id=args.stream_id,
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
