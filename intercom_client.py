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
DEFAULT_FRAME_MS = 320


@dataclass(slots=True)
class IntercomAudioStream:
    stream_id: str
    home_client_id: str
    format: AudioFormat
    source: AudioSource
    sink: AudioSink
    connection: JsonConnection
    sequence: int = 0
    task: Optional[asyncio.Task[None]] = field(default=None, repr=False)


class IntercomClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        client_id: str,
        audio_format: AudioFormat,
        source_factory: Callable[[AudioFormat], AudioSource],
        sink_factory: Callable[[AudioFormat], AudioSink],
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.format = audio_format
        self._source_factory = source_factory
        self._sink_factory = sink_factory
        self._connection: Optional[JsonConnection] = None
        self._streams: Dict[str, IntercomAudioStream] = {}
        self._frame_interval = self.format.frame_ms / 1000.0

    async def run(self) -> None:
        try:
            connection = await open_connection(self.host, self.port)
            await register_client(connection, role="intercom", client_id=self.client_id)
            self._connection = connection
            logging.info("Registered intercom '%s'", self.client_id)
            await self._listen_loop(connection)
        except ConnectionClosed as exc:
            logging.error("Connection closed: %s", exc)
        finally:
            await self._shutdown()

    async def _listen_loop(self, connection: JsonConnection) -> None:
        while True:
            try:
                message = await connection.receive()
            except ConnectionClosed:
                break
            message_type = message.get("type")
            if message_type == "command":
                await self._handle_command(message)
            elif message_type == "audio_frame":
                await self._handle_audio_frame(message)
            elif message_type == "error":
                await self._handle_error(message)
            elif message_type == "close":
                logging.info("Received close control message from host")
                break
            else:
                logging.debug("Unhandled message type '%s'", message_type)

    async def _handle_command(self, message: Dict) -> None:
        command = message.get("command")
        command_id = message.get("command_id")
        if not isinstance(command, str) or not isinstance(command_id, str):
            logging.warning("Malformed command: %s", message)
            return

        payload = message.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        try:
            if command == "start_audio":
                await self._cmd_start_audio(command_id, message.get("origin_id"), payload)
            elif command == "stop_audio":
                await self._cmd_stop_audio(command_id, payload)
            else:
                await self._send_response(command_id, status="error", payload={"reason": "unknown_command"})
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.exception("Failed to execute command '%s'", command)
            await self._send_response(
                command_id,
                status="error",
                payload={"reason": "exception", "details": str(exc)},
            )

    async def _cmd_start_audio(self, command_id: str, origin_id: Optional[str], payload: Dict) -> None:
        if not isinstance(origin_id, str):
            await self._send_response(command_id, status="error", payload={"reason": "missing_origin"})
            return

        requested_stream_id = payload.get("stream_id")
        stream_id = requested_stream_id if isinstance(requested_stream_id, str) and requested_stream_id else str(uuid.uuid4())
        if stream_id in self._streams:
            await self._send_response(command_id, status="error", payload={"reason": "stream_exists"})
            return

        requested_encoding = payload.get("encoding")
        if requested_encoding is not None and requested_encoding != self.format.encoding:
            await self._send_response(command_id, status="error", payload={"reason": "unsupported_encoding"})
            return

        requested_sample_rate = payload.get("sample_rate")
        if requested_sample_rate is not None and requested_sample_rate != self.format.sample_rate:
            await self._send_response(command_id, status="error", payload={"reason": "unsupported_sample_rate"})
            return

        requested_channels = payload.get("channels")
        if requested_channels is not None and requested_channels != self.format.channels:
            await self._send_response(command_id, status="error", payload={"reason": "unsupported_channels"})
            return

        source = self._source_factory(self.format)
        sink = self._sink_factory(self.format)
        try:
            await source.start()
            await sink.start()
        except Exception as exc:
            await self._send_response(
                command_id,
                status="error",
                payload={"reason": "audio_init_failed", "details": str(exc)},
            )
            await source.stop()
            await sink.stop()
            return

        connection = self._connection
        if connection is None:
            await self._send_response(command_id, status="error", payload={"reason": "connection_lost"})
            await source.stop()
            await sink.stop()
            return

        stream = IntercomAudioStream(
            stream_id=stream_id,
            home_client_id=origin_id,
            format=self.format,
            source=source,
            sink=sink,
            connection=connection
        )
        task = asyncio.create_task(self._source_loop(stream))
        stream.task = task
        self._streams[stream_id] = stream
        logging.info("Started audio stream %s for client %s", stream_id, origin_id)

        response_payload = {
            "stream_id": stream_id,
            "encoding": self.format.encoding,
            "sample_rate": self.format.sample_rate,
            "channels": self.format.channels,
        }
        await self._send_response(command_id, status="ok", payload=response_payload)

    async def _cmd_stop_audio(self, command_id: str, payload: Dict) -> None:
        stream_id = payload.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            await self._send_response(command_id, status="error", payload={"reason": "invalid_stream_id"})
            return
        stream = self._streams.get(stream_id)
        if stream is None:
            await self._send_response(command_id, status="error", payload={"reason": "unknown_stream_id"})
            return

        await self._terminate_stream(stream_id)
        await self._send_response(
            command_id,
            status="ok",
            payload={"stream_id": stream_id},
        )

    async def _send_response(self, command_id: str, *, status: str, payload: Dict) -> None:
        connection = self._connection
        if connection is None:
            return
        await connection.send(
            {
                "type": "response",
                "command_id": command_id,
                "status": status,
                "payload": payload,
            }
        )

    async def _handle_audio_frame(self, message: Dict) -> None:
        stream_id = message.get("stream_id")
        if not isinstance(stream_id, str):
            return
        stream = self._streams.get(stream_id)
        if stream is None:
            logging.debug("Ignoring audio frame for unknown stream %s", stream_id)
            return
        direction = message.get("direction")
        if direction not in {"client_to_intercom", "intercom_to_client"}:
            logging.debug("Ignoring audio frame with direction %s", direction)
            return
        if direction != "client_to_intercom":
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

    async def _handle_error(self, message: Dict) -> None:
        details = message.get("details", {})
        if not isinstance(details, dict):
            details = {}
        stream_id = details.get("stream_id")
        if isinstance(stream_id, str) and stream_id in self._streams:
            logging.warning("Host reported error for stream %s: %s", stream_id, message.get("reason"))
            await self._terminate_stream(stream_id)

    async def _source_loop(self, stream: IntercomAudioStream) -> None:
        global filters
        notch = NotchFilter(stream.format, freq=filters.get("notch", 50))        # for hum
        highpass = HighPassFilter(stream.format, cutoff=filters.get("highpass", 100))  # for clunks/pops
        try:
            while True:
                frame = await stream.source.read_frame()
                if "notch" in filters.keys():
                    frame = notch.process(frame)
                if "highpass" in filters.keys():
                    frame = highpass.process(frame)
                await stream.connection.send(
                    {
                        "type": "audio_frame",
                        "stream_id": stream.stream_id,
                        "sequence": stream.sequence,
                        "encoding": stream.format.encoding,
                        "sample_rate": stream.format.sample_rate,
                        "channels": stream.format.channels,
                        "direction": "intercom_to_client",
                        "data": encode_audio_payload(frame),
                    }
                )
                stream.sequence += 1
                await asyncio.sleep(self._frame_interval)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            logging.info("Connection closed while streaming audio for %s", stream.stream_id)
        except Exception as exc:
            logging.exception("Audio source loop failed for stream %s", stream.stream_id)
            await self._send_stream_error(stream.stream_id, str(exc))
        finally:
            await stream.source.stop()
            await stream.sink.stop()
            if self._streams.get(stream.stream_id) is stream:
                self._streams.pop(stream.stream_id, None)
            logging.info("Stopped audio stream %s", stream.stream_id)

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
            logging.debug("Unable to report stream error for %s due to closed connection", stream_id)

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
        if self._connection is not None:
            self._connection.close()
        tasks = [self._terminate_stream(stream_id) for stream_id in list(self._streams.keys())]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _build_source_factory(args: argparse.Namespace) -> Callable[[AudioFormat], AudioSource]:
    if args.source_file:
        path = args.source_file
        return lambda fmt: FileAudioSource(fmt, path, loop=True)
    if args.mic:
        device = args.mic_device
        return lambda fmt: MicrophoneAudioSource(fmt, device=device)
    if args.silence:
        return lambda fmt: SilenceAudioSource(fmt)
    frequency = args.tone_frequency
    amplitude = args.tone_amplitude
    return lambda fmt: ToneAudioSource(fmt, frequency=frequency, amplitude=amplitude)


def _build_sink_factory(args: argparse.Namespace) -> Callable[[AudioFormat], AudioSink]:
    if args.sink_file:
        path = args.sink_file
        return lambda fmt: WaveFileSink(fmt, path)
    if args.mute:
        return lambda fmt: NullAudioSink(fmt)
    return lambda fmt: SimpleAudioSink(fmt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedded intercom client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--client-id", default="intercom-1", help="Client identifier")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Audio sample rate")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="Audio channel count")
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS, help="Audio frame duration in ms")
    parser.add_argument("--source-file", help="Path to a WAV file used as intercom outbound audio")
    parser.add_argument("--mic", action="store_true", help="Capture intercom audio from the default microphone")
    parser.add_argument("--mic-device", help="Optional sounddevice input identifier")
    parser.add_argument("--sink-file", help="Write inbound audio to WAV file")
    parser.add_argument("--tone-frequency", type=float, default=440.0, help="Tone generator frequency")
    parser.add_argument("--tone-amplitude", type=float, default=0.2, help="Tone amplitude (0-1)")
    parser.add_argument("--silence", action="store_true", help="Use silence as outbound audio source")
    parser.add_argument("--mute", action="store_true", help="Discard inbound audio instead of playing it")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--notch", type=float, default=50.0, help="Notch filter frequency (Hz)")
    parser.add_argument("--highpass", type=float, default=100.0, help="High-pass filter cutoff (Hz)")
    parser.add_argument("--enable-notch", action="store_true", help="Enable notch filter")
    parser.add_argument("--enable-highpass", action="store_true", help="Enable high-pass filter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    audio_format = AudioFormat(
        sample_rate=args.sample_rate,
        channels=args.channels,
        frame_ms=args.frame_ms,
    )
    global filters
    filters = {}
    if args.enable_notch:
        filters["notch"] = args.notch
    if args.enable_highpass:
        filters["highpass"] = args.highpass
    client = IntercomClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        audio_format=audio_format,
        source_factory=_build_source_factory(args),
        sink_factory=_build_sink_factory(args),
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
