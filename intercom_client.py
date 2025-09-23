from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol

from audio_utils import (
    AudioFormat,
    EchoCanceller,
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
    LiveKitAudioProcessor,
    LiveKitAPMConfig,
    WebRTCAudioProcessor,
    WebRTCProcessorConfig,
    decode_audio_payload,
    encode_audio_payload,
)
from protocol_client import ConnectionClosed, JsonConnection, open_connection, register_client

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_FRAME_MS = 320


@dataclass(slots=True)
class AudioProcessingConfig:
    enable_notch: bool = False
    notch_frequency: float = 50.0
    enable_highpass: bool = False
    highpass_cutoff: float = 100.0
    enable_echo_cancel: bool = False
    echo_filter_length: int = 1024
    echo_adaptation: float = 0.05
    echo_leakage: float = 0.999
    echo_min_power: float = 1e-3
    use_livekit_apm: bool = False
    livekit_config: LiveKitAPMConfig = field(default_factory=LiveKitAPMConfig)
    use_webrtc: bool = False
    webrtc_config: WebRTCProcessorConfig = field(default_factory=WebRTCProcessorConfig)


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
    echo_canceller: Optional[EchoCanceller] = field(default=None, repr=False)
    processor: Optional["AudioProcessor"] = field(default=None, repr=False)


class AudioProcessor(Protocol):
    def process_capture_frame(self, frame: bytes) -> bytes:
        ...

    def process_render_frame(self, frame: bytes) -> None:
        ...


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
        processing_config: Optional[AudioProcessingConfig] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.format = audio_format
        self._source_factory = source_factory
        self._sink_factory = sink_factory
        self._processing = processing_config or AudioProcessingConfig()
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

        if self._processing.use_livekit_apm and self._processing.use_webrtc:
            logging.debug(
                "Both LiveKit APM and WebRTC processing requested; LiveKit will be tried first",
            )

        audio_processor: Optional[AudioProcessor] = None
        if self._processing.use_livekit_apm:
            try:
                audio_processor = LiveKitAudioProcessor(self.format, self._processing.livekit_config)
                logging.debug("LiveKit APM enabled for stream %s", stream_id)
            except Exception as exc:
                logging.warning("LiveKit APM disabled for stream %s: %s", stream_id, exc)

        if audio_processor is None and self._processing.use_webrtc:
            try:
                audio_processor = WebRTCAudioProcessor(self.format, self._processing.webrtc_config)
                logging.debug("WebRTC audio processing enabled for stream %s", stream_id)
            except Exception as exc:
                logging.warning("WebRTC audio processing disabled for stream %s: %s", stream_id, exc)

        echo_canceller: Optional[EchoCanceller] = None
        if self._processing.enable_echo_cancel:
            try:
                echo_canceller = EchoCanceller(
                    self.format,
                    filter_length=self._processing.echo_filter_length,
                    adaptation_rate=self._processing.echo_adaptation,
                    leakage=self._processing.echo_leakage,
                    min_power=self._processing.echo_min_power,
                )
                logging.debug(
                    "Echo cancellation enabled for stream %s with filter_length=%d",
                    stream_id,
                    self._processing.echo_filter_length,
                )
            except Exception as exc:
                logging.warning("Echo cancellation disabled for stream %s: %s", stream_id, exc)

        if audio_processor is not None and echo_canceller is not None:
            logging.info(
                "Disabling standalone echo canceller for stream %s because APM echo cancellation is active",
                stream_id,
            )
            echo_canceller = None

        stream = IntercomAudioStream(
            stream_id=stream_id,
            home_client_id=origin_id,
            format=self.format,
            source=source,
            sink=sink,
            connection=connection,
            echo_canceller=echo_canceller,
            processor=audio_processor,
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
        if stream.processor is not None:
            stream.processor.process_render_frame(frame)
        if stream.echo_canceller is not None:
            stream.echo_canceller.update_far(frame)
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
        config = self._processing
        notch = NotchFilter(stream.format, freq=config.notch_frequency) if config.enable_notch else None
        highpass = HighPassFilter(stream.format, cutoff=config.highpass_cutoff) if config.enable_highpass else None
        echo = stream.echo_canceller
        processor = stream.processor
        try:
            while True:
                frame = await stream.source.read_frame()
                if processor is not None:
                    frame = processor.process_capture_frame(frame)
                if echo is not None:
                    frame = echo.cancel(frame)
                if highpass is not None:
                    frame = highpass.process(frame)
                if notch is not None:
                    frame = notch.process(frame)
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
    device = args.speaker_device or None
    return lambda fmt: SimpleAudioSink(fmt, device=device)


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
    parser.add_argument("--speaker-device", help="Optional sounddevice output identifier")
    parser.add_argument("--tone-frequency", type=float, default=440.0, help="Tone generator frequency")
    parser.add_argument("--tone-amplitude", type=float, default=0.2, help="Tone amplitude (0-1)")
    parser.add_argument("--silence", action="store_true", help="Use silence as outbound audio source")
    parser.add_argument("--mute", action="store_true", help="Discard inbound audio instead of playing it")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--notch", type=float, default=50.0, help="Notch filter frequency (Hz)")
    parser.add_argument("--highpass", type=float, default=100.0, help="High-pass filter cutoff (Hz)")
    parser.add_argument("--enable-notch", action="store_true", help="Enable notch filter")
    parser.add_argument("--enable-highpass", action="store_true", help="Enable high-pass filter")
    parser.add_argument("--enable-echo-cancel", action="store_true", help="Enable adaptive echo cancellation")
    parser.add_argument("--echo-filter-length", type=int, default=1024, help="Echo canceller filter length in samples")
    parser.add_argument("--echo-adaptation", type=float, default=0.05, help="Echo canceller adaptation rate (0-1)")
    parser.add_argument("--echo-leakage", type=float, default=0.999, help="Echo canceller leakage (0-1)")
    parser.add_argument("--echo-min-power", type=float, default=1e-3, help="Minimum far-end power used for normalization")
    parser.add_argument("--enable-livekit-apm", action="store_true", help="Use LiveKit AudioProcessingModule pipeline")
    parser.add_argument("--livekit-disable-aec", action="store_true", help="Disable LiveKit acoustic echo cancellation")
    parser.add_argument("--livekit-disable-ns", action="store_true", help="Disable LiveKit noise suppression")
    parser.add_argument("--livekit-disable-hpf", action="store_true", help="Disable LiveKit high-pass filter stage")
    parser.add_argument("--livekit-disable-agc", action="store_true", help="Disable LiveKit automatic gain control")
    parser.add_argument("--enable-webrtc", action="store_true", help="Use WebRTC audio processing pipeline")
    parser.add_argument("--webrtc-disable-aec", action="store_true", help="Disable WebRTC acoustic echo cancellation")
    parser.add_argument("--webrtc-enable-agc", action="store_true", help="Enable WebRTC automatic gain control")
    parser.add_argument("--webrtc-disable-ns", action="store_true", help="Disable WebRTC noise suppression")
    parser.add_argument("--webrtc-ns-level", default="high", help="WebRTC noise suppression level (e.g. low, moderate, high, very_high)")
    parser.add_argument("--webrtc-agc-mode", default="adaptive_digital", help="WebRTC AGC mode (e.g. adaptive_digital, adaptive_analog)")
    parser.add_argument("--webrtc-agc-target", type=int, default=3, help="WebRTC AGC target level in dBFS")
    parser.add_argument("--webrtc-agc-compression", type=int, default=9, help="WebRTC AGC compression gain in dB")
    parser.add_argument("--webrtc-disable-hpf", action="store_true", help="Disable WebRTC high-pass filter stage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    audio_format = AudioFormat(
        sample_rate=args.sample_rate,
        channels=args.channels,
        frame_ms=args.frame_ms,
    )
    processing_config = AudioProcessingConfig(
        enable_notch=args.enable_notch,
        notch_frequency=args.notch,
        enable_highpass=args.enable_highpass,
        highpass_cutoff=args.highpass,
        enable_echo_cancel=args.enable_echo_cancel,
        echo_filter_length=max(1, args.echo_filter_length),
        echo_adaptation=max(args.echo_adaptation, 1e-6),
        echo_leakage=args.echo_leakage,
        echo_min_power=max(args.echo_min_power, 1e-9),
        use_livekit_apm=args.enable_livekit_apm,
        livekit_config=LiveKitAPMConfig(
            echo_cancellation=not args.livekit_disable_aec,
            noise_suppression=not args.livekit_disable_ns,
            high_pass_filter=not args.livekit_disable_hpf,
            auto_gain_control=not args.livekit_disable_agc,
        ),
        use_webrtc=args.enable_webrtc,
        webrtc_config=WebRTCProcessorConfig(
            enable_aec=not args.webrtc_disable_aec,
            enable_agc=args.webrtc_enable_agc,
            enable_noise_suppression=not args.webrtc_disable_ns,
            enable_high_pass_filter=not args.webrtc_disable_hpf,
            noise_suppression_level=args.webrtc_ns_level,
            agc_mode=args.webrtc_agc_mode,
            agc_target_level_dbfs=args.webrtc_agc_target,
            agc_compression_gain_db=args.webrtc_agc_compression,
        ),
    )
    client = IntercomClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        audio_format=audio_format,
        source_factory=_build_source_factory(args),
        sink_factory=_build_sink_factory(args),
        processing_config=processing_config,
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
