"""Asyncio-based host server for the Home Assistant intercom integration.

The server accepts two types of clients:
- A single embedded intercom client that bridges to the physical hardware.
- Multiple Home Assistant clients that consume events and issue commands.

Messages are exchanged as newline-delimited JSON documents following the
handshake and routing rules defined in PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

REGISTER_TIMEOUT_SECONDS = 10
ENCODING = "utf-8"


class ProtocolError(Exception):
    """Raised when a client sends a malformed or disallowed message."""


@dataclass
class ClientSession:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    role: str
    client_id: str
    address: str
    pending_commands: Set[str] = field(default_factory=set)

    @property
    def is_intercom(self) -> bool:
        return self.role == "intercom"

    @property
    def is_home_assistant(self) -> bool:
        return self.role == "home_assistant"


@dataclass
class PendingCommand:
    client_id: str
    command: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioSession:
    stream_id: str
    home_client_id: str
    intercom_client_id: str
    encoding: str
    sample_rate: int
    channels: int


class IntercomHostServer:
    def __init__(self) -> None:
        self._state_lock = asyncio.Lock()
        self._home_clients: Dict[str, ClientSession] = {}
        self._intercom_session: Optional[ClientSession] = None
        self._pending_commands: Dict[str, PendingCommand] = {}
        self._audio_sessions: Dict[str, AudioSession] = {}

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        address = self._format_address(writer)
        logging.info("Connection accepted from %s", address)

        try:
            session = await self._register_client(reader, writer)
        except ProtocolError as exc:
            logging.warning("Registration failed for %s: %s", address, exc)
            await self._send_json(writer, {"type": "error", "reason": str(exc)})
            self._close_writer(writer)
            return
        except asyncio.CancelledError:
            self._close_writer(writer)
            raise
        except Exception:
            logging.exception("Unexpected error during registration from %s", address)
            await self._send_json(writer, {"type": "error", "reason": "internal_error"})
            self._close_writer(writer)
            return

        logging.info(
            "Registered %s client '%s' from %s",
            session.role,
            session.client_id,
            session.address,
        )

        try:
            await self._client_loop(session)
        except asyncio.CancelledError:
            raise
        finally:
            await self._unregister_client(session)
            self._close_writer(writer)
            logging.info("Connection closed for %s", session.address)

    async def _register_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> ClientSession:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=REGISTER_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise ProtocolError("registration_timeout") from exc

        if not raw:
            raise ProtocolError("connection_closed")

        message = self._decode_message(raw)
        if message.get("type") != "register":
            raise ProtocolError("expected_register")

        role = message.get("role")
        client_id = message.get("client_id")

        if role not in {"intercom", "home_assistant"}:
            raise ProtocolError("invalid_role")
        if not isinstance(client_id, str) or not client_id:
            raise ProtocolError("invalid_client_id")

        session = ClientSession(
            reader=reader,
            writer=writer,
            role=role,
            client_id=client_id,
            address=self._format_address(writer),
        )

        async with self._state_lock:
            if role == "intercom":
                if self._intercom_session is not None:
                    raise ProtocolError("intercom_already_registered")
                self._intercom_session = session
            else:
                if client_id in self._home_clients:
                    raise ProtocolError("duplicate_client_id")
                self._home_clients[client_id] = session

        await self._send_json(
            writer,
            {
                "type": "registered",
                "status": "ok",
                "role": role,
                "client_id": client_id,
            },
        )

        return session

    async def _client_loop(self, session: ClientSession) -> None:
        reader = session.reader
        while True:
            raw = await reader.readline()
            if not raw:
                break

            try:
                message = self._decode_message(raw)
                should_continue = await self._dispatch_message(session, message)
                if not should_continue:
                    break
            except ProtocolError as exc:
                logging.warning(
                    "Protocol error from %s (%s): %s",
                    session.client_id,
                    session.role,
                    exc,
                )
                await self._send_error(session, reason=str(exc))
                break
            except json.JSONDecodeError as exc:
                logging.warning("Invalid JSON from %s: %s", session.client_id, exc)
                await self._send_error(session, reason="invalid_json")
            except Exception:
                logging.exception("Unhandled error while processing message from %s", session.client_id)
                await self._send_error(session, reason="internal_error")
                break

    async def _dispatch_message(self, session: ClientSession, message: Dict) -> bool:
        message_type = message.get("type")
        if message_type == "command" and session.is_home_assistant:
            await self._handle_home_command(session, message)
        elif message_type == "response" and session.is_intercom:
            await self._handle_intercom_response(message)
        elif message_type == "event" and session.is_intercom:
            await self._handle_intercom_event(message)
        elif message_type == "audio_frame":
            await self._handle_audio_frame(session, message)
        elif message_type == "close":
            return False
        else:
            raise ProtocolError(f"unsupported_message:{message_type}")

        return True

    async def _handle_home_command(self, session: ClientSession, message: Dict) -> None:
        command = message.get("command")
        if not isinstance(command, str) or not command:
            raise ProtocolError("invalid_command")

        payload = message.get("payload", {})
        if not isinstance(payload, dict):
            raise ProtocolError("invalid_payload")

        requested_command_id = message.get("command_id")
        if requested_command_id is not None and not isinstance(requested_command_id, str):
            raise ProtocolError("invalid_command_id")

        async with self._state_lock:
            intercom_session = self._intercom_session
            if intercom_session is None:
                raise ProtocolError("intercom_unavailable")

            context: Dict[str, Any] = {}
            if command == "start_audio":
                requested_stream_id = payload.get("stream_id")
                if requested_stream_id is not None:
                    if not isinstance(requested_stream_id, str) or not requested_stream_id:
                        raise ProtocolError("invalid_stream_id")
                    if requested_stream_id in self._audio_sessions:
                        raise ProtocolError("stream_id_in_use")
                    context["requested_stream_id"] = requested_stream_id

                if "direction" in payload:
                    raise ProtocolError("audio_direction_unsupported")

                if any(
                    audio.home_client_id == session.client_id
                    for audio in self._audio_sessions.values()
                ):
                    raise ProtocolError("audio_session_exists")
            elif command == "stop_audio":
                stream_id = payload.get("stream_id")
                if not isinstance(stream_id, str) or not stream_id:
                    raise ProtocolError("invalid_stream_id")
                audio_session = self._audio_sessions.get(stream_id)
                if audio_session is None or audio_session.home_client_id != session.client_id:
                    raise ProtocolError("unknown_stream_id")
                context["stream_id"] = stream_id

            command_id = requested_command_id or str(uuid.uuid4())
            self._pending_commands[command_id] = PendingCommand(
                client_id=session.client_id,
                command=command,
                context=context,
            )
            session.pending_commands.add(command_id)

        ack = {
            "type": "command_ack",
            "command_id": command_id,
            "generated": requested_command_id is None,
        }
        await self._send_json(session.writer, ack)

        relay_message = {
            "type": "command",
            "command": command,
            "payload": payload,
            "command_id": command_id,
            "origin_id": session.client_id,
        }
        await self._send_json(intercom_session.writer, relay_message)

    async def _handle_intercom_response(self, message: Dict) -> None:
        command_id = message.get("command_id")
        status = message.get("status")
        payload = message.get("payload", {})

        if not isinstance(command_id, str) or not command_id:
            raise ProtocolError("missing_command_id")
        if status not in {"ok", "error"}:
            raise ProtocolError("invalid_response_status")
        if not isinstance(payload, dict):
            raise ProtocolError("invalid_payload")

        async with self._state_lock:
            pending = self._pending_commands.pop(command_id, None)
            if pending is None:
                raise ProtocolError("unknown_command_id")
            target_session = self._home_clients.get(pending.client_id)
            if target_session:
                target_session.pending_commands.discard(command_id)

            if pending.command == "start_audio" and status == "ok":
                stream_id = payload.get("stream_id") or pending.context.get("requested_stream_id")
                encoding = payload.get("encoding")
                sample_rate = payload.get("sample_rate")
                channels = payload.get("channels")

                if "direction" in payload:
                    raise ProtocolError("audio_direction_unsupported")

                if not isinstance(stream_id, str) or not stream_id:
                    raise ProtocolError("missing_stream_id")
                if not isinstance(encoding, str):
                    raise ProtocolError("invalid_audio_encoding")
                if not isinstance(sample_rate, int):
                    raise ProtocolError("invalid_sample_rate")
                if not isinstance(channels, int):
                    raise ProtocolError("invalid_channels")

                if stream_id in self._audio_sessions:
                    raise ProtocolError("duplicate_stream_id")

                intercom_session = self._intercom_session
                if intercom_session is None or target_session is None:
                    if intercom_session is None:
                        target_session = None
                else:
                    self._audio_sessions[stream_id] = AudioSession(
                        stream_id=stream_id,
                        home_client_id=pending.client_id,
                        intercom_client_id=intercom_session.client_id,
                        encoding=encoding,
                        sample_rate=sample_rate,
                        channels=channels,
                    )

            if pending.command == "stop_audio" and status == "ok":
                stream_id = payload.get("stream_id") or pending.context.get("stream_id")
                if isinstance(stream_id, str):
                    self._audio_sessions.pop(stream_id, None)

        if target_session is None:
            raise ProtocolError("origin_client_not_found")

        await self._send_json(
            target_session.writer,
            {
                "type": "response",
                "command_id": command_id,
                "status": status,
                "payload": payload,
            },
        )

    async def _handle_intercom_event(self, message: Dict) -> None:
        event_name = message.get("event")
        payload = message.get("payload", {})

        if not isinstance(event_name, str) or not event_name:
            raise ProtocolError("invalid_event")
        if not isinstance(payload, dict):
            raise ProtocolError("invalid_payload")

        async with self._state_lock:
            sessions = list(self._home_clients.values())

        if not sessions:
            return

        broadcast = {
            "type": "event",
            "event": event_name,
            "payload": payload,
        }

        await asyncio.gather(
            *(self._send_json(session.writer, broadcast) for session in sessions),
            return_exceptions=True,
        )

    async def _handle_audio_frame(self, session: ClientSession, message: Dict) -> None:
        stream_id = message.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            raise ProtocolError("invalid_stream_id")

        data = message.get("data")
        if not isinstance(data, str) or not data:
            raise ProtocolError("invalid_audio_data")

        sequence = message.get("sequence")
        if sequence is not None and not isinstance(sequence, int):
            raise ProtocolError("invalid_sequence")

        encoding = message.get("encoding")
        sample_rate = message.get("sample_rate")
        channels = message.get("channels")

        if not isinstance(encoding, str) or not encoding:
            raise ProtocolError("invalid_audio_encoding")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ProtocolError("invalid_sample_rate")
        if not isinstance(channels, int) or channels <= 0:
            raise ProtocolError("invalid_channels")

        async with self._state_lock:
            audio_session = self._audio_sessions.get(stream_id)
            if audio_session is None:
                raise ProtocolError("unknown_stream_id")

            if session.is_intercom:
                if audio_session.intercom_client_id != session.client_id:
                    raise ProtocolError("stream_not_owned")
                target_session = self._home_clients.get(audio_session.home_client_id)
                direction = "intercom_to_client"
            elif session.is_home_assistant:
                if audio_session.home_client_id != session.client_id:
                    raise ProtocolError("stream_not_owned")
                target_session = self._intercom_session
                direction = "client_to_intercom"
            else:
                raise ProtocolError("invalid_role")

        if target_session is None:
            await self._send_error(
                session,
                reason="destination_unavailable",
                details={"stream_id": stream_id},
            )
            return

        frame = dict(message)
        frame["direction"] = direction
        frame["origin_id"] = session.client_id

        await self._send_json(target_session.writer, frame)

    async def _unregister_client(self, session: ClientSession) -> None:
        if session.is_home_assistant:
            await self._handle_home_disconnect(session)
        elif session.is_intercom:
            await self._handle_intercom_disconnect(session)

    async def _handle_home_disconnect(self, session: ClientSession) -> None:
        async with self._state_lock:
            self._home_clients.pop(session.client_id, None)
            pending_for_client = [
                command_id
                for command_id, pending in self._pending_commands.items()
                if pending.client_id == session.client_id
            ]
            for command_id in pending_for_client:
                self._pending_commands.pop(command_id, None)

            disconnected_streams = [
                stream_id
                for stream_id, audio in self._audio_sessions.items()
                if audio.home_client_id == session.client_id
            ]
            for stream_id in disconnected_streams:
                self._audio_sessions.pop(stream_id, None)

            intercom_session = self._intercom_session

        if intercom_session is None:
            return

        tasks = []
        for command_id in pending_for_client:
            tasks.append(
                self._send_json(
                    intercom_session.writer,
                    {
                        "type": "error",
                        "reason": "origin_client_disconnected",
                        "details": {"client_id": session.client_id, "command_id": command_id},
                    },
                )
            )

        for stream_id in disconnected_streams:
            tasks.append(
                self._send_json(
                    intercom_session.writer,
                    {
                        "type": "error",
                        "reason": "audio_client_disconnected",
                        "details": {"client_id": session.client_id, "stream_id": stream_id},
                    },
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_intercom_disconnect(self, session: ClientSession) -> None:
        async with self._state_lock:
            if self._intercom_session is session:
                self._intercom_session = None
            pending = list(self._pending_commands.items())
            self._pending_commands.clear()
            audio_sessions = list(self._audio_sessions.values())
            self._audio_sessions.clear()

        tasks = []
        for command_id, pending_cmd in pending:
            client_session = self._home_clients.get(pending_cmd.client_id)
            if client_session is None:
                continue
            client_session.pending_commands.discard(command_id)
            tasks.append(
                self._send_json(
                    client_session.writer,
                    {
                        "type": "response",
                        "command_id": command_id,
                        "status": "error",
                        "payload": {"reason": "intercom_disconnected"},
                    },
                )
            )

        for audio_session in audio_sessions:
            client_session = self._home_clients.get(audio_session.home_client_id)
            if client_session is None:
                continue
            tasks.append(
                self._send_json(
                    client_session.writer,
                    {
                        "type": "error",
                        "reason": "audio_intercom_disconnected",
                        "details": {
                            "stream_id": audio_session.stream_id,
                            "encoding": audio_session.encoding,
                        },
                    },
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _decode_message(raw: bytes) -> Dict:
        text = raw.decode(ENCODING).strip()
        if not text:
            raise ProtocolError("empty_message")
        message = json.loads(text)
        if not isinstance(message, dict):
            raise ProtocolError("message_not_object")
        return message

    async def _send_json(self, writer: asyncio.StreamWriter, message: Dict) -> None:
        if writer.is_closing():
            return
        data = json.dumps(message, separators=(",", ":")) + "\n"
        writer.write(data.encode(ENCODING))
        try:
            await writer.drain()
        except ConnectionResetError:
            self._close_writer(writer)

    async def _send_error(self, session: ClientSession, *, reason: str, details: Optional[Dict] = None) -> None:
        payload = {"type": "error", "reason": reason}
        if details:
            payload["details"] = details
        await self._send_json(session.writer, payload)

    @staticmethod
    def _format_address(writer: asyncio.StreamWriter) -> str:
        peer = writer.get_extra_info("peername")
        if not peer:
            return "unknown"
        if isinstance(peer, tuple):
            return f"{peer[0]}:{peer[1]}"
        return str(peer)

    @staticmethod
    def _close_writer(writer: asyncio.StreamWriter) -> None:
        if not writer.is_closing():
            writer.close()


async def _serve(host: str, port: int) -> None:
    state = IntercomHostServer()

    async def client_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await state.handle_client(reader, writer)

    server = await asyncio.start_server(client_handler, host, port)
    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    logging.info("Serving on %s", sockets)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logging.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Signal handlers are not available on some platforms (e.g., Windows).
            pass

    async with server:
        await stop_event.wait()
        server.close()
        await server.wait_closed()

    logging.info("Server shut down")


def main() -> None:
    parser = argparse.ArgumentParser(description="Intercom host server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Listen port")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(_serve(args.host, args.port))
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
