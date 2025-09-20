from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

ENCODING = "utf-8"


class ConnectionClosed(Exception):
    pass


@dataclass(slots=True)
class JsonConnection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    async def send(self, message: Dict[str, Any]) -> None:
        data = json.dumps(message, separators=(",", ":")) + "\n"
        self.writer.write(data.encode(ENCODING))
        try:
            await self.writer.drain()
        except ConnectionResetError as exc:  # pragma: no cover - handled by caller
            raise ConnectionClosed("connection reset") from exc

    async def receive(self) -> Dict[str, Any]:
        line = await self.reader.readline()
        if not line:
            raise ConnectionClosed("connection closed by peer")
        message = json.loads(line.decode(ENCODING).strip())
        if not isinstance(message, dict):
            raise ValueError("expected object message")
        return message

    def is_closing(self) -> bool:
        return self.writer.is_closing()

    def close(self) -> None:
        if not self.writer.is_closing():
            self.writer.close()


async def open_connection(host: str, port: int) -> JsonConnection:
    reader, writer = await asyncio.open_connection(host, port)
    return JsonConnection(reader=reader, writer=writer)


async def register_client(
    connection: JsonConnection,
    *,
    role: str,
    client_id: str,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": "register", "role": role, "client_id": client_id}
    if token is not None:
        payload["token"] = token
    await connection.send(payload)
    reply = await connection.receive()
    if reply.get("type") != "registered" or reply.get("status") != "ok":
        raise ConnectionClosed(f"registration failed: {reply}")
    return reply
