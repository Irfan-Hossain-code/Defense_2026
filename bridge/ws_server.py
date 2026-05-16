"""WebSocket bridge: Python tracker → React HUD."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol


class HudBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._clients: Set[WebSocketServerProtocol] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._latest: dict[str, Any] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ws-bridge")
        self._thread.start()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"[hud] WebSocket listening ws://{self.host}:{self.port}")
            await asyncio.Future()

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        try:
            if self._latest:
                await ws.send(json.dumps(self._latest))
            async for _msg in ws:
                pass  # React is receive-only for now
        finally:
            self._clients.discard(ws)

    def broadcast(self, payload: dict[str, Any]) -> None:
        self._latest = payload
        if not self._loop or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._send_all(payload), self._loop)

    async def _send_all(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return
        msg = json.dumps(payload)
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
