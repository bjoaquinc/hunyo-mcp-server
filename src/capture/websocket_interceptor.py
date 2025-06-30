#!/usr/bin/env python3
"""
Marimo WebSocket Interceptor - Primary execution monitoring strategy
Intercepts WebSocket communication between marimo frontend and backend
to capture ALL cell execution events, not just DataFrame operations.

Based on the comprehensive strategy document for external marimo monitoring.
"""

import asyncio
import json
import threading
import time
import uuid
from datetime import UTC, datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import websockets

# Import the logger
from .logger import get_logger

# Module logger
ws_logger = get_logger("hunyo.websocket")

# Global tracking state
_proxy_server = None
_active_connections = {}
_message_history = []
_config = {
    "frontend_port": 2718,
    "backend_port": 2719,
    "proxy_port": 2720,
    "log_messages": True,
    "log_data_messages": False,
}


class MarimoWebSocketProxy:
    """WebSocket proxy server for intercepting marimo communication"""

    def __init__(
        self,
        frontend_port: int = 2718,
        backend_port: int = 2719,
        proxy_port: int = 2720,
        output_file: str = "marimo_websocket_events.jsonl",
    ):
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.proxy_port = proxy_port
        self.output_file = Path(output_file)
        self.session_id = str(uuid.uuid4())[:8]
        self.server = None
        self.running = False

        # Ensure output file exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.touch()

        ws_logger.status("Marimo WebSocket Proxy v1.0")
        ws_logger.config(f"Frontend port: {self.frontend_port}")
        ws_logger.config(f"Backend port: {self.backend_port}")
        ws_logger.config(f"Proxy port: {self.proxy_port}")
        ws_logger.file_op(f"Event log: {self.output_file.name}")
        ws_logger.config(f"Session: {self.session_id}")

    async def start_proxy(self):
        """Start the WebSocket proxy server"""
        if self.running:
            ws_logger.warning("Proxy server already running")
            return

        try:
            ws_logger.startup("Starting WebSocket proxy server...")
            self.server = await websockets.serve(
                self._handle_proxy_connection, "localhost", self.proxy_port
            )
            self.running = True
            ws_logger.success(f"Proxy server listening on ws://localhost:{self.proxy_port}")

        except Exception as e:
            ws_logger.error(f"Failed to start proxy server: {e}")
            raise

    async def _handle_proxy_connection(self, websocket, path):
        """Handle incoming proxy connections"""
        connection_id = str(uuid.uuid4())[:8]
        ws_logger.tracking(f"New connection: {connection_id}")

        try:
            # Connect to actual marimo backend
            backend_uri = f"ws://localhost:{self.backend_port}"
            async with websockets.connect(backend_uri) as backend_ws:
                ws_logger.success(f"Connected to backend: {backend_uri}")

                # Register connection
                _active_connections[connection_id] = {
                    "frontend": websocket,
                    "backend": backend_ws,
                    "connected_at": datetime.now().isoformat(),
                }

                # Start bidirectional forwarding
                await self._forward_messages(websocket, backend_ws, connection_id)

        except Exception as e:
            ws_logger.error(f"Connection error: {e}")
        finally:
            # Clean up connection
            if connection_id in _active_connections:
                del _active_connections[connection_id]
            ws_logger.tracking(f"Connection closed: {connection_id}")

    async def _forward_messages(self, frontend_ws, backend_ws, connection_id):
        """Forward messages bidirectionally between frontend and backend"""

        async def forward_frontend_to_backend():
            """Forward messages from frontend to backend"""
            try:
                async for message in frontend_ws:
                    # Log and process message
                    await self._log_message(
                        "frontend_to_backend", message, connection_id
                    )

                    # Forward to backend
                    await backend_ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                ws_logger.tracking(f"Frontend connection closed: {connection_id}")
            except Exception as e:
                ws_logger.warning(f"Frontend forwarding error: {e}")

        async def forward_backend_to_frontend():
            """Forward messages from backend to frontend"""
            try:
                async for message in backend_ws:
                    # Log and process message
                    await self._log_message(
                        "backend_to_frontend", message, connection_id
                    )

                    # Forward to frontend
                    await frontend_ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                ws_logger.tracking(f"Backend connection closed: {connection_id}")
            except Exception as e:
                ws_logger.warning(f"Backend forwarding error: {e}")

        # Run both forwarding tasks concurrently
        await asyncio.gather(
            forward_frontend_to_backend(),
            forward_backend_to_frontend(),
            return_exceptions=True,
        )

    async def _log_message(self, direction: str, message: str, connection_id: str):
        """Log WebSocket message with metadata"""
        try:
            # Parse message if it's JSON
            message_data = None
            try:
                message_data = json.loads(message)
            except json.JSONDecodeError:
                message_data = {"raw_message": message}

            # Create event record
            event = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "connection_id": connection_id,
                "direction": direction,
                "message_type": message_data.get("type", "unknown"),
                "message_size": len(message),
                "message_data": message_data if _config["log_data_messages"] else None,
            }

            # Add to memory history
            _message_history.append(event)
            if len(_message_history) > 1000:  # Keep last 1000 messages
                _message_history.pop(0)

            # Write to file
            with open(self.output_file, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")

            # Log message summary if enabled
            if _config["log_messages"]:
                msg_type = event["message_type"]
                size = event["message_size"]
                ws_logger.tracking(
                    f"{direction}: {msg_type} ({size} bytes) - {connection_id}"
                )

        except Exception as e:
            ws_logger.warning(f"Failed to log message: {e}")

    async def stop_proxy(self):
        """Stop the proxy server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.running = False
            ws_logger.config("Proxy server stopped")

    def get_session_summary(self):
        """Get summary of current proxy session"""
        return {
            "session_id": self.session_id,
            "proxy_port": self.proxy_port,
            "frontend_port": self.frontend_port,
            "backend_port": self.backend_port,
            "active_connections": len(_active_connections),
            "total_messages": len(_message_history),
            "running": self.running,
            "output_file": str(self.output_file),
        }


# Global proxy management functions
def start_websocket_proxy(
    frontend_port: int = 2718,
    backend_port: int = 2719,
    proxy_port: int = 2720,
    output_file: str = "marimo_websocket_events.jsonl",
):
    """Start the global WebSocket proxy"""
    global _proxy_server

    if _proxy_server and _proxy_server.running:
        ws_logger.warning("WebSocket proxy already running")
        return _proxy_server

    _proxy_server = MarimoWebSocketProxy(
        frontend_port, backend_port, proxy_port, output_file
    )

    # Start proxy in a separate thread
    def run_proxy():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_proxy_server.start_proxy())
            loop.run_forever()
        except KeyboardInterrupt:
            ws_logger.config("Proxy interrupted by user")
        finally:
            loop.close()

    proxy_thread = threading.Thread(target=run_proxy, daemon=True)
    proxy_thread.start()

    # Wait a moment for startup
    time.sleep(0.5)
    return _proxy_server


def stop_websocket_proxy():
    """Stop the global WebSocket proxy"""
    global _proxy_server

    if _proxy_server:
        asyncio.run(_proxy_server.stop_proxy())
        _proxy_server = None
        ws_logger.config("WebSocket proxy stopped")
    else:
        ws_logger.warning("No active WebSocket proxy to stop")


def get_proxy_summary():
    """Get summary of the current proxy session"""
    if _proxy_server:
        return _proxy_server.get_session_summary()
    else:
        return {"status": "not_running"}


# Testing and demonstration
if __name__ == "__main__":
    ws_logger.config("WebSocket Proxy Test Mode")
    ws_logger.config("=" * 50)
    ws_logger.startup("Starting test WebSocket proxy...")
    ws_logger.config("Connect your marimo frontend to ws://localhost:2720")
    ws_logger.tracking("Press Ctrl+C to stop")

    try:
        proxy = start_websocket_proxy()
        input("Press Enter to stop...")
    except KeyboardInterrupt:
        ws_logger.config("Stopping proxy...")
    finally:
        stop_websocket_proxy()
