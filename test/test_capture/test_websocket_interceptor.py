from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from test.mocks import MockWebSocketInterceptor as WebSocketInterceptor


class TestWebSocketInterceptor:
    """Tests for WebSocketInterceptor following marimo patterns"""

    def test_interceptor_initialization(self, config_with_temp_dir):
        """Test interceptor initializes properly"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        assert interceptor.config == config_with_temp_dir
        assert hasattr(interceptor, "websocket_url")
        assert hasattr(interceptor, "connection")

    @pytest.mark.asyncio
    async def test_websocket_connection_establishment(self, config_with_temp_dir):
        """Test WebSocket connection is established properly"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket = AsyncMock()

        with patch("websockets.connect", return_value=mock_websocket) as mock_connect:
            await interceptor.connect()

            mock_connect.assert_called_once()
            assert interceptor.connection == mock_websocket

    @pytest.mark.asyncio
    async def test_message_sending(self, config_with_temp_dir):
        """Test sending messages through WebSocket"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket = AsyncMock()
        interceptor.connection = mock_websocket

        test_message = {
            "type": "dataframe_operation",
            "df_name": "test_df",
            "operation": "merge",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        await interceptor.send_message(test_message)

        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        sent_message = json.loads(sent_data)

        assert sent_message["type"] == test_message["type"]
        assert sent_message["df_name"] == test_message["df_name"]

    @pytest.mark.asyncio
    async def test_message_receiving(self, config_with_temp_dir):
        """Test receiving messages from WebSocket"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket = AsyncMock()

        # Mock incoming messages
        test_messages = [
            json.dumps({"type": "session_start", "session_id": "test-123"}),
            json.dumps({"type": "cell_execution", "cell_id": "cell-1"}),
        ]

        mock_websocket.recv.side_effect = test_messages
        interceptor.connection = mock_websocket

        # Test receiving first message
        message1 = await interceptor.receive_message()
        assert message1["type"] == "session_start"
        assert message1["session_id"] == "test-123"

        # Test receiving second message
        message2 = await interceptor.receive_message()
        assert message2["type"] == "cell_execution"
        assert message2["cell_id"] == "cell-1"

    @pytest.mark.asyncio
    async def test_marimo_session_tracking(
        self, config_with_temp_dir, mock_marimo_session
    ):
        """Test tracking of marimo session events"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket = AsyncMock()
        interceptor.connection = mock_websocket

        with patch.object(
            interceptor, "_get_marimo_session", return_value=mock_marimo_session
        ):
            await interceptor.track_session_event("session_start")

            # Should send session event
            mock_websocket.send.assert_called_once()
            sent_data = json.loads(mock_websocket.send.call_args[0][0])

            assert sent_data["type"] == "session_start"
            assert sent_data["session_id"] == mock_marimo_session.session_id

    @pytest.mark.asyncio
    async def test_cell_execution_tracking(self, config_with_temp_dir):
        """Test tracking of cell execution events"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket = AsyncMock()
        interceptor.connection = mock_websocket

        cell_info = {
            "cell_id": "cell-42",
            "code": "df = pd.DataFrame({'a': [1, 2, 3]})",
            "execution_time": 0.123,
        }

        await interceptor.track_cell_execution(cell_info)

        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])

        assert sent_data["type"] == "cell_execution"
        assert sent_data["cell_id"] == cell_info["cell_id"]
        assert sent_data["code"] == cell_info["code"]

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, config_with_temp_dir):
        """Test error handling for connection failures"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        with patch(
            "websockets.connect", side_effect=ConnectionError("Connection failed")
        ):
            # Should handle connection errors gracefully
            try:
                await interceptor.connect()
            except ConnectionError:
                pytest.fail("Should handle connection errors gracefully")

            # Connection should be None after failed connection
            assert interceptor.connection is None

    @pytest.mark.asyncio
    async def test_reconnection_logic(self, config_with_temp_dir):
        """Test automatic reconnection on connection loss"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()

        # First connection succeeds
        with patch(
            "websockets.connect", side_effect=[mock_websocket1, mock_websocket2]
        ):
            await interceptor.connect()
            assert interceptor.connection == mock_websocket1

            # Simulate connection loss
            mock_websocket1.send.side_effect = ConnectionError("Connection lost")

            # Should attempt reconnection
            with patch.object(interceptor, "connect") as mock_reconnect:
                mock_reconnect.return_value = None
                interceptor.connection = mock_websocket2

                test_message = {"type": "test"}
                await interceptor.send_message(test_message)

                # Should attempt reconnection on failure
                mock_reconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_buffering_on_disconnection(self, config_with_temp_dir):
        """Test message buffering when connection is lost"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        # No connection established
        interceptor.connection = None

        test_messages = [
            {"type": "buffer_test_1", "data": "message1"},
            {"type": "buffer_test_2", "data": "message2"},
        ]

        # Should buffer messages when disconnected
        for message in test_messages:
            await interceptor.send_message(message)

        # Check that messages are buffered
        assert len(interceptor.message_buffer) == len(test_messages)

        # When connection is restored, should send buffered messages
        mock_websocket = AsyncMock()
        interceptor.connection = mock_websocket

        await interceptor._flush_message_buffer()

        # Should send all buffered messages
        assert mock_websocket.send.call_count == len(test_messages)

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, config_with_temp_dir):
        """Test handling of concurrent messages"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket = AsyncMock()
        interceptor.connection = mock_websocket

        # Send multiple messages concurrently
        messages = [
            {"type": f"concurrent_test_{i}", "data": f"message_{i}"} for i in range(10)
        ]

        tasks = [interceptor.send_message(message) for message in messages]

        # All messages should be sent successfully
        await asyncio.gather(*tasks)

        assert mock_websocket.send.call_count == len(messages)

    @pytest.mark.asyncio
    async def test_message_serialization_edge_cases(self, config_with_temp_dir):
        """Test message serialization with edge cases"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket = AsyncMock()
        interceptor.connection = mock_websocket

        # Test with various data types
        edge_case_messages = [
            {"type": "unicode", "data": "Hello 🌍"},
            {"type": "large_number", "data": 1234567890123456789},
            {"type": "nested", "data": {"nested": {"deeply": {"nested": "value"}}}},
            {"type": "empty", "data": {}},
            {"type": "null_value", "data": None},
        ]

        for message in edge_case_messages:
            await interceptor.send_message(message)

            # Should serialize without errors
            assert mock_websocket.send.called

            # Verify JSON serialization worked
            sent_data = mock_websocket.send.call_args[0][0]
            deserialized = json.loads(sent_data)
            assert deserialized["type"] == message["type"]

    @pytest.mark.asyncio
    async def test_cleanup_on_shutdown(self, config_with_temp_dir):
        """Test proper cleanup on interceptor shutdown"""
        interceptor = WebSocketInterceptor(config_with_temp_dir)

        mock_websocket = AsyncMock()
        interceptor.connection = mock_websocket

        # Add some buffered messages
        interceptor.message_buffer = [{"type": "test"}]

        await interceptor.shutdown()

        # Should close connection
        mock_websocket.close.assert_called_once()

        # Should clear buffers
        assert len(interceptor.message_buffer) == 0
        assert interceptor.connection is None
