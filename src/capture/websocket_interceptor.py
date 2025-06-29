#!/usr/bin/env python3
"""
Marimo WebSocket Interceptor - Primary execution monitoring strategy
Intercepts WebSocket communication between marimo frontend and backend
to capture ALL cell execution events, not just DataFrame operations.

Based on the comprehensive strategy document for external marimo monitoring.
"""

import asyncio
import json
import logging
import socket
import threading
import time
import uuid
import websockets
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarimoWebSocketInterceptor:
    """
    WebSocket-level interceptor for marimo cell execution monitoring
    Implements Layer 2 from the strategy document: WebSocket Communication Interception
    """
    
    def __init__(self, 
                 marimo_port: int = 8001,
                 proxy_port: int = 8002, 
                 output_file: str = "marimo_websocket_events.jsonl"):
        self.marimo_port = marimo_port
        self.proxy_port = proxy_port
        self.output_file = output_file
        self.session_id = str(uuid.uuid4())[:8]
        
        # Event tracking
        self.captured_events = []
        self.cell_executions = {}
        self._lock = threading.Lock()
        
        # WebSocket connections
        self.client_connections = set()
        self.marimo_websocket = None
        
        print(f"🌐 Marimo WebSocket Interceptor v1.0")
        print(f"🎯 Monitoring marimo on port {marimo_port}")
        print(f"🔗 Proxy running on port {proxy_port}")
        print(f"📁 Events log: {output_file}")
        print(f"🔧 Session: {session_id}")
    
    async def start_interceptor(self):
        """Start the WebSocket proxy server"""
        print("🚀 Starting WebSocket interceptor...")
        
        try:
            # Start the proxy server
            start_server = websockets.serve(
                self.handle_client_connection,
                "localhost",
                self.proxy_port
            )
            
            await start_server
            print(f"✅ WebSocket proxy active on ws://localhost:{self.proxy_port}")
            print(f"📋 Configure marimo to connect through proxy for full monitoring")
            
            # Keep the server running
            await asyncio.Future()  # Run forever
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket interceptor: {e}")
            raise
    
    async def handle_client_connection(self, websocket, path):
        """Handle incoming client WebSocket connections"""
        client_id = str(uuid.uuid4())[:8]
        print(f"🔗 Client {client_id} connected from {websocket.remote_address}")
        
        self.client_connections.add(websocket)
        
        try:
            # Connect to actual marimo server
            marimo_uri = f"ws://localhost:{self.marimo_port}{path}"
            
            async with websockets.connect(marimo_uri) as marimo_ws:
                self.marimo_websocket = marimo_ws
                print(f"🎯 Connected to marimo server at {marimo_uri}")
                
                # Start bidirectional forwarding with interception
                await asyncio.gather(
                    self.forward_client_to_marimo(websocket, marimo_ws, client_id),
                    self.forward_marimo_to_client(marimo_ws, websocket, client_id)
                )
                
        except Exception as e:
            logger.error(f"Connection error for client {client_id}: {e}")
        finally:
            self.client_connections.discard(websocket)
            print(f"🔌 Client {client_id} disconnected")
    
    async def forward_client_to_marimo(self, client_ws, marimo_ws, client_id):
        """Forward messages from client to marimo with interception"""
        try:
            async for message in client_ws:
                # Intercept client -> marimo messages
                await self.intercept_client_message(message, client_id)
                
                # Forward to marimo
                await marimo_ws.send(message)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client {client_id} closed connection")
        except Exception as e:
            logger.error(f"Error forwarding client->marimo for {client_id}: {e}")
    
    async def forward_marimo_to_client(self, marimo_ws, client_ws, client_id):
        """Forward messages from marimo to client with interception"""
        try:
            async for message in marimo_ws:
                # Intercept marimo -> client messages
                await self.intercept_marimo_message(message, client_id)
                
                # Forward to client
                await client_ws.send(message)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Marimo server closed connection for client {client_id}")
        except Exception as e:
            logger.error(f"Error forwarding marimo->client for {client_id}: {e}")
    
    async def intercept_client_message(self, message: str, client_id: str):
        """Intercept and analyze messages from client to marimo"""
        try:
            data = json.loads(message)
            
            # Look for cell execution requests
            if self.is_cell_execution_request(data):
                await self.handle_cell_execution_start(data, client_id)
            
            # Log all client messages for analysis
            self.emit_event({
                'event_type': 'websocket_client_message',
                'client_id': client_id,
                'message_type': data.get('type', 'unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': data
            })
            
        except json.JSONDecodeError:
            # Non-JSON message
            self.emit_event({
                'event_type': 'websocket_client_raw',
                'client_id': client_id,
                'message': message[:200],
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error intercepting client message: {e}")
    
    async def intercept_marimo_message(self, message: str, client_id: str):
        """Intercept and analyze messages from marimo to client"""
        try:
            data = json.loads(message)
            
            # Look for cell execution results
            if self.is_cell_execution_result(data):
                await self.handle_cell_execution_complete(data, client_id)
            
            # Look for execution errors
            elif self.is_execution_error(data):
                await self.handle_cell_execution_error(data, client_id)
            
            # Log all marimo messages for analysis
            self.emit_event({
                'event_type': 'websocket_marimo_message',
                'client_id': client_id,
                'message_type': data.get('type', 'unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': data
            })
            
        except json.JSONDecodeError:
            # Non-JSON message
            self.emit_event({
                'event_type': 'websocket_marimo_raw',
                'client_id': client_id,
                'message': message[:200],
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error intercepting marimo message: {e}")
    
    def is_cell_execution_request(self, data: Dict) -> bool:
        """Detect cell execution request patterns"""
        # Common patterns for marimo cell execution requests
        patterns = [
            data.get('type') == 'execute_cell',
            data.get('type') == 'run_cell', 
            data.get('type') == 'cell_execution_request',
            'cell_id' in data and 'code' in data,
            'cell_id' in data and 'source' in data
        ]
        return any(patterns)
    
    def is_cell_execution_result(self, data: Dict) -> bool:
        """Detect cell execution result patterns"""
        patterns = [
            data.get('type') == 'execution_result',
            data.get('type') == 'cell_execution_complete',
            data.get('type') == 'cell_result',
            'cell_id' in data and 'outputs' in data,
            'cell_id' in data and 'result' in data
        ]
        return any(patterns)
    
    def is_execution_error(self, data: Dict) -> bool:
        """Detect cell execution error patterns"""
        patterns = [
            data.get('type') == 'execution_error',
            data.get('type') == 'cell_error',
            'error' in data and 'cell_id' in data,
            data.get('status') == 'error'
        ]
        return any(patterns)
    
    async def handle_cell_execution_start(self, data: Dict, client_id: str):
        """Handle cell execution start event"""
        execution_id = str(uuid.uuid4())[:8]
        cell_id = data.get('cell_id', 'unknown')
        cell_code = data.get('code') or data.get('source', '')
        
        # Store execution context
        with self._lock:
            self.cell_executions[execution_id] = {
                'execution_id': execution_id,
                'cell_id': cell_id,
                'client_id': client_id,
                'start_time': time.time(),
                'cell_code': cell_code,
                'status': 'running'
            }
        
        # Emit standardized event
        self.emit_event({
            'event_type': 'cell_execution_start',
            'execution_id': execution_id,
            'cell_id': cell_id,
            'client_id': client_id,
            'cell_source': cell_code,
            'cell_source_lines': cell_code.count('\n') + 1 if cell_code else 0,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'websocket_data': data
        })
        
        print(f"🚀 Cell execution started: {execution_id} ({cell_id})")
    
    async def handle_cell_execution_complete(self, data: Dict, client_id: str):
        """Handle cell execution completion event"""
        cell_id = data.get('cell_id', 'unknown')
        
        # Find matching execution
        execution_context = None
        with self._lock:
            for exec_id, ctx in self.cell_executions.items():
                if ctx['cell_id'] == cell_id and ctx['status'] == 'running':
                    execution_context = ctx
                    ctx['status'] = 'completed'
                    break
        
        if execution_context:
            duration = time.time() - execution_context['start_time']
            execution_id = execution_context['execution_id']
            
            # Emit standardized event
            self.emit_event({
                'event_type': 'cell_execution_complete',
                'execution_id': execution_id,
                'cell_id': cell_id,
                'client_id': client_id,
                'duration_seconds': round(duration, 3),
                'success': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'outputs': data.get('outputs', []),
                'result': data.get('result'),
                'websocket_data': data
            })
            
            print(f"✅ Cell execution completed: {execution_id} ({duration:.3f}s)")
        else:
            print(f"⚠️  Received completion for unknown cell: {cell_id}")
    
    async def handle_cell_execution_error(self, data: Dict, client_id: str):
        """Handle cell execution error event"""
        cell_id = data.get('cell_id', 'unknown')
        
        # Find matching execution
        execution_context = None
        with self._lock:
            for exec_id, ctx in self.cell_executions.items():
                if ctx['cell_id'] == cell_id and ctx['status'] == 'running':
                    execution_context = ctx
                    ctx['status'] = 'error'
                    break
        
        if execution_context:
            duration = time.time() - execution_context['start_time']
            execution_id = execution_context['execution_id']
            
            # Emit standardized event
            self.emit_event({
                'event_type': 'cell_execution_complete',
                'execution_id': execution_id,
                'cell_id': cell_id,
                'client_id': client_id,
                'duration_seconds': round(duration, 3),
                'success': False,
                'error': data.get('error', 'Unknown error'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'websocket_data': data
            })
            
            print(f"❌ Cell execution failed: {execution_id} ({duration:.3f}s)")
        else:
            print(f"⚠️  Received error for unknown cell: {cell_id}")
    
    def emit_event(self, event: Dict):
        """Emit event to output file"""
        try:
            event['session_id'] = self.session_id
            event['emitted_at'] = datetime.now(timezone.utc).isoformat()
            
            with self._lock:
                self.captured_events.append(event)
                
                # Write to file
                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(event) + '\n')
                    
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")

class MarimoWebSocketMonitor:
    """
    High-level monitor that can run the WebSocket interceptor
    """
    
    def __init__(self, **kwargs):
        self.interceptor = MarimoWebSocketInterceptor(**kwargs)
    
    def start_monitoring(self):
        """Start monitoring in a background thread"""
        def run_interceptor():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.interceptor.start_interceptor())
        
        thread = threading.Thread(target=run_interceptor, daemon=True)
        thread.start()
        print("🌐 WebSocket monitor started in background")
        return thread

# Convenience functions
def start_websocket_monitoring(**kwargs):
    """Start WebSocket monitoring for marimo"""
    monitor = MarimoWebSocketMonitor(**kwargs)
    return monitor.start_monitoring()

def run_websocket_interceptor(**kwargs):
    """Run WebSocket interceptor directly (blocking)"""
    interceptor = MarimoWebSocketInterceptor(**kwargs)
    asyncio.run(interceptor.start_interceptor())

if __name__ == "__main__":
    print("🌐 Starting Marimo WebSocket Interceptor...")
    print("📋 This will monitor ALL marimo cell executions via WebSocket interception")
    print("🎯 Connect marimo frontend to ws://localhost:8002 for monitoring")
    print("⚡ Press Ctrl+C to stop")
    
    try:
        run_websocket_interceptor()
    except KeyboardInterrupt:
        print("\n🛑 WebSocket interceptor stopped") 