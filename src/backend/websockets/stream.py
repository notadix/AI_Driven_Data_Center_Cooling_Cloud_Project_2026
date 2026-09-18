"""
WebSocket Telemetry Stream — /ws/stream

Broadcasts live telemetry from the IoT local bus to all connected clients.
Supports optional CRAC / facility filtering via query parameters.

Usage:
  ws://localhost:8000/ws/stream
  ws://localhost:8000/ws/stream?facility_id=DC-EAST-01
  ws://localhost:8000/ws/stream?facility_id=DC-EAST-01&crac_id=CRAC-01
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.aws.iot.iot_publisher import local_bus

logger = logging.getLogger(__name__)
router = APIRouter()


async def _stream_messages(
    websocket: WebSocket,
    facility_id: Optional[str],
    crac_id: Optional[str],
    max_rate_hz: float = 4.0,
) -> None:
    """Reads from the local bus queue and forwards matching messages to the WebSocket client."""
    min_interval = 1.0 / max(0.1, max_rate_hz)
    sub_queue = local_bus.subscribe()
    client = websocket.client.host if websocket.client else "unknown"
    logger.info("WS client connected: %s (facility=%s, crac=%s)", client, facility_id, crac_id)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(sub_queue.get(), timeout=min_interval)
            except asyncio.TimeoutError:
                # Send heartbeat ping to keep connection alive
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                continue

            payload = msg.get("payload", {})

            # Apply filters
            if facility_id and payload.get("facility_id") != facility_id:
                continue
            if crac_id and payload.get("crac_id") != crac_id:
                continue

            # Annotate with ASHRAE status
            inlet = payload.get("server_inlet_temp_c")
            if inlet is not None:
                inlet = float(inlet)
                if inlet >= 32.0:
                    ashrae_status = "CRITICAL"
                elif inlet > 27.0 or inlet < 18.0:
                    ashrae_status = "SLA_BREACH"
                else:
                    ashrae_status = "NORMAL"
                payload = {**payload, "ashrae_status": ashrae_status}

            envelope = {"type": "telemetry", "topic": msg.get("topic", ""), "payload": payload}
            await websocket.send_text(json.dumps(envelope))

    except WebSocketDisconnect:
        logger.info("WS client disconnected: %s", client)
    except Exception as e:
        logger.error("WS stream error for %s: %s", client, e)
    finally:
        local_bus.unsubscribe(sub_queue)


@router.websocket("/stream")
async def telemetry_stream(
    websocket: WebSocket,
    facility_id: Optional[str] = Query(None, description="Filter by facility ID"),
    crac_id: Optional[str] = Query(None, description="Filter by CRAC ID"),
    max_rate_hz: float = Query(4.0, ge=0.1, le=10.0, description="Max message rate (Hz)"),
) -> None:
    """
    WebSocket endpoint streaming live telemetry from the IoT local bus.
    Message envelope: {"type": "telemetry", "topic": "...", "payload": {...}}
    Heartbeat: {"type": "heartbeat"} sent every second when no data matches filters.
    """
    await websocket.accept()
    await _stream_messages(websocket, facility_id, crac_id, max_rate_hz)
