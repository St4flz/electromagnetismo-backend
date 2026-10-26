"""Pruebas de la infraestructura de streaming (frames binarios, hilo de trabajo, cancelación)."""

import json
import time

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from api.websockets.stream import decode_field_frame, encode_field_frame, run_stream

app = FastAPI()


@app.websocket("/ws/test-stream")
async def ws_test(ws: WebSocket):
    await ws.accept()
    n = json.loads(await ws.receive_text())["frames"]

    def producer():
        for k in range(n):
            time.sleep(0.01)  # cálculo bloqueante simulado
            yield k, np.full((4, 6), float(k)), {"energy": float(k)}

    await run_stream(ws, producer, target_frame_ms=1.0)


client = TestClient(app)


def test_frame_roundtrip_quantization_error_below_1_percent():
    rng = np.random.default_rng(0)
    field = rng.normal(size=(20, 30))
    step, back = decode_field_frame(encode_field_frame(7, field))
    assert step == 7 and back.shape == field.shape
    assert np.max(np.abs(back - field)) / np.max(np.abs(field)) < 0.01


def test_stream_completes_with_text_and_binary_frames():
    with client.websocket_connect("/ws/test-stream") as ws:
        ws.send_text(json.dumps({"frames": 3}))
        steps = []
        for _ in range(3):
            meta = json.loads(ws.receive_text())
            assert meta["type"] == "stream_frame" and meta["computeTimeMs"] >= 10.0
            step, arr = decode_field_frame(ws.receive_bytes())
            assert step == meta["step"] and arr.shape == (4, 6)
            steps.append(step)
        assert steps == [0, 1, 2]
        end = json.loads(ws.receive_text())
        assert end == {"type": "stream_end", "completed": True}


def test_stream_can_be_cancelled():
    with client.websocket_connect("/ws/test-stream") as ws:
        ws.send_text(json.dumps({"frames": 10_000}))
        json.loads(ws.receive_text())
        ws.receive_bytes()
        ws.send_text(json.dumps({"type": "cancel"}))
        while True:
            msg = ws.receive()
            if msg.get("text") and json.loads(msg["text"]).get("type") == "stream_end":
                assert json.loads(msg["text"])["completed"] is False
                break
