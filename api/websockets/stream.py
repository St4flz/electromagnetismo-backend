"""Infraestructura reutilizable de streaming para simulaciones pesadas (FDTD, optimizador).

Resuelve tres problemas del WebSocket original:
  1. El cálculo pesado corría dentro del event loop → ahora corre en un hilo
     (``asyncio.to_thread``), el servidor sigue atendiendo a otros clientes.
  2. Los campos viajaban como listas JSON anidadas → ahora como bytes (uint8).
  3. No había forma de detener un cálculo → el cliente puede enviar ``{"type": "cancel"}``.

Formato del frame binario (little-endian):
    offset  tipo      campo
    0       uint32    step        índice de paso de la simulación
    4       float32   scale       valor físico correspondiente a |127| (decodificar: v = (u8-128)/127*scale)
    8       uint16    rows
    10      uint16    cols
    12      uint8[]   payload     rows*cols bytes, orden C (fila mayor)

Cada frame binario va precedido de un mensaje de texto JSON con las métricas del mismo paso.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import Any, Callable, Iterator

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

HEADER = struct.Struct("<IfHH")

# Un productor es un generador síncrono: cada ``yield`` entrega (paso, campo 2D | None, métricas).
Producer = Callable[[], Iterator[tuple[int, "np.ndarray | None", dict[str, Any]]]]


def encode_field_frame(step: int, field: np.ndarray) -> bytes:
    """Cuantiza un campo 2D a uint8 simétrico alrededor de 128 y lo empaqueta con su escala."""
    scale = float(np.max(np.abs(field))) or 1.0
    q = np.clip(np.rint(field / scale * 127.0) + 128.0, 0, 255).astype(np.uint8)
    rows, cols = q.shape
    return HEADER.pack(step, scale, rows, cols) + q.tobytes(order="C")


def decode_field_frame(frame: bytes) -> tuple[int, np.ndarray]:
    """Inversa de ``encode_field_frame`` (usada en pruebas; el frontend replica esta lógica)."""
    step, scale, rows, cols = HEADER.unpack_from(frame, 0)
    q = np.frombuffer(frame, dtype=np.uint8, offset=HEADER.size).reshape(rows, cols)
    return step, (q.astype(np.float64) - 128.0) / 127.0 * scale


async def run_stream(
    websocket: WebSocket,
    producer: Producer,
    target_frame_ms: float = 33.0,
) -> bool:
    """Ejecuta ``producer`` en un hilo y transmite sus frames hasta terminar o ser cancelado.

    Devuelve True si terminó de forma natural y False si el cliente canceló.
    """
    cancel = asyncio.Event()

    async def listen_for_cancel() -> None:
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                if msg.get("type") == "cancel":
                    cancel.set()
                    return
        except (WebSocketDisconnect, json.JSONDecodeError):
            cancel.set()

    listener = asyncio.create_task(listen_for_cancel())
    gen = producer()
    done = object()
    completed = True
    try:
        while not cancel.is_set():
            t0 = time.perf_counter()
            item = await asyncio.to_thread(next, gen, done)
            if item is done:
                break
            step, field, metrics = item
            metrics = {**metrics, "step": step, "computeTimeMs": (time.perf_counter() - t0) * 1000.0}
            await websocket.send_text(json.dumps({"type": "stream_frame", **metrics}))
            if field is not None:
                await websocket.send_bytes(encode_field_frame(step, field))
            # Respeta la cadencia objetivo sin bloquear el event loop.
            spare = target_frame_ms / 1000.0 - (time.perf_counter() - t0)
            if spare > 0:
                await asyncio.sleep(spare)
        completed = not cancel.is_set()
    finally:
        listener.cancel()
        gen.close()
    await websocket.send_text(json.dumps({"type": "stream_end", "completed": completed}))
    return completed
