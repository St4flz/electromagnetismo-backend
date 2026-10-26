"""WebSocket endpoint /ws/optimizer para streaming en tiempo real del Algoritmo Genético (Capítulo 12)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from domain.optimization.search_space import SearchSpace, ParameterBounds
from domain.optimization.evaluator import FDTDEvaluator
from domain.optimization.genetic import evolve_optimizer

router = APIRouter(prefix="/ws", tags=["WebSocket Optimizer"])


@router.websocket("/optimizer")
async def optimizer_websocket_endpoint(websocket: WebSocket):
    """Permite iniciar, pausar y cancelar la optimización genética recibiendo reportes por generación."""
    await websocket.accept()

    # Estado de la conexión
    state = {
        "running": False,
        "paused": False,
        "stop_requested": False,
    }

    async def listen_commands():
        nonlocal state
        try:
            while True:
                msg_text = await websocket.receive_text()
                try:
                    payload = json.loads(msg_text)
                    action = payload.get("action")
                    if action == "stop":
                        state["stop_requested"] = True
                        state["running"] = False
                    elif action == "pause":
                        state["paused"] = True
                    elif action == "resume":
                        state["paused"] = False
                except Exception:
                    pass
        except WebSocketDisconnect:
            state["stop_requested"] = True
            state["running"] = False

    cmd_task = asyncio.create_task(listen_commands())

    try:
        # Esperar mensaje de inicio
        init_raw = await websocket.receive_text()
        init_data = json.loads(init_raw)

        if init_data.get("action") != "start":
            await websocket.send_json({"type": "error", "message": "Primer comando debe ser 'start'"})
            await websocket.close()
            cmd_task.cancel()
            return

        cfg = init_data.get("config", {})
        family = cfg.get("family", "square_ring")
        pop_size = int(cfg.get("pop_size", 16))
        generations = int(cfg.get("generations", 15))
        seed = int(cfg.get("seed", 42))
        f0_ghz = float(cfg.get("f0_ghz", 10.0))
        periods = int(cfg.get("periods", 3))
        dx_mm = float(cfg.get("dx_mm", 0.5))

        space = SearchSpace(
            family=family,
            a_bounds=ParameterBounds(
                min_val=float(cfg.get("a_min", 6.0)),
                max_val=float(cfg.get("a_max", 16.0)),
                step=1.0,
            ),
            d_bounds=ParameterBounds(
                min_val=float(cfg.get("d_min", 2.0)),
                max_val=float(cfg.get("d_max", 8.0)),
                step=0.5,
            ),
            w_bounds=ParameterBounds(
                min_val=float(cfg.get("w_min", 0.5)),
                max_val=float(cfg.get("w_max", 2.5)),
                step=0.25,
            ),
            dx_mm=dx_mm,
        )

        evaluator = FDTDEvaluator(
            family=family,
            periods=periods,
            is_pec=bool(cfg.get("is_pec", True)),
            relative_permittivity=float(cfg.get("relative_permittivity", 1.0)),
            dx_mm=dx_mm,
            f0_ghz=f0_ghz,
            max_steps=1400,
        )

        state["running"] = True
        t0 = time.perf_counter()

        # Ciclo de evolución
        for report in evolve_optimizer(
            evaluator=evaluator,
            space=space,
            pop_size=pop_size,
            generations=generations,
            seed=seed,
        ):
            if state["stop_requested"]:
                break

            while state["paused"] and not state["stop_requested"]:
                await asyncio.sleep(0.1)

            # Enviar actualización de la generación
            await websocket.send_json({
                "type": "generation_update",
                "data": {
                    "generation": report.generation,
                    "max_generations": generations,
                    "best_fitness": report.best_fitness,
                    "mean_fitness": report.mean_fitness,
                    "best_individual": report.best_individual,
                    "best_design_point": report.best_design_point.__dict__,
                    "population": report.population,
                    "fitness_values": report.fitness_values,
                    "total_evaluations": report.total_evaluations,
                    "cache_hits": report.cache_hits,
                    "cache_hit_rate": round(evaluator.stats.hit_rate_percent, 1),
                }
            })

            # Ceder brevemente el control al event loop
            await asyncio.sleep(0.01)

        total_elapsed = time.perf_counter() - t0
        if not state["stop_requested"]:
            await websocket.send_json({
                "type": "optimization_completed",
                "data": {
                    "status": "completed",
                    "total_time_seconds": round(total_elapsed, 2),
                    "total_evaluations": evaluator.stats.total_requested,
                    "cache_hits": evaluator.stats.cache_hits,
                    "cache_hit_rate": round(evaluator.stats.hit_rate_percent, 1),
                }
            })
        else:
            await websocket.send_json({
                "type": "optimization_stopped",
                "data": {
                    "status": "cancelled",
                    "total_time_seconds": round(total_elapsed, 2),
                }
            })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        cmd_task.cancel()
