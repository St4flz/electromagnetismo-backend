"""WebSocket endpoint /ws/fdtd para simulación interactiva con streaming binario y control por el servidor."""

from __future__ import annotations

import json
from typing import Any, Iterator
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.websockets.stream import run_stream
from domain.fdtd import (
    FDTDSimulation,
    UnitCellGeometry,
    CellFamily,
    rasterize_unit_cell,
)

router = APIRouter(prefix="/ws", tags=["WebSocket FDTD"])


def create_fdtd_producer(
    config_dict: dict[str, Any],
    steps_per_frame: int = 8,
) -> tuple[FDTDSimulation, Any]:
    """Crea una simulación FDTD configurada y un generador que produce frames de visualización."""
    geom_cfg = config_dict.get("geometry", {})
    dx = float(config_dict.get("dx_mm", 0.5)) * 1e-3
    dy = dx
    a_m = float(geom_cfg.get("a_mm", 10.0)) * 1e-3
    Ny = max(int(round(a_m / dy)), 4)

    periods = int(geom_cfg.get("periods", 3))
    cell_len_m = a_m * periods
    cell_cells_x = int(round(cell_len_m / dx))
    pml_thickness = int(config_dict.get("pml_thickness", 16))
    margin_x = int(round(0.040 / dx))
    Nx = pml_thickness * 2 + 30 + margin_x + cell_cells_x

    sim = FDTDSimulation(
        Nx=Nx,
        Ny=Ny,
        dx=dx,
        dy=dy,
        courant_factor=float(config_dict.get("courant_factor", 0.70)),
        pml_thickness=pml_thickness,
        f_min_ghz=float(config_dict.get("f_min_ghz", 5.0)),
        f_max_ghz=float(config_dict.get("f_max_ghz", 15.0)),
    )

    # 1. Calibración en vacío
    sim.calibrate_incident_wave(max_steps=1400)

    # 2. Configurar estructura con reparación automática ante valores límite
    fam = CellFamily.ROD_ARRAY if geom_cfg.get("family") == "rod_array" else CellFamily.SQUARE_RING
    a_mm = max(2.0, float(geom_cfg.get("a_mm", 10.0)))
    d_mm = max(0.0, float(geom_cfg.get("d_mm", 3.0)))
    w_mm = max(0.0, min(a_mm, float(geom_cfg.get("w_mm", 1.0))))

    if fam == CellFamily.SQUARE_RING and w_mm > 0:
        if (a_mm - d_mm) <= 2.0 * w_mm:
            d_mm = max(0.5, a_mm - 2.0 * w_mm - 0.5)
            if (a_mm - d_mm) <= 2.0 * w_mm:
                w_mm = max(0.25, (a_mm - d_mm - 0.5) / 2.0)

    geom = UnitCellGeometry(
        a_mm=a_mm,
        d_mm=d_mm,
        w_mm=w_mm,
        periods=periods,
        family=fam,
        relative_permittivity=float(geom_cfg.get("relative_permittivity", 9.8)),
        conductivity=float(geom_cfg.get("conductivity", 0.0)),
        is_pec=bool(geom_cfg.get("is_pec", False)),
    )
    x_start = sim.i_tfsf + 15
    rasterize_unit_cell(geom, sim.materials, x_start_idx=x_start)
    sim.reset()

    max_steps = int(config_dict.get("max_steps", 2000))

    def producer() -> Iterator[tuple[int, np.ndarray | None, dict[str, Any]]]:
        current_step = 0
        while current_step < max_steps:
            # Avanzar bloques de pasos para maximizar FPS y fluidez de render
            for _ in range(steps_per_frame):
                sim.step()
                current_step += 1

            u_ratio = (
                (sim.energy_history[-1] / sim.max_energy_recorded)
                if sim.max_energy_recorded > 1e-18
                else 1.0
            )

            metrics = {
                "step": current_step,
                "max_steps": max_steps,
                "time_ns": round(sim.grid.current_time * 1e9, 3),
                "energy_ratio": round(float(u_ratio), 5),
                "max_field": round(float(np.max(np.abs(sim.grid.Ez))), 4),
            }

            # Enviar el campo Ez 2D completo (orden C)
            yield current_step, sim.grid.Ez.copy(), metrics

            # Criterio físico de decaimiento
            transit_steps = int(sim.Nx / max(sim.grid.courant_factor, 0.1))
            if current_step > transit_steps + 100 and u_ratio < 1e-4:
                break

    return sim, producer


@router.websocket("/fdtd")
async def websocket_fdtd_stream(websocket: WebSocket):
    """Maneja sesiones interactivas FDTD bidireccionales con streaming de frames binarios."""
    await websocket.accept()
    await websocket.send_text(json.dumps({
        "type": "connection_ready",
        "message": "FDTD v2 Real-Time Engine ready. Send {'type': 'start', 'config': {...}}",
    }))

    try:
        while True:
            text = await websocket.receive_text()
            data = json.loads(text)
            action = data.get("type")

            if action == "start":
                cfg = data.get("config", {})
                steps_per_frame = int(data.get("steps_per_frame", 6))
                try:
                    sim, producer = create_fdtd_producer(cfg, steps_per_frame=steps_per_frame)
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Error configurando FDTD: {exc}",
                    }))
                    continue

                # Notificar inicialización completada con metadatos de malla
                await websocket.send_text(json.dumps({
                    "type": "simulation_started",
                    "grid": {
                        "Nx": sim.Nx,
                        "Ny": sim.Ny,
                        "dx": sim.dx,
                        "dy": sim.dy,
                        "dt_ps": round(sim.grid.dt * 1e12, 3),
                        "courant": sim.grid.courant_factor,
                        "i_tfsf": sim.i_tfsf,
                        "x_refl": sim.monitor_refl.x_idx,
                        "x_trans": sim.monitor_trans.x_idx,
                    },
                }))

                completed = await run_stream(websocket, producer, target_frame_ms=25.0)

                if completed:
                    spec = sim.get_spectrum()
                    await websocket.send_text(json.dumps({
                        "type": "spectrum_ready",
                        "spectrum": {
                            "frequencies_ghz": spec.frequencies_ghz,
                            "transmission": spec.transmission,
                            "reflection": spec.reflection,
                            "absorption": spec.absorption,
                            "transmission_db": spec.transmission_db,
                        },
                    }))

            elif action == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
