"""WebSocket endpoint para streaming y cálculo interactivo de física en tiempo real.

Permite que cuando el usuario arrastre partículas en el canvas 2D o en la escena 3D,
el backend resuelva instantáneamente las leyes de Coulomb o el tensor/grilla
de campo eléctrico E(x, y), garantizando precisión numérica a 60 fps.
"""

from __future__ import annotations

import json
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from domain.vectors import Vector2D, Vector3D
from domain.particles import Particle
from domain.coulomb import all_forces
from domain.electric_field import compute_field_grid_2d
from api.schemas.physics import (
    PhysicsRequest,
    PhysicsResponse,
    CoulombForceSchema,
    FieldPointSchema,
)

logger = logging.getLogger("physics_ws")
ws_router = APIRouter()


def _schema_to_particle(schema) -> Particle:
    pos = schema.position
    if hasattr(pos, "z") and pos.z is not None:
        vec = Vector3D(x=pos.x, y=pos.y, z=pos.z)
    else:
        vec = Vector2D(x=pos.x, y=pos.y)

    return Particle(
        position=vec,
        charge=schema.charge,
        label=schema.label,
        id=schema.id,
    )


@ws_router.websocket("/ws/physics")
async def websocket_physics_endpoint(websocket: WebSocket):
    """Maneja la conexión WebSocket bidireccional continua con el cliente."""
    await websocket.accept()
    logger.info("Cliente WebSocket conectado a /ws/physics")

    try:
        while True:
            raw_text = await websocket.receive_text()
            t0 = time.perf_counter()

            try:
                data = json.loads(raw_text)
                request = PhysicsRequest.model_validate(data)
            except Exception as parse_err:
                await websocket.send_text(
                    json.dumps({"error": f"Payload inválido: {str(parse_err)}"})
                )
                continue

            domain_particles = [_schema_to_particle(p) for p in request.particles]

            if request.type == "coulomb":
                if len(domain_particles) < 2:
                    response = PhysicsResponse(
                        type="coulomb_result",
                        forces=[],
                        computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                    )
                else:
                    try:
                        forces_domain = all_forces(domain_particles)
                        schema_forces = [
                            CoulombForceSchema(
                                force=f.force.to_dict(),
                                forceMagnitude=f.force_magnitude,
                                distance=f.distance,
                                unitVector=f.unit_vector.to_dict(),
                                isAttractive=f.is_attractive,
                                particleFromId=f.particle_from_id,
                                particleToId=f.particle_to_id,
                            )
                            for f in forces_domain
                        ]
                        response = PhysicsResponse(
                            type="coulomb_result",
                            forces=schema_forces,
                            computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                        )
                    except ValueError as math_err:
                        # Si las partículas colisionan o r -> 0
                        await websocket.send_text(
                            json.dumps({"error": str(math_err), "code": "SINGULARITY"})
                        )
                        continue

            elif request.type == "electric_field":
                query = request.fieldQuery
                x_range = query.xRange if query else (-0.2, 0.2)
                y_range = query.yRange if query else (-0.2, 0.2)
                resolution = query.resolution if query else 20

                field_pts = compute_field_grid_2d(
                    sources=domain_particles,
                    x_range=x_range,
                    y_range=y_range,
                    resolution=resolution,
                )

                schema_pts = [
                    FieldPointSchema(
                        point=fp.point.to_dict(),
                        E=fp.E.to_dict(),
                        magnitude=fp.magnitude,
                    )
                    for fp in field_pts
                ]

                response = PhysicsResponse(
                    type="electric_field_result",
                    field=schema_pts,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )
            else:
                await websocket.send_text(
                    json.dumps({"error": f"Tipo de cálculo desconocido: {request.type}"})
                )
                continue

            await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        logger.info("Cliente WebSocket desconectado")
    except Exception as e:
        logger.error(f"Error inesperado en WebSocket: {e}", exc_info=True)
