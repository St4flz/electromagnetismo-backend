"""Endpoints REST para consultas y cálculos de física electromagnética.

Permite consultar constantes universales y realizar cálculos deterministas
de fuerzas de Coulomb y superposición de campo eléctrico.
"""

from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException

from domain.vectors import Vector2D, Vector3D
from domain.particles import Particle
from domain.coulomb import all_forces, CoulombResult
from domain.electric_field import (
    compute_field_grid_2d,
    electric_field_superposition,
    FieldPoint,
)
from domain import units

from api.schemas.physics import (
    ParticleSchema,
    CoulombForceSchema,
    FieldPointSchema,
    FieldGridQuery,
    PhysicsResponse,
)

router = APIRouter(prefix="/physics", tags=["Physics"])


def schema_to_particle(schema: ParticleSchema) -> Particle:
    """Convierte un ParticleSchema a la entidad de dominio Particle."""
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


def coulomb_result_to_schema(res: CoulombResult) -> CoulombForceSchema:
    """Convierte un CoulombResult del dominio al schema de API."""
    return CoulombForceSchema(
        force=res.force.to_dict(),
        forceMagnitude=res.force_magnitude,
        distance=res.distance,
        unitVector=res.unit_vector.to_dict(),
        isAttractive=res.is_attractive,
        particleFromId=res.particle_from_id,
        particleToId=res.particle_to_id,
    )


def field_point_to_schema(fp: FieldPoint) -> FieldPointSchema:
    """Convierte un FieldPoint del dominio al schema de API."""
    return FieldPointSchema(
        point=fp.point.to_dict(),
        E=fp.E.to_dict(),
        magnitude=fp.magnitude,
    )


@router.get("/constants")
def get_constants():
    """Retorna las constantes fundamentales del electromagnetismo con su significado físico y unidades."""
    return {
        "coulomb_constant": {
            "symbol": "k",
            "value": units.COULOMB_CONSTANT,
            "unit": "N·m²/C²",
            "formula": "1 / (4πε₀)",
            "description": "Constante de Coulomb. Define la escala de la fuerza electrostática en el vacío.",
        },
        "epsilon_0": {
            "symbol": "ε₀",
            "value": units.EPSILON_0,
            "unit": "F/m",
            "description": "Permitividad eléctrica del vacío. Determina cómo se propaga el campo eléctrico.",
        },
        "mu_0": {
            "symbol": "μ₀",
            "value": units.MU_0,
            "unit": "H/m",
            "description": "Permeabilidad magnética del vacío. Determina cómo se propaga el campo magnético.",
        },
        "speed_of_light": {
            "symbol": "c",
            "value": units.SPEED_OF_LIGHT,
            "unit": "m/s",
            "formula": "1 / √(ε₀μ₀)",
            "description": "Velocidad de la luz y de cualquier onda electromagnética en el vacío.",
        },
        "elementary_charge": {
            "symbol": "e",
            "value": units.ELEMENTARY_CHARGE,
            "unit": "C",
            "description": "Magnitud de la carga del electrón/protón.",
        },
    }


@router.post("/coulomb", response_model=PhysicsResponse)
def calculate_coulomb_forces(particles: list[ParticleSchema]):
    """Calcula todas las interacciones de fuerza electrostática entre las cargas suministradas."""
    t0 = time.perf_counter()
    if len(particles) < 2:
        return PhysicsResponse(
            type="coulomb_result",
            forces=[],
            computeTimeMs=(time.perf_counter() - t0) * 1000.0,
        )

    domain_particles = [schema_to_particle(p) for p in particles]
    try:
        forces_domain = all_forces(domain_particles)
        schema_forces = [coulomb_result_to_schema(f) for f in forces_domain]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    compute_time_ms = (time.perf_counter() - t0) * 1000.0
    return PhysicsResponse(
        type="coulomb_result",
        forces=schema_forces,
        computeTimeMs=compute_time_ms,
    )


@router.post("/electric-field/grid", response_model=PhysicsResponse)
def calculate_electric_field_grid(
    particles: list[ParticleSchema],
    grid: FieldGridQuery,
):
    """Calcula el campo vectorial eléctrico E en una rejilla 2D debido a la superposición de cargas."""
    t0 = time.perf_counter()
    domain_particles = [schema_to_particle(p) for p in particles]

    field_domain = compute_field_grid_2d(
        sources=domain_particles,
        x_range=grid.xRange,
        y_range=grid.yRange,
        resolution=grid.resolution,
    )

    schema_points = [field_point_to_schema(fp) for fp in field_domain]
    compute_time_ms = (time.perf_counter() - t0) * 1000.0

    return PhysicsResponse(
        type="electric_field_result",
        field=schema_points,
        computeTimeMs=compute_time_ms,
    )
