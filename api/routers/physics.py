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
from domain.dynamics import (
    DynamicParticle,
    UniformElectricField,
    TrajectorySnapshot,
    simulate_trajectory,
)
from domain import units

from api.schemas.physics import (
    ParticleSchema,
    CoulombForceSchema,
    FieldPointSchema,
    FieldGridQuery,
    DynamicParticleSchema,
    UniformFieldSchema,
    TrajectorySnapshotSchema,
    TrajectoryQuery,
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


def schema_to_dynamic_particle(schema: DynamicParticleSchema) -> DynamicParticle:
    """Convierte DynamicParticleSchema a la entidad de dominio DynamicParticle."""
    part = schema_to_particle(schema.particle)
    v = schema.velocity
    if hasattr(v, "z") and v.z is not None:
        vel = Vector3D(x=v.x, y=v.y, z=v.z)
    else:
        vel = Vector2D(x=v.x, y=v.y)
    return DynamicParticle(particle=part, mass=schema.mass, velocity=vel)


def schema_to_uniform_field(schema: UniformFieldSchema) -> UniformElectricField:
    """Convierte UniformFieldSchema a entidad de dominio UniformElectricField."""
    fv = schema.fieldVector
    if hasattr(fv, "z") and fv.z is not None:
        vec = Vector3D(x=fv.x, y=fv.y, z=fv.z)
    else:
        vec = Vector2D(x=fv.x, y=fv.y)
    return UniformElectricField(
        field_vector=vec,
        x_bounds=schema.xBounds or (-float("inf"), float("inf")),
        y_bounds=schema.yBounds or (-float("inf"), float("inf")),
        z_bounds=schema.zBounds or (-float("inf"), float("inf")),
    )


def snapshot_to_schema(snap: TrajectorySnapshot) -> TrajectorySnapshotSchema:
    """Convierte TrajectorySnapshot a TrajectorySnapshotSchema."""
    return TrajectorySnapshotSchema(
        time=snap.time,
        position=snap.position.to_dict(),
        velocity=snap.velocity.to_dict(),
        acceleration=snap.acceleration.to_dict(),
        speed=snap.speed,
        kineticEnergy=snap.kinetic_energy,
        potentialEnergy=snap.potential_energy,
        totalEnergy=snap.total_energy,
    )


@router.post("/trajectory", response_model=PhysicsResponse)
def calculate_trajectory(
    query: TrajectoryQuery,
    sources: list[ParticleSchema] | None = None,
):
    """Calcula la trayectoria dinámica completa de una carga bajo campos y fuerzas de Coulomb."""
    t0 = time.perf_counter()
    dyn_particle = schema_to_dynamic_particle(query.dynamicParticle)
    domain_sources = [schema_to_particle(s) for s in sources] if sources else []
    domain_uniform = (
        [schema_to_uniform_field(u) for u in query.uniformFields]
        if query.uniformFields
        else []
    )

    snapshots = simulate_trajectory(
        dynamic_particle=dyn_particle,
        sources=domain_sources,
        uniform_fields=domain_uniform,
        total_time=query.totalTime,
        dt=query.dt,
    )

    schema_snapshots = [snapshot_to_schema(s) for s in snapshots]
    compute_time_ms = (time.perf_counter() - t0) * 1000.0

    return PhysicsResponse(
        type="trajectory_result",
        trajectory=schema_snapshots,
        computeTimeMs=compute_time_ms,
    )


# ═══════════════════════════════════════════════════════════════════
# CAPÍTULO 5: THOMSON
# ═══════════════════════════════════════════════════════════════════
from domain.dynamics import verify_thomson_experiment
from api.schemas.physics import ThomsonVerificationQuery

@router.post("/ch5/thomson", response_model=PhysicsResponse)
def verify_thomson(query: ThomsonVerificationQuery):
    t0 = time.perf_counter()
    res = verify_thomson_experiment(
        charge=query.charge,
        mass=query.mass,
        v0x=query.v0x,
        Ey=query.Ey,
        length=query.length,
        dt=query.dt,
    )
    return PhysicsResponse(
        type="ch5_thomson_result",
        thomsonResult=res,
        computeTimeMs=(time.perf_counter() - t0) * 1000.0,
    )


# ═══════════════════════════════════════════════════════════════════
# CAPÍTULO 6: LORENTZ Y CICLOTRÓN
# ═══════════════════════════════════════════════════════════════════
from domain.magnetic import (
    simulate_lorentz_trajectory_2d,
    verify_cyclotron_physics,
    verify_wien_filter,
)
from api.schemas.physics import (
    LorentzQuery,
    LorentzSnapshotSchema,
    CyclotronVerificationQuery,
    WienFilterQuery,
    Vector2DSchema,
)

@router.post("/ch6/lorentz", response_model=PhysicsResponse)
def calculate_lorentz(query: LorentzQuery):
    t0 = time.perf_counter()
    p0 = Vector2D(query.initialPos.x, query.initialPos.y)
    v0 = Vector2D(query.initialVel.x, query.initialVel.y)
    e_vec = Vector2D(query.E.x, query.E.y)

    snaps = simulate_lorentz_trajectory_2d(
        charge=query.charge,
        mass=query.mass,
        initial_pos=p0,
        initial_vel=v0,
        E=e_vec,
        Bz=query.Bz,
        total_time=query.totalTime,
        dt=query.dt,
    )

    schema_snaps = [
        LorentzSnapshotSchema(
            time=s.time,
            position=Vector2DSchema(x=s.position.x, y=s.position.y),
            velocity=Vector2DSchema(x=s.velocity.x, y=s.velocity.y),
            acceleration=Vector2DSchema(x=s.acceleration.x, y=s.acceleration.y),
            speed=s.speed,
            kineticEnergy=s.kinetic_energy,
            workDone=s.work_done,
        )
        for s in snaps
    ]

    return PhysicsResponse(
        type="ch6_magnetic_result",
        lorentzTrajectory=schema_snaps,
        computeTimeMs=(time.perf_counter() - t0) * 1000.0,
    )

@router.post("/ch6/cyclotron-verify", response_model=PhysicsResponse)
def verify_cyclotron(query: CyclotronVerificationQuery):
    t0 = time.perf_counter()
    res = verify_cyclotron_physics(query.mass, query.charge, query.speed, query.Bz, query.dt)
    return PhysicsResponse(
        type="ch6_cyclotron_result",
        cyclotronResult=res,
        computeTimeMs=(time.perf_counter() - t0) * 1000.0,
    )


# ═══════════════════════════════════════════════════════════════════
# CAPÍTULO 7: ONDA ELECTROMAGNÉTICA EN VACÍO
# ═══════════════════════════════════════════════════════════════════
from domain.em_wave import (
    PlaneWave2D,
    evaluate_wave_grid_2d,
    verify_vacuum_wave_speed,
    verify_orthogonality_and_equipartition,
    verify_poynting_impedance,
)
from api.schemas.physics import EMWaveGridQuery, EMWaveSnapshotSchema, EMWaveVerificationQuery

@router.post("/ch7/em-wave", response_model=PhysicsResponse)
def calculate_em_wave(query: EMWaveGridQuery):
    t0 = time.perf_counter()
    wave = PlaneWave2D(
        amplitude_E0=query.amplitudeE0,
        frequency=query.frequency,
        propagation_angle_rad=query.propagationAngleRad,
    )
    grid_points = evaluate_wave_grid_2d(wave, query.xRange, query.yRange, query.time, query.resolution)
    schema_grid = [
        EMWaveSnapshotSchema(
            point=Vector2DSchema(x=gp.point.x, y=gp.point.y),
            Ez=gp.Ez,
            B=Vector2DSchema(x=gp.B.x, y=gp.B.y),
            BMagnitude=gp.B_magnitude,
            poynting=Vector2DSchema(x=gp.poynting.x, y=gp.poynting.y),
            poyntingMagnitude=gp.poynting_magnitude,
            energyDensityE=gp.energy_density_E,
            energyDensityB=gp.energy_density_B,
            totalEnergyDensity=gp.total_energy_density,
        )
        for gp in grid_points
    ]
    return PhysicsResponse(
        type="ch7_em_wave_result",
        emWaveGrid=schema_grid,
        computeTimeMs=(time.perf_counter() - t0) * 1000.0,
    )


# ═══════════════════════════════════════════════════════════════════
# CAPÍTULO 8: METAMATERIALES
# ═══════════════════════════════════════════════════════════════════
from domain.metamaterials import (
    MetamaterialUnitCell,
    compute_spectral_response,
    verify_energy_conservation_spectrum,
    verify_veselago_negative_refraction,
    verify_bragg_bandgap,
)
from api.schemas.physics import MetamaterialsQuery, SpectralPointSchema

@router.post("/ch8/metamaterials", response_model=PhysicsResponse)
def calculate_metamaterials(query: MetamaterialsQuery):
    t0 = time.perf_counter()
    cell = MetamaterialUnitCell(
        a=query.cell.a,
        d=query.cell.d,
        w=query.cell.w,
        periods=query.cell.periods,
        loss_factor=query.cell.lossFactor,
    )
    pts = compute_spectral_response(cell, query.freqMinGhz, query.freqMaxGhz, query.points)
    schema_spec = [
        SpectralPointSchema(
            frequencyGhz=sp.frequency_ghz,
            transmission=sp.transmission,
            reflection=sp.reflection,
            absorption=sp.absorption,
            transmissionDb=sp.transmission_db,
            epsReal=sp.eps_real,
            epsImag=sp.eps_imag,
            muReal=sp.mu_real,
            muImag=sp.mu_imag,
            nReal=sp.n_real,
            nImag=sp.n_imag,
            isLeftHanded=sp.is_left_handed,
        )
        for sp in pts
    ]
    return PhysicsResponse(
        type="ch8_metamaterials_result",
        spectrum=schema_spec,
        computeTimeMs=(time.perf_counter() - t0) * 1000.0,
    )


# ═══════════════════════════════════════════════════════════════════
# CAPÍTULO 9: FDTD
# ═══════════════════════════════════════════════════════════════════
from domain.fdtd2d import verify_cfl_stability_condition, verify_fdtd_pulse_velocity

@router.get("/ch9/fdtd-verify", response_model=PhysicsResponse)
def verify_fdtd():
    t0 = time.perf_counter()
    cfl_res = verify_cfl_stability_condition(0.5)
    cfl_unstable = verify_cfl_stability_condition(1.05)
    pulse_res = verify_fdtd_pulse_velocity(Nx=60, Ny=30, dx=0.003)

    combined_fdtd = {
        "cfl_stable_test": cfl_res,
        "cfl_unstable_test": cfl_unstable,
        "pulse_speed_test": pulse_res,
        "is_all_verified": cfl_res["theoretically_stable"] and pulse_res["is_verified"],
    }

    return PhysicsResponse(
        type="ch9_fdtd_verify_result",
        fdtdVerifyResult=combined_fdtd,
        computeTimeMs=(time.perf_counter() - t0) * 1000.0,
    )


