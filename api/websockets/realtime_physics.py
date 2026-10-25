"""WebSocket endpoint para streaming y cálculo interactivo de física en tiempo real.

Permite que cuando el usuario interactúe con los controles, mueva cargas, lance partículas
en ciclotrones, ajuste celdas de metamateriales o pulse la onda en la rejilla Yee FDTD,
el backend resuelva instantáneamente las leyes electromagnéticas garantizando rigor
matemático, estabilidad numérica y verificación analítica a 60 fps.
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
from domain.dynamics import (
    DynamicParticle,
    UniformElectricField,
    simulate_trajectory,
    verify_thomson_experiment,
)
from domain.magnetic import (
    simulate_lorentz_trajectory_2d,
    verify_cyclotron_physics,
    verify_wien_filter,
)
from domain.em_wave import (
    PlaneWave2D,
    evaluate_wave_grid_2d,
    verify_vacuum_wave_speed,
    verify_orthogonality_and_equipartition,
    verify_poynting_impedance,
)
from domain.metamaterials import (
    MetamaterialUnitCell,
    compute_spectral_response,
    verify_energy_conservation_spectrum,
    verify_veselago_negative_refraction,
    verify_bragg_bandgap,
)
from domain.fdtd2d import (
    FDTD2DSimulator,
    verify_cfl_stability_condition,
    verify_fdtd_pulse_velocity,
)
from api.schemas.physics import (
    PhysicsRequest,
    PhysicsResponse,
    CoulombForceSchema,
    FieldPointSchema,
    TrajectorySnapshotSchema,
    LorentzSnapshotSchema,
    EMWaveSnapshotSchema,
    SpectralPointSchema,
    Vector2DSchema,
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


def _schema_to_dynamic_particle(schema) -> DynamicParticle:
    part = _schema_to_particle(schema.particle)
    v = schema.velocity
    if hasattr(v, "z") and v.z is not None:
        vel = Vector3D(x=v.x, y=v.y, z=v.z)
    else:
        vel = Vector2D(x=v.x, y=v.y)
    return DynamicParticle(particle=part, mass=schema.mass, velocity=vel)


def _schema_to_uniform_field(schema) -> UniformElectricField:
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


@ws_router.websocket("/ws/physics")
async def websocket_physics_endpoint(websocket: WebSocket):
    """Maneja la conexión WebSocket bidireccional continua con el cliente."""
    await websocket.accept()
    logger.info("Cliente WebSocket conectado a /ws/physics")

    # Instancia de simulación FDTD en memoria por cada conexión activa
    fdtd_sim: FDTD2DSimulator | None = None

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

            # ═════════════════════════════════════════════════════════
            # CAPÍTULOS 1, 2, 3: COULOMB
            # ═════════════════════════════════════════════════════════
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
                        await websocket.send_text(
                            json.dumps({"error": str(math_err), "code": "SINGULARITY"})
                        )
                        continue

            # ═════════════════════════════════════════════════════════
            # CAPÍTULO 4: CAMPO ELÉCTRICO CONTINUO
            # ═════════════════════════════════════════════════════════
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

            # ═════════════════════════════════════════════════════════
            # CAPÍTULO 5: CINEMÁTICA Y DEFLEXIÓN
            # ═════════════════════════════════════════════════════════
            elif request.type == "trajectory_sim":
                if not request.trajectoryQuery:
                    await websocket.send_text(
                        json.dumps({"error": "trajectoryQuery es requerido para type 'trajectory_sim'"})
                    )
                    continue

                t_query = request.trajectoryQuery
                dyn_p = _schema_to_dynamic_particle(t_query.dynamicParticle)
                uniform_fields = (
                    [_schema_to_uniform_field(u) for u in t_query.uniformFields]
                    if t_query.uniformFields
                    else []
                )

                snapshots = simulate_trajectory(
                    dynamic_particle=dyn_p,
                    sources=domain_particles,
                    uniform_fields=uniform_fields,
                    total_time=t_query.totalTime,
                    dt=t_query.dt,
                )

                schema_snaps = [
                    TrajectorySnapshotSchema(
                        time=s.time,
                        position=s.position.to_dict(),
                        velocity=s.velocity.to_dict(),
                        acceleration=s.acceleration.to_dict(),
                        speed=s.speed,
                        kineticEnergy=s.kinetic_energy,
                        potentialEnergy=s.potential_energy,
                        totalEnergy=s.total_energy,
                    )
                    for s in snapshots
                ]

                response = PhysicsResponse(
                    type="trajectory_result",
                    trajectory=schema_snaps,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            elif request.type == "ch5_thomson":
                t_q = request.thomsonQuery
                charge = t_q.charge if t_q else -1.602e-19
                mass = t_q.mass if t_q else 9.109e-31
                v0x = t_q.v0x if t_q else 1e7
                Ey = t_q.Ey if t_q else 2000.0
                length = t_q.length if t_q else 0.05
                dt = t_q.dt if t_q else 1e-11

                thomson_res = verify_thomson_experiment(
                    charge=charge,
                    mass=mass,
                    v0x=v0x,
                    Ey=Ey,
                    length=length,
                    dt=dt,
                )

                response = PhysicsResponse(
                    type="ch5_thomson_result",
                    thomsonResult=thomson_res,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            # ═════════════════════════════════════════════════════════
            # CAPÍTULO 6: ONDAS Y CAMPO MAGNÉTICO (2D)
            # ═════════════════════════════════════════════════════════
            elif request.type == "ch6_magnetic_sim":
                l_q = request.lorentzQuery
                if not l_q:
                    await websocket.send_text(json.dumps({"error": "lorentzQuery es requerido"}))
                    continue

                p0 = Vector2D(l_q.initialPos.x, l_q.initialPos.y)
                v0 = Vector2D(l_q.initialVel.x, l_q.initialVel.y)
                e_vec = Vector2D(l_q.E.x, l_q.E.y)

                snaps = simulate_lorentz_trajectory_2d(
                    charge=l_q.charge,
                    mass=l_q.mass,
                    initial_pos=p0,
                    initial_vel=v0,
                    E=e_vec,
                    Bz=l_q.Bz,
                    total_time=l_q.totalTime,
                    dt=l_q.dt,
                )

                schema_l = [
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

                response = PhysicsResponse(
                    type="ch6_magnetic_result",
                    lorentzTrajectory=schema_l,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            elif request.type == "ch6_cyclotron_verify":
                c_q = request.cyclotronQuery
                mass = c_q.mass if c_q else 1e-6
                charge = c_q.charge if c_q else 1e-6
                speed = c_q.speed if c_q else 50.0
                Bz = c_q.Bz if c_q else 0.5
                dt = c_q.dt if c_q else 0.00005

                res_cyc = verify_cyclotron_physics(mass, charge, speed, Bz, dt)
                response = PhysicsResponse(
                    type="ch6_cyclotron_result",
                    cyclotronResult=res_cyc,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            elif request.type == "ch6_wien_verify":
                w_q = request.wienQuery
                mass = w_q.mass if w_q else 1e-6
                charge = w_q.charge if w_q else 1e-6
                speed = w_q.speed if w_q else 1000.0
                Ey = w_q.Ey if w_q else 100.0
                Bz = w_q.Bz if w_q else 0.1
                length = w_q.length if w_q else 0.2
                dt = w_q.dt if w_q else 0.00005

                res_wien = verify_wien_filter(mass, charge, speed, Ey, Bz, length, dt)
                response = PhysicsResponse(
                    type="ch6_wien_result",
                    wienResult=res_wien,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            # ═════════════════════════════════════════════════════════
            # CAPÍTULO 7: ONDA ELECTROMAGNÉTICA IDEAL EN EL VACÍO (2D)
            # ═════════════════════════════════════════════════════════
            elif request.type == "ch7_em_wave_grid":
                w_q = request.emWaveQuery
                amp = w_q.amplitudeE0 if w_q else 100.0
                freq = w_q.frequency if w_q else 1.0e9
                angle = w_q.propagationAngleRad if w_q else 0.0
                t_val = w_q.time if w_q else 0.0
                res = w_q.resolution if w_q else 16
                xr = w_q.xRange if w_q else (-0.15, 0.15)
                yr = w_q.yRange if w_q else (-0.15, 0.15)

                wave = PlaneWave2D(
                    amplitude_E0=amp,
                    frequency=freq,
                    propagation_angle_rad=angle,
                )
                grid_points = evaluate_wave_grid_2d(wave, xr, yr, t_val, res)

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

                response = PhysicsResponse(
                    type="ch7_em_wave_result",
                    emWaveGrid=schema_grid,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            elif request.type == "ch7_em_verify":
                v_q = request.emWaveVerifyQuery
                amp = v_q.amplitudeE0 if v_q else 100.0
                freq = v_q.frequency if v_q else 1.0e9
                ang = v_q.angleRad if v_q else 0.5236

                speed_res = verify_vacuum_wave_speed(freq)
                ortho_res = verify_orthogonality_and_equipartition(amp, freq, ang)
                poynt_res = verify_poynting_impedance(amp, freq)

                combined_res = {
                    "wave_speed": speed_res,
                    "orthogonality_equipartition": ortho_res,
                    "poynting_impedance": poynt_res,
                    "is_all_verified": speed_res["is_verified"] and ortho_res["is_ratio_verified"] and poynt_res["is_verified"],
                }

                response = PhysicsResponse(
                    type="ch7_em_verify_result",
                    emWaveVerifyResult=combined_res,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            # ═════════════════════════════════════════════════════════
            # CAPÍTULO 8: INTRODUCCIÓN A METAMATERIALES
            # ═════════════════════════════════════════════════════════
            elif request.type == "ch8_metamaterials_spectrum":
                m_q = request.metaQuery
                if not m_q:
                    await websocket.send_text(json.dumps({"error": "metaQuery es requerido"}))
                    continue

                cell = MetamaterialUnitCell(
                    a=m_q.cell.a,
                    d=m_q.cell.d,
                    w=m_q.cell.w,
                    periods=m_q.cell.periods,
                    loss_factor=m_q.cell.lossFactor,
                )

                spectrum_pts = compute_spectral_response(
                    cell=cell,
                    freq_min_ghz=m_q.freqMinGhz,
                    freq_max_ghz=m_q.freqMaxGhz,
                    points=m_q.points,
                )

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
                    for sp in spectrum_pts
                ]

                response = PhysicsResponse(
                    type="ch8_metamaterials_result",
                    spectrum=schema_spec,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            elif request.type == "ch8_metamaterials_verify":
                m_q = request.metaQuery
                c_sch = m_q.cell if m_q else None
                cell = MetamaterialUnitCell(
                    a=c_sch.a if c_sch else 10.0,
                    d=c_sch.d if c_sch else 2.0,
                    w=c_sch.w if c_sch else 1.0,
                    periods=c_sch.periods if c_sch else 5,
                    loss_factor=c_sch.lossFactor if c_sch else 0.002,
                )

                energy_res = verify_energy_conservation_spectrum(cell)
                veselago_res = verify_veselago_negative_refraction(cell, incident_angle_deg=30.0)
                bragg_res = verify_bragg_bandgap(cell_size_mm=cell.a, host_eps_r=2.2)

                combined_meta = {
                    "energy_conservation": energy_res,
                    "veselago_refraction": veselago_res,
                    "bragg_bandgap": bragg_res,
                    "is_all_verified": energy_res["is_conserved"] and veselago_res["is_negative_refraction"],
                }

                response = PhysicsResponse(
                    type="ch8_metamaterials_verify_result",
                    metamaterialsVerifyResult=combined_meta,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            # ═════════════════════════════════════════════════════════
            # CAPÍTULO 9: INTRODUCCIÓN A FDTD
            # ═════════════════════════════════════════════════════════
            elif request.type == "ch9_fdtd_step":
                f_q = request.fdtdQuery
                nx = f_q.Nx if f_q else 80
                ny = f_q.Ny if f_q else 50
                dx = f_q.dx if f_q else 0.002
                courant = f_q.courantFactor if f_q else 0.5
                s_type = f_q.sourceType if f_q else "gaussian"
                s_freq = f_q.sourceFreqGhz if f_q else 10.0
                steps_count = f_q.steps if f_q else 5
                should_reset = f_q.reset if f_q else False

                if fdtd_sim is None or should_reset or fdtd_sim.Nx != nx or fdtd_sim.Ny != ny:
                    fdtd_sim = FDTD2DSimulator(
                        Nx=nx,
                        Ny=ny,
                        dx=dx,
                        courant_factor=courant,
                        source_type=s_type,
                        source_freq_ghz=s_freq,
                    )
                    if f_q and f_q.addSlab:
                        fdtd_sim.add_dielectric_slab(f_q.slabStart, f_q.slabEnd, f_q.slabEps)
                    elif f_q and f_q.addMetamaterial:
                        fdtd_sim.add_metamaterial_periodic_array(start_x=30, periods=4)

                fdtd_sim.step(steps_count)
                slice_grid = fdtd_sim.get_field_slice_subsampled(step_stride=2)
                fft_spec = fdtd_sim.compute_transmission_spectrum()

                response = PhysicsResponse(
                    type="ch9_fdtd_result",
                    fdtdGridEz=slice_grid,
                    fdtdTimeStep=fdtd_sim.time_step,
                    fdtdCurrentTime=fdtd_sim.current_time,
                    fdtdDt=fdtd_sim.dt,
                    fdtdSpectrum=fft_spec,
                    computeTimeMs=(time.perf_counter() - t0) * 1000.0,
                )

            elif request.type == "ch9_fdtd_verify":
                cfl_res = verify_cfl_stability_condition(0.5)
                cfl_unstable = verify_cfl_stability_condition(1.05)
                pulse_res = verify_fdtd_pulse_velocity(Nx=60, Ny=30, dx=0.003)

                combined_fdtd = {
                    "cfl_stable_test": cfl_res,
                    "cfl_unstable_test": cfl_unstable,
                    "pulse_speed_test": pulse_res,
                    "is_all_verified": cfl_res["theoretically_stable"] and pulse_res["is_verified"],
                }

                response = PhysicsResponse(
                    type="ch9_fdtd_verify_result",
                    fdtdVerifyResult=combined_fdtd,
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
