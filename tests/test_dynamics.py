"""Pruebas unitarias para cinemática, dinámica y conservación de energía (Capítulo 5)."""

import math
from domain.vectors import Vector2D, Vector3D
from domain.particles import Particle
from domain.units import ELECTRON_MASS, ELEMENTARY_CHARGE
from domain.dynamics import (
    DynamicParticle,
    UniformElectricField,
    compute_net_force_at,
    simulate_trajectory,
    compute_drift_velocity,
    compute_current_density,
    compute_current,
)


def test_dynamic_particle_properties():
    """Valida cálculo de rapidez, momentum, energía cinética y relación q/m."""
    p = Particle(position=Vector2D(0.0, 0.0), charge=-ELEMENTARY_CHARGE, label="e-")
    dyn = DynamicParticle(
        particle=p,
        mass=ELECTRON_MASS,
        velocity=Vector2D(1e6, 0.0),  # 1,000 km/s
    )

    assert math.isclose(dyn.speed, 1e6)
    expected_ke = 0.5 * ELECTRON_MASS * (1e6 ** 2)
    assert math.isclose(dyn.kinetic_energy, expected_ke, rel_tol=1e-6)

    expected_q_over_m = -ELEMENTARY_CHARGE / ELECTRON_MASS
    assert math.isclose(dyn.charge_to_mass_ratio, expected_q_over_m, rel_tol=1e-6)


def test_ballistic_motion_under_uniform_electric_field():
    """Compara la trayectoria numérica (Verlet) contra la solución analítica de Thomson."""
    # Partícula de prueba lanzada horizontalmente con vx = 2.0 m/s
    # Campo eléctrico vertical uniforme: Ey = 100 V/m
    # Carga q = 1e-6 C, masa m = 1e-6 kg => a_y = q*Ey/m = 100 m/s²
    charge = 1e-6
    mass = 1e-6
    v0_x = 2.0
    field_y = 100.0

    p = Particle(position=Vector2D(0.0, 0.0), charge=charge, label="q")
    dyn = DynamicParticle(particle=p, mass=mass, velocity=Vector2D(v0_x, 0.0))

    uniform_e = UniformElectricField(field_vector=Vector2D(0.0, field_y))

    total_time = 0.05  # 50 ms
    dt = 0.001        # 1 ms
    snapshots = simulate_trajectory(
        dynamic_particle=dyn,
        uniform_fields=[uniform_e],
        total_time=total_time,
        dt=dt,
    )

    last = snapshots[-1]
    t_final = last.time

    # Solución analítica exacta:
    # x(t) = v0_x * t
    # y(t) = 0.5 * a_y * t^2
    # vy(t) = a_y * t
    ay_expected = (charge * field_y) / mass
    x_expected = v0_x * t_final
    y_expected = 0.5 * ay_expected * (t_final ** 2)
    vy_expected = ay_expected * t_final

    assert math.isclose(last.position.x, x_expected, rel_tol=1e-5)
    assert math.isclose(last.position.y, y_expected, rel_tol=1e-5)
    assert math.isclose(last.velocity.y, vy_expected, rel_tol=1e-5)


def test_energy_conservation_in_conservative_field():
    """Verifica que el integrador Velocity-Verlet conserve la energía total mecánica."""
    # Partícula cargada que se aproxima a una carga fija del mismo signo (repulsión Coulomb)
    source = Particle(position=Vector2D(0.1, 0.0), charge=2e-6, label="Q_fija")
    probe = Particle(position=Vector2D(0.0, 0.0), charge=1e-6, label="q_movil")

    # Lanzada hacia la fuente con velocidad inicial vx = 5.0 m/s
    dyn = DynamicParticle(particle=probe, mass=1e-4, velocity=Vector2D(5.0, 0.0))

    snapshots = simulate_trajectory(
        dynamic_particle=dyn,
        sources=[source],
        total_time=0.015,
        dt=0.0001,
    )

    initial_energy = snapshots[0].total_energy

    # Verificar que en cada paso de tiempo la energía total se conserve dentro de un margen estrecho
    for snap in snapshots:
        assert math.isclose(snap.total_energy, initial_energy, rel_tol=1e-2), (
            f"Deriva energética en t={snap.time}: {snap.total_energy} vs inicial {initial_energy}"
        )


def test_current_and_drift_velocity():
    """Verifica las ecuaciones de conducción macroscópica de cargas."""
    mobility = 0.0045        # [m²/(V·s)] típico en silicio
    E_mag = 1000.0           # [V/m]
    vd = compute_drift_velocity(mobility, E_mag)
    assert math.isclose(vd, 4.5)  # [m/s]

    carrier_density = 1e20   # [m⁻³]
    carrier_charge = ELEMENTARY_CHARGE
    J = compute_current_density(carrier_density, carrier_charge, vd)
    expected_J = 1e20 * ELEMENTARY_CHARGE * 4.5
    assert math.isclose(J, expected_J, rel_tol=1e-6)

    area = 1e-4  # 1 cm² = 1e-4 m²
    I = compute_current(J, area)
    assert math.isclose(I, expected_J * area, rel_tol=1e-6)


def test_verify_thomson_experiment_script():
    from domain.dynamics import verify_thomson_experiment
    res = verify_thomson_experiment(
        charge=-1.602e-19,
        mass=9.109e-31,
        v0x=1e7,
        Ey=2000.0,
        length=0.05,
        dt=1e-11,
    )
    assert res["is_verified"]
    assert res["relative_error_percent"] < 0.5

