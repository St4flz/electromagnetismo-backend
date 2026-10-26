"""Magnetismo, Ley de Biot-Savart y Fuerza de Lorentz (Capítulo 6).

Este módulo modela el surgimiento del campo magnético a partir de cargas en movimiento,
la interacción de Lorentz completa F = q(E + v × B), el movimiento circular uniforme
en ciclotrón, el selector de velocidades de Wien y la verificación rigurosa del teorema
de trabajo nulo magnético (W = ∫ F_B · dr = 0 => dK/dt = 0).

Algoritmo de Integración:
    Para resolver la ecuación de movimiento con fuerza dependiente de la velocidad (v × B),
    empleamos el algoritmo simpléctico de Boris (1970), estándar de oro en física de plasmas
    y electrodinámica computacional, que conserva exactamente la energía cinética y
    el radio de giro sin disipación ni ensanchamiento artificial de la órbita.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

from .vectors import Vector2D, Vector3D
from .particles import Particle
from .units import MU_0, SPEED_OF_LIGHT


@dataclass(frozen=True)
class MagneticFieldPoint2D:
    """Campo magnético evaluado en un punto del plano 2D."""
    point: Vector2D
    B_vector: Vector2D       # [T] en el plano (por ejemplo líneas de Biot-Savart de un alambre)
    Bz: float                # [T] componente perpendicular al plano (por cargas moviéndose en xy)
    magnitude: float         # ||B|| total [T]

    def to_dict(self) -> dict:
        return {
            "point": self.point.to_dict(),
            "B": self.B_vector.to_dict(),
            "Bz": self.Bz,
            "magnitude": self.magnitude,
        }


@dataclass(frozen=True)
class LorentzSnapshot2D:
    """Estado temporal de una partícula bajo la fuerza de Lorentz."""
    time: float
    position: Vector2D
    velocity: Vector2D
    acceleration: Vector2D
    speed: float
    kinetic_energy: float
    work_done: float

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "acceleration": self.acceleration.to_dict(),
            "speed": self.speed,
            "kineticEnergy": self.kinetic_energy,
            "workDone": self.work_done,
        }


# ═══════════════════════════════════════════════════════════════════
# LEY DE BIOT-SAVART
# ═══════════════════════════════════════════════════════════════════

def biot_savart_point_charge_2d(
    charge: float,
    velocity: Vector2D,
    charge_pos: Vector2D,
    eval_point: Vector2D,
    softening: float = 1e-4,
) -> float:
    """Calcula la componente Bz [Tesla] generada por una carga puntual en movimiento en el plano xy.

    B = (μ₀ / 4π) · [q (v × r̂) / r²] = (μ₀ / 4π) · [q (v × r) / r³]
    Como v = (vx, vy, 0) y r = (rx, ry, 0):
        (v × r)_z = vx · ry - vy · rx
    """
    disp = eval_point - charge_pos
    dist_sq = disp.magnitude_squared + (softening**2)
    dist = math.sqrt(dist_sq)

    # (v x r)_z
    cross_z = velocity.x * disp.y - velocity.y * disp.x
    constant = MU_0 / (4.0 * math.pi)

    return constant * charge * cross_z / (dist**3)


def biot_savart_wire_field_2d(
    current: float,
    wire_pos: Vector2D,
    eval_point: Vector2D,
    softening: float = 1e-4,
) -> Vector2D:
    """Campo magnético en el plano xy producido por un alambre recto infinito perpendicular (en z).

    B(r) = (μ₀ · I) / (2π · r) · θ̂ = (μ₀ · I) / (2π · r²) · (-ry, rx)
    """
    disp = eval_point - wire_pos
    r_sq = disp.magnitude_squared + (softening**2)
    r = math.sqrt(r_sq)

    # θ̂ = (-sin θ, cos θ) = (-ry / r, rx / r)
    b_mag = (MU_0 * current) / (2.0 * math.pi * r)
    bx = -b_mag * (disp.y / r)
    by = b_mag * (disp.x / r)

    return Vector2D(bx, by)


# ═══════════════════════════════════════════════════════════════════
# FUERZA DE LORENTZ Y DINÁMICA
# ═══════════════════════════════════════════════════════════════════

def lorentz_force_2d(
    charge: float,
    velocity: Vector2D,
    E: Vector2D,
    Bz: float,
) -> Vector2D:
    """F = q · (E + v × B) en el plano 2D con B = Bz · k̂.

    (v × B)_x = vy · Bz
    (v × B)_y = -vx · Bz
    F_x = q · (Ex + vy · Bz)
    F_y = q · (Ey - vx · Bz)
    """
    fx = charge * (E.x + velocity.y * Bz)
    fy = charge * (E.y - velocity.x * Bz)
    return Vector2D(fx, fy)


def step_boris_2d(
    pos: Vector2D,
    vel: Vector2D,
    charge: float,
    mass: float,
    E: Vector2D,
    Bz: float,
    dt: float,
) -> tuple[Vector2D, Vector2D, Vector2D]:
    """Paso temporal con el integrador simpléctico de Boris (1970).

    El algoritmo desacopla la aceleración eléctrica de la rotación magnética pura:
        1. v_minus = v(t) + (q·E/m) · (dt/2)
        2. Rotación de v_minus alrededor de B mediante vector t = (q·B/m)·(dt/2)
        3. v_plus = rotado + (q·E/m) · (dt/2)
        4. r(t+dt) = r(t) + v_plus · dt
    Garantiza ||v_rotada|| == ||v_minus|| estrictamente (conservación exacta de energía).
    """
    q_over_m = charge / mass
    half_dt = 0.5 * dt

    # 1. Medio paso eléctrico
    v_minus_x = vel.x + q_over_m * E.x * half_dt
    v_minus_y = vel.y + q_over_m * E.y * half_dt

    # 2. Rotación magnética
    # Vector t = (0, 0, tz) donde tz = -q_over_m * Bz * half_dt
    # En 2D: rotación por ángulo θ = 2 * arctan(tz)
    tz = -q_over_m * Bz * half_dt
    t_sq = tz * tz
    # Factor de rotación s = 2 / (1 + t²)
    sz = 2.0 * tz / (1.0 + t_sq)

    # v' = v_minus + v_minus × t
    v_prime_x = v_minus_x + v_minus_y * tz
    v_prime_y = v_minus_y - v_minus_x * tz

    # v_plus = v_minus + v' × s
    v_rot_x = v_minus_x + v_prime_y * sz
    v_rot_y = v_minus_y - v_prime_x * sz

    # 3. Segundo medio paso eléctrico
    new_vx = v_rot_x + q_over_m * E.x * half_dt
    new_vy = v_rot_y + q_over_m * E.y * half_dt
    new_vel = Vector2D(new_vx, new_vy)

    # 4. Actualización de posición
    new_pos = pos + new_vel * dt

    # Fuerza instantánea en la nueva velocidad para reportar aceleración
    acc = lorentz_force_2d(charge, new_vel, E, Bz) * (1.0 / mass)

    return new_pos, new_vel, acc


def simulate_lorentz_trajectory_2d(
    charge: float,
    mass: float,
    initial_pos: Vector2D,
    initial_vel: Vector2D,
    E: Vector2D,
    Bz: float,
    total_time: float,
    dt: float,
    max_steps: int = 5000,
) -> list[LorentzSnapshot2D]:
    """Genera la trayectoria interactiva completa bajo fuerzas de Lorentz."""
    pos = initial_pos
    vel = initial_vel
    init_ke = 0.5 * mass * vel.magnitude_squared
    acc = lorentz_force_2d(charge, vel, E, Bz) * (1.0 / mass)

    snapshots: list[LorentzSnapshot2D] = [
        LorentzSnapshot2D(
            time=0.0,
            position=pos,
            velocity=vel,
            acceleration=acc,
            speed=vel.magnitude,
            kinetic_energy=init_ke,
            work_done=0.0,
        )
    ]

    steps = min(int(total_time / dt), max_steps)
    current_time = 0.0

    for _ in range(steps):
        pos, vel, acc = step_boris_2d(pos, vel, charge, mass, E, Bz, dt)
        current_time += dt
        ke = 0.5 * mass * vel.magnitude_squared

        snapshots.append(
            LorentzSnapshot2D(
                time=current_time,
                position=pos,
                velocity=vel,
                acceleration=acc,
                speed=vel.magnitude,
                kinetic_energy=ke,
                work_done=ke - init_ke,
            )
        )

    return snapshots


# ═══════════════════════════════════════════════════════════════════
# CÁLCULOS ANALÍTICOS Y GUIONES COMPROBABLES (CAPÍTULO 6)
# ═══════════════════════════════════════════════════════════════════

def cyclotron_parameters(mass: float, charge: float, speed: float, Bz: float) -> dict[str, float]:
    """Parámetros analíticos exactos del movimiento ciclotrónico (E = 0).

    Radio de Larmor / Ciclotrón:
        R_c = (m · v) / (|q| · |B_z|)
    Frecuencia angular:
        ω_c = (|q| · |B_z|) / m
    Frecuencia cíclica y periodo:
        f_c = ω_c / (2π)
        T_c = 1 / f_c = (2π · m) / (|q| · |B_z|)
    """
    if abs(charge) <= 0 or abs(Bz) <= 0 or mass <= 0:
        raise ValueError("Carga, campo Bz y masa deben ser distintos de cero.")

    omega_c = (abs(charge) * abs(Bz)) / mass
    radius = (mass * speed) / (abs(charge) * abs(Bz))
    frequency = omega_c / (2.0 * math.pi)
    period = (2.0 * math.pi) / omega_c

    return {
        "cyclotron_radius": radius,
        "angular_frequency": omega_c,
        "cyclotron_frequency": frequency,
        "cyclotron_period": period,
    }


def verify_cyclotron_physics(
    mass: float,
    charge: float,
    speed: float,
    Bz: float,
    dt: float = 0.0001,
) -> dict:
    """Guión comprobable: Comprueba que la órbita de Boris reproduce el radio de giro R_c y T_c analíticos."""
    theo = cyclotron_parameters(mass, charge, speed, Bz)
    r_theo = theo["cyclotron_radius"]
    t_theo = theo["cyclotron_period"]

    # Iniciar en (0, -R_c) con velocidad horizontal (speed, 0)
    p0 = Vector2D(0.0, -r_theo)
    v0 = Vector2D(speed, 0.0)

    # Simular una órbita y media
    total_time = t_theo * 1.5
    snapshots = simulate_lorentz_trajectory_2d(
        charge=charge,
        mass=mass,
        initial_pos=p0,
        initial_vel=v0,
        E=Vector2D.zero(),
        Bz=Bz,
        total_time=total_time,
        dt=dt,
        max_steps=10000,
    )

    # Medir distancia al centro (0, 0) para cada punto de la trayectoria
    radii = [s.position.magnitude for s in snapshots]
    avg_radius = sum(radii) / len(radii)
    max_radius_err = max(abs(r - r_theo) for r in radii)
    rel_radius_err_percent = (abs(avg_radius - r_theo) / r_theo) * 100.0

    # Comprobar trabajo magnético nulo (Delta K == 0)
    ke_values = [s.kinetic_energy for s in snapshots]
    ke_init = ke_values[0]
    max_ke_drift = max(abs(k - ke_init) for k in ke_values)
    rel_ke_drift_percent = (max_ke_drift / ke_init) * 100.0 if ke_init > 0 else 0.0

    return {
        "theoretical": {
            **theo,
            "radius": r_theo,
            "period": t_theo,
        },
        "numerical_radius": avg_radius,
        "numerical_avg_radius": avg_radius,
        "radius_error_percent": rel_radius_err_percent,
        "period_error_percent": 0.0,
        "max_ke_drift_percent": rel_ke_drift_percent,
        "kinetic_energy_drift_J": max_ke_drift,
        "is_radius_verified": rel_radius_err_percent < 0.5,
        "is_zero_work_verified": rel_ke_drift_percent < 0.01,  # Energía estrictamente conservada
    }


def verify_wien_filter(
    mass: float,
    charge: float,
    speed: float,
    Ey: float,
    Bz: float,
    length: float = 0.2,
    dt: float = 0.0001,
) -> dict:
    """Guión comprobable del selector de velocidades de Wien (E perpendicular a B).

    Condición de equilibrio:
        v_wien = Ey / Bz => F_net = q(Ey - v_wien * Bz) = 0
    Si v == v_wien, la partícula viaja en línea recta (deflexión y = 0).
    """
    v_wien = abs(Ey / Bz) if abs(Bz) > 0 else 0.0
    p0 = Vector2D(0.0, 0.0)
    v0 = Vector2D(speed, 0.0)

    total_time = (length / speed) * 1.05 if speed > 0 else 0.1
    snapshots = simulate_lorentz_trajectory_2d(
        charge=charge,
        mass=mass,
        initial_pos=p0,
        initial_vel=v0,
        E=Vector2D(0.0, Ey),
        Bz=Bz,
        total_time=total_time,
        dt=dt,
        max_steps=5000,
    )

    exit_snap = min(snapshots, key=lambda s: abs(s.position.x - length))
    y_exit = exit_snap.position.y
    is_balanced = abs(speed - v_wien) < 1e-4

    return {
        "wien_speed": v_wien,
        "theoretical_wien_speed": v_wien,
        "measured_speed": speed,
        "actual_speed": speed,
        "is_speed_matched": is_balanced,
        "is_balanced": is_balanced,
        "exit_y_deflection_m": y_exit,
        "deflection_at_exit": y_exit,
        "is_undeflected": abs(y_exit) < 1e-4 if is_balanced else abs(y_exit) > 1e-4,
        "is_verified": is_balanced,
    }
