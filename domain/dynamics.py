"""Cinemática y Dinámica de Partículas Cargadas (Capítulo 5).

En electrostática pura, las cargas se asumen estacionarias. En la física real,
toda carga con masa m inmersa en un campo eléctrico E experimenta una fuerza
neta F = q·E y acelera de acuerdo con la Segunda Ley de Newton:

    F = m · a  =>  a = (q / m) · E

Si la partícula interactúa con un conjunto de fuentes fijas o un campo uniforme,
la fuerza total se obtiene mediante el principio de superposición:

    F_total = q · E_uniforme + Σ [k · q · q_i / |r - r_i|²] · r̂_i

Para resolver numéricamente las trayectorias sin deriva artificial de energía,
implementamos el algoritmo simpléctico de Velocity-Verlet (orden O(dt²)).

Unidades SI:
    - Posición: metros [m]
    - Velocidad: m/s
    - Aceleración: m/s²
    - Masa: kilogramos [kg]
    - Carga: Coulombs [C]
    - Fuerza: Newtons [N]
    - Energía: Joules [J]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Union

from .vectors import Vector2D, Vector3D
from .particles import Particle
from .units import COULOMB_CONSTANT, ELECTRON_MASS, PROTON_MASS


# Masa por defecto para partículas visuales a escala de laboratorio (1 miligramo)
DEFAULT_SIMULATION_MASS: float = 1e-6  # [kg]


@dataclass
class DynamicParticle:
    """Una partícula cargada con propiedades mecánicas (masa, velocidad, aceleración)."""

    particle: Particle
    mass: float = DEFAULT_SIMULATION_MASS  # [kg]
    velocity: Union[Vector2D, Vector3D] = field(default_factory=Vector2D.zero)

    def __post_init__(self):
        if self.mass <= 0:
            raise ValueError(f"La masa debe ser estrictamente positiva. Recibido: {self.mass}")

        # Asegurar coherencia dimensional entre posición y velocidad
        if isinstance(self.particle.position, Vector3D) and isinstance(self.velocity, Vector2D):
            self.velocity = Vector3D(self.velocity.x, self.velocity.y, 0.0)
        elif isinstance(self.particle.position, Vector2D) and isinstance(self.velocity, Vector3D):
            self.velocity = Vector2D(self.velocity.x, self.velocity.y)

    @property
    def speed(self) -> float:
        """Rapidez escalar ||v|| [m/s]."""
        return self.velocity.magnitude

    @property
    def kinetic_energy(self) -> float:
        """Energía cinética clásica: K = 0.5 * m * v² [J]."""
        return 0.5 * self.mass * (self.velocity.magnitude_squared)

    @property
    def momentum(self) -> Union[Vector2D, Vector3D]:
        """Cantidad de movimiento lineal p = m * v [kg·m/s]."""
        return self.velocity * self.mass

    @property
    def charge_to_mass_ratio(self) -> float:
        """Relación carga/masa q/m [C/kg], factor crítico de deflexión."""
        return self.particle.charge / self.mass


@dataclass(frozen=True)
class UniformElectricField:
    """Campo eléctrico uniforme delimitado en una región rectangular/cúbica."""

    field_vector: Union[Vector2D, Vector3D]  # [V/m]
    x_bounds: tuple[float, float] = (-float("inf"), float("inf"))
    y_bounds: tuple[float, float] = (-float("inf"), float("inf"))
    z_bounds: tuple[float, float] = (-float("inf"), float("inf"))

    def contains(self, point: Union[Vector2D, Vector3D]) -> bool:
        """Evalúa si un punto del espacio se encuentra dentro de la zona de campo activo."""
        in_x = self.x_bounds[0] <= point.x <= self.x_bounds[1]
        in_y = self.y_bounds[0] <= point.y <= self.y_bounds[1]
        if isinstance(point, Vector3D):
            in_z = self.z_bounds[0] <= point.z <= self.z_bounds[1]
            return in_x and in_y and in_z
        return in_x and in_y

    def field_at(self, point: Union[Vector2D, Vector3D]) -> Union[Vector2D, Vector3D]:
        """Retorna el campo E en el punto dado, o cero si está fuera de los límites."""
        if self.contains(point):
            return self.field_vector
        return Vector3D.zero() if isinstance(point, Vector3D) else Vector2D.zero()


@dataclass(frozen=True)
class TrajectorySnapshot:
    """Estado instantáneo de la partícula durante su vuelo."""

    time: float                                # [s]
    position: Union[Vector2D, Vector3D]        # [m]
    velocity: Union[Vector2D, Vector3D]        # [m/s]
    acceleration: Union[Vector2D, Vector3D]    # [m/s²]
    speed: float                               # [m/s]
    kinetic_energy: float                      # [J]
    potential_energy: float                    # [J]
    total_energy: float                        # [J]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "acceleration": self.acceleration.to_dict(),
            "speed": self.speed,
            "kineticEnergy": self.kinetic_energy,
            "potentialEnergy": self.potential_energy,
            "totalEnergy": self.total_energy,
        }


def compute_net_force_at(
    position: Union[Vector2D, Vector3D],
    charge: float,
    sources: list[Particle],
    uniform_fields: list[UniformElectricField] | None = None,
) -> tuple[Union[Vector2D, Vector3D], float]:
    """Calcula la fuerza neta electrostática y el potencial eléctrico acumulado.

    Args:
        position: Dónde se encuentra la carga de prueba [m].
        charge: Carga puntual sometida a la fuerza [C].
        sources: Fuentes puntuales fijas generadoras de campo.
        uniform_fields: Campos eléctricos uniformes delimitados opcionales.

    Returns:
        (Fuerza neta [N], Energía potencial electrostática U [J]).
    """
    is_3d = isinstance(position, Vector3D)
    net_force = Vector3D.zero() if is_3d else Vector2D.zero()
    potential_energy = 0.0

    # 1. Contribución de fuentes puntuales de Coulomb
    for src in sources:
        displacement = position - src.position
        dist = displacement.magnitude

        # Evitar singularidad si la partícula colisiona con una fuente
        if dist < 1e-9:
            continue

        unit_vec = displacement.unit
        force_mag = COULOMB_CONSTANT * charge * src.charge / (dist**2)
        net_force = net_force + (unit_vec * force_mag)
        potential_energy += COULOMB_CONSTANT * charge * src.charge / dist

    # 2. Contribución de campos uniformes externos
    if uniform_fields:
        for u_field in uniform_fields:
            if u_field.contains(position):
                e_vec = u_field.field_vector
                f_uniform = e_vec * charge
                net_force = net_force + f_uniform
                # U = -q * (E · r)
                potential_energy -= charge * e_vec.dot(position)

    return net_force, potential_energy


def step_velocity_verlet(
    position: Union[Vector2D, Vector3D],
    velocity: Union[Vector2D, Vector3D],
    acceleration: Union[Vector2D, Vector3D],
    charge: float,
    mass: float,
    force_calculator: Callable[[Union[Vector2D, Vector3D]], tuple[Union[Vector2D, Vector3D], float]],
    dt: float,
) -> tuple[Union[Vector2D, Vector3D], Union[Vector2D, Vector3D], Union[Vector2D, Vector3D], float]:
    """Realiza un paso de integración simpléctica Velocity-Verlet.

    Garantiza estabilidad energética temporal sin disipación ni explosión artificial:
        1. r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
        2. a(t+dt) = F(r(t+dt)) / m
        3. v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

    Returns:
        (nueva_posicion, nueva_velocidad, nueva_aceleracion, nueva_energia_potencial)
    """
    half_dt = 0.5 * dt
    dt_sq_half = 0.5 * dt * dt

    # 1. Actualizar posición
    new_position = position + (velocity * dt) + (acceleration * dt_sq_half)

    # 2. Evaluar nueva fuerza y aceleración en la nueva posición
    new_force, new_potential = force_calculator(new_position)
    new_acceleration = new_force * (1.0 / mass)

    # 3. Actualizar velocidad usando el promedio de aceleraciones
    new_velocity = velocity + ((acceleration + new_acceleration) * half_dt)

    return new_position, new_velocity, new_acceleration, new_potential


def simulate_trajectory(
    dynamic_particle: DynamicParticle,
    sources: list[Particle] | None = None,
    uniform_fields: list[UniformElectricField] | None = None,
    total_time: float = 0.1,
    dt: float = 0.0005,
    max_steps: int = 2000,
) -> list[TrajectorySnapshot]:
    """Genera la trayectoria temporal completa de una partícula cargada.

    Args:
        dynamic_particle: Partícula con masa, carga, posición y velocidad inicial.
        sources: Cargas fijas externas en la escena.
        uniform_fields: Placas deflectoras o campos uniformes.
        total_time: Duración total de la simulación en segundos.
        dt: Paso de integración temporal en segundos.
        max_steps: Límite superior de pasos para evitar bucles infinitos.

    Returns:
        Lista de instantáneas de la trayectoria para animación y graficado.
    """
    sources_list = sources or []
    uniform_list = uniform_fields or []

    pos = dynamic_particle.particle.position
    vel = dynamic_particle.velocity
    mass = dynamic_particle.mass
    charge = dynamic_particle.particle.charge

    def calc_f(p):
        return compute_net_force_at(p, charge, sources_list, uniform_list)

    init_force, init_potential = calc_f(pos)
    acc = init_force * (1.0 / mass)

    snapshots: list[TrajectorySnapshot] = []
    current_time = 0.0

    # Registro de estado inicial t = 0
    init_ke = 0.5 * mass * vel.magnitude_squared
    snapshots.append(
        TrajectorySnapshot(
            time=0.0,
            position=pos,
            velocity=vel,
            acceleration=acc,
            speed=vel.magnitude,
            kinetic_energy=init_ke,
            potential_energy=init_potential,
            total_energy=init_ke + init_potential,
        )
    )

    steps = min(int(total_time / dt), max_steps)

    for _ in range(steps):
        pos, vel, acc, pot = step_velocity_verlet(
            position=pos,
            velocity=vel,
            acceleration=acc,
            charge=charge,
            mass=mass,
            force_calculator=calc_f,
            dt=dt,
        )
        current_time += dt
        ke = 0.5 * mass * vel.magnitude_squared

        snapshots.append(
            TrajectorySnapshot(
                time=current_time,
                position=pos,
                velocity=vel,
                acceleration=acc,
                speed=vel.magnitude,
                kinetic_energy=ke,
                potential_energy=pot,
                total_energy=ke + pot,
            )
        )

    return snapshots


# ═══════════════════════════════════════════════════════════════════
# MODELOS DE CORRIENTE Y VELOCIDAD DE DERIVA
# ═══════════════════════════════════════════════════════════════════

def compute_drift_velocity(mobility: float, electric_field_mag: float) -> float:
    """Velocidad media de deriva vd = μ * E en un medio conductor [m/s]."""
    return mobility * electric_field_mag


def compute_current_density(
    carrier_density: float,
    carrier_charge: float,
    drift_velocity: float,
) -> float:
    """Densidad de corriente eléctrica J = n * q * vd [A/m²]."""
    return carrier_density * carrier_charge * drift_velocity


def compute_current(current_density: float, cross_sectional_area: float) -> float:
    """Intensidad de corriente total I = J * A [A] (Amperios)."""
    return current_density * cross_sectional_area


# ═══════════════════════════════════════════════════════════════════
# GUIONES DE COMPROBACIÓN FÍSICA ANALÍTICA (CAPÍTULO 5)
# ═══════════════════════════════════════════════════════════════════

def theoretical_thomson_deflection(
    charge: float,
    mass: float,
    v0x: float,
    Ey: float,
    length: float,
) -> dict[str, float]:
    """Cálculo analítico exacto de la deflexión balística de J.J. Thomson.

    Una partícula cargada con velocidad horizontal v0x atraviesa placas de longitud L
    con campo eléctrico vertical constante Ey:
        ay = (q * Ey) / m
        t_vuelo = L / v0x
        y(L) = 0.5 * ay * t_vuelo² = (q * Ey * L²) / (2 * m * v0x²)
        vy(L) = ay * t_vuelo = (q * Ey * L) / (m * v0x)
        tan(θ) = vy / vx = (q * Ey * L) / (m * v0x²)
    """
    if v0x <= 0 or mass <= 0 or length <= 0:
        raise ValueError("v0x, masa y longitud deben ser estrictamente positivos.")

    ay = (charge * Ey) / mass
    time_of_flight = length / v0x
    y_deflection = 0.5 * ay * (time_of_flight**2)
    vy_exit = ay * time_of_flight
    deflection_angle_rad = math.atan2(vy_exit, v0x)
    deflection_angle_deg = math.degrees(deflection_angle_rad)

    return {
        "acceleration_y": ay,
        "time_of_flight": time_of_flight,
        "y_deflection": y_deflection,
        "vy_exit": vy_exit,
        "deflection_angle_rad": deflection_angle_rad,
        "deflection_angle_deg": deflection_angle_deg,
    }


def verify_thomson_experiment(
    charge: float,
    mass: float,
    v0x: float,
    Ey: float,
    length: float,
    dt: float = 0.0001,
) -> dict:
    """Guión comprobable que compara la simulación numérica (Velocity-Verlet) contra la fórmula analítica."""
    theoretical = theoretical_thomson_deflection(charge, mass, v0x, Ey, length)

    p0 = Particle(position=Vector2D(0.0, 0.0), charge=charge, label="electrón_prueba")
    dyn_p = DynamicParticle(particle=p0, mass=mass, velocity=Vector2D(v0x, 0.0))
    plates = UniformElectricField(
        field_vector=Vector2D(0.0, Ey),
        x_bounds=(0.0, length),
        y_bounds=(-float("inf"), float("inf")),
    )

    total_time = (length / v0x) * 1.05
    snapshots = simulate_trajectory(
        dynamic_particle=dyn_p,
        sources=[],
        uniform_fields=[plates],
        total_time=total_time,
        dt=dt,
        max_steps=10000,
    )

    # Buscar el snapshot más cercano a x = length
    exit_snap = min(snapshots, key=lambda s: abs(s.position.x - length))
    num_y = exit_snap.position.y
    theo_y = theoretical["y_deflection"]
    abs_err = abs(num_y - theo_y)
    rel_err = (abs_err / abs(theo_y)) * 100.0 if abs(theo_y) > 1e-12 else 0.0

    return {
        "theoretical": theoretical,
        "numerical_y": num_y,
        "numerical_vy": exit_snap.velocity.y,
        "numerical_time": exit_snap.time,
        "absolute_error_m": abs_err,
        "relative_error_percent": rel_err,
        "is_verified": rel_err < 0.5,  # Tolerancia estricta del 0.5%
    }

