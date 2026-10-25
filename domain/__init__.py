"""Dominio físico del simulador electromagnético.

Este paquete contiene la física pura: vectores, partículas, leyes y campos.
No hay HTTP, no hay API, no hay frameworks — solo matemáticas y física rigurosa.

Estructura por capítulos:
    vectors       — Vector2D, Vector3D (la abstracción fundamental)
    units         — Constantes del SI universales (c, ε₀, μ₀, e, k)
    particles     — Carga puntual: posición + carga eléctrica
    coulomb       — Ley de Coulomb: F = k·q₁·q₂/r²·r̂ (Capítulo 2 y 3)
    electric_field — Campo eléctrico y grillas espaciales (Capítulo 4)
    dynamics      — Cinemática, Velocity-Verlet y deflexión de Thomson (Capítulo 5)
    magnetic      — Biot-Savart, Lorentz Boris 2D y Ciclotrón (Capítulo 6)
    em_wave       — Ondas en el vacío 2D, Poynting y equipartición (Capítulo 7)
    metamaterials — Celdas periódicas (a,d,w), Veselago DNG y TMM (Capítulo 8)
    fdtd2d        — Malla escalonada de Yee 2D, CFL y Mur ABC (Capítulo 9)
"""

from .vectors import Vector2D, Vector3D
from .units import (
    COULOMB_CONSTANT,
    EPSILON_0,
    MU_0,
    SPEED_OF_LIGHT,
    ELEMENTARY_CHARGE,
    ELECTRON_MASS,
    PROTON_MASS,
)
from .particles import Particle
from .coulomb import CoulombResult, all_forces, coulomb_force, net_force_on
from .electric_field import FieldPoint, electric_field_at, electric_field_superposition, compute_field_grid_2d
from .dynamics import (
    DynamicParticle,
    UniformElectricField,
    TrajectorySnapshot,
    simulate_trajectory,
    theoretical_thomson_deflection,
    verify_thomson_experiment,
)
from .magnetic import (
    MagneticFieldPoint2D,
    LorentzSnapshot2D,
    biot_savart_point_charge_2d,
    biot_savart_wire_field_2d,
    lorentz_force_2d,
    step_boris_2d,
    simulate_lorentz_trajectory_2d,
    cyclotron_parameters,
    verify_cyclotron_physics,
    verify_wien_filter,
)
from .em_wave import (
    PlaneWave2D,
    EMWaveSnapshot2D,
    VACUUM_IMPEDANCE,
    evaluate_wave_grid_2d,
    verify_vacuum_wave_speed,
    verify_orthogonality_and_equipartition,
    verify_poynting_impedance,
)
from .metamaterials import (
    MetamaterialUnitCell,
    SpectralPoint,
    compute_effective_parameters,
    compute_spectral_response,
    evaluate_objective_function,
    verify_energy_conservation_spectrum,
    verify_veselago_negative_refraction,
    verify_bragg_bandgap,
)
from .fdtd2d import (
    FDTD2DSimulator,
    verify_cfl_stability_condition,
    verify_fdtd_pulse_velocity,
)

__all__ = [
    "Vector2D",
    "Vector3D",
    "COULOMB_CONSTANT",
    "EPSILON_0",
    "MU_0",
    "SPEED_OF_LIGHT",
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS",
    "PROTON_MASS",
    "Particle",
    "CoulombResult",
    "all_forces",
    "coulomb_force",
    "net_force_on",
    "FieldPoint",
    "electric_field_at",
    "electric_field_superposition",
    "compute_field_grid_2d",
    "DynamicParticle",
    "UniformElectricField",
    "TrajectorySnapshot",
    "simulate_trajectory",
    "theoretical_thomson_deflection",
    "verify_thomson_experiment",
    "MagneticFieldPoint2D",
    "LorentzSnapshot2D",
    "biot_savart_point_charge_2d",
    "biot_savart_wire_field_2d",
    "lorentz_force_2d",
    "step_boris_2d",
    "simulate_lorentz_trajectory_2d",
    "cyclotron_parameters",
    "verify_cyclotron_physics",
    "verify_wien_filter",
    "PlaneWave2D",
    "EMWaveSnapshot2D",
    "VACUUM_IMPEDANCE",
    "evaluate_wave_grid_2d",
    "verify_vacuum_wave_speed",
    "verify_orthogonality_and_equipartition",
    "verify_poynting_impedance",
    "MetamaterialUnitCell",
    "SpectralPoint",
    "compute_effective_parameters",
    "compute_spectral_response",
    "evaluate_objective_function",
    "verify_energy_conservation_spectrum",
    "verify_veselago_negative_refraction",
    "verify_bragg_bandgap",
    "FDTD2DSimulator",
    "verify_cfl_stability_condition",
    "verify_fdtd_pulse_velocity",
]
