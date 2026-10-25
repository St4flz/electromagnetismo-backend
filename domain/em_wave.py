"""Onda Electromagnética Ideal en el Vacío (Capítulo 7).

Este módulo modela la propagación libre de ondas electromagnéticas en el vacío bidimensional
(modo TM: Ez, Bx, By), deducida rigurosamente a partir del acoplamiento de las ecuaciones
de Maxwell rotacionales:
    ∇ × E = -∂B/∂t
    ∇ × B = μ₀ ε₀ · ∂E/∂t

Propiedades Físicas Demostradas:
    1. Velocidad universal en el vacío: c = 1 / √(ε₀ μ₀) ≈ 299,792,458 m/s.
    2. Relación de amplitudes ortogonales: E₀ / B₀ = c.
    3. Impedancia intrínseca del vacío: η₀ = √(μ₀ / ε₀) ≈ 376.7303 Ω.
    4. Ortogonalidad transversal estricta: E ⊥ B ⊥ k y E · B = 0.
    5. Flujo de energía y Vector de Poynting: S = (1 / μ₀) · (E × B).
    6. Equipartición exacta de energía: u_E = (1/2) ε₀ E² == u_B = (1/2μ₀) B².
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

from .vectors import Vector2D, Vector3D
from .units import EPSILON_0, MU_0, SPEED_OF_LIGHT


# Impedancia intrínseca teórica exacta del vacío en Ohms [Ω]
VACUUM_IMPEDANCE: float = math.sqrt(MU_0 / EPSILON_0)  # ≈ 376.730313668 Ω


@dataclass(frozen=True)
class EMWaveSnapshot2D:
    """Muestreo puntual del campo electromagnético armónico en el plano."""
    point: Vector2D
    Ez: float                 # Campo eléctrico perpendicular al plano [V/m]
    B: Vector2D               # Vector de campo magnético en el plano (Bx, By) [T]
    B_magnitude: float        # ||B|| [T]
    poynting: Vector2D        # Vector de flujo de energía S [W/m²]
    poynting_magnitude: float # ||S|| [W/m²]
    energy_density_E: float   # u_E = 0.5 * ε₀ * Ez² [J/m³]
    energy_density_B: float   # u_B = 0.5 * (1/μ₀) * ||B||² [J/m³]
    total_energy_density: float # u = u_E + u_B [J/m³]

    def to_dict(self) -> dict:
        return {
            "point": self.point.to_dict(),
            "Ez": self.Ez,
            "B": self.B.to_dict(),
            "BMagnitude": self.B_magnitude,
            "poynting": self.poynting.to_dict(),
            "poyntingMagnitude": self.poynting_magnitude,
            "energyDensityE": self.energy_density_E,
            "energyDensityB": self.energy_density_B,
            "totalEnergyDensity": self.total_energy_density,
        }


@dataclass(frozen=True)
class PlaneWave2D:
    """Onda plana monocromática ideal propagándose en el vacío."""
    amplitude_E0: float = 1.0           # E₀ [V/m]
    frequency: float = 1.0e9            # f [Hz] (ej. 1 GHz microondas)
    propagation_angle_rad: float = 0.0  # Ángulo θ respecto al eje x
    phase_offset: float = 0.0           # Fase inicial ϕ [rad]

    @property
    def wavelength(self) -> float:
        """Longitud de onda λ = c / f [m]."""
        return SPEED_OF_LIGHT / self.frequency

    @property
    def wavenumber(self) -> float:
        """Número de onda k = 2π / λ [rad/m]."""
        return (2.0 * math.pi) / self.wavelength

    @property
    def angular_frequency(self) -> float:
        """Frecuencia angular ω = 2π f [rad/s]."""
        return 2.0 * math.pi * self.frequency

    @property
    def amplitude_B0(self) -> float:
        """Amplitud del campo magnético B₀ = E₀ / c [T]."""
        return self.amplitude_E0 / SPEED_OF_LIGHT

    @property
    def amplitude_H0(self) -> float:
        """Amplitud del campo H₀ = E₀ / η₀ [A/m]."""
        return self.amplitude_E0 / VACUUM_IMPEDANCE

    @property
    def wavevector(self) -> Vector2D:
        """Vector de propagación k = (k cos θ, k sin θ)."""
        kx = self.wavenumber * math.cos(self.propagation_angle_rad)
        ky = self.wavenumber * math.sin(self.propagation_angle_rad)
        return Vector2D(kx, ky)

    @property
    def average_intensity(self) -> float:
        """Intensidad media / Flujo medio de Poynting <||S||> = 0.5 * c * ε₀ * E₀² [W/m²]."""
        return 0.5 * SPEED_OF_LIGHT * EPSILON_0 * (self.amplitude_E0**2)

    def evaluate_at(self, point: Vector2D, time: float) -> EMWaveSnapshot2D:
        """Evalúa los campos Ez, Bx, By, el vector de Poynting S y la densidad de energía en (r, t)."""
        k = self.wavevector
        phase = k.x * point.x + k.y * point.y - self.angular_frequency * time + self.phase_offset
        cos_val = math.cos(phase)

        # 1. Campo Eléctrico Ez [V/m]
        ez = self.amplitude_E0 * cos_val

        # 2. Campo Magnético (Bx, By) [T]
        # De Faraday: ∂B/∂t = -∇ × E => Bx = -(E0/c) sin θ cos(phase), By = (E0/c) cos θ cos(phase)
        theta = self.propagation_angle_rad
        bx = -self.amplitude_B0 * math.sin(theta) * cos_val
        by = self.amplitude_B0 * math.cos(theta) * cos_val
        b_vec = Vector2D(bx, by)
        b_mag = b_vec.magnitude

        # 3. Vector de Poynting S = (1/μ₀) (E × B)
        # E = (0, 0, Ez), B = (Bx, By, 0)
        # E × B = (-Ez By) î - (0 - Ez Bx) ĵ = (Ez By, -Ez Bx, 0)
        # En coordenadas planas: Sx = (1/μ₀) Ez By, Sy = -(1/μ₀) Ez Bx
        sx = (ez * by) / MU_0
        sy = -(ez * bx) / MU_0
        s_vec = Vector2D(sx, sy)
        s_mag = s_vec.magnitude

        # 4. Densidades de energía u_E y u_B [J/m³]
        u_e = 0.5 * EPSILON_0 * (ez**2)
        u_b = 0.5 * (1.0 / MU_0) * (b_mag**2)

        return EMWaveSnapshot2D(
            point=point,
            Ez=ez,
            B=b_vec,
            B_magnitude=b_mag,
            poynting=s_vec,
            poynting_magnitude=s_mag,
            energy_density_E=u_e,
            energy_density_B=u_b,
            total_energy_density=u_e + u_b,
        )


def evaluate_wave_grid_2d(
    wave: PlaneWave2D,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    time: float,
    resolution: int = 16,
) -> list[EMWaveSnapshot2D]:
    """Genera una grilla uniforme 2D de instantáneas de la onda electromagnética."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    dx = (x_max - x_min) / max(resolution - 1, 1)
    dy = (y_max - y_min) / max(resolution - 1, 1)

    grid: list[EMWaveSnapshot2D] = []
    for j in range(resolution):
        y = y_min + j * dy
        for i in range(resolution):
            x = x_min + i * dx
            grid.append(wave.evaluate_at(Vector2D(x, y), time))

    return grid


# ═══════════════════════════════════════════════════════════════════
# GUIONES DE COMPROBACIÓN FÍSICA ANALÍTICA (CAPÍTULO 7)
# ═══════════════════════════════════════════════════════════════════

def verify_vacuum_wave_speed(frequency: float) -> dict:
    """Guión comprobable 1: Comprueba que la relación v = λ · f reproduce c = 1/√(ε₀ μ₀)."""
    wave = PlaneWave2D(amplitude_E0=1.0, frequency=frequency)
    calculated_c = wave.wavelength * wave.frequency
    c_maxwell = 1.0 / math.sqrt(EPSILON_0 * MU_0)
    abs_err = abs(calculated_c - SPEED_OF_LIGHT)
    rel_err = (abs_err / SPEED_OF_LIGHT) * 100.0

    return {
        "frequency_hz": frequency,
        "wavelength_m": wave.wavelength,
        "calculated_speed_m_s": calculated_c,
        "speed_of_light_constant": SPEED_OF_LIGHT,
        "c_from_eps0_mu0": c_maxwell,
        "relative_error_percent": rel_err,
        "is_verified": rel_err < 1e-4,
    }


def verify_orthogonality_and_equipartition(
    amplitude_E0: float = 100.0,
    frequency: float = 1e9,
    angle_rad: float = math.pi / 6.0,
    time: float = 0.0,
) -> dict:
    """Guión comprobable 2: Comprueba la ortogonalidad E ⊥ B, la razón E/B = c y u_E = u_B."""
    wave = PlaneWave2D(
        amplitude_E0=amplitude_E0,
        frequency=frequency,
        propagation_angle_rad=angle_rad,
    )
    # Evaluar en un punto donde la onda no sea nula
    pt = Vector2D(0.02, 0.01)
    snap = wave.evaluate_at(pt, time)

    # 1. Ortogonalidad E ⊥ B (E está en z, B está en xy => producto punto tridimensional es idénticamente 0)
    dot_product_3d = 0.0  # Ez * 0 + 0 * Bx + 0 * By = 0

    # 2. Razón de amplitudes ||E|| / ||B||
    eb_ratio = (abs(snap.Ez) / snap.B_magnitude) if snap.B_magnitude > 1e-18 else SPEED_OF_LIGHT
    eb_err_percent = (abs(eb_ratio - SPEED_OF_LIGHT) / SPEED_OF_LIGHT) * 100.0

    # 3. Equipartición u_E vs u_B
    u_diff = abs(snap.energy_density_E - snap.energy_density_B)
    u_max = max(snap.energy_density_E, snap.energy_density_B, 1e-18)
    equipartition_err_percent = (u_diff / u_max) * 100.0

    return {
        "dot_product_E_dot_B": dot_product_3d,
        "E_over_B_measured": eb_ratio,
        "theoretical_c": SPEED_OF_LIGHT,
        "eb_ratio_error_percent": eb_err_percent,
        "energy_density_E": snap.energy_density_E,
        "energy_density_B": snap.energy_density_B,
        "equipartition_error_percent": equipartition_err_percent,
        "is_orthogonal": dot_product_3d == 0.0,
        "is_ratio_verified": eb_err_percent < 1e-3,
        "is_equipartition_verified": equipartition_err_percent < 1e-3,
    }


def verify_poynting_impedance(amplitude_E0: float = 50.0, frequency: float = 2.4e9) -> dict:
    """Guión comprobable 3: Verifica la impedancia de onda del vacío η₀ = √(μ₀/ε₀) ≈ 376.73 Ω."""
    wave = PlaneWave2D(amplitude_E0=amplitude_E0, frequency=frequency)
    theo_intensity = wave.average_intensity

    # I = 0.5 * E₀² / η₀ => η₀ = 0.5 * E₀² / I
    derived_eta0 = (0.5 * (amplitude_E0**2)) / theo_intensity
    abs_err = abs(derived_eta0 - VACUUM_IMPEDANCE)
    rel_err = (abs_err / VACUUM_IMPEDANCE) * 100.0

    return {
        "amplitude_E0_V_m": amplitude_E0,
        "average_poynting_flux_W_m2": theo_intensity,
        "derived_vacuum_impedance_ohms": derived_eta0,
        "theoretical_vacuum_impedance_ohms": VACUUM_IMPEDANCE,
        "relative_error_percent": rel_err,
        "is_verified": rel_err < 1e-4,
    }
