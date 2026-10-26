"""Soluciones analíticas exactas y benchmarks de referencia para validación física."""

from __future__ import annotations

import cmath
import math
import numpy as np

from domain.units import SPEED_OF_LIGHT, MU_0, EPSILON_0


def airy_dielectric_slab_transmission(
    relative_permittivity: float,
    thickness_meters: float,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Fórmula analítica exacta de Airy (Fabry-Pérot) para una lámina dieléctrica homogénea en el vacío.
    
    T(f) = [ 1 + (1/4) * (n - 1/n)^2 * sin^2(n * k0 * L) ]^(-1)
    """
    n = math.sqrt(relative_permittivity)
    delta_factor = (n - (1.0 / n)) ** 2
    t_vals = []

    for f in frequencies_hz:
        k0 = (2.0 * math.pi * f) / SPEED_OF_LIGHT
        phase = n * k0 * thickness_meters
        denom = 1.0 + 0.25 * delta_factor * (math.sin(phase) ** 2)
        t_vals.append(1.0 / denom)

    return np.array(t_vals, dtype=np.float64)


def tmm_multilayer_transmission(
    eps_layers: list[float],
    thicknesses_meters: list[float],
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Método de Matriz de Transferencia (TMM) exacto para multicapas y espejos de Bragg 1D."""
    t_vals = []
    layers_n = [math.sqrt(eps) for eps in eps_layers]

    for f in frequencies_hz:
        k0 = (2.0 * math.pi * f) / SPEED_OF_LIGHT
        # Matriz identidad 2x2 inicial
        M11 = 1.0 + 0.0j
        M12 = 0.0 + 0.0j
        M21 = 0.0 + 0.0j
        M22 = 1.0 + 0.0j

        for n_j, d_j in zip(layers_n, thicknesses_meters):
            delta = n_j * k0 * d_j
            cos_d = cmath.cos(delta)
            sin_d = cmath.sin(delta)
            # Matriz de capa j para modo normal
            m11 = cos_d
            m12 = -1j * sin_d / n_j
            m21 = -1j * n_j * sin_d
            m22 = cos_d

            # Multiplicación M = M * m_j
            new_M11 = M11 * m11 + M12 * m21
            new_M12 = M11 * m12 + M12 * m22
            new_M21 = M21 * m11 + M22 * m21
            new_M22 = M21 * m12 + M22 * m22

            M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22

        t_coeff = 2.0 / (M11 + M12 + M21 + M22)
        t_vals.append(abs(t_coeff) ** 2)

    return np.array(t_vals, dtype=np.float64)


def yee_numerical_phase_velocity(
    frequency_hz: float,
    dx: float,
    dt: float,
) -> float:
    """Velocidad de fase teórica predicha por la relación de dispersión numérica de Yee en 1D/2D.
    
    sin(omega * dt / 2) / (c * dt) = sin(k_tilde * dx / 2) / dx
    => k_tilde = (2 / dx) * arcsin[ (dx / (c * dt)) * sin(omega * dt / 2) ]
    => v_tilde = omega / k_tilde
    """
    omega = 2.0 * math.pi * frequency_hz
    arg = (dx / (SPEED_OF_LIGHT * dt)) * math.sin(omega * dt / 2.0)
    if abs(arg) > 1.0:
        return 0.0  # Frecuencia fuera del límite de corte numérico
    k_tilde = (2.0 / dx) * math.asin(arg)
    return omega / k_tilde if k_tilde > 0 else SPEED_OF_LIGHT


def wire_medium_plasma_frequency(
    period_a_meters: float,
    wire_radius_meters: float,
) -> float:
    """Frecuencia de plasma teórica de un medio de varillas metálicas (Pendry 1996, Belov 2002).
    
    (k_p * a)^2 = 2pi / [ ln(a / (2pi * r)) + 0.5275 ]
    f_p = (c * k_p) / (2pi)
    """
    a = period_a_meters
    r = wire_radius_meters
    ln_term = math.log(a / (2.0 * math.pi * r)) + 0.5275
    kp_sq = (2.0 * math.pi) / ((a ** 2) * ln_term)
    kp = math.sqrt(max(kp_sq, 1e-12))
    fp = (SPEED_OF_LIGHT * kp) / (2.0 * math.pi)
    return fp
