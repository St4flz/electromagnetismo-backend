"""Física y Modelado de Metamateriales Electromagnéticos (Capítulo 8).

Este módulo implementa el rigor teórico de estructuras periódicas artificiales y medios
efectivos con propiedades electromagnéticas no convencionales (medios zurdos DNG con
permitividad negativa ε < 0 y permeabilidad negativa μ < 0, Veselago 1968, Pendry 2000,
Smith 2000), tal como se especifica en la Segunda Entrega del Proyecto:
    - Parametrización geométrica de celda unitaria: tamaño de celda (a), separación (d), ancho (w).
    - Frecuencia objetivo de diseño: f₀ = 10 GHz (Banda de microondas).
    - Modelos constitutivos de Drude-Lorentz para resonadores tipo Split-Ring (SRR) y varillas.
    - Método de Matriz de Transferencia (TMM) para cálculo analítico exacto de T(f), R(f) y A(f).
    - Conservación rigurosa de energía: R(f) + T(f) + A(f) = 1.0.
    - Evaluación de la Función Objetivo F(p) = T(f₀; p) para la posterior optimización computacional.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Literal

from .units import SPEED_OF_LIGHT, EPSILON_0, MU_0


@dataclass(frozen=True)
class MetamaterialUnitCell:
    """Parámetros geométricos y constitutivos de la celda unitaria periódica."""

    a: float  # Tamaño de celda en milímetros [mm] (ej. 10.0 mm = 0.01 m)
    d: float  # Separación entre elementos en milímetros [mm] (ej. 2.0 mm)
    w: float  # Ancho del elemento conductor/dieléctrico en mm [mm] (ej. 1.0 mm)
    periods: int = 5  # Número de celdas repetidas en la dirección de propagación
    loss_factor: float = 0.002  # Factor de amortiguamiento / pérdidas óhmicas (γ)

    @property
    def a_meters(self) -> float:
        return self.a * 1e-3

    @property
    def d_meters(self) -> float:
        return self.d * 1e-3

    @property
    def w_meters(self) -> float:
        return self.w * 1e-3

    @property
    def total_thickness_meters(self) -> float:
        return self.periods * self.a_meters

    @property
    def filling_factor(self) -> float:
        """Fracción volumétrica ocupada por el elemento resonante F ≈ w / a."""
        return min(max(self.w / max(self.a, 1e-6), 0.05), 0.95)

    @property
    def magnetic_resonance_frequency(self) -> float:
        """Frecuencia de resonancia magnética f₀m [Hz] escalada por la geometría (a, d, w).

        El tamaño de la espira resonante (a - d) y la capacitancia del gap (d, w)
        determinan la frecuencia LC según: f₀m ∝ c / (2 * a * √(1 + d/w)).
        Calibrada para situarse en la banda de microondas (alrededor de 10 GHz).
        """
        # Escala fundamental en microondas: a = 10mm => ~10 GHz
        geom_scale = (self.a + 0.5 * self.d) * 1e-3
        f_base = SPEED_OF_LIGHT / (3.0 * geom_scale)
        # Factor de ajuste por gap de capacitancia y ancho w
        gap_ratio = math.sqrt(max(self.d / max(self.w, 0.1), 0.1))
        return f_base * (1.0 + 0.15 * gap_ratio)


@dataclass(frozen=True)
class SpectralPoint:
    """Respuesta electromagnética evaluada a una frecuencia específica."""

    frequency_ghz: float
    transmission: float      # T(f) ∈ [0, 1]
    reflection: float        # R(f) ∈ [0, 1]
    absorption: float        # A(f) ∈ [0, 1]
    transmission_db: float   # 10 * log10(T) [dB]
    eps_real: float          # Re(ε_r)
    eps_imag: float          # Im(ε_r)
    mu_real: float           # Re(μ_r)
    mu_imag: float           # Im(μ_r)
    n_real: float            # Re(n_eff) (índice de refracción efectivo)
    n_imag: float            # Im(n_eff)
    is_left_handed: bool     # True si Re(ε) < 0 y Re(μ) < 0 (Medio Zurdos / Veselago)

    def to_dict(self) -> dict:
        return {
            "frequencyGhz": self.frequency_ghz,
            "transmission": self.transmission,
            "reflection": self.reflection,
            "absorption": self.absorption,
            "transmissionDb": self.transmission_db,
            "epsReal": self.eps_real,
            "epsImag": self.eps_imag,
            "muReal": self.mu_real,
            "muImag": self.mu_imag,
            "nReal": self.n_real,
            "nImag": self.n_imag,
            "isLeftHanded": self.is_left_handed,
        }


def compute_effective_parameters(
    cell: MetamaterialUnitCell,
    frequency_hz: float,
) -> tuple[complex, complex, complex]:
    """Calcula ε_r(f), μ_r(f) y el índice de refracción efectivo n_eff(f).

    Utiliza el modelo combinado Drude-Lorentz (Smith & Pendry, 2000):
        ε_r(f) = 1 - (f_pe² / [f · (f + i · γ_e)])
        μ_r(f) = 1 - (F · f² / [f² - f₀m² + i · γ_m · f])
    """
    f = frequency_hz
    f_0m = cell.magnetic_resonance_frequency
    # Frecuencia de plasma eléctrica sintonizada por encima de la resonancia magnética
    f_pe = 1.35 * f_0m
    gamma_e = cell.loss_factor * f_pe
    gamma_m = cell.loss_factor * f_0m
    f_factor = cell.filling_factor

    # 1. Permitividad eléctrica efectiva ε_r(f) (Modelo tipo Drude de varillas delgadas)
    eps_denom = complex(f * f, f * gamma_e)
    eps_r = 1.0 - (f_pe**2) / eps_denom

    # 2. Permeabilidad magnética efectiva μ_r(f) (Modelo resonante Lorentz de SRR)
    mu_denom = complex(f**2 - f_0m**2, gamma_m * f)
    mu_r = 1.0 - (f_factor * (f**2)) / mu_denom

    # 3. Índice de refracción efectivo n_eff = ± √(ε_r · μ_r)
    # Condición fundamental de Veselago: si Re(ε_r) < 0 y Re(μ_r) < 0, se debe tomar la rama negativa:
    product = eps_r * mu_r
    sqrt_prod = cmath.sqrt(product)

    if eps_r.real < 0.0 and mu_r.real < 0.0:
        # Medio Zurdo (DNG) -> Re(n_eff) es negativa
        n_real = -abs(sqrt_prod.real)
        n_imag = abs(sqrt_prod.imag)
        n_eff = complex(n_real, n_imag)
    else:
        # Medio Convencional (DPS) -> Re(n_eff) es positiva
        n_real = abs(sqrt_prod.real)
        n_imag = abs(sqrt_prod.imag)
        n_eff = complex(n_real, n_imag)

    return eps_r, mu_r, n_eff


def compute_spectral_response(
    cell: MetamaterialUnitCell,
    freq_min_ghz: float = 5.0,
    freq_max_ghz: float = 15.0,
    points: int = 100,
) -> list[SpectralPoint]:
    """Calcula el espectro continuo de transmisión T(f), reflexión R(f) y absorción A(f) mediante TMM."""
    results: list[SpectralPoint] = []
    df = (freq_max_ghz - freq_min_ghz) / max(points - 1, 1)
    length = cell.total_thickness_meters

    for idx in range(points):
        f_ghz = freq_min_ghz + idx * df
        f_hz = f_ghz * 1e9

        eps_r, mu_r, n_eff = compute_effective_parameters(cell, f_hz)

        # Impedancia relativa de onda Z = √(μ_r / ε_r)
        z_wave = cmath.sqrt(mu_r / eps_r) if abs(eps_r) > 1e-9 else complex(1.0, 0.0)

        # Desfase de propagación δ = n_eff · k₀ · L
        k0 = (2.0 * math.pi * f_hz) / SPEED_OF_LIGHT
        delta = n_eff * k0 * length

        # Coeficientes de transmisión t y reflexión r de lámina homogénea efectiva
        # t = 2 / [2 cos(δ) - i (Z + 1/Z) sin(δ)]
        cos_delta = cmath.cos(delta)
        sin_delta = cmath.sin(delta)
        denom = 2.0 * cos_delta - 1j * (z_wave + 1.0 / z_wave) * sin_delta

        t_coeff = 2.0 / denom
        r_coeff = (-1j * (z_wave - 1.0 / z_wave) * sin_delta) / denom

        # T y R se calculan de forma independiente a partir de |t|² y |r|².
        # NO se recortan ni se renormalizan: la absorción A = 1 - T - R es el
        # balance de potencia y debe cumplir A ≥ 0 en un medio pasivo. Si algún día
        # A < 0, es una violación real que verify_energy_conservation_spectrum reporta.
        trans = abs(t_coeff) ** 2
        refl = abs(r_coeff) ** 2
        absorp = 1.0 - trans - refl

        # dB
        t_db = 10.0 * math.log10(max(trans, 1e-12))
        is_dng = (eps_r.real < 0.0) and (mu_r.real < 0.0)

        results.append(
            SpectralPoint(
                frequency_ghz=f_ghz,
                transmission=trans,
                reflection=refl,
                absorption=absorp,
                transmission_db=t_db,
                eps_real=eps_r.real,
                eps_imag=eps_r.imag,
                mu_real=mu_r.real,
                mu_imag=mu_r.imag,
                n_real=n_eff.real,
                n_imag=n_eff.imag,
                is_left_handed=is_dng,
            )
        )

    return results


def evaluate_objective_function(
    cell: MetamaterialUnitCell,
    target_frequency_ghz: float = 10.0,
    mode: Literal["minimize_transmission", "maximize_transmission"] = "minimize_transmission",
) -> float:
    """Función objetivo F(p) para optimización computacional (Segunda Entrega, Fase 4).

    F(a, d, w) = T(f₀; a, d, w)
    Para blindaje/filtrado notch: minimizar T(f₀).
    """
    f_hz = target_frequency_ghz * 1e9
    eps_r, mu_r, n_eff = compute_effective_parameters(cell, f_hz)
    z_wave = cmath.sqrt(mu_r / eps_r)
    k0 = (2.0 * math.pi * f_hz) / SPEED_OF_LIGHT
    delta = n_eff * k0 * cell.total_thickness_meters

    cos_delta = cmath.cos(delta)
    sin_delta = cmath.sin(delta)
    denom = 2.0 * cos_delta - 1j * (z_wave + 1.0 / z_wave) * sin_delta
    trans = min(max(abs(2.0 / denom)**2, 0.0), 1.0)

    if mode == "minimize_transmission":
        return trans
    return 1.0 - trans


# ═══════════════════════════════════════════════════════════════════
# GUIONES DE COMPROBACIÓN FÍSICA ANALÍTICA (CAPÍTULO 8)
# ═══════════════════════════════════════════════════════════════════

def verify_energy_conservation_spectrum(cell: MetamaterialUnitCell) -> dict:
    """Guión comprobable 1: balance de potencia del modelo de medio efectivo.

    Dos pruebas independientes (ninguna fuerza el resultado):
      (a) Límite sin pérdidas (γ = 0): R(f) + T(f) = 1 → se mide max |R + T − 1|.
      (b) Medio con pérdidas: pasividad A(f) = 1 − R − T ≥ 0 → se mide min A(f).
    """
    lossless = MetamaterialUnitCell(
        a=cell.a, d=cell.d, w=cell.w, periods=cell.periods, loss_factor=0.0
    )
    spec_lossless = compute_spectral_response(lossless, 5.0, 15.0, points=50)
    max_dev = max(abs(p.transmission + p.reflection - 1.0) for p in spec_lossless)

    spec_lossy = compute_spectral_response(cell, 5.0, 15.0, points=50)
    min_abs = min(p.absorption for p in spec_lossy)

    return {
        "tested_points": len(spec_lossy),
        "max_energy_deviation": max_dev,  # lossless |R+T-1|
        "min_absorption_lossy": min_abs,  # debe ser >= 0 (pasividad)
        "is_conserved": max_dev < 1e-9 and min_abs > -1e-9,
    }


def verify_veselago_negative_refraction(
    cell: MetamaterialUnitCell,
    incident_angle_deg: float = 30.0,
) -> dict:
    """Guión comprobable 2: Verifica la Ley de Snell Negativa (n₁ sin θ₁ = n₂ sin θ₂ con n₂ < 0).

    En la banda DNG, el haz se refracta al mismo lado de la normal (θ₂ < 0).
    """
    # Encontrar la frecuencia donde Re(n_eff) es más negativa
    spectrum = compute_spectral_response(cell, freq_min_ghz=5.0, freq_max_ghz=15.0, points=100)
    dng_points = [p for p in spectrum if p.is_left_handed and p.n_real < -0.1]

    if not dng_points:
        # Usar el punto con n más bajo
        best_pt = min(spectrum, key=lambda p: p.n_real)
    else:
        best_pt = min(dng_points, key=lambda p: p.n_real)

    n_metamaterial = best_pt.n_real
    theta1_rad = math.radians(incident_angle_deg)
    # sin(θ₂) = (n₁ / n₂) * sin(θ₁)  con n₁ = 1 (vacío)
    sin_theta2 = (1.0 / n_metamaterial) * math.sin(theta1_rad)
    sin_theta2 = max(min(sin_theta2, 1.0), -1.0)
    theta2_rad = math.asin(sin_theta2)
    theta2_deg = math.degrees(theta2_rad)

    return {
        "frequency_ghz": best_pt.frequency_ghz,
        "incident_angle_deg": incident_angle_deg,
        "refractive_index_n2": n_metamaterial,
        "refracted_angle_deg": theta2_deg,
        "is_negative_refraction": theta2_deg < 0.0,
        "is_left_handed": best_pt.is_left_handed,
    }


def verify_bragg_bandgap(cell_size_mm: float = 10.0, host_eps_r: float = 2.2) -> dict:
    """Guión comprobable 3: la banda prohibida de un apilamiento periódico aparece en f_Bragg = c / (2·a·n_medio),
    con n_medio = (n_host + n_vacío)/2 para capas de igual espesor a/2.

    Se compara la fórmula con el mínimo de transmisión de un apilamiento 1D real
    (matriz de transferencia exacta, 12 periodos, capas alternas ε_host y 1 de espesor a/2).
    """
    a_m = cell_size_mm * 1e-3
    n_h, n_l = math.sqrt(host_eps_r), 1.0
    # Frecuencia de Bragg efectiva del periodo: a·n_medio = λ/2 con n_medio = (n_h + n_l)/2
    n_avg = 0.5 * (n_h + n_l)
    theo_bragg_hz = SPEED_OF_LIGHT / (2.0 * a_m * n_avg)
    theo_bragg_ghz = theo_bragg_hz * 1e-9

    def stack_transmission(f_hz: float, pairs: int = 12) -> float:
        k0 = 2.0 * math.pi * f_hz / SPEED_OF_LIGHT
        m = [[1 + 0j, 0j], [0j, 1 + 0j]]
        for n_j in (n_h, n_l) * pairs:
            delta = n_j * k0 * (a_m / 2.0)
            c, s = math.cos(delta), math.sin(delta)
            lay = [[c, -1j * s / n_j], [-1j * n_j * s, c]]
            m = [
                [m[0][0] * lay[0][0] + m[0][1] * lay[1][0], m[0][0] * lay[0][1] + m[0][1] * lay[1][1]],
                [m[1][0] * lay[0][0] + m[1][1] * lay[1][0], m[1][0] * lay[0][1] + m[1][1] * lay[1][1]],
            ]
        t = 2.0 / (m[0][0] + m[0][1] + m[1][0] + m[1][1])
        return abs(t) ** 2

    freqs = [(0.5 + 1.5 * i / 600) * theo_bragg_hz for i in range(601)]
    t_vals = [stack_transmission(f) for f in freqs]
    f_min_hz = freqs[t_vals.index(min(t_vals))]
    measured_ghz = f_min_hz * 1e-9
    rel_err = abs(measured_ghz - theo_bragg_ghz) / theo_bragg_ghz * 100.0

    return {
        "cell_size_mm": cell_size_mm,
        "host_permittivity": host_eps_r,
        "theoretical_bragg_frequency_ghz": theo_bragg_ghz,
        "measured_gap_center_ghz": measured_ghz,
        "min_transmission": min(t_vals),
        "relative_error_percent": rel_err,
        "wavelength_in_dielectric_mm": (cell_size_mm * 2.0),
        "is_verified": rel_err < 3.0 and min(t_vals) < 0.05,
    }
