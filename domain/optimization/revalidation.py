"""Módulo de re-validación en malla fina (Δx/2 = 0.25 mm) del diseño óptimo (Capítulo 12)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from domain.sweep import simulate_single_design, SweepDesignPoint


@dataclass
class RevalidationReport:
    """Reporte formal de re-validación de malla fina vs malla rápida."""
    design_id: int
    a_mm: float
    d_mm: float
    w_mm: float
    periods: int
    family: str
    f0_ghz: float

    # Resultados en malla rápida (dx = 0.5 mm)
    dx_fast_mm: float
    t_f0_fast: float
    t_f0_fast_db: float
    f2_fast: float
    time_fast_s: float

    # Resultados en malla fina (dx = 0.25 mm)
    dx_fine_mm: float
    t_f0_fine: float
    t_f0_fine_db: float
    f2_fine: float
    time_fine_s: float

    # Comparativa y criterio de rigor físico
    delta_t_db: float
    delta_t_linear: float
    passed_tolerance: bool  # |ΔT(f0)| < 2.0 dB
    tolerance_threshold_db: float

    # Curvas espectrales completas para overlay en UI
    freqs_ghz: list[float]
    t_fast_spectrum: list[float]
    t_fine_spectrum: list[float]
    r_fast_spectrum: list[float]
    r_fine_spectrum: list[float]


def revalidate_winner_design(
    a_mm: float,
    d_mm: float,
    w_mm: float,
    periods: int = 3,
    family_str: str = "square_ring",
    is_pec: bool = True,
    relative_permittivity: float = 1.0,
    f0_ghz: float = 10.0,
    tolerance_db: float = 2.0,
) -> RevalidationReport:
    """Simula el diseño óptimo tanto en malla rápida (0.50 mm) como en malla fina de validación (0.25 mm),
    verificando si la respuesta electromagnética converge con un error menor a 2 dB.
    """
    import time

    # 1. Simulación en malla rápida
    t0_fast = time.perf_counter()
    pt_fast, freqs_fast, t_fast_spec, r_fast_spec = simulate_single_design(
        a_mm=a_mm,
        d_mm=d_mm,
        w_mm=w_mm,
        periods=periods,
        family_str=family_str,
        is_pec=is_pec,
        relative_permittivity=relative_permittivity,
        dx_mm=0.50,
        f0_ghz=f0_ghz,
        max_steps=1400,
        design_id=1,
    )
    dt_fast = time.perf_counter() - t0_fast

    # 2. Simulación en malla fina (Δx = 0.25 mm, mayor resolución espacial y temporal)
    t0_fine = time.perf_counter()
    pt_fine, freqs_fine, t_fine_spec, r_fine_spec = simulate_single_design(
        a_mm=a_mm,
        d_mm=d_mm,
        w_mm=w_mm,
        periods=periods,
        family_str=family_str,
        is_pec=is_pec,
        relative_permittivity=relative_permittivity,
        dx_mm=0.25,
        f0_ghz=f0_ghz,
        max_steps=2200,
        design_id=2,
    )
    dt_fine = time.perf_counter() - t0_fine

    # 3. Interpolar la curva fina a los puntos de frecuencia de la curva rápida para coincidencia 1 a 1
    t_fine_interp = np.interp(freqs_fast, freqs_fine, t_fine_spec)
    r_fine_interp = np.interp(freqs_fast, freqs_fine, r_fine_spec)

    # 4. Cálculo del error en f0
    delta_db = abs(pt_fine.t_f0_db - pt_fast.t_f0_db)
    delta_lin = abs(pt_fine.t_f0 - pt_fast.t_f0)
    passed = bool(delta_db <= tolerance_db)

    return RevalidationReport(
        design_id=1,
        a_mm=a_mm,
        d_mm=d_mm,
        w_mm=w_mm,
        periods=periods,
        family=family_str,
        f0_ghz=f0_ghz,
        dx_fast_mm=0.50,
        t_f0_fast=round(float(pt_fast.t_f0), 5),
        t_f0_fast_db=round(float(pt_fast.t_f0_db), 2),
        f2_fast=round(float(pt_fast.f2_score), 5),
        time_fast_s=round(float(dt_fast), 3),
        dx_fine_mm=0.25,
        t_f0_fine=round(float(pt_fine.t_f0), 5),
        t_f0_fine_db=round(float(pt_fine.t_f0_db), 2),
        f2_fine=round(float(pt_fine.f2_score), 5),
        time_fine_s=round(float(dt_fine), 3),
        delta_t_db=round(float(delta_db), 3),
        delta_t_linear=round(float(delta_lin), 5),
        passed_tolerance=passed,
        tolerance_threshold_db=tolerance_db,
        freqs_ghz=[round(float(f), 3) for f in freqs_fast],
        t_fast_spectrum=[round(float(v), 5) for v in t_fast_spec],
        t_fine_spectrum=[round(float(v), 5) for v in t_fine_interp],
        r_fast_spectrum=[round(float(v), 5) for v in r_fast_spec],
        r_fine_spectrum=[round(float(v), 5) for v in r_fine_interp],
    )
