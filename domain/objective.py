"""Funciones objetivo y métricas de desempeño para filtros y metamateriales en banda X."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np

from domain.fdtd.monitors import PoyntingSpectrum


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """Resultado detallado de la evaluación de la función objetivo sobre una simulación FDTD."""

    objective_value: float              # Valor final F(p) (a minimizar)
    objective_type: str                 # "F1_pure_transmission" o "F2_bandstop_filter"
    t_f0: float                         # Transmisión en la frecuencia central f0 [0, 1]
    t_f0_db: float                      # Transmisión en f0 en decibelios [dB]
    t_pass_mean: float                  # Transmisión media en la banda de paso [0, 1]
    rejection_depth_db: float           # Profundidad de rechazo en dB: 10 * log10(T_pass_mean / max(T_f0, 1e-6))
    penalty: float                      # Penalización geométrica o de llenado C(p)
    is_valid_geometry: bool             # Cumple restricciones de fabricación
    bandwidth_3db_ghz: float            # Ancho de banda a -3 dB alrededor de f0 [GHz]
    q_factor: float                     # Factor de calidad Q = f0 / Delta_f


def evaluate_f1_pure_transmission(
    spectrum: PoyntingSpectrum,
    f0_ghz: float = 10.0,
) -> ObjectiveEvaluation:
    """Función Objetivo F1: Minimización pura de la transmisión en f0.

    NOTA DE RIGOR (Trampa Física):
    Con conductores perfectos (PEC), minimizar puramente T(f0) conduce a la
    solución trivial de cerrar la celda y formar una pared metálica continua
    (T ≈ 0 en todas las frecuencias). Sirve como caso didáctico de comparación.
    """
    freqs = np.asarray(spectrum.frequencies_ghz, dtype=np.float64)
    trans = np.asarray(spectrum.transmission, dtype=np.float64)

    # Interpolación en f0
    idx_closest = int(np.argmin(np.abs(freqs - f0_ghz)))
    t_f0 = float(trans[idx_closest])
    t_f0_db = float(10.0 * math.log10(max(t_f0, 1e-6)))

    return ObjectiveEvaluation(
        objective_value=round(t_f0, 6),
        objective_type="F1_pure_transmission",
        t_f0=round(t_f0, 5),
        t_f0_db=round(t_f0_db, 2),
        t_pass_mean=round(float(np.mean(trans)), 5),
        rejection_depth_db=0.0,
        penalty=0.0,
        is_valid_geometry=True,
        bandwidth_3db_ghz=0.0,
        q_factor=0.0,
    )


def evaluate_f2_bandstop_filter(
    spectrum: PoyntingSpectrum,
    f0_ghz: float = 10.0,
    delta_f_reject_ghz: float = 0.5,    # Semi-ancho de banda de rechazo: [f0 - delta, f0 + delta] = [9.5, 10.5] GHz
    w1: float = 0.60,                    # Peso para transmisión nula en f0: T(f0) -> 0
    w2: float = 0.30,                    # Peso para máxima transmisión en banda de paso: (1 - T_pass) -> 0
    w3: float = 0.10,                    # Peso para penalización geométrica
    penalty_value: float = 0.0,
    is_valid_geometry: bool = True,
) -> ObjectiveEvaluation:
    """Función Objetivo F2: Filtro rechaza-banda con selectividad de frecuencia.

    F2(p) = w1 * T(f0) + w2 * (1 - T_pass) + w3 * C(p)

    Exige simultáneamente:
      1. Rechazo profundo en la frecuencia de diseño f0 = 10 GHz (T(f0) << 1).
      2. Alta transparencia fuera de la banda de corte (T_pass ≈ 1).
      3. Penalización C(p) si el diseño viola límites físicos o fabricación.
    """
    if not is_valid_geometry:
        # Penalización severa para geometrías imposibles (e.g. grosor w > (a-d)/2)
        return ObjectiveEvaluation(
            objective_value=999.0,
            objective_type="F2_bandstop_filter",
            t_f0=1.0,
            t_f0_db=0.0,
            t_pass_mean=0.0,
            rejection_depth_db=0.0,
            penalty=1.0,
            is_valid_geometry=False,
            bandwidth_3db_ghz=0.0,
            q_factor=0.0,
        )

    freqs = np.asarray(spectrum.frequencies_ghz, dtype=np.float64)
    trans = np.asarray(spectrum.transmission, dtype=np.float64)

    # Identificar banda de rechazo [f0 - delta, f0 + delta]
    reject_mask = (freqs >= (f0_ghz - delta_f_reject_ghz)) & (freqs <= (f0_ghz + delta_f_reject_ghz))
    pass_mask = ~reject_mask

    idx_f0 = int(np.argmin(np.abs(freqs - f0_ghz)))
    t_f0 = float(trans[idx_f0])
    t_f0_db = float(10.0 * math.log10(max(t_f0, 1e-6)))

    t_pass_mean = float(np.mean(trans[pass_mask])) if np.any(pass_mask) else 0.0

    # Profundidad de rechazo respecto a la banda de paso
    rejection_ratio = max(t_pass_mean, 1e-4) / max(t_f0, 1e-6)
    rejection_depth_db = float(10.0 * math.log10(rejection_ratio))

    # Cálculo de ancho de banda de rechazo a -3 dB
    threshold_3db = max(t_f0 * 2.0, t_f0 + 0.15)
    sub_threshold = freqs[reject_mask & (trans < threshold_3db)]
    bw_ghz = float(np.ptp(sub_threshold)) if len(sub_threshold) > 1 else 0.2
    q_factor = float(f0_ghz / bw_ghz) if bw_ghz > 0 else 0.0

    # Valor de la función objetivo multiobjetivo
    f2_val = (
        w1 * t_f0
        + w2 * max(0.0, 1.0 - t_pass_mean)
        + w3 * penalty_value
    )

    return ObjectiveEvaluation(
        objective_value=round(f2_val, 6),
        objective_type="F2_bandstop_filter",
        t_f0=round(t_f0, 5),
        t_f0_db=round(t_f0_db, 2),
        t_pass_mean=round(t_pass_mean, 5),
        rejection_depth_db=round(rejection_depth_db, 2),
        penalty=round(penalty_value, 4),
        is_valid_geometry=True,
        bandwidth_3db_ghz=round(bw_ghz, 3),
        q_factor=round(q_factor, 2),
    )


def compute_spearman_rank_correlation(ranks_a: list[float], ranks_b: list[float]) -> float:
    """Calcula el coeficiente de correlación de rangos de Spearman entre dos evaluaciones (e.g. coarse vs fine)."""
    n = len(ranks_a)
    if n < 2:
        return 1.0

    def get_ranks(vals: list[float]) -> np.ndarray:
        order = np.argsort(vals)
        r = np.empty(n, dtype=np.float64)
        r[order] = np.arange(n)
        return r

    r_a = get_ranks(ranks_a)
    r_b = get_ranks(ranks_b)

    d_sq = np.sum((r_a - r_b) ** 2)
    rho = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1.0))
    return float(rho)
