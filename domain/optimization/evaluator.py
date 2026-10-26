"""Evaluador de individuos FDTD con caché de genomas cuantizados (Capítulo 12)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from domain.sweep import simulate_single_design, SweepDesignPoint
from domain.optimization.search_space import SearchSpace


@dataclass
class EvaluationCacheStats:
    total_requested: int = 0
    cache_hits: int = 0
    simulations_executed: int = 0

    @property
    def hit_rate_percent(self) -> float:
        return (self.cache_hits / max(self.total_requested, 1)) * 100.0


class FDTDEvaluator:
    """Evalúa la función objetivo F2 sobre candidatos (a, d, w) reutilizando soluciones previas vía caché."""

    def __init__(
        self,
        family: str = "square_ring",
        periods: int = 3,
        is_pec: bool = True,
        relative_permittivity: float = 1.0,
        dx_mm: float = 0.5,
        f0_ghz: float = 10.0,
        max_steps: int = 1400,
    ):
        self.family = family
        self.periods = periods
        self.is_pec = is_pec
        self.relative_permittivity = relative_permittivity
        self.dx_mm = dx_mm
        self.f0_ghz = f0_ghz
        self.max_steps = max_steps

        # Caché de resultados: key = (family, round(a, 2), round(d, 2), round(w, 2)) -> (f2_score, SweepDesignPoint)
        self._cache: dict[tuple, tuple[float, SweepDesignPoint]] = {}
        self.stats = EvaluationCacheStats()

    def _make_key(self, a_mm: float, d_mm: float, w_mm: float) -> tuple:
        return (
            self.family,
            round(float(a_mm), 2),
            round(float(d_mm), 2),
            round(float(w_mm), 2),
            self.is_pec,
            self.dx_mm,
        )

    def evaluate_individual(self, ind: Sequence[float], design_id: int = 0) -> tuple[float, SweepDesignPoint]:
        """Evalúa un individuo único (a, d, w). Si ya fue calculado, retorna de caché en 0 ms."""
        a, d, w = float(ind[0]), float(ind[1]), float(ind[2])
        key = self._make_key(a, d, w)
        self.stats.total_requested += 1

        if key in self._cache:
            self.stats.cache_hits += 1
            score, cached_pt = self._cache[key]
            # Clonar con el ID actual
            pt = SweepDesignPoint(
                design_id=design_id,
                a_mm=cached_pt.a_mm,
                d_mm=cached_pt.d_mm,
                w_mm=cached_pt.w_mm,
                periods=cached_pt.periods,
                family=cached_pt.family,
                is_pec=cached_pt.is_pec,
                relative_permittivity=cached_pt.relative_permittivity,
                f0_ghz=cached_pt.f0_ghz,
                t_f0=cached_pt.t_f0,
                t_f0_db=cached_pt.t_f0_db,
                r_f0=cached_pt.r_f0,
                t_pass_mean=cached_pt.t_pass_mean,
                rejection_depth_db=cached_pt.rejection_depth_db,
                f2_score=score,
            )
            return score, pt

        # Simulación física FDTD real
        self.stats.simulations_executed += 1
        pt, _, _, _ = simulate_single_design(
            a_mm=a,
            d_mm=d,
            w_mm=w,
            periods=self.periods,
            family_str=self.family,
            is_pec=self.is_pec,
            relative_permittivity=self.relative_permittivity,
            dx_mm=self.dx_mm,
            f0_ghz=self.f0_ghz,
            max_steps=self.max_steps,
            design_id=design_id,
        )

        score = pt.f2_score
        self._cache[key] = (score, pt)
        return score, pt

    def evaluate_batch(self, population: np.ndarray, base_id: int = 0) -> tuple[np.ndarray, list[SweepDesignPoint]]:
        """Evalúa un conjunto de individuos y retorna los puntajes de aptitud (fitness vector)."""
        pop_size = len(population)
        fitness = np.empty(pop_size, dtype=np.float64)
        points: list[SweepDesignPoint] = []

        for i in range(pop_size):
            score, pt = self.evaluate_individual(population[i], design_id=base_id + i + 1)
            fitness[i] = score
            points.append(pt)

        return fitness, points
