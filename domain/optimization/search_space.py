"""Definición del espacio de búsqueda paramétrico y operadores de reparación geométrica (Capítulo 12)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class ParameterBounds:
    min_val: float
    max_val: float
    step: float


@dataclass
class SearchSpace:
    """Espacio de búsqueda tridimensional (a, d, w) con cuantización a la malla FDTD y restricciones físicas."""

    family: str = "square_ring"
    a_bounds: ParameterBounds = ParameterBounds(min_val=6.0, max_val=16.0, step=1.0)
    d_bounds: ParameterBounds = ParameterBounds(min_val=2.0, max_val=8.0, step=0.5)
    w_bounds: ParameterBounds = ParameterBounds(min_val=0.5, max_val=2.5, step=0.25)
    dx_mm: float = 0.5

    @property
    def span(self) -> np.ndarray:
        """Rango de búsqueda de cada gen [span_a, span_d, span_w]."""
        return np.array([
            self.a_bounds.max_val - self.a_bounds.min_val,
            self.d_bounds.max_val - self.d_bounds.min_val,
            self.w_bounds.max_val - self.w_bounds.min_val,
        ], dtype=np.float64)

    def quantize_value(self, val: float, bounds: ParameterBounds) -> float:
        """Cuantiza un valor al múltiplo más cercano del paso de resolución o malla."""
        step = max(bounds.step, self.dx_mm)
        q = round((val - bounds.min_val) / step) * step + bounds.min_val
        return float(np.clip(q, bounds.min_val, bounds.max_val))

    def is_feasible(self, ind: Sequence[float]) -> bool:
        """Verifica si el genoma (a, d, w) satisface estrictamente las leyes geométricas."""
        a, d, w = ind[0], ind[1], ind[2]
        if a < self.a_bounds.min_val or a > self.a_bounds.max_val:
            return False
        if d < self.d_bounds.min_val or d > self.d_bounds.max_val:
            return False
        if w < self.w_bounds.min_val or w > self.w_bounds.max_val:
            return False

        if self.family == "square_ring":
            # Para anillo cuadrado: hueco interior requiere (a - d) > 2*w y grosor positivo
            if (a - d) <= 2.0 * w + 0.1:
                return False
        elif self.family == "rod_array":
            # Para varillas: lado d < a
            if d >= a - 0.5:
                return False

        return True

    def repair_individual(self, ind: Sequence[float]) -> np.ndarray:
        """Proyecta un individuo al espacio factible cuantizado más cercano."""
        a = self.quantize_value(ind[0], self.a_bounds)
        d = self.quantize_value(ind[1], self.d_bounds)
        w = self.quantize_value(ind[2], self.w_bounds)

        if self.family == "square_ring":
            # Garantizar que (a - d) > 2*w
            max_d = a - 2.0 * w - 0.5
            if d > max_d:
                d = max(self.d_bounds.min_val, self.quantize_value(max_d, self.d_bounds))
            # Si aún viola, reducir w
            max_w = (a - d - 0.5) / 2.0
            if w > max_w:
                w = max(self.w_bounds.min_val, self.quantize_value(max_w, self.w_bounds))
        elif self.family == "rod_array":
            if d >= a:
                d = max(self.d_bounds.min_val, a - 1.0)
            w = d # En varillas cuadradas w es igual a d

        return np.array([a, d, w], dtype=np.float64)

    def repair(self, population: np.ndarray) -> np.ndarray:
        """Repara una matriz de individuos [pop_size, 3]."""
        repaired = np.empty_like(population)
        for i in range(len(population)):
            repaired[i] = self.repair_individual(population[i])
        return repaired

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Genera una población inicial uniforme de individuos factibles reproducibles."""
        pop = np.empty((size, 3), dtype=np.float64)
        count = 0
        while count < size:
            raw_a = rng.uniform(self.a_bounds.min_val, self.a_bounds.max_val)
            raw_d = rng.uniform(self.d_bounds.min_val, self.d_bounds.max_val)
            raw_w = rng.uniform(self.w_bounds.min_val, self.w_bounds.max_val)
            cand = self.repair_individual([raw_a, raw_d, raw_w])
            if self.is_feasible(cand):
                pop[count] = cand
                count += 1
        return pop
